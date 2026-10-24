import copy
import hashlib
import io
from http.client import IncompleteRead
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from scripts.datasets.sample import sample, OfficialRedirect
from scripts.datasets import cli


class Response(io.BytesIO):
    def __init__(self, body=b"abcdef", status=200, headers=None, url="https://physionet.org/files/sample"):
        super().__init__(body)
        self.status = status
        self.headers = {"Content-Length": str(len(body))} if headers is None else headers
        self.url = url

    def geturl(self):
        return self.url


class SampleDownloadTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.selected = "data/raw/sample/signal.txt"
        self.target = self.root / self.selected
        self.partial = self.root / (self.selected + ".part")
        self.entry = {"path": self.selected, "source": "https://physionet.org/files/sample",
                      "source_verified": True, "exact_bytes": 6,
                      "official_checksums": {"sha256": hashlib.sha256(b"abcdef").hexdigest()},
                      "checksum_source": "https://physionet.org/files/SHA256SUMS.txt"}
        self.manifest = {"id": "fixture", "license": {"verified": True,
                         "source": "https://physionet.org/LICENSE.txt"}, "files": [self.entry]}

    def run_download(self, response):
        opener = Mock()
        opener.open.return_value = response
        with patch("scripts.datasets.sample.build_opener", return_value=opener):
            result = sample(self.root, self.manifest, self.selected, True)
        return result, opener

    def seed_partial(self, body=b"abc"):
        self.partial.parent.mkdir(parents=True)
        self.partial.write_bytes(body)

    def test_dry_run_never_opens_or_writes(self):
        with patch("scripts.datasets.sample.build_opener", side_effect=AssertionError("network forbidden")):
            result = sample(self.root, self.manifest, self.selected)
        self.assertFalse(result["network"])
        self.assertEqual(list(self.root.iterdir()), [])

    def test_success_publishes_only_verified_file(self):
        result, _ = self.run_download(Response())
        self.assertEqual(self.target.read_bytes(), b"abcdef")
        self.assertFalse(self.partial.exists())
        self.assertEqual(result["status"], "downloaded and officially verified")

    def test_resume_checks_range_and_appends(self):
        self.seed_partial()
        result, opener = self.run_download(Response(b"def", 206, {"Content-Range": "bytes 3-5/6", "Content-Length": "3"}))
        self.assertEqual(opener.open.call_args.args[0].get_header("Range"), "bytes=3-")
        self.assertEqual(result["resumed_bytes"], 3)
        self.assertEqual(self.target.read_bytes(), b"abcdef")

    def test_zero_length_partial_can_restart(self):
        self.seed_partial(b"")
        result, opener = self.run_download(Response())
        self.assertIsNone(opener.open.call_args.args[0].get_header('Range'))
        self.assertEqual(result['resumed_bytes'], 0)
        self.assertEqual(self.target.read_bytes(), b'abcdef')

    def test_interrupt_flushes_received_bytes_for_next_resume(self):
        response = Response()
        response.read1 = Mock(side_effect=[b'abc', KeyboardInterrupt()])
        with self.assertRaises(KeyboardInterrupt):
            self.run_download(response)
        self.assertEqual(self.partial.read_bytes(), b'abc')
        self.assertFalse(self.target.exists())

    def test_http_interruption_preserves_received_prefix(self):
        response = Response()
        response.read1 = Mock(side_effect=[b'abc',IncompleteRead(b'',3)])
        with self.assertRaises(IncompleteRead):
            self.run_download(response)
        self.assertEqual(self.partial.read_bytes(),b'abc')
        self.assertFalse(self.target.exists())

    def test_bad_resume_responses_preserve_partial(self):
        self.seed_partial()
        for status, headers in [(200, {}), (206, {"Content-Range": "bytes 0-2/6"}),
                                (206, {"Content-Range": "bytes 3-5/7"})]:
            with self.subTest(status=status, headers=headers), self.assertRaises(ValueError):
                self.run_download(Response(b"def", status, headers))
            self.assertEqual(self.partial.read_bytes(), b"abc")
            self.assertFalse(self.target.exists())

    def test_wrong_checksum_never_publishes(self):
        with self.assertRaises(ValueError):
            self.run_download(Response(b"abcdeg"))
        self.assertEqual(self.partial.read_bytes(), b"abcdeg")
        self.assertFalse(self.target.exists())

    def test_truncated_and_oversize_responses_never_publish(self):
        for body in (b"abc", b"abcdefg"):
            if self.partial.exists():
                self.partial.unlink()
            with self.subTest(body=body), self.assertRaises(ValueError):
                self.run_download(Response(body, headers={}))
            self.assertFalse(self.target.exists())
            self.assertLessEqual(self.partial.stat().st_size, 6)

    def test_metadata_gates_before_network(self):
        mutations = [lambda m: m["license"].update(verified=False),
                     lambda m: m["files"][0].update(source_verified=False),
                     lambda m: m["files"][0].update(exact_bytes=None),
                     lambda m: m["files"][0].update(exact_bytes=500000001),
                     lambda m: m["files"][0].update(source="https://physionet.org/file (description)"),
                     lambda m: m["files"][0].update(source="http://physionet.org/file"),
                     lambda m: m["files"][0].update(official_checksums={})]
        for mutate in mutations:
            manifest = copy.deepcopy(self.manifest)
            mutate(manifest)
            with self.assertRaises(ValueError), patch("scripts.datasets.sample.build_opener", side_effect=AssertionError("network forbidden")):
                sample(self.root, manifest, self.selected, True)

    def test_large_file_plan_requires_download_path_but_never_fetches(self):
        self.entry['exact_bytes'] = 710738976
        with self.assertRaises(ValueError):
            sample(self.root, self.manifest, self.selected)
        with patch('scripts.datasets.sample.build_opener', side_effect=AssertionError('network forbidden')):
            result = sample(self.root, self.manifest, self.selected, confirmed_large=True)
        self.assertFalse(result['network'])
        self.assertEqual(result['exact_bytes'], 710738976)

    def test_large_execution_without_confirmation_is_rejected(self):
        import contextlib
        with contextlib.redirect_stderr(io.StringIO()), patch('scripts.datasets.sample.sample', side_effect=AssertionError('must not execute')):
            self.assertEqual(cli.main(['download', '--dataset', 'fixture', '--file', self.selected, '--execute']), 2)
            self.assertEqual(cli.main(['sample', '--dataset', 'fixture', '--file', self.selected, '--confirm-large-download']), 2)

    def test_existing_verified_file_does_not_download(self):
        self.target.parent.mkdir(parents=True)
        self.target.write_bytes(b"abcdef")
        with patch("scripts.datasets.sample.build_opener", side_effect=AssertionError("network forbidden")):
            result = sample(self.root, self.manifest, self.selected, True)
        self.assertEqual(result["status"], "already verified")

    def test_existing_mismatch_is_not_overwritten(self):
        self.target.parent.mkdir(parents=True)
        self.target.write_bytes(b"user file")
        with self.assertRaises(ValueError):
            self.run_download(Response())
        self.assertEqual(self.target.read_bytes(), b"user file")

    def test_disk_shortage_rejected_before_network(self):
        with patch("scripts.datasets.sample.shutil.disk_usage", return_value=Mock(free=0)):
            with self.assertRaisesRegex(ValueError, "disk space"):
                self.run_download(Response())
        self.assertFalse(self.partial.exists())

    def test_redirect_and_encoded_response_rejected(self):
        for response in (Response(url="https://other.example/data"),
                         Response(headers={"Content-Encoding": "gzip"})):
            with self.assertRaises(ValueError):
                self.run_download(response)
        handler = OfficialRedirect("physionet.org")
        with self.assertRaises(ValueError):
            handler.redirect_request(None, None, 302, "", {}, "http://physionet.org/data")


if __name__ == "__main__":
    unittest.main()
