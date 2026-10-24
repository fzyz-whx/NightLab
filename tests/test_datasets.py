import contextlib
import copy
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.datasets import cli


class DatasetToolsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "data/raw").mkdir(parents=True)
        self.content = b"a small signal fixture\n"
        self.entry = {"path": "data/raw/fixture/signal.bin", "exact_bytes": len(self.content), "official_checksums": {"sha256": hashlib.sha256(self.content).hexdigest(), "md5": hashlib.md5(self.content).hexdigest()}}
        self.manifest = {"schema_version": 1, "id": "fixture", "version": "1", "license": {"name": "test"}, "sources": [], "reported_size": "rounded only", "inventory_complete": False, "files": [self.entry]}

    def create_file(self, content=None):
        path = self.root / self.entry["path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(self.content if content is None else content)
        return path

    def save_manifest(self, manifest=None):
        directory = self.root / "data/manifests"
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "fixture.json"
        path.write_text(json.dumps(self.manifest if manifest is None else manifest), encoding="utf-8")
        return path

    def test_missing_data_and_empty_inventory_fail(self):
        self.assertFalse(cli.verify(self.root, self.manifest)["ok"])
        self.manifest["files"] = []
        self.assertFalse(cli.verify(self.root, self.manifest)["ok"])

    def test_streaming_hashes_match_and_detect_corruption(self):
        path = self.create_file()
        with patch.object(cli, "CHUNK_SIZE", 3):
            result = cli.verify(self.root, self.manifest)
        self.assertTrue(result["ok"])
        self.assertTrue(result["official_sha256_verified"])
        self.assertEqual(result["files"][0]["local_checksums"], self.entry["official_checksums"])
        path.write_bytes(b"x" * len(self.content))
        errors = cli.verify(self.root, self.manifest)["files"][0]["errors"]
        self.assertIn("official sha256 mismatch", errors)
        self.assertIn("official md5 mismatch", errors)

    def test_size_mismatch_and_empty_file_fail(self):
        self.create_file(b"")
        errors = cli.verify(self.root, self.manifest)["files"][0]["errors"]
        self.assertIn("exact size mismatch", errors)
        self.assertIn("empty data file", errors)

    def test_local_digest_has_explicit_provenance(self):
        self.create_file()
        self.entry["exact_bytes"] = None
        self.entry["official_checksums"] = {}
        result = cli.verify(self.root, self.manifest)
        self.assertTrue(result["ok"])
        self.assertFalse(result["official_sha256_verified"])
        self.assertIn("official authenticity unverified", result["files"][0]["checksum_provenance"])
        self.assertIn("listed files only", result["scope"])
        self.assertFalse(result["inventory_complete"])

    def test_reject_unsafe_paths(self):
        for value in ("../signal", "/data/raw/signal", "C:/data/raw/signal", "data/raw/../../secret", "data/raw", "data/processed/signal", "data\\raw\\signal", None):
            with self.subTest(value=value), self.assertRaises(cli.ManifestError):
                cli.raw_path(self.root, value)

    def test_symlink_escape_rejected(self):
        outside = self.root / "outside"
        outside.mkdir()
        link = self.root / "data/raw/link"
        try:
            link.symlink_to(outside, target_is_directory=True)
        except OSError as exc:
            self.skipTest(f"OS does not permit symlink creation: {exc}")
        with self.assertRaises(cli.ManifestError):
            cli.raw_path(self.root, "data/raw/link/signal")

    def test_resolved_escape_rejected_without_os_symlink_privileges(self):
        # Windows may disallow creating symbolic links; exercise the resolved
        # path boundary deterministically as well as the real-link test above.
        original_resolve = Path.resolve
        raw = self.root / "data/raw"
        target = raw / "link/signal"

        def redirected(path, *args, **kwargs):
            if path == target:
                return self.root / "outside/signal"
            return original_resolve(path, *args, **kwargs)

        with patch.object(Path, "resolve", redirected), self.assertRaises(cli.ManifestError):
            cli.raw_path(self.root, "data/raw/link/signal")

        def raw_redirected(path, *args, **kwargs):
            if path == raw:
                return self.root / "outside"
            return original_resolve(path, *args, **kwargs)

        with patch.object(Path, "resolve", raw_redirected), self.assertRaises(cli.ManifestError):
            cli.raw_path(self.root, "data/raw/signal")

    def test_manifest_rejects_invalid_checksums_sizes_and_duplicates(self):
        for mutation in (lambda m: m["files"][0].update(exact_bytes=True), lambda m: m["files"][0].update(official_checksums={"sha256": "fake"}), lambda m: m["files"].append(copy.deepcopy(m["files"][0]))):
            manifest = copy.deepcopy(self.manifest)
            mutation(manifest)
            with self.assertRaises(cli.ManifestError):
                cli.load_manifest(self.save_manifest(manifest), self.root)

    def test_default_audit_and_plan_do_not_write_or_hash(self):
        self.save_manifest()
        before = sorted(str(p.relative_to(self.root)) for p in self.root.rglob("*"))
        for args in ([], ["plan"]):
            with patch.object(cli, "ROOT", self.root), patch.object(cli, "verify", side_effect=AssertionError("must not hash")), contextlib.redirect_stdout(io.StringIO()) as output:
                self.assertEqual(cli.main(args), 0)
                self.assertIn("offline", json.loads(output.getvalue())["mode"])
        self.assertEqual(before, sorted(str(p.relative_to(self.root)) for p in self.root.rglob("*")))

    def test_verify_exit_status_missing_and_present(self):
        self.save_manifest()
        with patch.object(cli, "ROOT", self.root), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(cli.main(["verify", "--dataset", "fixture"]), 1)
            self.create_file()
            self.assertEqual(cli.main(["verify", "--dataset", "fixture"]), 0)

    def test_bad_manifest_and_dataset_id_fail(self):
        with patch.object(cli, "ROOT", self.root), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(cli.main(["audit", "--dataset", "../secret"]), 2)
            self.assertEqual(cli.main(["audit", "--dataset", "absent"]), 2)

    def test_actual_repository_manifests_load(self):
        root = Path(__file__).resolve().parents[1]
        paths = sorted((root / "data/manifests").glob("*.json"))
        self.assertTrue({"sleepaccel", "sciencedb-19070", "bidsleep", "psg-audio"}.issubset({p.stem for p in paths}))
        for path in paths:
            with self.subTest(manifest=path.name):
                manifest = cli.load_manifest(path, root)
                self.assertEqual(manifest["id"], path.stem)
                self.assertFalse(manifest["inventory_complete"])


if __name__ == "__main__":
    unittest.main()
