"""Explicit, bounded single-file sample retrieval (standard library only)."""

import hashlib
import os
from pathlib import Path
import re
import shutil
import time
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

MAX_BYTES = 500_000_000
CHUNK_SIZE = 64 * 1024
SOCKET_TIMEOUT = 30
TOTAL_TIMEOUT = 300


def https_url(value):
    if not isinstance(value, str) or any(c.isspace() for c in value):
        raise ValueError("source must be an unannotated HTTPS URL")
    parsed = urlsplit(value)
    if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password or parsed.fragment:
        raise ValueError("source must be an unannotated HTTPS URL")
    return parsed


class OfficialRedirect(HTTPRedirectHandler):
    def __init__(self, host):
        self.host = host

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        if https_url(newurl).hostname != self.host:
            raise ValueError("redirect leaves audited official host")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def matching_file(path, entry):
    if not path.is_file() or path.stat().st_size != entry["exact_bytes"]:
        return False
    digests = {name: hashlib.new(name, usedforsecurity=False) for name in entry["official_checksums"]}
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(CHUNK_SIZE), b""):
            for digest in digests.values():
                digest.update(chunk)
    return all(digests[name].hexdigest() == value.lower() for name, value in entry["official_checksums"].items())


def sample(root, manifest, selected, execute=False, confirmed_large=False, progress=None, stop_requested=None):
    # Imported here to keep the CLI's manifest/path validation authoritative.
    from .cli import raw_path

    entries = [entry for entry in manifest["files"] if entry["path"] == selected]
    if len(entries) != 1:
        raise ValueError("--file must exactly select one manifest path")
    entry = entries[0]
    license_info = manifest["license"]
    if not isinstance(license_info, dict) or license_info.get("verified") is not True:
        raise ValueError("sample requires explicitly verified dataset license")
    https_url(license_info.get("source"))
    source = https_url(entry.get("source"))
    if entry.get("source_verified") is not True:
        raise ValueError("sample requires explicitly verified official file source")
    https_url(entry.get("checksum_source"))
    size = entry.get("exact_bytes")
    if type(size) is not int or size <= 0:
        raise ValueError("download requires positive known exact_bytes")
    if size > MAX_BYTES and not confirmed_large:
        raise ValueError("sample size exceeds 500000000; use download with explicit large-download confirmation")
    if not entry.get("official_checksums"):
        raise ValueError("sample requires official SHA-256 or MD5")
    target = raw_path(root, selected)
    partial = raw_path(root, selected + ".part")
    # Reject aliases rather than mutating the destination of a file symlink.
    if (Path(root) / selected).is_symlink() or (Path(root) / (selected + ".part")).is_symlink():
        raise ValueError("sample target and partial must not be symlinks")
    operation = 'confirmed download' if confirmed_large else 'sample'
    result = {"mode": operation + (" execute" if execute else " dry-run"), "dataset": manifest["id"],
              "path": selected, "source": entry["source"], "exact_bytes": size,
              "official_checksums": entry["official_checksums"], "network": execute, "writes": execute}
    if not execute:
        return result
    if stop_requested is not None and stop_requested():
        raise InterruptedError('retrieval cancelled; partial preserved')
    if target.exists():
        if not matching_file(target, entry):
            raise ValueError("existing target differs; refusing overwrite")
        return {**result, "status": "already verified", "network": False, "writes": False}
    if partial.exists() and not partial.is_file():
        raise ValueError("partial is not a regular file")
    offset = partial.stat().st_size if partial.exists() else 0
    if offset > size:
        raise ValueError("partial exceeds exact_bytes; preserve for inspection")
    ancestor = target.parent
    while not ancestor.exists():
        ancestor = ancestor.parent
    if shutil.disk_usage(ancestor).free < size - offset:
        raise ValueError("insufficient disk space")
    if offset < size:
        headers = {"Accept-Encoding": "identity"}
        if offset:
            headers["Range"] = f"bytes={offset}-"
        started = time.monotonic()
        opener = build_opener(OfficialRedirect(source.hostname))
        with opener.open(Request(entry["source"], headers=headers), timeout=SOCKET_TIMEOUT) as response:
            if https_url(response.geturl()).hostname != source.hostname:
                raise ValueError("response leaves audited official host")
            status = response.status
            if offset or status == 206:
                match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", response.headers.get("Content-Range", ""))
                if status != 206 or not match or tuple(map(int, match.groups())) != (offset, size - 1, size):
                    raise ValueError("resume requires matching HTTP 206 Content-Range; partial preserved")
            elif status != 200:
                raise ValueError(f"unexpected HTTP status {status}")
            length = response.headers.get("Content-Length")
            if length is not None and (not length.isdigit() or int(length) != size - offset):
                raise ValueError("Content-Length differs from exact remaining bytes")
            if response.headers.get("Content-Encoding", "identity").lower() != "identity":
                raise ValueError("encoded response cannot be resumed safely")
            target.parent.mkdir(parents=True, exist_ok=True)
            with partial.open("ab" if partial.exists() else "xb") as stream:
                total = offset
                while True:
                    if stop_requested is not None and stop_requested():
                        raise InterruptedError('retrieval cancelled; partial preserved')
                    if time.monotonic() - started > TOTAL_TIMEOUT:
                        raise TimeoutError("sample deadline exceeded; partial preserved")
                    read = getattr(response, "read1", response.read)
                    chunk = read(min(CHUNK_SIZE, size - total + 1))
                    if not chunk:
                        break
                    if total + len(chunk) > size:
                        raise ValueError("response exceeds exact_bytes; partial preserved")
                    stream.write(chunk)
                    total += len(chunk)
                    if progress is not None:
                        progress(total,size)
                stream.flush()
                os.fsync(stream.fileno())
    if not matching_file(partial, entry):
        raise ValueError("download size or official checksum mismatch; partial preserved")
    # Atomic publication with exclusive destination creation, including on POSIX.
    # A racing existing target can never be silently overwritten.
    os.link(partial, target)
    partial.unlink()
    return {**result, "status": "downloaded and officially verified", "resumed_bytes": offset}
