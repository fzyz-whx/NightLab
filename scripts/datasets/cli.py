"""Dataset metadata, integrity checks, and explicitly selected sample retrieval."""

import argparse
import hashlib
from http.client import HTTPException
import json
from pathlib import Path, PurePosixPath
import re
import sys

ROOT = Path(__file__).resolve().parents[2]
CHUNK_SIZE = 1024 * 1024


class ManifestError(ValueError):
    pass


def raw_path(root, value):
    """Require portable repository-relative paths and resolve existing symlinks."""
    if not isinstance(value, str) or "\\" in value or ":" in value:
        raise ManifestError("path must be a portable path under data/raw")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or path.parts[:2] != ("data", "raw") or len(path.parts) < 3:
        raise ManifestError("path must stay under data/raw")
    # Also reject a data/raw symlink that redirects the whole allowed tree.
    root = Path(root).resolve()
    allowed = (root / "data" / "raw").resolve()
    if not allowed.is_relative_to(root / "data" / "raw"):
        raise ManifestError("data/raw resolves outside its repository location")
    target = (root / path).resolve()
    if not target.is_relative_to(allowed):
        raise ManifestError("path resolves outside data/raw")
    return target


def load_manifest(path, root):
    try:
        manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ManifestError(f"cannot read manifest: {exc}") from exc
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 1:
        raise ManifestError("expected manifest schema_version 1")
    for key in ("id", "version", "license", "sources", "reported_size", "inventory_complete", "files"):
        if key not in manifest:
            raise ManifestError(f"missing manifest field: {key}")
    if not isinstance(manifest["files"], list):
        raise ManifestError("files must be a list")
    seen = set()
    for entry in manifest["files"]:
        if not isinstance(entry, dict):
            raise ManifestError("file entry must be an object")
        target = raw_path(root, entry.get("path"))
        if target in seen:
            raise ManifestError("duplicate resolved file path")
        seen.add(target)
        size = entry.get("exact_bytes")
        if size is not None and (type(size) is not int or size < 0):
            raise ManifestError("exact_bytes must be a nonnegative integer or null")
        checksums = entry.get("official_checksums", {})
        if not isinstance(checksums, dict):
            raise ManifestError("official_checksums must be an object")
        for algorithm, digest in checksums.items():
            length = {"sha256": 64, "md5": 32}.get(algorithm)
            if length is None or not isinstance(digest, str) or not re.fullmatch(f"[0-9a-fA-F]{{{length}}}", digest):
                raise ManifestError("invalid official checksum")
    return manifest


def file_state(root, entry):
    path = raw_path(root, entry["path"])
    state = {"path": entry["path"], "present": path.is_file()}
    if state["present"]:
        state["actual_bytes"] = path.stat().st_size
    return state


def verify(root, manifest):
    results = []
    for entry in manifest["files"]:
        state = file_state(root, entry)
        state["errors"] = []
        state["official_sha256_verified"] = False
        if not state["present"]:
            state["errors"].append("missing regular file")
        else:
            path = raw_path(root, entry["path"])
            expected = entry.get("exact_bytes")
            if expected is not None and state["actual_bytes"] != expected:
                state["errors"].append("exact size mismatch")
            if state["actual_bytes"] == 0:
                state["errors"].append("empty data file")
            # Stream even large recordings; never read the entire dataset into RAM.
            digests = {"sha256": hashlib.sha256()}
            if "md5" in entry.get("official_checksums", {}):
                digests["md5"] = hashlib.md5(usedforsecurity=False)
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(CHUNK_SIZE), b""):
                    for digest in digests.values():
                        digest.update(chunk)
            state["local_checksums"] = {name: digest.hexdigest() for name, digest in digests.items()}
            official = entry.get("official_checksums", {})
            for name, digest in official.items():
                if state["local_checksums"][name] != digest.lower():
                    state["errors"].append(f"official {name} mismatch")
            state["official_sha256_verified"] = "sha256" in official and not state["errors"]
            state["checksum_provenance"] = "official checksum compared" if official else "local SHA-256 only; official authenticity unverified"
        state["ok"] = not state["errors"]
        results.append(state)
    ok = bool(results) and all(item["ok"] for item in results)
    return {"dataset": manifest["id"], "scope": "listed files only; no format, labels, or time alignment validation", "inventory_complete": manifest["inventory_complete"], "ok": ok, "official_sha256_verified": bool(results) and all(item["official_sha256_verified"] for item in results), "files": results, "errors": [] if results else ["no concrete files available to verify"]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", nargs="?", choices=("audit", "plan", "verify", "sample", "download"), default="audit")
    parser.add_argument("--dataset", help="manifest id (default: all)")
    parser.add_argument("--file", help="exact manifest path for one sample file")
    parser.add_argument("--execute", action="store_true", help="enable sample network and writes")
    parser.add_argument("--confirm-large-download", action="store_true", help="confirm the explicitly selected download file")
    args = parser.parse_args(argv)
    directory = ROOT / "data" / "manifests"
    try:
        if args.command in {"sample", "download"} and (not args.dataset or not args.file):
            raise ManifestError("retrieval requires --dataset and --file")
        if args.command not in {"sample", "download"} and (args.file or args.execute):
            raise ManifestError("--file and --execute are only valid for sample/download")
        if args.confirm_large_download and args.command != 'download':
            raise ManifestError('--confirm-large-download is only valid for download')
        if args.command == 'download' and args.execute and not args.confirm_large_download:
            raise ManifestError('download execution requires --confirm-large-download after user authorization')
        if args.dataset and not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", args.dataset):
            raise ManifestError("invalid dataset id")
        paths = [directory / f"{args.dataset}.json"] if args.dataset else sorted(directory.glob("*.json"))
        if not paths:
            raise ManifestError("no dataset manifests found")
        manifests = [load_manifest(path, ROOT) for path in paths]
        if args.command in {"sample", "download"}:
            from .sample import sample
            output = sample(ROOT, manifests[0], args.file, args.execute,
                            confirmed_large=args.command == 'download' and (not args.execute or args.confirm_large_download))
        elif args.command == "plan":
            output = {"mode": "offline read-only plan", "network": False, "writes": False, "download_policy": "Retrieval selects one audited file, default dry-run. sample caps at 500 MB; download execution additionally requires --confirm-large-download after user authorization. No bulk download or prepare implemented. BIDSleep is optional.", "datasets": manifests}
        elif args.command == "audit":
            output = {"mode": "offline read-only metadata and local inventory", "datasets": [{**manifest, "local_files": [file_state(ROOT, entry) for entry in manifest["files"]]} for manifest in manifests]}
        else:
            results = [verify(ROOT, manifest) for manifest in manifests]
            output = {"mode": "offline read-only integrity verification", "ok": all(item["ok"] for item in results), "datasets": results}
        print(json.dumps(output, ensure_ascii=False, indent=2))
        return 1 if args.command == "verify" and not output["ok"] else 0
    except (ValueError, OSError, HTTPException) as exc:
        print(f"dataset error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("sample interrupted; partial file preserved", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
