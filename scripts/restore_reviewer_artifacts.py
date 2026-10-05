"""Safely restore and verify a SAGE-QA reviewer artifact package."""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any, Sequence


MANIFEST_NAME = "REVIEWER_ARTIFACT_MANIFEST.json"


def safe_relative(value: str) -> Path:
    pure = PurePosixPath(value)
    if pure.is_absolute() or ".." in pure.parts or not pure.parts:
        raise ValueError(f"Unsafe package member: {value}")
    return Path(*pure.parts)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def restore(package: Path, repo_root: Path) -> dict[str, Any]:
    with zipfile.ZipFile(package) as archive:
        manifest = json.loads(archive.read(MANIFEST_NAME).decode("utf-8"))
        declared = {entry["path"]: entry for entry in manifest["files"]}
        payload = {name for name in archive.namelist() if name != MANIFEST_NAME}
        if payload != set(declared):
            raise ValueError("Package payload differs from its manifest")
        restored = 0
        reused = 0
        for relative, entry in declared.items():
            target = repo_root / safe_relative(relative)
            if target.is_file():
                if target.stat().st_size == entry["size_bytes"] and sha256(target) == entry["sha256"]:
                    reused += 1
                    continue
                raise FileExistsError(f"Existing file differs from package: {target}")
            target.parent.mkdir(parents=True, exist_ok=True)
            digest = hashlib.sha256()
            size = 0
            with archive.open(relative) as source, target.open("xb") as output:
                for chunk in iter(lambda: source.read(1024 * 1024), b""):
                    output.write(chunk)
                    digest.update(chunk)
                    size += len(chunk)
            if size != entry["size_bytes"] or digest.hexdigest() != entry["sha256"]:
                target.unlink(missing_ok=True)
                raise ValueError(f"Restored file failed validation: {target}")
            restored += 1
    return {
        "status": "verified_restored",
        "package": str(package),
        "lineage": manifest["lineage"],
        "file_count": manifest["file_count"],
        "restored": restored,
        "already_present_and_verified": reused,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package", type=Path)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    print(json.dumps(restore(args.package.resolve(), args.repo_root.resolve()), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
