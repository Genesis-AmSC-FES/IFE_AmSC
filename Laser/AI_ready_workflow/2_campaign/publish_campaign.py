#!/usr/bin/env python3
"""Publish completed Laser ACA archives and the ACX index to the shared project store."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import shutil
import stat
import subprocess
from pathlib import Path
from typing import Any

DEFAULT_SPEC = Path(__file__).with_name("campaign_spec.json")


def load_spec(path: Path) -> dict[str, Any]:
    """Load and validate campaign publication settings."""
    with path.open(encoding="utf-8") as stream:
        spec = json.load(stream)

    required = {
        "CAMPAIGN_STORE": str,
        "PUBLISH_CAMPAIGN_STORE": str,
        "ARCHIVE_PREFIX": str,
        "CAMPAIGN_INDEX": str,
    }
    for key, expected_type in required.items():
        if key not in spec:
            raise ValueError(f"Missing required setting: {key}")
        if not isinstance(spec[key], expected_type):
            raise TypeError(
                f"Setting {key!r} must be {expected_type.__name__}, "
                f"not {type(spec[key]).__name__}"
            )
        if not spec[key]:
            raise ValueError(f"Setting {key!r} must not be empty")

    return spec


def sha256_digest(path: Path) -> str:
    """Return the SHA-256 digest of a file without loading it into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def publication_files(spec: dict[str, Any]) -> tuple[Path, Path, list[Path]]:
    """Resolve and validate the source, destination, and files to publish."""
    source_store = Path(spec["CAMPAIGN_STORE"]).expanduser().resolve()
    publish_store = Path(spec["PUBLISH_CAMPAIGN_STORE"]).expanduser().resolve()
    archive_prefix = spec["ARCHIVE_PREFIX"]
    index_name = spec["CAMPAIGN_INDEX"]

    if not source_store.is_dir():
        raise FileNotFoundError(f"Campaign store does not exist: {source_store}")
    if source_store == publish_store:
        raise ValueError("CAMPAIGN_STORE and PUBLISH_CAMPAIGN_STORE must differ")

    archives = sorted(source_store.glob(f"{archive_prefix}-*.aca"), key=str)
    if not archives:
        raise FileNotFoundError(
            f"No archives matching {archive_prefix!r}-*.aca in {source_store}"
        )

    index = source_store / index_name
    if not index.is_file():
        raise FileNotFoundError(f"Campaign index does not exist: {index}")

    return source_store, publish_store, [*archives, index]


def publish_campaign(spec: dict[str, Any], *, dry_run: bool = False) -> None:
    """Publish campaign artifacts and verify every resulting project copy."""
    source_store, publish_store, artifacts = publication_files(spec)

    if shutil.which("rsync") is None:
        raise FileNotFoundError("rsync is not available on PATH")

    command = [
        "rsync",
        "-aH",
        "--checksum",
        "--delay-updates",
        "--itemize-changes",
        "--no-owner",
        "--no-group",
        "--chmod=F660,D2770",
        *(str(artifact) for artifact in artifacts),
        f"{publish_store}/",
    ]

    print(f"Source campaign store : {source_store}")
    print(f"Shared campaign store : {publish_store}")
    print(f"Publishing {len(artifacts) - 1} ACA archive(s) and 1 ACX index:")
    for artifact in artifacts:
        print(f"  {artifact.name}")
    print(f"Command: {shlex.join(command)}")

    if dry_run:
        print("Dry run complete; no directory or files were changed.")
        return

    publish_store.mkdir(parents=True, exist_ok=True)
    publish_store.chmod(0o2770)
    subprocess.run(command, check=True)

    destination_gid = publish_store.stat().st_gid
    for source in artifacts:
        published = publish_store / source.name
        if not published.is_file():
            raise FileNotFoundError(f"Published artifact is missing: {published}")
        if sha256_digest(source) != sha256_digest(published):
            raise ValueError(f"Published artifact failed verification: {published}")
        published.chmod(0o660)
        if published.stat().st_gid != destination_gid:
            os.chown(published, -1, destination_gid)
        mode = stat.S_IMODE(published.stat().st_mode)
        print(f"Verified: {published} (mode {mode:o})")

    print(
        f"Campaign publication complete: {len(artifacts)} artifact(s) in "
        f"{publish_store}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Publish completed Laser ACA archives and ACX index to the shared project store."
    )
    parser.add_argument(
        "--spec",
        type=Path,
        default=DEFAULT_SPEC,
        help=f"Campaign JSON specification (default: {DEFAULT_SPEC})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List ACA/ACX artifacts and print rsync without changing files.",
    )
    args = parser.parse_args()
    publish_campaign(load_spec(args.spec), dry_run=args.dry_run)


if __name__ == "__main__":
    main()
