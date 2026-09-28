"""Publish completed RHINO campaign archives to the shared project store."""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import shutil
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
    }
    for key, expected_type in required.items():
        if key not in spec:
            raise ValueError(f"Missing required setting: {key}")
        if not isinstance(spec[key], expected_type):
            raise TypeError(
                f"Setting '{key}' must be {expected_type.__name__}, "
                f"not {type(spec[key]).__name__}"
            )
        if not spec[key]:
            raise ValueError(f"Setting '{key}' must not be empty")

    return spec


def sha256_digest(path: Path) -> str:
    """Return the SHA-256 digest of a file without loading it into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def publish_campaign(spec: dict[str, Any], *, dry_run: bool = False) -> None:
    """Publish completed ACA files and verify the resulting project copies."""
    source_store = Path(spec["CAMPAIGN_STORE"]).expanduser().resolve()
    publish_store = Path(spec["PUBLISH_CAMPAIGN_STORE"]).expanduser().resolve()
    archive_prefix = spec["ARCHIVE_PREFIX"]

    if not source_store.is_dir():
        raise FileNotFoundError(f"Campaign store does not exist: {source_store}")
    if source_store == publish_store:
        raise ValueError("CAMPAIGN_STORE and PUBLISH_CAMPAIGN_STORE must differ")

    archives = sorted(source_store.glob(f"{archive_prefix}-*.aca"), key=str)
    if not archives:
        raise FileNotFoundError(
            f"No archives matching '{archive_prefix}-*.aca' in {source_store}"
        )
    if shutil.which("rsync") is None:
        raise FileNotFoundError("rsync is not available on PATH")

    command = [
        "rsync",
        "-a",
        "--checksum",
        "--itemize-changes",
        *(str(archive) for archive in archives),
        f"{publish_store}/",
    ]

    print(f"Source campaign store: {source_store}")
    print(f"Published campaign store: {publish_store}")
    print(f"Publishing {len(archives)} archive(s):")
    for archive in archives:
        print(f"  {archive.name}")
    print(f"Command: {shlex.join(command)}")

    if dry_run:
        print("Dry run complete; no directory or files were changed.")
        return

    publish_store.mkdir(parents=True, exist_ok=True)
    subprocess.run(command, check=True)

    for source in archives:
        published = publish_store / source.name
        if not published.is_file():
            raise FileNotFoundError(f"Published archive is missing: {published}")
        if sha256_digest(source) != sha256_digest(published):
            raise ValueError(f"Published archive failed verification: {published}")
        print(f"Verified: {published}")

    print(
        f"Campaign publication complete: {len(archives)} archive(s) in "
        f"{publish_store}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Publish completed RHINO ACA files to the project store."
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
        help="List ACA files and print rsync without changing files.",
    )
    args = parser.parse_args()
    publish_campaign(load_spec(args.spec), dry_run=args.dry_run)


if __name__ == "__main__":
    main()
