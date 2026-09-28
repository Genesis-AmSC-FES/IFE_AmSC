"""Build a RHINO campaign index from settings in a JSON specification."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
from pathlib import Path
from typing import Any


DEFAULT_SPEC = Path(__file__).with_name("campaign_spec.json")


def load_spec(path: Path) -> dict[str, Any]:
    """Load and validate the index-related campaign settings."""
    with path.open(encoding="utf-8") as stream:
        spec = json.load(stream)

    required = {
        "PUBLISH_CAMPAIGN_STORE": str,
        "ARCHIVE_PREFIX": str,
        "CAMPAIGN_INDEX": str,
    }
    for key, expected_type in required.items():
        if key not in spec:
            raise ValueError(f"Missing required setting: {key}")
        if not isinstance(spec[key], expected_type):
            raise TypeError(
                f"Setting '{key}' must be {expected_type.__name__}, "
                f"not {type(spec[key]).__name__}"
            )

    return spec


def create_index(spec: dict[str, Any], *, dry_run: bool = False) -> None:
    """Build the authoritative index from published campaign archives."""
    campaign_store = Path(spec["PUBLISH_CAMPAIGN_STORE"]).expanduser()
    archive_prefix = spec["ARCHIVE_PREFIX"]
    configured_index = Path(spec["CAMPAIGN_INDEX"])

    if not campaign_store.is_dir():
        raise FileNotFoundError(
            f"Campaign store does not exist: {campaign_store}"
        )

    archives = sorted(
        campaign_store.glob(f"{archive_prefix}*.aca"),
        key=lambda path: str(path),
    )
    if not archives:
        raise FileNotFoundError(
            f"No archives matching '{archive_prefix}*.aca' in {campaign_store}"
        )

    index_path = (
        configured_index
        if configured_index.is_absolute()
        else campaign_store / configured_index
    )
    archive_arguments = [str(path.relative_to(campaign_store)) for path in archives]
    index_argument = (
        str(configured_index)
        if not configured_index.is_absolute()
        else str(index_path)
    )

    print(f"Published campaign store: {campaign_store}")
    print(f"Index: {index_path}")
    print(f"Found {len(archives)} archive(s):")
    for archive in archives:
        print(f"  {archive}")

    add_command = [
        "hpc_campaign",
        "index",
        index_argument,
        "add",
        *archive_arguments,
    ]
    inspect_command = ["hpc_campaign", "index", index_argument, "ls"]

    if index_path.exists():
        print(f"Replacing existing index: {index_path}")
        if not dry_run:
            index_path.unlink()

    print(f"Command: {shlex.join(add_command)}")
    if not dry_run:
        subprocess.run(add_command, cwd=campaign_store, check=True)

    print(f"Command: {shlex.join(inspect_command)}")
    if not dry_run:
        subprocess.run(inspect_command, cwd=campaign_store, check=True)

    print(f"Campaign index creation complete: {index_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a campaign index from RHINO campaign archives."
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
        help="Print index operations without changing files or running commands.",
    )
    args = parser.parse_args()

    create_index(load_spec(args.spec), dry_run=args.dry_run)


if __name__ == "__main__":
    main()
