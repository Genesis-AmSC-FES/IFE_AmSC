#!/usr/bin/env python
"""Build the RHINO model-training ACA from completed MLflow run records."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shlex
import subprocess
from typing import Any


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = BASE_DIR / "campaign_spec.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create rhino-model-training.aca from the durable provenance "
            "records written after successful MLflow uploads."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help=f"Campaign configuration (default: {DEFAULT_CONFIG})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate records and print hpc_campaign commands without writing.",
    )
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        config = json.load(stream)
    required = (
        "CAMPAIGN_STORE",
        "MODEL_CAMPAIGN_ARCHIVE",
        "MODEL_PROVENANCE_ROOT",
        "MODEL_RUN_RECORD_DIR",
    )
    missing = [name for name in required if not config.get(name)]
    if missing:
        raise ValueError(f"Missing campaign configuration: {', '.join(missing)}")
    return config


def load_records(record_dir: Path) -> list[tuple[Path, dict[str, Any]]]:
    records: list[tuple[Path, dict[str, Any]]] = []
    seen_run_ids: set[str] = set()
    for path in sorted(record_dir.glob("*.json")):
        with path.open(encoding="utf-8") as stream:
            record = json.load(stream)
        run_id = record.get("run_id")
        mlflow = record.get("mlflow")
        if record.get("schema_version") != 1:
            raise ValueError(f"Unsupported schema_version in {path}")
        if record.get("record_type") != "rhino_model_training":
            raise ValueError(f"Unexpected record_type in {path}")
        if record.get("status") != "completed":
            raise ValueError(f"Run is not completed in {path}")
        if not isinstance(run_id, str) or not run_id:
            raise ValueError(f"Missing run_id in {path}")
        if run_id in seen_run_ids:
            raise ValueError(f"Duplicate MLflow run_id: {run_id}")
        if not isinstance(mlflow, dict):
            raise ValueError(f"Missing MLflow metadata in {path}")
        if not mlflow.get("run_url") or not mlflow.get("model_uri"):
            raise ValueError(f"Missing MLflow run_url or model_uri in {path}")
        seen_run_ids.add(run_id)
        records.append((path.resolve(), record))
    return records


def campaign_command(
    campaign_store: Path,
    archive_name: str,
    record_path: Path,
    dataset_name: str,
    *,
    truncate: bool,
) -> list[str]:
    command = [
        "hpc_campaign",
        "manager",
        "--campaign_store",
        str(campaign_store),
        archive_name,
    ]
    if truncate:
        command.append("--truncate")
    command.extend(["data", str(record_path), "--name", dataset_name])
    return command


def main() -> None:
    args = parse_args()
    config = load_config(args.config.resolve())
    campaign_store = Path(config["CAMPAIGN_STORE"]).resolve()
    archive_name = str(config["MODEL_CAMPAIGN_ARCHIVE"])
    provenance_root = Path(config["MODEL_PROVENANCE_ROOT"]).resolve()
    record_dir = Path(config["MODEL_RUN_RECORD_DIR"]).resolve()

    if not record_dir.is_dir():
        message = f"No MLflow registry directory exists yet: {record_dir}"
        if args.dry_run:
            print(message)
            return
        raise FileNotFoundError(message)

    records = load_records(record_dir)
    if not records:
        message = f"No completed MLflow run records found in {record_dir}"
        if args.dry_run:
            print(message)
            return
        raise RuntimeError(message)

    if not args.dry_run:
        campaign_store.mkdir(parents=True, exist_ok=True)

    for index, (record_path, record) in enumerate(records):
        try:
            relative_record = record_path.relative_to(provenance_root)
        except ValueError as exc:
            raise ValueError(
                f"Record {record_path} is outside MODEL_PROVENANCE_ROOT "
                f"{provenance_root}"
            ) from exc
        dataset_name = f"mlflow-run-{record['run_id']}"
        command = campaign_command(
            campaign_store,
            archive_name,
            relative_record,
            dataset_name,
            truncate=index == 0,
        )
        print(shlex.join(command))
        if not args.dry_run:
            subprocess.run(command, cwd=provenance_root, check=True)

    action = "Would create" if args.dry_run else "Created"
    print(f"{action} {archive_name} with {len(records)} completed MLflow run(s).")


if __name__ == "__main__":
    main()
