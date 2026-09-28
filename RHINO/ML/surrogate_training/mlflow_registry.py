"""Write durable, credential-free records for completed MLflow runs."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


def _mlflow_url(tracking_uri: str, experiment_id: str, run_id: str) -> str:
    return f"{tracking_uri.rstrip('/')}/#/experiments/{experiment_id}/runs/{run_id}"


def write_run_record(
    registry_dir: Path,
    *,
    run_id: str,
    run_name: str | None,
    experiment_id: str,
    experiment_name: str,
    tracking_uri: str,
    model_uri: str,
    registered_model_name: str | None,
    registered_model_version: str | int | None,
    run_manifest: Mapping[str, Any],
    metrics: Mapping[str, Any],
    run_bundle: Path,
) -> Path:
    """Atomically persist one completed run for the model-training campaign."""
    registry_dir.mkdir(parents=True, exist_ok=True)
    tracking_uri = tracking_uri.rstrip("/")
    record = {
        "schema_version": 1,
        "record_type": "rhino_model_training",
        "status": "completed",
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "run_id": run_id,
        "run_name": run_name or run_id,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "mlflow": {
            "tracking_uri": tracking_uri,
            "experiment_id": experiment_id,
            "experiment_name": experiment_name,
            "experiment_url": f"{tracking_uri}/#/experiments/{experiment_id}",
            "run_url": _mlflow_url(tracking_uri, experiment_id, run_id),
            "model_uri": model_uri,
            "registered_model_name": registered_model_name,
            "registered_model_version": registered_model_version,
        },
        "run_bundle": str(run_bundle.resolve()),
        "metrics": dict(metrics),
        "training_run_manifest": dict(run_manifest),
    }
    destination = registry_dir / f"{run_id}.json"
    temporary = registry_dir / f".{run_id}.{os.getpid()}.tmp"
    temporary.write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)
    return destination
