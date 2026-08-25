"""Dataset fingerprint, sidecar, and MLflow-input helpers."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def sha256_file(path: Path) -> str:
    """Return a stable SHA-256 fingerprint for dataset provenance checks."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_dataset_manifest(
    features_path: Path,
    configured_path: Path | None,
) -> tuple[Path | None, dict[str, Any] | None]:
    """Load and validate an explicit or adjacent feature-dataset manifest."""
    if configured_path is not None:
        candidates = [configured_path.expanduser().resolve()]
        required = True
    else:
        candidates = [
            features_path.parent / "dataset_manifest.json",
            features_path.with_suffix(".manifest.json"),
        ]
        required = False

    manifest_path = next((path for path in candidates if path.is_file()), None)
    if manifest_path is None:
        if required:
            raise FileNotFoundError(f"Dataset manifest does not exist: {candidates[0]}")
        return None, None

    with manifest_path.open(encoding="utf-8") as stream:
        manifest = json.load(stream)
    if not isinstance(manifest, dict):
        raise TypeError(f"Dataset manifest must be a JSON object: {manifest_path}")
    expected_digest = manifest.get("sha256")
    actual_digest = sha256_file(features_path)
    if expected_digest != actual_digest:
        raise ValueError(
            f"Dataset manifest SHA-256 does not match feature CSV: {manifest_path}"
        )
    expected_rows = manifest.get("rows")
    if expected_rows is not None and (
        isinstance(expected_rows, bool) or not isinstance(expected_rows, int)
    ):
        raise TypeError("Dataset manifest rows must be an integer")
    return manifest_path, manifest


def log_mlflow_dataset_inputs(
    mlflow_module: Any,
    frame: pd.DataFrame,
    features_path: Path,
    features_digest: str,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    dataset_name: str,
) -> dict[str, dict[str, str]]:
    """Log source, training, and validation datasets to the active MLflow run."""
    source = features_path.expanduser().resolve().as_uri()
    datasets = {
        "source": mlflow_module.data.from_pandas(
            frame,
            source=source,
            name=dataset_name,
        ),
        "training": mlflow_module.data.from_pandas(
            frame.iloc[train_idx],
            source=source,
            name=f"{dataset_name}-training",
        ),
        "validation": mlflow_module.data.from_pandas(
            frame.iloc[val_idx],
            source=source,
            name=f"{dataset_name}-validation",
        ),
    }
    for context, dataset in datasets.items():
        mlflow_module.log_input(dataset, context=context)
    return {
        context: {
            "name": dataset.name,
            "digest": dataset.digest,
            "source": source,
            **({"sha256": features_digest} if context == "source" else {}),
        }
        for context, dataset in datasets.items()
    }
