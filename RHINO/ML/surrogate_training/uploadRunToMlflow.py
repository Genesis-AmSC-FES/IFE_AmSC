#!/usr/bin/env python
"""Upload a completed, portable RHINO training bundle to MLflow."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import numpy as np
import pandas as pd
import torch

from dataset_tracking import log_mlflow_dataset_inputs, sha256_file
from mlflow_model import RhinoSurrogatePyFunc
from mlflow_registry import write_run_record
from trainSurrogate import BASE_DIR, SurrogateMLP


def parse_args() -> argparse.Namespace:
    """Parse deferred-upload options."""
    parser = argparse.ArgumentParser(
        description="Upload a completed RHINO run bundle to an MLflow server."
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Directory containing run_manifest.json and training artifacts.",
    )
    parser.add_argument(
        "--mlflow-tracking-uri",
        help="Destination server URI; otherwise require MLFLOW_TRACKING_URI.",
    )
    parser.add_argument(
        "--experiment",
        default="rhino-surrogate",
        help="Destination experiment name (default: rhino-surrogate).",
    )
    parser.add_argument("--run-name", help="Optional MLflow run name.")
    parser.add_argument(
        "--registered-model-name",
        default="rhino-surrogate",
        help="Registry name (default: rhino-surrogate).",
    )
    parser.add_argument(
        "--skip-model-registration",
        action="store_true",
        help="Log the serving model without creating a registry version.",
    )
    parser.add_argument(
        "--features",
        type=Path,
        help=(
            "Optional current location of the original feature CSV. When it is "
            "available and its digest matches, replay MLflow dataset inputs."
        ),
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    """Load and validate a JSON object."""
    if not path.is_file():
        raise FileNotFoundError(f"Required run-bundle file does not exist: {path}")
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def verify_bundle(run_dir: Path, manifest: dict[str, Any]) -> None:
    """Reject incomplete or modified core training artifacts."""
    if manifest.get("schema_version") != 1:
        raise ValueError("Unsupported or missing run-bundle schema_version")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError("Run manifest has no artifacts mapping")
    for name, digest_name in (
        ("model", "model_sha256"),
        ("training_history", "training_history_sha256"),
        ("splits", "splits_sha256"),
    ):
        path = run_dir / str(artifacts.get(name, ""))
        expected = artifacts.get(digest_name)
        if not path.is_file() or not expected:
            raise ValueError(f"Run bundle is missing {name}")
        if sha256_file(path) != expected:
            raise ValueError(f"Run-bundle checksum mismatch for {path.name}")


def flatten_test_metrics(metrics: dict[str, Any]) -> dict[str, float]:
    """Flatten evaluation metrics into names accepted by MLflow."""
    flattened = {
        f"test_{name}": float(value)
        for name, value in metrics.items()
        if name != "per_output"
    }
    for output, values in metrics.get("per_output", {}).items():
        safe_output = re.sub(r"[^A-Za-z0-9_.-]+", "_", output)
        for name, value in values.items():
            flattened[f"test_{safe_output}_{name}"] = float(value)
    return flattened


def public_tracking_uri(uri: str) -> str:
    """Remove credentials, query parameters, and fragments from a URI."""
    parts = urlsplit(uri)
    if not parts.scheme or not parts.netloc:
        return uri
    hostname = parts.hostname or ""
    if parts.port is not None:
        hostname = f"{hostname}:{parts.port}"
    return urlunsplit((parts.scheme, hostname, parts.path, "", ""))


def resolve_features(
    override: Path | None,
    split_metadata: dict[str, Any],
) -> Path | None:
    """Resolve an optional feature CSV without requiring NERSC paths later."""
    candidate = override
    if candidate is None and split_metadata.get("features_file"):
        candidate = Path(split_metadata["features_file"])
    if candidate is None:
        return None
    path = candidate.expanduser().resolve()
    if not path.is_file():
        if override is not None:
            raise FileNotFoundError(f"Feature CSV does not exist: {path}")
        return None
    expected = split_metadata.get("features_sha256")
    if expected and sha256_file(path) != expected:
        raise ValueError(f"Feature CSV SHA-256 does not match the training run: {path}")
    return path


def main() -> None:
    """Create a remote run, replay metrics, and log/register the saved model."""
    args = parse_args()
    tracking_uri = args.mlflow_tracking_uri or os.environ.get("MLFLOW_TRACKING_URI")
    if not tracking_uri:
        raise ValueError(
            "Choose an upload destination with --mlflow-tracking-uri or "
            "MLFLOW_TRACKING_URI"
        )

    try:
        import mlflow
        from mlflow.models import infer_signature
    except ImportError as error:
        raise RuntimeError("Deferred upload requires MLflow") from error

    run_dir = args.run_dir.expanduser().resolve()
    manifest = load_json(run_dir / "run_manifest.json")
    verify_bundle(run_dir, manifest)
    artifact_names = manifest["artifacts"]
    split_metadata = load_json(run_dir / artifact_names["splits"])
    checkpoint = torch.load(run_dir / artifact_names["model"], map_location="cpu")
    input_example = pd.read_csv(run_dir / artifact_names["input_example"])
    input_columns = checkpoint["input_columns"]
    output_columns = checkpoint["output_columns"]
    if list(input_example.columns) != input_columns:
        raise ValueError("Input example does not match the checkpoint input columns")

    config = checkpoint["model_config"]
    model = SurrogateMLP(
        input_dim=config["input_dim"],
        output_dim=config["output_dim"],
        hidden_dim=config["hidden_dim"],
        num_hidden_layers=config["num_hidden_layers"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    serving_model = RhinoSurrogatePyFunc(
        model=model,
        input_columns=input_columns,
        output_columns=output_columns,
        x_mean=checkpoint["x_mean"].detach().cpu().numpy(),
        x_std=checkpoint["x_std"].detach().cpu().numpy(),
        y_mean=checkpoint["y_mean"].detach().cpu().numpy(),
        y_std=checkpoint["y_std"].detach().cpu().numpy(),
    )
    input_example = input_example.astype(np.float32)
    output_example = serving_model.predict(None, input_example)
    signature = infer_signature(input_example, output_example)
    features_path = resolve_features(args.features, split_metadata)

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(args.experiment)
    default_run_name = manifest.get("scheduler", {}).get(
        "slurm_job_id", run_dir.name
    )
    registered_name = (
        None if args.skip_model_registration else args.registered_model_name
    )
    training = manifest["training"]
    dataset = manifest["dataset"]
    runtime = manifest["runtime"]
    source = manifest.get("source", {})
    git = source.get("git", {})

    with mlflow.start_run(run_name=args.run_name or str(default_run_name)) as run:
        mlflow.log_params(
            {
                "epochs": training["epochs"],
                "batch_size": training["batch_size"],
                "hidden_dim": training["hidden_dim"],
                "hidden_layers": training["hidden_layers"],
                "learning_rate": training["learning_rate"],
                "seed": training["seed"],
                "train_rows": training["train_rows"],
                "validation_rows": training["validation_rows"],
                "test_rows": training["test_rows"],
                "input_columns": json.dumps(input_columns),
                "output_columns": json.dumps(output_columns),
            }
        )
        tags = {
            "deferred_upload": "true",
            "features_sha256": dataset["features_sha256"],
            "model_type": "multi_output_regression",
            "training_device": training["device"],
            "python_version": runtime["python"],
            "torch_version": runtime["torch"],
            "dataset_inputs_replayed": str(features_path is not None).lower(),
        }
        if git.get("commit"):
            tags["git_commit"] = git["commit"]
            tags["git_dirty"] = str(git.get("dirty", False)).lower()
        for name in ("feature_spec_sha256", "dataset_manifest_sha256"):
            if dataset.get(name):
                tags[name] = dataset[name]
        for name, value in manifest.get("scheduler", {}).items():
            tags[name] = str(value)
        mlflow.set_tags(tags)

        history = pd.read_csv(run_dir / artifact_names["training_history"])
        for step, row in enumerate(history.itertuples(index=False), start=1):
            mlflow.log_metrics(
                {
                    "train_loss": float(row.train_loss),
                    "val_loss": float(row.val_loss),
                },
                step=step,
            )
        mlflow.log_metric("best_val_loss", float(checkpoint["best_val_loss"]))

        if features_path is not None:
            frame = pd.read_csv(features_path)
            log_mlflow_dataset_inputs(
                mlflow,
                frame,
                features_path,
                split_metadata["features_sha256"],
                np.asarray(split_metadata["train_idx"], dtype=int),
                np.asarray(split_metadata["val_idx"], dtype=int),
                str(features_path.stem),
            )

        metrics_path = run_dir / "metrics.json"
        metrics = load_json(metrics_path) if metrics_path.is_file() else {}
        if metrics:
            mlflow.log_metrics(flatten_test_metrics(metrics))
            mlflow.set_tag("evaluation_complete", "true")

        torch_version = str(torch.__version__).split("+", maxsplit=1)[0]
        model_info = mlflow.pyfunc.log_model(
            name="model",
            python_model=serving_model,
            input_example=input_example,
            signature=signature,
            registered_model_name=registered_name,
            code_paths=[
                str(BASE_DIR / "trainSurrogate.py"),
                str(BASE_DIR / "dataset_tracking.py"),
                str(BASE_DIR / "mlflow_model.py"),
            ],
            pip_requirements=[
                f"mlflow=={mlflow.__version__}",
                f"numpy=={np.__version__}",
                f"pandas=={pd.__version__}",
                f"torch=={torch_version}",
            ],
            metadata={
                "input_columns": input_columns,
                "output_columns": output_columns,
                "features_sha256": dataset["features_sha256"],
            },
        )
        tracking_public = public_tracking_uri(tracking_uri)
        campaign_record = write_run_record(
            run_dir.parent / "mlflow_registry",
            run_id=run.info.run_id,
            run_name=args.run_name or str(default_run_name),
            experiment_id=run.info.experiment_id,
            experiment_name=args.experiment,
            tracking_uri=tracking_public,
            model_uri=model_info.model_uri,
            registered_model_name=registered_name,
            registered_model_version=model_info.registered_model_version,
            run_manifest=manifest,
            metrics=metrics,
            run_bundle=run_dir,
        )
        receipt = {
            "tracking_uri": tracking_public,
            "experiment": args.experiment,
            "run_id": run.info.run_id,
            "mlflow_run_url": (
                f"{tracking_public.rstrip('/')}/#/experiments/"
                f"{run.info.experiment_id}/runs/{run.info.run_id}"
            ),
            "model_uri": model_info.model_uri,
            "registered_model_name": registered_name,
            "registered_model_version": model_info.registered_model_version,
            "campaign_record": str(campaign_record),
        }
        receipt_path = run_dir / f"mlflow_upload_{run.info.run_id}.json"
        with receipt_path.open("w", encoding="utf-8") as stream:
            json.dump(receipt, stream, indent=2)
        mlflow.log_artifacts(str(run_dir), artifact_path="run_bundle")

    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
