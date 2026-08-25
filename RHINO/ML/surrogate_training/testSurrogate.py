#!/usr/bin/env python
"""Evaluate a saved RHINO surrogate on its held-out feature-table rows."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch

from dataset_tracking import sha256_file
from trainSurrogate import SurrogateMLP, load_feature_arrays


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_ARTIFACT_DIR = BASE_DIR / "artifacts"


def parse_args() -> argparse.Namespace:
    """Parse surrogate-evaluation command-line options."""
    parser = argparse.ArgumentParser(
        description="Evaluate a trained RHINO surrogate on its test split."
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_ARTIFACT_DIR / "rhino_surrogate.pt",
        help="Path to the trained surrogate artifact.",
    )
    parser.add_argument(
        "--splits",
        type=Path,
        default=DEFAULT_ARTIFACT_DIR / "split_indices.json",
        help="Path to the saved train/validation/test split metadata.",
    )
    parser.add_argument(
        "--features",
        type=Path,
        help="Optional replacement path for the exact CSV used during training.",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=DEFAULT_ARTIFACT_DIR,
        help="Directory where evaluation results will be saved.",
    )
    parser.add_argument(
        "--mlflow",
        action="store_true",
        help="Attach test metrics and artifacts to the originating MLflow run.",
    )
    parser.add_argument(
        "--mlflow-tracking-uri",
        help="MLflow server URI; otherwise use MLFLOW_TRACKING_URI/default.",
    )
    parser.add_argument(
        "--model-alias",
        help="Optional registry alias to assign after successful evaluation.",
    )
    return parser.parse_args()


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute aggregate scalar regression-error metrics."""
    error = y_pred - y_true
    return {
        "mse": float(np.mean(error**2)),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mae": float(np.mean(np.abs(error))),
        "max_abs_error": float(np.max(np.abs(error))),
    }


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_columns: Sequence[str],
) -> dict[str, object]:
    """Compute overall and per-output regression metrics."""
    return {
        **regression_metrics(y_true, y_pred),
        "per_output": {
            column: regression_metrics(y_true[:, index], y_pred[:, index])
            for index, column in enumerate(output_columns)
        },
    }


def mlflow_metrics(metrics: dict[str, object]) -> dict[str, float]:
    """Flatten overall and per-output metrics into MLflow-safe names."""
    flattened = {
        f"test_{name}": float(value)
        for name, value in metrics.items()
        if name != "per_output"
    }
    per_output = metrics.get("per_output", {})
    if not isinstance(per_output, dict):
        raise TypeError("per_output metrics must be a mapping")
    for output, output_metrics in per_output.items():
        safe_output = re.sub(r"[^A-Za-z0-9_.-]+", "_", output)
        if not isinstance(output_metrics, dict):
            raise TypeError(f"Metrics for output {output!r} must be a mapping")
        for name, value in output_metrics.items():
            flattened[f"test_{safe_output}_{name}"] = float(value)
    return flattened


def save_parity_plots(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_columns: Sequence[str],
    outdir: Path,
) -> None:
    """Save one predicted-versus-true parity panel for each model output."""
    n_outputs = len(output_columns)
    figure, axes = plt.subplots(
        nrows=1,
        ncols=n_outputs,
        figsize=(5 * n_outputs, 4),
        dpi=300,
        squeeze=False,
    )

    for index, column in enumerate(output_columns):
        axis = axes[0, index]
        true_values = y_true[:, index]
        predicted_values = y_pred[:, index]
        axis.scatter(true_values, predicted_values, s=15)
        minimum = min(true_values.min(), predicted_values.min())
        maximum = max(true_values.max(), predicted_values.max())
        axis.plot([minimum, maximum], [minimum, maximum])
        axis.set_xlabel("True")
        axis.set_ylabel("Predicted")
        axis.set_title(column)

    figure.tight_layout()
    figure.savefig(outdir / "parity_plots.png")
    plt.close(figure)


def resolve_features_path(
    configured_path: str,
    override: Path | None,
) -> Path:
    """Resolve the training CSV path, allowing an explicit moved-file override."""
    path = override if override is not None else Path(configured_path)
    return path.expanduser().resolve()


def main() -> None:
    """Load the saved split and model, evaluate it, and write diagnostics."""
    args = parse_args()
    if args.model_alias and not args.mlflow:
        raise ValueError("--model-alias requires --mlflow")
    outdir = args.outdir.expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    with args.splits.expanduser().open(encoding="utf-8") as stream:
        split_indices = json.load(stream)
    artifact = torch.load(args.model.expanduser(), map_location="cpu")

    input_columns = split_indices["input_columns"]
    output_columns = split_indices["output_columns"]
    if artifact["input_columns"] != input_columns:
        raise ValueError("Model and split metadata disagree about input columns")
    if artifact["output_columns"] != output_columns:
        raise ValueError("Model and split metadata disagree about output columns")

    features_path = resolve_features_path(
        split_indices["features_file"],
        args.features,
    )
    expected_digest = split_indices.get("features_sha256")
    if expected_digest and sha256_file(features_path) != expected_digest:
        raise ValueError(
            "Feature CSV differs from the file used during training; refusing "
            "to apply saved row indexes"
        )

    frame, x, y = load_feature_arrays(
        features_path,
        input_columns,
        output_columns,
    )
    expected_rows = split_indices.get("row_count")
    if expected_rows is not None and len(frame) != expected_rows:
        raise ValueError(
            f"Feature CSV has {len(frame)} rows; training metadata expects "
            f"{expected_rows}"
        )

    test_idx = np.asarray(split_indices["test_idx"], dtype=int)
    if test_idx.size == 0 or test_idx.min() < 0 or test_idx.max() >= len(frame):
        raise ValueError("Saved test indexes are empty or outside the feature table")
    x_test = x[test_idx]
    y_test = y[test_idx]

    x_mean = artifact["x_mean"].numpy()
    x_std = artifact["x_std"].numpy()
    y_mean = artifact["y_mean"].numpy()
    y_std = artifact["y_std"].numpy()
    x_test_n = (x_test - x_mean) / x_std

    config = artifact["model_config"]
    if config["input_dim"] != len(input_columns):
        raise ValueError("Saved model input dimension does not match its columns")
    if config["output_dim"] != len(output_columns):
        raise ValueError("Saved model output dimension does not match its columns")
    model = SurrogateMLP(
        input_dim=config["input_dim"],
        output_dim=config["output_dim"],
        hidden_dim=config["hidden_dim"],
        num_hidden_layers=config["num_hidden_layers"],
    )
    model.load_state_dict(artifact["model_state_dict"])
    model.eval()

    with torch.no_grad():
        normalized_prediction = model(
            torch.tensor(x_test_n, dtype=torch.float32)
        ).numpy()
    y_prediction = normalized_prediction * y_std + y_mean
    metrics = compute_metrics(y_test, y_prediction, output_columns)

    with (outdir / "metrics.json").open("w", encoding="utf-8") as stream:
        json.dump(metrics, stream, indent=2)

    metadata_columns = [
        column
        for column in split_indices.get("metadata_columns", [])
        if column in frame.columns
    ]
    context_columns = [*metadata_columns, *input_columns]
    predictions = frame.iloc[test_idx][context_columns].reset_index(drop=True)
    predictions.insert(len(predictions.columns), "source_row", test_idx)
    for index, column in enumerate(output_columns):
        predictions[f"true_{column}"] = y_test[:, index]
        predictions[f"pred_{column}"] = y_prediction[:, index]
        predictions[f"abs_error_{column}"] = np.abs(
            y_prediction[:, index] - y_test[:, index]
        )
    predictions.to_csv(outdir / "test_predictions.csv", index=False)
    save_parity_plots(y_test, y_prediction, output_columns, outdir)

    if args.mlflow:
        try:
            import mlflow
        except ImportError as error:
            raise RuntimeError(
                "MLflow logging was requested but mlflow is not installed"
            ) from error

        run_id = split_indices.get("mlflow_run_id")
        if not run_id:
            raise ValueError(
                "Split metadata has no MLflow run ID; retrain with --mlflow"
            )
        if args.mlflow_tracking_uri:
            mlflow.set_tracking_uri(args.mlflow_tracking_uri)
        with mlflow.start_run(run_id=run_id):
            source_dataset = split_indices.get("mlflow_datasets", {}).get(
                "source", {}
            )
            source_name = source_dataset.get("name", features_path.stem)
            test_dataset = mlflow.data.from_pandas(
                frame.iloc[test_idx],
                source=features_path.as_uri(),
                name=f"{source_name}-testing",
            )
            mlflow.log_input(test_dataset, context="testing")
            mlflow.log_metrics(mlflow_metrics(metrics))
            mlflow.set_tags(
                {
                    "evaluation_complete": "true",
                    "test_dataset_digest": test_dataset.digest,
                }
            )
            mlflow.log_artifact(outdir / "metrics.json", artifact_path="evaluation")
            mlflow.log_artifact(
                outdir / "test_predictions.csv",
                artifact_path="evaluation",
            )
            mlflow.log_artifact(
                outdir / "parity_plots.png",
                artifact_path="evaluation",
            )

        if args.model_alias:
            model_name = split_indices.get("mlflow_registered_model_name")
            model_version = split_indices.get("mlflow_registered_model_version")
            if not model_name or model_version is None:
                raise ValueError(
                    "No registered model version is recorded; retrain without "
                    "--skip-model-registration"
                )
            mlflow.MlflowClient().set_registered_model_alias(
                model_name,
                args.model_alias,
                str(model_version),
            )

    print("\nTest metrics:")
    print(json.dumps(metrics, indent=2))
    print("\nSaved:")
    print(outdir / "metrics.json")
    print(outdir / "test_predictions.csv")
    print(outdir / "parity_plots.png")


if __name__ == "__main__":
    main()
