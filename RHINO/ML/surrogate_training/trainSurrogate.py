#!/usr/bin/env python
"""Train and save a RHINO neural-network surrogate from a feature-layer CSV."""

from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from dataset_tracking import (
    log_mlflow_dataset_inputs,
    resolve_dataset_manifest,
    sha256_file,
)


BASE_DIR = Path(__file__).resolve().parent
RHINO_ROOT = BASE_DIR.parents[1]
DEFAULT_FEATURE_SPEC = (
    RHINO_ROOT
    / "AI_ready_workflow"
    / "3_feature_extraction"
    / "feature_spec.json"
)
DEFAULT_OUTDIR = BASE_DIR / "artifacts"
KNOWN_METADATA_COLUMNS = [
    "archive",
    "datasetid",
    "run_id",
    "simulation_date",
    "simulation_time",
    "simulation_datetime",
]


def git_metadata(repository: Path) -> dict[str, Any]:
    """Return reproducibility metadata without making Git a hard dependency."""
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=repository,
                check=True,
                capture_output=True,
                text=True,
                timeout=10,
            ).stdout.strip()
        )
        return {"commit": commit, "dirty": dirty}
    except (FileNotFoundError, subprocess.SubprocessError):
        return {"commit": None, "dirty": None}


def copy_provenance_file(source: Path | None, destination: Path) -> Path | None:
    """Copy a small provenance file into the portable run bundle."""
    if source is None or not source.is_file():
        return None
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return destination


class SurrogateMLP(nn.Module):
    """Fully connected regression network used for the RHINO surrogate."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 100,
        num_hidden_layers: int = 3,
    ) -> None:
        """Construct an MLP with ReLU-activated hidden layers."""
        super().__init__()

        layers: list[nn.Module] = []
        in_dim = input_dim
        for _ in range(num_hidden_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim

        layers.append(nn.Linear(in_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return predictions for one normalized input batch."""
        return self.net(x)


def load_feature_roles(spec_path: Path) -> tuple[list[str], list[str]]:
    """Read default model input and output keys from the feature JSON spec."""
    with spec_path.open(encoding="utf-8") as stream:
        specification = json.load(stream)

    try:
        inputs = [entry["key"] for entry in specification["inputs"]]
        outputs = [entry["key"] for entry in specification["outputs"]]
    except (KeyError, TypeError) as error:
        raise ValueError(
            f"Invalid input/output declarations in feature spec: {spec_path}"
        ) from error

    if not inputs or not outputs:
        raise ValueError("Feature specification must define inputs and outputs")
    return inputs, outputs


def load_feature_arrays(
    features_path: Path,
    input_columns: Sequence[str],
    output_columns: Sequence[str],
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Load the CSV and return validated finite input and output arrays."""
    if not features_path.is_file():
        raise FileNotFoundError(f"Feature CSV does not exist: {features_path}")
    if len(set(input_columns)) != len(input_columns):
        raise ValueError("Input column names must be unique")
    if len(set(output_columns)) != len(output_columns):
        raise ValueError("Output column names must be unique")
    overlap = sorted(set(input_columns) & set(output_columns))
    if overlap:
        raise ValueError(f"Columns cannot be both inputs and outputs: {overlap}")

    frame = pd.read_csv(features_path)
    required = [*input_columns, *output_columns]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(
            "Feature CSV is missing required column(s): " + ", ".join(missing)
        )
    if frame.empty:
        raise ValueError("Feature CSV contains no rows")

    try:
        values = frame[required].apply(pd.to_numeric, errors="raise")
    except (TypeError, ValueError) as error:
        raise ValueError("Model input and output columns must be numeric") from error

    invalid = ~np.isfinite(values.to_numpy(dtype=np.float64))
    if invalid.any():
        bad_columns = [
            column
            for position, column in enumerate(required)
            if invalid[:, position].any()
        ]
        bad_rows = int(invalid.any(axis=1).sum())
        raise ValueError(
            f"Feature CSV has missing or non-finite model values in {bad_rows} "
            f"row(s), in column(s): {', '.join(bad_columns)}"
        )

    input_count = len(input_columns)
    matrix = values.to_numpy(dtype=np.float32)
    return frame, matrix[:, :input_count], matrix[:, input_count:]


def split_dataset(
    x: np.ndarray,
    y: np.ndarray,
    train_frac: float = 0.80,
    val_frac: float = 0.15,
    seed: int = 42,
) -> tuple[np.ndarray, ...]:
    """Randomly split aligned arrays and return data plus source row indexes."""
    if not 0 < train_frac < 1 or not 0 < val_frac < 1:
        raise ValueError("Training and validation fractions must be between 0 and 1")
    if train_frac + val_frac >= 1:
        raise ValueError("Training and validation fractions must sum to less than 1")
    if x.shape[0] != y.shape[0]:
        raise ValueError("Input and output arrays must have the same row count")

    rng = np.random.default_rng(seed)
    indexes = np.arange(x.shape[0])
    rng.shuffle(indexes)
    n_train = int(train_frac * len(indexes))
    n_val = int(val_frac * len(indexes))
    if min(n_train, n_val, len(indexes) - n_train - n_val) == 0:
        raise ValueError(
            "Dataset is too small to create non-empty train/val/test splits"
        )

    train_idx = indexes[:n_train]
    val_idx = indexes[n_train : n_train + n_val]
    test_idx = indexes[n_train + n_val :]
    return (
        x[train_idx],
        y[train_idx],
        x[val_idx],
        y[val_idx],
        x[test_idx],
        y[test_idx],
        train_idx,
        val_idx,
        test_idx,
    )


def normalize(
    train: np.ndarray,
    val: np.ndarray,
    test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Normalize all splits using means and standard deviations from training."""
    mean = train.mean(axis=0, keepdims=True)
    std = train.std(axis=0, keepdims=True)
    std = np.where(std < 1e-12, 1.0, std)
    return (
        (train - mean) / std,
        (val - mean) / std,
        (test - mean) / std,
        mean,
        std,
    )


def make_loader(
    x: np.ndarray,
    y: np.ndarray,
    batch_size: int = 64,
    shuffle: bool = False,
) -> DataLoader:
    """Create a PyTorch loader from two aligned NumPy arrays."""
    dataset = TensorDataset(
        torch.tensor(x, dtype=torch.float32),
        torch.tensor(y, dtype=torch.float32),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    device: str | torch.device = "cpu",
) -> float:
    """Run one training or validation epoch and return sample-weighted loss."""
    is_training = optimizer is not None
    model.train(is_training)
    total_loss = 0.0
    total_samples = 0

    with torch.set_grad_enabled(is_training):
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            if optimizer is not None:
                optimizer.zero_grad()

            loss = criterion(model(xb), yb)
            if optimizer is not None:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * xb.size(0)
            total_samples += xb.size(0)

    return total_loss / total_samples


def parse_args() -> argparse.Namespace:
    """Parse surrogate-training command-line options."""
    parser = argparse.ArgumentParser(
        description="Train RHINO surrogate model from feature-layer CSV."
    )
    parser.add_argument(
        "--features",
        type=Path,
        required=True,
        help="Immutable feature-layer CSV to use for this training run.",
    )
    parser.add_argument(
        "--feature-spec",
        type=Path,
        default=DEFAULT_FEATURE_SPEC,
        help=(
            "Feature JSON used for default model columns "
            f"(default: {DEFAULT_FEATURE_SPEC})."
        ),
    )
    parser.add_argument(
        "--dataset-manifest",
        type=Path,
        help=(
            "Feature-dataset provenance manifest. If omitted, use an adjacent "
            "dataset_manifest.json or <features>.manifest.json when present."
        ),
    )
    parser.add_argument(
        "--inputs",
        nargs="+",
        help="Input columns; defaults to the input keys in --feature-spec.",
    )
    parser.add_argument(
        "--outputs",
        nargs="+",
        help="Output columns; defaults to the output keys in --feature-spec.",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=DEFAULT_OUTDIR,
        help=f"Model artifact directory (default: {DEFAULT_OUTDIR}).",
    )
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=100)
    parser.add_argument("--hidden-layers", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--mlflow",
        action="store_true",
        help="Log this training run and its serving model to MLflow.",
    )
    parser.add_argument(
        "--mlflow-tracking-uri",
        help="MLflow server URI; otherwise use MLFLOW_TRACKING_URI/default.",
    )
    parser.add_argument(
        "--mlflow-experiment",
        default="rhino-surrogate",
        help="MLflow experiment name (default: rhino-surrogate).",
    )
    parser.add_argument(
        "--mlflow-run-name",
        help="Optional human-readable MLflow run name.",
    )
    parser.add_argument(
        "--registered-model-name",
        default="rhino-surrogate",
        help="MLflow registered model name (default: rhino-surrogate).",
    )
    parser.add_argument(
        "--skip-model-registration",
        action="store_true",
        help="Log the MLflow model without creating a registry version.",
    )
    return parser.parse_args()


def train_and_save(
    *,
    args: argparse.Namespace,
    frame: pd.DataFrame,
    x: np.ndarray,
    y: np.ndarray,
    input_columns: Sequence[str],
    output_columns: Sequence[str],
    features_path: Path,
    outdir: Path,
    dataset_manifest_path: Path | None = None,
    dataset_manifest: dict[str, Any] | None = None,
    mlflow_module: Any | None = None,
    active_run: Any | None = None,
) -> None:
    """Train, persist, and optionally log one surrogate experiment."""
    (
        x_train,
        y_train,
        x_val,
        y_val,
        x_test,
        y_test,
        train_idx,
        val_idx,
        test_idx,
    ) = split_dataset(x, y, seed=args.seed)

    x_train_n, x_val_n, _, x_mean, x_std = normalize(x_train, x_val, x_test)
    y_train_n, y_val_n, _, y_mean, y_std = normalize(y_train, y_val, y_test)
    train_loader = make_loader(x_train_n, y_train_n, args.batch_size, shuffle=True)
    val_loader = make_loader(x_val_n, y_val_n, args.batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SurrogateMLP(
        input_dim=x.shape[1],
        output_dim=y.shape[1],
        hidden_dim=args.hidden_dim,
        num_hidden_layers=args.hidden_layers,
    ).to(device)

    metadata_columns = [
        column for column in KNOWN_METADATA_COLUMNS if column in frame.columns
    ]
    feature_spec_path = args.feature_spec.expanduser().resolve()
    feature_spec_exists = feature_spec_path.is_file()
    features_digest = sha256_file(features_path)
    split_indices = {
        "train_idx": train_idx.tolist(),
        "val_idx": val_idx.tolist(),
        "test_idx": test_idx.tolist(),
        "features_file": str(features_path),
        "features_sha256": features_digest,
        "row_count": len(frame),
        "feature_spec": str(feature_spec_path) if feature_spec_exists else None,
        "feature_spec_sha256": (
            sha256_file(feature_spec_path) if feature_spec_exists else None
        ),
        "metadata_columns": metadata_columns,
        "input_columns": list(input_columns),
        "output_columns": list(output_columns),
        "seed": args.seed,
        "dataset_manifest": (
            str(dataset_manifest_path) if dataset_manifest_path is not None else None
        ),
        "dataset_manifest_sha256": (
            sha256_file(dataset_manifest_path)
            if dataset_manifest_path is not None
            else None
        ),
        "mlflow_run_id": active_run.info.run_id if active_run else None,
    }

    if mlflow_module is not None:
        dataset_name = str(
            (dataset_manifest or {}).get("dataset_name", features_path.stem)
        )
        split_indices["mlflow_datasets"] = log_mlflow_dataset_inputs(
            mlflow_module,
            frame,
            features_path,
            features_digest,
            train_idx,
            val_idx,
            dataset_name,
        )
        mlflow_module.log_params(
            {
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "hidden_dim": args.hidden_dim,
                "hidden_layers": args.hidden_layers,
                "learning_rate": args.lr,
                "seed": args.seed,
                "train_rows": len(train_idx),
                "validation_rows": len(val_idx),
                "test_rows": len(test_idx),
                "input_columns": json.dumps(list(input_columns)),
                "output_columns": json.dumps(list(output_columns)),
            }
        )
        tags = {
            "features_sha256": features_digest,
            "model_type": "multi_output_regression",
            "training_device": str(device),
        }
        if split_indices["feature_spec_sha256"]:
            tags["feature_spec_sha256"] = split_indices["feature_spec_sha256"]
        if split_indices["dataset_manifest_sha256"]:
            tags["dataset_manifest_sha256"] = split_indices[
                "dataset_manifest_sha256"
            ]
        mlflow_module.set_tags(tags)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    best_val_loss = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    history = {"train_loss": [], "val_loss": []}

    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = run_epoch(model, val_loader, criterion, device=device)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        if mlflow_module is not None:
            mlflow_module.log_metrics(
                {"train_loss": train_loss, "val_loss": val_loss},
                step=epoch,
            )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
        if epoch % 20 == 0:
            print(
                f"Epoch {epoch:4d} | train_loss={train_loss:.6e} | "
                f"val_loss={val_loss:.6e}"
            )

    model.load_state_dict(best_state)
    artifact = {
        "model_state_dict": model.state_dict(),
        "model_config": {
            "input_dim": x.shape[1],
            "output_dim": y.shape[1],
            "hidden_dim": args.hidden_dim,
            "num_hidden_layers": args.hidden_layers,
        },
        "input_columns": list(input_columns),
        "output_columns": list(output_columns),
        "x_mean": torch.tensor(x_mean, dtype=torch.float32),
        "x_std": torch.tensor(x_std, dtype=torch.float32),
        "y_mean": torch.tensor(y_mean, dtype=torch.float32),
        "y_std": torch.tensor(y_std, dtype=torch.float32),
        "best_val_loss": best_val_loss,
    }
    checkpoint_path = outdir / "rhino_surrogate.pt"
    history_path = outdir / "training_history.csv"
    split_path = outdir / "split_indices.json"
    input_example_path = outdir / "input_example.csv"
    run_manifest_path = outdir / "run_manifest.json"
    torch.save(artifact, checkpoint_path)
    pd.DataFrame(history).to_csv(history_path, index=False)
    frame[list(input_columns)].head(5).to_csv(input_example_path, index=False)

    bundled_feature_spec = copy_provenance_file(
        feature_spec_path if feature_spec_exists else None,
        outdir / "provenance" / "feature_spec.json",
    )
    bundled_dataset_manifest = copy_provenance_file(
        dataset_manifest_path,
        outdir / "provenance" / "dataset_manifest.json",
    )

    if mlflow_module is not None:
        from mlflow.models import infer_signature

        from mlflow_model import RhinoSurrogatePyFunc

        serving_model = RhinoSurrogatePyFunc(
            model=model,
            input_columns=input_columns,
            output_columns=output_columns,
            x_mean=x_mean,
            x_std=x_std,
            y_mean=y_mean,
            y_std=y_std,
        )
        input_example = frame[list(input_columns)].head(5).astype(np.float32)
        output_example = serving_model.predict(None, input_example)
        signature = infer_signature(input_example, output_example)
        registered_name = (
            None if args.skip_model_registration else args.registered_model_name
        )
        version = str(torch.__version__).split("+", maxsplit=1)[0]
        model_info = mlflow_module.pyfunc.log_model(
            name="model",
            python_model=serving_model,
            input_example=input_example,
            signature=signature,
            registered_model_name=registered_name,
            code_paths=[
                str(Path(__file__).resolve()),
                str(BASE_DIR / "dataset_tracking.py"),
                str(BASE_DIR / "mlflow_model.py"),
            ],
            pip_requirements=[
                f"mlflow=={mlflow_module.__version__}",
                f"numpy=={np.__version__}",
                f"pandas=={pd.__version__}",
                f"torch=={version}",
            ],
            metadata={
                "input_columns": list(input_columns),
                "output_columns": list(output_columns),
                "features_sha256": features_digest,
            },
        )
        split_indices["mlflow_model_uri"] = model_info.model_uri
        split_indices["mlflow_registered_model_name"] = registered_name
        split_indices["mlflow_registered_model_version"] = (
            model_info.registered_model_version
        )
        mlflow_module.log_metric("best_val_loss", best_val_loss)
        mlflow_module.log_artifact(checkpoint_path, artifact_path="training")
        mlflow_module.log_artifact(history_path, artifact_path="training")
        if feature_spec_exists:
            mlflow_module.log_artifact(feature_spec_path, artifact_path="provenance")
        if dataset_manifest_path is not None:
            mlflow_module.log_artifact(
                dataset_manifest_path,
                artifact_path="provenance",
            )

    with split_path.open("w", encoding="utf-8") as stream:
        json.dump(split_indices, stream, indent=2)

    run_manifest = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "training": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "hidden_dim": args.hidden_dim,
            "hidden_layers": args.hidden_layers,
            "learning_rate": args.lr,
            "seed": args.seed,
            "train_rows": len(train_idx),
            "validation_rows": len(val_idx),
            "test_rows": len(test_idx),
            "input_columns": list(input_columns),
            "output_columns": list(output_columns),
            "device": str(device),
        },
        "dataset": {
            "features_file": str(features_path),
            "features_sha256": features_digest,
            "row_count": len(frame),
            "feature_spec_sha256": split_indices["feature_spec_sha256"],
            "dataset_manifest_sha256": split_indices[
                "dataset_manifest_sha256"
            ],
        },
        "source": {
            "repository": str(RHINO_ROOT.parent),
            "git": git_metadata(RHINO_ROOT.parent),
            "command": sys.argv,
        },
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
        },
        "scheduler": {
            key.lower(): os.environ[key]
            for key in (
                "SLURM_JOB_ID",
                "SLURM_JOB_NAME",
                "SLURM_CLUSTER_NAME",
                "NERSC_HOST",
            )
            if key in os.environ
        },
        "artifacts": {
            "model": checkpoint_path.name,
            "model_sha256": sha256_file(checkpoint_path),
            "training_history": history_path.name,
            "training_history_sha256": sha256_file(history_path),
            "splits": split_path.name,
            "splits_sha256": sha256_file(split_path),
            "input_example": input_example_path.name,
            "feature_spec": (
                str(bundled_feature_spec.relative_to(outdir))
                if bundled_feature_spec is not None
                else None
            ),
            "dataset_manifest": (
                str(bundled_dataset_manifest.relative_to(outdir))
                if bundled_dataset_manifest is not None
                else None
            ),
        },
    }
    with run_manifest_path.open("w", encoding="utf-8") as stream:
        json.dump(run_manifest, stream, indent=2)

    if mlflow_module is not None:
        mlflow_module.log_artifact(split_path, artifact_path="training")
        mlflow_module.log_artifact(run_manifest_path, artifact_path="training")
        mlflow_module.log_artifact(input_example_path, artifact_path="training")

    print(f"Training device: {device}")
    print(
        f"Rows: {len(frame)}; inputs: {len(input_columns)}; "
        f"outputs: {len(output_columns)}"
    )
    print("\nSaved:")
    print(checkpoint_path)
    print(history_path)
    print(split_path)
    print(run_manifest_path)
    print(input_example_path)
    if active_run is not None:
        print(f"MLflow run: {active_run.info.run_id}")
        if split_indices.get("mlflow_model_uri"):
            print(f"MLflow model: {split_indices['mlflow_model_uri']}")


def main() -> None:
    """Train the surrogate locally and optionally track it with MLflow."""
    args = parse_args()
    default_inputs: list[str] = []
    default_outputs: list[str] = []
    if args.inputs is None or args.outputs is None:
        default_inputs, default_outputs = load_feature_roles(args.feature_spec)
    input_columns = args.inputs if args.inputs is not None else default_inputs
    output_columns = args.outputs if args.outputs is not None else default_outputs
    features_path = args.features.expanduser().resolve()
    outdir = args.outdir.expanduser().resolve()

    if min(args.epochs, args.batch_size, args.hidden_dim, args.hidden_layers) <= 0:
        raise ValueError(
            "Epochs, batch size, hidden dimension, and hidden layers must be positive"
        )
    if args.lr <= 0:
        raise ValueError("Learning rate must be greater than zero")
    if args.mlflow and not args.skip_model_registration:
        if not args.registered_model_name.strip():
            raise ValueError("Registered model name must not be empty")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    outdir.mkdir(parents=True, exist_ok=True)
    frame, x, y = load_feature_arrays(
        features_path,
        input_columns,
        output_columns,
    )
    dataset_manifest_path, dataset_manifest = resolve_dataset_manifest(
        features_path,
        args.dataset_manifest,
    )
    if dataset_manifest is not None:
        expected_rows = dataset_manifest.get("rows")
        if expected_rows is not None and expected_rows != len(frame):
            raise ValueError(
                f"Dataset manifest records {expected_rows} rows, but the feature "
                f"CSV contains {len(frame)}"
            )

    train_arguments = {
        "args": args,
        "frame": frame,
        "x": x,
        "y": y,
        "input_columns": input_columns,
        "output_columns": output_columns,
        "features_path": features_path,
        "outdir": outdir,
        "dataset_manifest_path": dataset_manifest_path,
        "dataset_manifest": dataset_manifest,
    }
    if not args.mlflow:
        train_and_save(**train_arguments)
        return

    try:
        import mlflow
    except ImportError as error:
        raise RuntimeError(
            "MLflow logging was requested but mlflow is not installed"
        ) from error

    if args.mlflow_tracking_uri:
        mlflow.set_tracking_uri(args.mlflow_tracking_uri)
    mlflow.set_experiment(args.mlflow_experiment)
    with mlflow.start_run(run_name=args.mlflow_run_name) as active_run:
        train_and_save(
            **train_arguments,
            mlflow_module=mlflow,
            active_run=active_run,
        )


if __name__ == "__main__":
    main()
