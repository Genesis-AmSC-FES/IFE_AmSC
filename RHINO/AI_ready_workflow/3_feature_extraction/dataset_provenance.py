"""Persist immutable feature datasets and their extraction provenance."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import pandas as pd


DATASET_MANIFEST_SCHEMA_VERSION = 1
DATASET_FILENAME = "rhino_features.csv"
MANIFEST_FILENAME = "dataset_manifest.json"
DATASET_ID_LENGTH = 12


def sha256_file(path: str | Path) -> str:
    """Return the SHA-256 digest of one file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_frame_to_temporary(frame: pd.DataFrame, directory: Path) -> Path:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".rhino-features-", suffix=".csv.tmp", dir=directory
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        frame.to_csv(temporary_path, index=False)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise
    return temporary_path


def _write_json_exclusive(path: Path, document: Mapping[str, Any]) -> bool:
    """Create a JSON file without replacing an existing immutable manifest."""
    try:
        with path.open("x", encoding="utf-8") as stream:
            json.dump(document, stream, indent=2, sort_keys=True)
            stream.write("\n")
    except FileExistsError:
        return False
    return True


def persist_versioned_dataset(
    frame: pd.DataFrame,
    dataset_root: str | Path,
    provenance: Mapping[str, Any],
) -> tuple[Path, Path, str, bool]:
    """Store a content-addressed CSV and immutable sidecar manifest.

    Returns ``(dataset_path, manifest_path, sha256, reused)``. Repeated builds
    that produce byte-identical CSV content reuse the same dataset directory.
    """
    root = Path(dataset_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    temporary_path = _write_frame_to_temporary(frame, root)
    digest = sha256_file(temporary_path)
    dataset_id = digest[:DATASET_ID_LENGTH]
    version_dir = root / dataset_id
    version_dir.mkdir(exist_ok=True)
    dataset_path = version_dir / DATASET_FILENAME
    reused = dataset_path.exists()

    if reused:
        existing_digest = sha256_file(dataset_path)
        if existing_digest != digest:
            temporary_path.unlink(missing_ok=True)
            raise RuntimeError(
                f"SHA-256 prefix collision for dataset ID {dataset_id}: "
                f"existing file has digest {existing_digest}, new file has {digest}"
            )
        temporary_path.unlink()
    else:
        try:
            os.link(temporary_path, dataset_path)
        except FileExistsError:
            reused = True
            if sha256_file(dataset_path) != digest:
                raise RuntimeError(
                    f"Concurrent dataset creation produced invalid content: {dataset_path}"
                )
        finally:
            temporary_path.unlink(missing_ok=True)

    manifest_path = version_dir / MANIFEST_FILENAME
    manifest = {
        "schema_version": DATASET_MANIFEST_SCHEMA_VERSION,
        "dataset_name": "rhino-features",
        "dataset_id": dataset_id,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset_file": str(dataset_path),
        "sha256": digest,
        "rows": int(len(frame)),
        "columns": list(frame.columns),
        **dict(provenance),
    }
    _write_json_exclusive(manifest_path, manifest)
    return dataset_path, manifest_path, digest, reused


def persist_explicit_dataset(
    frame: pd.DataFrame,
    output_path: str | Path,
    provenance: Mapping[str, Any],
) -> tuple[Path, Path, str]:
    """Write a replaceable explicit CSV export and adjacent manifest."""
    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = _write_frame_to_temporary(frame, output.parent)
    os.replace(temporary_path, output)
    digest = sha256_file(output)
    manifest_path = output.with_suffix(".manifest.json")
    manifest = {
        "schema_version": DATASET_MANIFEST_SCHEMA_VERSION,
        "dataset_name": "rhino-features",
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset_file": str(output),
        "sha256": digest,
        "rows": int(len(frame)),
        "columns": list(frame.columns),
        **dict(provenance),
    }
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{manifest_path.name}.", suffix=".tmp", dir=manifest_path.parent
    )
    os.close(descriptor)
    temporary_manifest = Path(temporary_name)
    try:
        temporary_manifest.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary_manifest, manifest_path)
    finally:
        temporary_manifest.unlink(missing_ok=True)
    return output, manifest_path, digest
