"""Laser HDF5 -> openPMD/ADIOS2 BP5 conversion entry points."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Sequence

import h5py

from metadata import load_conversion_metadata, read_laser_h5
from utils import default_bp_filename, get_default_out_dir, remove_path
from writer import write_laser_models_to_openpmd


def _validate_h5_input(h5_path: Path) -> None:
    """Reject missing, unreadable, or non-laser HDF5 inputs before conversion."""
    if not h5_path.is_file():
        raise FileNotFoundError(f"Laser HDF5 input does not exist or is not a file: {h5_path}")

    try:
        with h5py.File(h5_path, "r") as h5_file:
            epoch = h5_file.get("epoch")
            if not isinstance(epoch, h5py.Dataset) or epoch.ndim != 1:
                raise ValueError(
                    f"Laser HDF5 input must contain a one-dimensional 'epoch' dataset: {h5_path}"
                )
    except OSError as exc:
        raise ValueError(f"Laser HDF5 input is unreadable or truncated: {h5_path}") from exc


def laser_run_to_adios(
    h5_paths: Sequence[str | os.PathLike[str]],
    out_bp_path: str | os.PathLike[str] | None = None,
    *,
    excel_candidates: list[str | os.PathLike[str]] | None = None,
    documentation_dir: str | os.PathLike[str] | None = None,
    domain_notes_source: str = "",
    overwrite: bool = True,
    strict: bool = False,
    verbose: bool = True,
) -> str:
    """Convert all HDF5 rollover segments for one run into one BP5 Series."""
    paths = [Path(path) for path in h5_paths]
    if not paths:
        raise ValueError("At least one Laser HDF5 input is required")
    for path in paths:
        _validate_h5_input(path)

    metadata_kwargs: dict[str, Any] = {
        "excel_candidates": excel_candidates,
        "domain_notes_source": domain_notes_source,
        "verbose": verbose,
    }
    if documentation_dir is not None:
        metadata_kwargs["documentation_dir"] = documentation_dir

    excel_meta, documentation_meta = load_conversion_metadata(**metadata_kwargs)
    models = [
        read_laser_h5(path, excel_meta=excel_meta, strict=strict)
        for path in paths
    ]

    if out_bp_path is None:
        out_bp_path = get_default_out_dir() / default_bp_filename(paths[0])
    out_bp_path = Path(out_bp_path)

    if out_bp_path.exists():
        if not overwrite:
            raise FileExistsError(f"Output already exists: {out_bp_path}")
        remove_path(out_bp_path)
    out_bp_path.parent.mkdir(parents=True, exist_ok=True)

    written = write_laser_models_to_openpmd(models, out_bp_path, documentation_meta)

    if verbose:
        total_valid = sum(model.num_valid_samples for model in models)
        total_allocated = sum(model.n_requested for model in models)
        print(
            "[done] Wrote Laser BP5: "
            f"{written} ({len(models)} source segment(s), "
            f"{total_valid}/{total_allocated} recorded/allocated events)"
        )

    return written


def laser_to_adios(
    h5_path: str | os.PathLike[str],
    out_bp_path: str | os.PathLike[str] | None = None,
    *,
    excel_candidates: list[str | os.PathLike[str]] | None = None,
    documentation_dir: str | os.PathLike[str] | None = None,
    domain_notes_source: str = "",
    overwrite: bool = True,
    strict: bool = False,
    verbose: bool = True,
) -> str:
    """Compatibility entry point for a run contained in one HDF5 file."""
    return laser_run_to_adios(
        [h5_path],
        out_bp_path=out_bp_path,
        excel_candidates=excel_candidates,
        documentation_dir=documentation_dir,
        domain_notes_source=domain_notes_source,
        overwrite=overwrite,
        strict=strict,
        verbose=verbose,
    )
