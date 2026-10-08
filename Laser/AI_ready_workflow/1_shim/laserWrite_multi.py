"""Batch conversion of Phoenix Laser HDF5 runs to BP5.

Only immediate YYYY-MM-DD folders and target_data are discovered. HDF5 files
with the same recorded run number and run date are rollover segments of one
logical run and are stacked into one continuous openPMD Series.
"""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

import h5py
import numpy as np

from laserWrite import laser_run_to_adios
from utils import (
    default_bp_filename,
    get_default_out_dir,
    infer_run_date_from_h5,
    infer_run_number_from_h5,
    parse_laser_filename,
)


_DATE_DIRECTORY_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_DEFAULT_EXCLUDE_DIRS = {
    "bp_output",
    "campaign_tar",
    "campaign-tar",
    "doc",
    "documentation",
    "mlrun",
    "older data",
    "older_data",
    "older-data",
    "__pycache__",
}


def _is_date_directory(path: Path) -> bool:
    return path.is_dir() and _DATE_DIRECTORY_RE.fullmatch(path.name) is not None


def _is_target_data_directory(path: Path) -> bool:
    return path.is_dir() and path.name.casefold() == "target_data"


def _is_excluded(candidate: Path, root: Path, exclude_dirs: set[str]) -> bool:
    try:
        parts = candidate.relative_to(root).parts
    except ValueError:
        parts = candidate.parts
    for part in parts[:-1]:
        normalized = part.casefold()
        if normalized in exclude_dirs or normalized.startswith("mlrun"):
            return True
    return False


def _discovery_roots(path: Path) -> list[Path]:
    """Return only approved acquisition roots for a directory input."""
    if _is_date_directory(path) or _is_target_data_directory(path):
        return [path]
    if not path.is_dir():
        return []
    return sorted(
        child
        for child in path.iterdir()
        if _is_date_directory(child) or _is_target_data_directory(child)
    )


def discover_h5_files(
    input_path: str | os.PathLike[str],
    recursive: bool = False,
    exclude_dirs: set[str] | None = None,
) -> list[Path]:
    """Discover HDF5 files only in date-named folders and target_data."""
    path = Path(input_path)
    if path.is_file():
        return [path] if path.suffix.casefold() == ".h5" else []

    excluded = {name.casefold() for name in (exclude_dirs or _DEFAULT_EXCLUDE_DIRS)}
    files: list[Path] = []
    for root in _discovery_roots(path):
        candidates = root.rglob("*.h5") if recursive else root.glob("*.h5")
        for candidate in candidates:
            if candidate.is_file() and not _is_excluded(candidate, root, excluded):
                files.append(candidate)
    return sorted(set(files))


def _logical_run_identity(path: Path) -> tuple[str, str, str]:
    """Return a grouping key based on recorded run number and calendar date."""
    info = parse_laser_filename(path)
    run_id = info.get("run_id")
    run_date = info.get("run_date")
    try:
        run_id = run_id or infer_run_number_from_h5(path)
        run_date = run_date or infer_run_date_from_h5(path)
    except (OSError, ValueError):
        # Keep unreadable inputs in the inventory so conversion can report the
        # failure without preventing unrelated runs from being processed.
        pass
    if run_id and run_date:
        normalized = re.sub(r"[^A-Za-z0-9._-]+", "_", str(run_id)).strip("_")
        return ("run", normalized, str(run_date))
    return ("file", str(path.resolve()), "")


def _first_valid_epoch(path: Path) -> float | None:
    """Read the first populated event time without loading the full run model."""
    try:
        with h5py.File(path, "r") as h5_file:
            for name in ("phoeniX:epoch", "epoch"):
                dataset = h5_file.get(name)
                if not isinstance(dataset, h5py.Dataset):
                    continue
                values = np.asarray(dataset[()]).reshape(-1)
                if not np.issubdtype(values.dtype, np.number):
                    continue
                valid = values[np.isfinite(values) & (values > 0)]
                if valid.size:
                    return float(valid[0])
    except (OSError, TypeError, ValueError):
        return None
    return None


def _natural_path_key(path: Path) -> tuple[object, ...]:
    parts = re.split(r"(\d+)", str(path).casefold())
    return tuple(int(part) if part.isdigit() else part for part in parts)


def _segment_sort_key(path: Path) -> tuple[object, ...]:
    epoch = _first_valid_epoch(path)
    return (epoch is None, epoch if epoch is not None else float("inf"), _natural_path_key(path))


def group_h5_files_by_run(
    h5_files: list[Path],
    out_dir: str | os.PathLike[str],
) -> list[tuple[Path, list[Path]]]:
    """Group rollover files and choose one canonical run/date BP5 output."""
    grouped: dict[tuple[str, str, str], list[Path]] = {}
    for h5_path in h5_files:
        grouped.setdefault(_logical_run_identity(h5_path), []).append(h5_path)

    planned: list[tuple[Path, list[Path]]] = []
    output_root = Path(out_dir)
    output_owners: dict[Path, tuple[str, str, str]] = {}
    for identity, sources in grouped.items():
        ordered_sources = sorted(sources, key=_segment_sort_key)
        if identity[0] == "run":
            output_name = f"run-{identity[1]}-{identity[2]}.bp5"
        else:
            output_name = default_bp_filename(ordered_sources[0])
        output_path = output_root / output_name
        previous = output_owners.get(output_path)
        if previous is not None and previous != identity:
            raise ValueError(f"Output-name collision for {output_path}: {previous} and {identity}")
        output_owners[output_path] = identity
        planned.append((output_path, ordered_sources))
    return sorted(planned, key=lambda item: str(item[0]))


def convert_many(
    input_path: str | os.PathLike[str],
    out_dir: str | os.PathLike[str] | None = None,
    *,
    recursive: bool = False,
    excel_candidates: list[str | os.PathLike[str]] | None = None,
    documentation_dir: str | os.PathLike[str] | None = None,
    domain_notes_source: str = "",
    overwrite: bool = True,
    skip_existing: bool = False,
    strict: bool = False,
    continue_on_error: bool = True,
    verbose: bool = True,
) -> list[str]:
    """Convert every discovered logical run into one output BP5 Series."""
    h5_files = discover_h5_files(input_path, recursive=recursive)
    out_dir = Path(out_dir) if out_dir is not None else get_default_out_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    run_groups = group_h5_files_by_run(h5_files, out_dir)

    written: list[str] = []
    failures: list[tuple[list[Path], Exception]] = []
    skipped: list[Path] = []

    for out_bp_path, source_files in run_groups:
        if skip_existing and out_bp_path.exists():
            skipped.append(out_bp_path)
            if verbose:
                print(f"[skip] Output already exists: {out_bp_path}")
            continue

        if verbose and len(source_files) > 1:
            print(
                f"[stack] {len(source_files)} HDF5 rollover segments -> {out_bp_path.name}"
            )
            for segment_index, source in enumerate(source_files):
                print(f"        {segment_index:04d}: {source}")

        try:
            written_path = laser_run_to_adios(
                h5_paths=source_files,
                out_bp_path=out_bp_path,
                excel_candidates=excel_candidates,
                documentation_dir=documentation_dir,
                domain_notes_source=domain_notes_source,
                overwrite=overwrite,
                strict=strict,
                verbose=verbose,
            )
            written.append(written_path)
        except Exception as exc:
            failures.append((source_files, exc))
            if not continue_on_error:
                raise
            print(f"[error] Failed to convert run from {source_files}: {exc}")

    if verbose:
        stacked_segments = sum(len(sources) for _, sources in run_groups)
        print(
            f"[summary] Converted {len(written)} run(s) from {stacked_segments} HDF5 file(s); "
            f"skipped {len(skipped)} existing run(s); failed {len(failures)} run(s)."
        )
    return written


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert Phoenix Laser HDF5 runs to openPMD/ADIOS2 BP5."
    )
    parser.add_argument("input_path", help="Input .h5 file or Laser data root directory.")
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Output directory for .bp5 files. Defaults to $LASER_BP_OUT_DIR or <LASER_ROOT>/bp_output.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively search approved date and target_data directories.",
    )
    parser.add_argument(
        "--excel",
        action="append",
        dest="excel_candidates",
        help="Optional PV metadata Excel path. May be repeated.",
    )
    parser.add_argument("--documentation-dir", help="Optional documentation directory.")
    parser.add_argument(
        "--domain-notes",
        default="",
        help="Optional path to laser-pv-details.docx for provenance.",
    )
    parser.add_argument("--no-overwrite", action="store_true", help="Fail if output already exists.")
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Leave existing BP5 outputs untouched and continue with remaining runs.",
    )
    parser.add_argument(
        "--strict", action="store_true", help="Raise on optional shape/classification problems."
    )
    parser.add_argument(
        "--stop-on-error", action="store_true", help="Stop at the first failed conversion."
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress routine conversion messages.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    convert_many(
        input_path=args.input_path,
        out_dir=args.out_dir,
        recursive=args.recursive,
        excel_candidates=args.excel_candidates,
        documentation_dir=args.documentation_dir,
        domain_notes_source=args.domain_notes,
        overwrite=not args.no_overwrite,
        skip_existing=args.skip_existing,
        strict=args.strict,
        continue_on_error=not args.stop_on_error,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
