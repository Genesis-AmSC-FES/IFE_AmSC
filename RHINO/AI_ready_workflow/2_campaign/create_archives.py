"""Create RHINO TAR and HPC Campaign archives from a JSON specification."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shlex
import subprocess
import tarfile
from pathlib import Path
from typing import Any


DEFAULT_SPEC = Path(__file__).with_name("campaign_spec.json")
RUN_ID_PATTERN = re.compile(r"\d{2}-\d{2}-\d{2}.*")
SUPPORTED_TAR_STORAGE_SYSTEMS = {"Kronos", "HPSS", "fs", "https", "S3"}


def load_spec(path: Path) -> dict[str, Any]:
    """Load and validate the archive-related campaign settings."""
    with path.open(encoding="utf-8") as stream:
        spec = json.load(stream)

    required = {
        "RHINO_DATA_ROOT": str,
        "CAMPAIGN_STORE": str,
        "CAMPAIGN_NAMESPACE": str,
        "ARCHIVE_PREFIX": str,
        "INPUT_DIRS": list,
        "TAR_OUTPUT_DIR": str,
        "TAR_PREFIX": str,
        "TAR_STORAGE_SYSTEM": str,
        "TAR_STORAGE_HOST": str,
    }
    for key, expected_type in required.items():
        if key not in spec:
            raise ValueError(f"Missing required setting: {key}")
        if not isinstance(spec[key], expected_type):
            raise TypeError(
                f"Setting '{key}' must be {expected_type.__name__}, "
                f"not {type(spec[key]).__name__}"
            )

    if not spec["INPUT_DIRS"]:
        raise ValueError("INPUT_DIRS must contain at least one directory")
    if not all(isinstance(item, str) and item for item in spec["INPUT_DIRS"]):
        raise TypeError("Every INPUT_DIRS entry must be a non-empty string")
    if not spec["TAR_PREFIX"]:
        raise ValueError("TAR_PREFIX must not be empty")
    if spec["TAR_STORAGE_SYSTEM"] not in SUPPORTED_TAR_STORAGE_SYSTEMS:
        allowed = ", ".join(sorted(SUPPORTED_TAR_STORAGE_SYSTEMS))
        raise ValueError(f"TAR_STORAGE_SYSTEM must be one of: {allowed}")
    if not spec["TAR_STORAGE_HOST"]:
        raise ValueError("TAR_STORAGE_HOST must not be empty")

    return spec


def resolve_from_root(data_root: Path, configured_path: str) -> Path:
    """Resolve a configured path relative to the RHINO data root."""
    path = Path(configured_path).expanduser()
    return path if path.is_absolute() else data_root / path


def validate_input_directories(
    data_root: Path, input_directories: list[str]
) -> list[Path]:
    """Resolve input directories and ensure each one exists under the data root."""
    resolved: list[Path] = []
    for configured_directory in input_directories:
        configured_path = Path(configured_directory)
        if configured_path.is_absolute():
            raise ValueError(
                f"INPUT_DIRS entries must be relative: {configured_directory}"
            )
        directory = (data_root / configured_path).resolve()
        try:
            directory.relative_to(data_root)
        except ValueError as error:
            raise ValueError(
                f"INPUT_DIRS entry escapes RHINO_DATA_ROOT: "
                f"{configured_directory}"
            ) from error
        if not directory.is_dir():
            raise FileNotFoundError(f"BP5 input directory does not exist: {directory}")
        resolved.append(directory)
    return resolved


def discover_datasets(input_directories: list[Path]) -> list[Path]:
    """Return all top-level BP5 datasets in deterministic order."""
    datasets: list[Path] = []
    for directory in input_directories:
        print(f"Scanning directory: {directory}")
        datasets.extend(directory.glob("*.bp5"))
    return sorted(datasets, key=lambda path: str(path))


def run_id_from_path(dataset: Path) -> str:
    """Derive a RHINO run identifier from a BP5 dataset name."""
    match = RUN_ID_PATTERN.search(dataset.stem)
    return match.group(0) if match else dataset.stem


def run_command(command: list[str], *, cwd: Path, dry_run: bool) -> None:
    """Display and optionally execute one external command."""
    print(f"  Command : {shlex.join(command)}")
    if not dry_run:
        subprocess.run(command, cwd=cwd, check=True)


def sha256_digest(path: Path) -> str:
    """Return the SHA-256 digest of a file without loading it into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_checksum(tar_path: Path) -> Path:
    """Write a sha256sum-compatible checksum file for a TAR archive."""
    checksum_path = tar_path.with_name(f"{tar_path.name}.sha256")
    checksum_path.write_text(
        f"{sha256_digest(tar_path)}  {tar_path.name}\n",
        encoding="utf-8",
    )
    return checksum_path


def verify_checksum(tar_path: Path) -> None:
    """Raise an error if a TAR archive does not match its checksum sidecar."""
    checksum_path = tar_path.with_name(f"{tar_path.name}.sha256")
    if not checksum_path.is_file():
        raise FileNotFoundError(
            f"Existing TAR has no checksum file: {checksum_path}. "
            "Use --rebuild-tars to replace it."
        )

    fields = checksum_path.read_text(encoding="utf-8").strip().split(maxsplit=1)
    if len(fields) != 2 or fields[1].lstrip(" *") != tar_path.name:
        raise ValueError(f"Invalid checksum file: {checksum_path}")
    actual = sha256_digest(tar_path)
    if actual != fields[0].lower():
        raise ValueError(
            f"SHA-256 verification failed for {tar_path}. "
            "Use --rebuild-tars to replace it."
        )


def tar_path_for_directory(
    input_directory: Path, tar_output_dir: Path, tar_prefix: str
) -> Path:
    """Return the TAR path associated with one configured input directory."""
    return tar_output_dir / f"{tar_prefix}-{input_directory.name}.tar"


def validate_tar_members(
    tar_path: Path, datasets: list[Path], data_root: Path
) -> None:
    """Ensure an existing TAR contains only the datasets assigned to its ACA."""
    expected = {
        str(dataset.relative_to(data_root)).rstrip("/") for dataset in datasets
    }
    found: set[str] = set()

    with tarfile.open(tar_path, "r") as archive:
        for member in archive:
            member_name = member.name.removeprefix("./").rstrip("/")
            matching_dataset = next(
                (
                    dataset
                    for dataset in expected
                    if member_name == dataset
                    or member_name.startswith(f"{dataset}/")
                ),
                None,
            )
            if matching_dataset is None:
                raise ValueError(
                    f"Existing TAR contains a path not present in its ACA: "
                    f"{member.name}. Use --rebuild-tars to replace {tar_path}."
                )
            found.add(matching_dataset)

    missing = expected - found
    if missing:
        missing_list = ", ".join(sorted(missing))
        raise ValueError(
            f"Existing TAR is missing ACA datasets: {missing_list}. "
            f"Use --rebuild-tars to replace {tar_path}."
        )


def create_tar_archives(
    *,
    archive_datasets: dict[str, list[Path]],
    data_root: Path,
    tar_output_dir: Path,
    tar_prefix: str,
    rebuild_tars: bool,
    dry_run: bool,
) -> dict[str, Path]:
    """Create, verify, and index one TAR from each ACA's dataset list."""
    tar_paths: dict[str, Path] = {}
    seen_paths: set[Path] = set()

    if not dry_run:
        tar_output_dir.mkdir(parents=True, exist_ok=True)

    for archive, datasets in archive_datasets.items():
        dataset_directories = {dataset.parent for dataset in datasets}
        if len(dataset_directories) != 1:
            raise ValueError(
                f"Campaign archive {archive} spans multiple date directories"
            )
        input_directory = next(iter(dataset_directories))
        relative_datasets = [dataset.relative_to(data_root) for dataset in datasets]
        tar_path = tar_path_for_directory(
            input_directory, tar_output_dir, tar_prefix
        )
        checksum_path = tar_path.with_name(f"{tar_path.name}.sha256")
        index_path = tar_path.with_name(f"{tar_path.name}.idx")

        if tar_path in seen_paths:
            raise ValueError(
                f"Multiple campaign archives produce the same TAR name: {tar_path}"
            )
        seen_paths.add(tar_path)
        tar_paths[archive] = tar_path
        if tar_output_dir.is_relative_to(input_directory):
            raise ValueError(
                f"TAR_OUTPUT_DIR cannot be inside a directory being archived: "
                f"{input_directory}"
            )

        print(f"\nPreparing TAR for {archive}:")
        print(f"  Datasets: {len(relative_datasets)}")
        create_tar = rebuild_tars or not tar_path.exists()
        if create_tar:
            run_command(
                [
                    "tar",
                    "-cf",
                    str(tar_path),
                    *(str(dataset) for dataset in relative_datasets),
                ],
                cwd=data_root,
                dry_run=dry_run,
            )
            print(f"  Checksum: {checksum_path}")
            if not dry_run:
                write_checksum(tar_path)
        else:
            print(f"  Reusing  : {tar_path}")
            if not checksum_path.is_file():
                raise FileNotFoundError(
                    f"Existing TAR has no checksum file: {checksum_path}. "
                    "Use --rebuild-tars to replace it."
                )
            if dry_run:
                print(f"  Verify   : {checksum_path}")
            else:
                verify_checksum(tar_path)
                validate_tar_members(tar_path, datasets, data_root)
                print(f"  Verified : {checksum_path}")
                print("  Contents : match ACA dataset list")

        index_is_stale = (
            tar_path.exists()
            and index_path.exists()
            and tar_path.stat().st_mtime_ns > index_path.stat().st_mtime_ns
        )
        if create_tar or not index_path.exists() or index_is_stale:
            run_command(
                ["hpc_campaign", "taridx", str(tar_path), str(index_path)],
                cwd=data_root,
                dry_run=dry_run,
            )
        else:
            print(f"  Reusing  : {index_path}")

    return tar_paths

def archive_name_for_directory(
    input_directory: Path, archive_prefix: str
) -> str:
    """Return the dated campaign archive name for one input directory."""
    return f"{archive_prefix}-{input_directory.name}.aca"


def create_campaign_archives(
    *,
    datasets: list[Path],
    input_directories: list[Path],
    data_root: Path,
    campaign_store: Path,
    archive_prefix: str,
    dry_run: bool,
) -> dict[str, list[Path]]:
    """Create one campaign archive for each dated input directory."""
    archive_datasets: dict[str, list[Path]] = {}

    for input_directory in input_directories:
        archive = archive_name_for_directory(input_directory, archive_prefix)
        dated_datasets = [
            dataset for dataset in datasets if dataset.parent == input_directory
        ]
        if not dated_datasets:
            print(f"\nSkipping empty date directory: {input_directory}")
            continue
        archive_datasets[archive] = dated_datasets

        for dataset_number, dataset in enumerate(dated_datasets):
            relative_dataset = dataset.relative_to(data_root)
            run_id = run_id_from_path(dataset)

            print("\nAdding dataset:")
            print(f"  File    : {relative_dataset}")
            print(f"  Run ID  : {run_id}")
            print(f"  Archive : {archive}")

            command = [
                "hpc_campaign",
                "manager",
                "--campaign_store",
                str(campaign_store),
                archive,
            ]
            if dataset_number == 0:
                command.append("--truncate")
            command.extend(["data", str(relative_dataset), "--name", run_id])
            run_command(command, cwd=data_root, dry_run=dry_run)

    return archive_datasets


def register_tar_replicas(
    *,
    archive_datasets: dict[str, list[Path]],
    tar_paths: dict[str, Path],
    data_root: Path,
    campaign_store: Path,
    storage_system: str,
    storage_host: str,
    dry_run: bool,
) -> None:
    """Register the one NERSC TAR copy associated with each campaign archive."""
    for archive in archive_datasets:
        tar_path = tar_paths[archive]
        index_path = tar_path.with_name(f"{tar_path.name}.idx")
        print("\nRegistering NERSC TAR replicas:")
        print(f"  Archive : {archive}")
        print(f"  TAR     : {tar_path}")
        command = [
            "hpc_campaign",
            "manager",
            "--campaign_store",
            str(campaign_store),
            archive,
            "add-archival-storage",
            storage_system,
            storage_host,
            str(tar_path.parent),
            tar_path.name,
            str(index_path),
        ]
        run_command(command, cwd=data_root, dry_run=dry_run)

def create_archives(
    spec: dict[str, Any],
    *,
    dry_run: bool = False,
    rebuild_tars: bool = False,
) -> None:
    """Create date-based ACAs, matching TARs, and NERSC TAR replicas."""
    data_root = Path(spec["RHINO_DATA_ROOT"]).expanduser().resolve()
    campaign_store = Path(spec["CAMPAIGN_STORE"]).expanduser()
    archive_prefix = spec["ARCHIVE_PREFIX"]

    if not data_root.is_dir():
        raise FileNotFoundError(f"RHINO data root does not exist: {data_root}")
    input_directories = validate_input_directories(
        data_root, spec["INPUT_DIRS"]
    )
    datasets = discover_datasets(input_directories)
    if not datasets:
        raise FileNotFoundError("No .bp5 datasets were discovered")

    print(f"Discovered {len(datasets)} dataset(s).")
    archive_datasets = create_campaign_archives(
        datasets=datasets,
        input_directories=input_directories,
        data_root=data_root,
        campaign_store=campaign_store,
        archive_prefix=archive_prefix,
        dry_run=dry_run,
    )
    archive_count = len(archive_datasets)
    print(f"Dated archive count: {archive_count}")

    tar_output_dir = resolve_from_root(
        data_root, spec["TAR_OUTPUT_DIR"]
    ).resolve()
    tar_paths = create_tar_archives(
        archive_datasets=archive_datasets,
        data_root=data_root,
        tar_output_dir=tar_output_dir,
        tar_prefix=spec["TAR_PREFIX"],
        rebuild_tars=rebuild_tars,
        dry_run=dry_run,
    )
    register_tar_replicas(
        archive_datasets=archive_datasets,
        tar_paths=tar_paths,
        data_root=data_root,
        campaign_store=campaign_store,
        storage_system=spec["TAR_STORAGE_SYSTEM"],
        storage_host=spec["TAR_STORAGE_HOST"],
        dry_run=dry_run,
    )

    print(
        f"\nCampaign archive creation complete: {archive_count} archive(s), "
        f"{len(tar_paths)} TAR file(s), one campaign archive per date."
    )

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Create RHINO TAR files and campaign archives from BP5 datasets."
        )
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
        help="Discover inputs and print all commands without changing files.",
    )
    parser.add_argument(
        "--rebuild-tars",
        action="store_true",
        help="Replace TAR, checksum, and TAR index files instead of reusing them.",
    )
    args = parser.parse_args()

    create_archives(
        load_spec(args.spec),
        dry_run=args.dry_run,
        rebuild_tars=args.rebuild_tars,
    )


if __name__ == "__main__":
    main()
