"""Create date-windowed Laser HPC Campaign archives and TAR replicas."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shlex
import subprocess
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, Sequence


DATE_RE = re.compile(r"(?<!\d)(20\d{2}-\d{2}-\d{2})(?!\d)")


@dataclass(frozen=True)
class Dataset:
    path: Path
    relative_path: Path
    run_date: date

    @property
    def name(self) -> str:
        return self.path.name.removesuffix(".bp5")


@dataclass(frozen=True)
class CampaignGroup:
    archive_name: str
    tar_path: Path
    datasets: tuple[Dataset, ...]

    @property
    def start_date(self) -> date:
        return self.datasets[0].run_date

    @property
    def end_date(self) -> date:
        return self.datasets[-1].run_date


def parse_dataset_date(path: Path) -> date:
    match = DATE_RE.search(path.name)
    if match is None:
        raise ValueError(f"No YYYY-MM-DD acquisition date in BP5 name: {path.name}")
    return date.fromisoformat(match.group(1))


def discover_datasets(bp5_input_dir: Path, data_root: Path) -> list[Dataset]:
    """Discover top-level BP5 file- or directory-based Series."""
    if not bp5_input_dir.is_dir():
        raise FileNotFoundError(f"BP5 input directory does not exist: {bp5_input_dir}")
    datasets: list[Dataset] = []
    skipped: list[Path] = []
    for path in sorted(bp5_input_dir.glob("*.bp5")):
        try:
            run_date = parse_dataset_date(path)
            relative_path = path.resolve().relative_to(data_root.resolve())
        except (ValueError, OSError) as exc:
            skipped.append(path)
            print(f"[skip] {path}: {exc}")
            continue
        datasets.append(Dataset(path.resolve(), relative_path, run_date))
    if skipped:
        print(f"[summary] Skipped {len(skipped)} BP5 item(s) without usable dates.")
    return sorted(datasets, key=lambda item: (item.run_date, item.path.name))


def group_datasets(
    datasets: Sequence[Dataset],
    *,
    window_days: int,
    max_datasets: int = 0,
) -> list[list[Dataset]]:
    """Group chronologically into windows anchored at each group's first date."""
    if window_days <= 0:
        raise ValueError("window_days must be positive")
    if max_datasets < 0:
        raise ValueError("max_datasets cannot be negative")

    groups: list[list[Dataset]] = []
    current: list[Dataset] = []
    window_end: date | None = None
    for dataset in datasets:
        date_overflow = window_end is not None and dataset.run_date >= window_end
        count_overflow = max_datasets > 0 and len(current) >= max_datasets
        if current and (date_overflow or count_overflow):
            groups.append(current)
            current = []
            window_end = None
        if not current:
            window_end = dataset.run_date + timedelta(days=window_days)
        current.append(dataset)
    if current:
        groups.append(current)
    return groups


def build_plan(
    datasets: Sequence[Dataset],
    *,
    archive_prefix: str,
    tar_prefix: str,
    tar_output_dir: Path,
    window_days: int,
    max_datasets: int,
) -> list[CampaignGroup]:
    raw_groups = group_datasets(
        datasets, window_days=window_days, max_datasets=max_datasets
    )
    stem_counts: dict[str, int] = {}
    plan: list[CampaignGroup] = []
    for datasets_in_group in raw_groups:
        first = datasets_in_group[0].run_date.isoformat()
        last = datasets_in_group[-1].run_date.isoformat()
        date_stem = f"{first}_to_{last}"
        stem_counts[date_stem] = stem_counts.get(date_stem, 0) + 1
        part = stem_counts[date_stem]
        part_suffix = f"-part{part:02d}" if part > 1 else ""
        archive_name = f"{archive_prefix}-{date_stem}{part_suffix}.aca"
        tar_path = tar_output_dir / f"{tar_prefix}-{date_stem}{part_suffix}.tar"
        plan.append(CampaignGroup(archive_name, tar_path, tuple(datasets_in_group)))
    return plan


def print_command(command: Sequence[str]) -> None:
    print("Command:", shlex.join(str(value) for value in command))


def run_command(
    command: Sequence[str], *, cwd: Path | None = None, dry_run: bool = False
) -> None:
    print_command(command)
    if not dry_run:
        subprocess.run(command, cwd=cwd, check=True)


def sha256_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_checksum(tar_path: Path) -> Path:
    checksum_path = tar_path.with_name(f"{tar_path.name}.sha256")
    checksum_path.write_text(f"{sha256_digest(tar_path)}  {tar_path.name}\n")
    return checksum_path


def verify_checksum(tar_path: Path) -> None:
    checksum_path = tar_path.with_name(f"{tar_path.name}.sha256")
    if not checksum_path.is_file():
        raise FileNotFoundError(
            f"Existing TAR has no checksum: {checksum_path}. Use --rebuild-tars."
        )
    expected = checksum_path.read_text().split()[0]
    actual = sha256_digest(tar_path)
    if actual != expected:
        raise ValueError(f"Checksum mismatch for {tar_path}: {actual} != {expected}")


def manifest_path(tar_path: Path) -> Path:
    return tar_path.with_name(f"{tar_path.name}.members.json")


def group_manifest(group: CampaignGroup) -> dict[str, object]:
    return {
        "archive": group.archive_name,
        "tar": group.tar_path.name,
        "startDate": group.start_date.isoformat(),
        "endDate": group.end_date.isoformat(),
        "datasetCount": len(group.datasets),
        "datasets": [dataset.relative_path.as_posix() for dataset in group.datasets],
    }


def ensure_tar(
    group: CampaignGroup,
    *,
    data_root: Path,
    rebuild_tars: bool,
    dry_run: bool,
) -> tuple[Path, Path]:
    tar_path = group.tar_path
    index_path = tar_path.with_name(f"{tar_path.name}.idx")
    expected_manifest = group_manifest(group)
    current_manifest_path = manifest_path(tar_path)

    rebuild = rebuild_tars or not tar_path.exists()
    if tar_path.exists() and not rebuild:
        if not current_manifest_path.is_file():
            raise FileNotFoundError(
                f"Existing TAR has no member manifest: {current_manifest_path}. "
                "Use --rebuild-tars."
            )
        current_manifest = json.loads(current_manifest_path.read_text())
        if current_manifest != expected_manifest:
            raise ValueError(
                f"Existing TAR membership is stale: {tar_path}. Use --rebuild-tars."
            )
        if dry_run:
            print(f"Verify existing TAR and checksum: {tar_path}")
        else:
            verify_checksum(tar_path)
    else:
        members = [dataset.relative_path.as_posix() for dataset in group.datasets]
        run_command(
            ["tar", "--create", "--file", str(tar_path), "--format=pax", "--", *members],
            cwd=data_root,
            dry_run=dry_run,
        )
        if not dry_run:
            current_manifest_path.write_text(
                json.dumps(expected_manifest, indent=2, sort_keys=True) + "\n"
            )
            write_checksum(tar_path)

    stale_index = (
        not index_path.exists()
        or (tar_path.exists() and index_path.exists() and tar_path.stat().st_mtime_ns > index_path.stat().st_mtime_ns)
    )
    if rebuild or stale_index:
        run_command(
            ["hpc_campaign", "taridx", str(tar_path), str(index_path)],
            cwd=data_root,
            dry_run=dry_run,
        )
    else:
        print(f"Reusing TAR index: {index_path}")
    return tar_path, index_path


def create_campaign_archive(
    group: CampaignGroup,
    *,
    data_root: Path,
    campaign_store: Path,
    dry_run: bool,
) -> None:
    for index, dataset in enumerate(group.datasets):
        command = [
            "hpc_campaign",
            "manager",
            "--campaign_store",
            str(campaign_store),
            group.archive_name,
        ]
        if index == 0:
            command.append("--truncate")
        command.extend(
            ["data", dataset.relative_path.as_posix(), "--name", dataset.name]
        )
        run_command(command, cwd=data_root, dry_run=dry_run)


def register_tar_replica(
    group: CampaignGroup,
    *,
    tar_path: Path,
    index_path: Path,
    campaign_store: Path,
    storage_system: str,
    storage_host: str,
    data_root: Path,
    dry_run: bool,
) -> None:
    command = [
        "hpc_campaign",
        "manager",
        "--campaign_store",
        str(campaign_store),
        group.archive_name,
        "add-archival-storage",
        storage_system,
        storage_host,
        str(tar_path.parent),
        tar_path.name,
        str(index_path),
    ]
    run_command(command, cwd=data_root, dry_run=dry_run)


def write_plan(plan: Sequence[CampaignGroup], output_dir: Path, *, dry_run: bool) -> Path:
    plan_path = output_dir / "laser_campaign_plan.json"
    payload = {
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "campaignCount": len(plan),
        "datasetCount": sum(len(group.datasets) for group in plan),
        "campaigns": [group_manifest(group) for group in plan],
    }
    if dry_run:
        print(f"Would write plan: {plan_path}")
    else:
        plan_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return plan_path


def create_archives(args: argparse.Namespace) -> list[CampaignGroup]:
    data_root = args.data_root.expanduser().resolve()
    bp5_input_dir = args.bp5_input_dir.expanduser().resolve()
    campaign_store = args.campaign_store.expanduser().resolve()
    tar_output_dir = args.tar_output_dir.expanduser().resolve()

    if not data_root.is_dir():
        raise FileNotFoundError(f"Laser data root does not exist: {data_root}")
    datasets = discover_datasets(bp5_input_dir, data_root)
    if not datasets:
        raise FileNotFoundError(f"No date-bearing BP5 datasets found in {bp5_input_dir}")

    plan = build_plan(
        datasets,
        archive_prefix=args.archive_prefix,
        tar_prefix=args.tar_prefix,
        tar_output_dir=tar_output_dir,
        window_days=args.window_days,
        max_datasets=args.max_datasets,
    )
    print(
        f"Discovered {len(datasets)} dataset(s); planned {len(plan)} "
        f"campaign archive(s) with {args.window_days}-day windows."
    )
    for group in plan:
        print(
            f"  {group.archive_name}: {group.start_date} through {group.end_date} "
            f"({len(group.datasets)} dataset(s))"
        )

    if not args.dry_run:
        campaign_store.mkdir(parents=True, exist_ok=True)
        # TAR output creation is temporarily disabled.
    write_plan(plan, campaign_store, dry_run=args.dry_run)

    for group in plan:
        print(f"\nPreparing {group.archive_name}")
        # TEMPORARILY DISABLED: TAR creation/checksum/index generation.
        # tar_path, index_path = ensure_tar(...)
        create_campaign_archive(
            group,
            data_root=data_root,
            campaign_store=campaign_store,
            dry_run=args.dry_run,
        )
        # TEMPORARILY DISABLED: registration of a TAR archival replica.
        # register_tar_replica(...)
    return plan


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build date-windowed Laser ACAs; TAR replicas are temporarily disabled."
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--bp5-input-dir", type=Path, required=True)
    parser.add_argument("--campaign-store", type=Path, required=True)
    parser.add_argument("--tar-output-dir", type=Path, required=True)
    parser.add_argument("--archive-prefix", default="laser")
    parser.add_argument("--tar-prefix", default="laser")
    parser.add_argument("--window-days", type=int, default=10)
    parser.add_argument("--max-datasets", type=int, default=0)
    parser.add_argument("--storage-system", default="fs")
    parser.add_argument("--storage-host", default="NERSC")
    parser.add_argument("--rebuild-tars", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    create_archives(parse_args())


if __name__ == "__main__":
    main()
