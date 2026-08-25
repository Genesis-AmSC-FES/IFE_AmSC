"""Command-line entry point for building the RHINO ML feature table."""

from __future__ import annotations

import argparse
from pathlib import Path

from dataset_provenance import (
    persist_explicit_dataset,
    persist_versioned_dataset,
    sha256_file,
)
from spec_loader import load_feature_spec


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_SPEC = BASE_DIR / "feature_spec.json"
DEFAULT_QUERY_DIR = BASE_DIR.parent / "2_campaign" / "queries"
DATASET_ROOT = Path(
    "/global/cfs/cdirs/m3239/2026_FES-AmSC/data/rhino/ml-datasets"
)


def parse_args() -> argparse.Namespace:
    """Parse feature-table build options."""
    parser = argparse.ArgumentParser(
        description="Build a RHINO feature table from an HPC Campaign index."
    )
    parser.add_argument(
        "--acx",
        type=Path,
        required=True,
        help="Path to the SQLite-backed campaign index (.acx).",
    )
    parser.add_argument(
        "--campaign-store",
        type=Path,
        required=True,
        help=(
            "Base directory used to resolve archive names stored in the index."
        ),
    )
    parser.add_argument(
        "--spec",
        type=Path,
        default=DEFAULT_SPEC,
        help=f"Feature JSON specification (default: {DEFAULT_SPEC}).",
    )
    parser.add_argument(
        "--query-dir",
        type=Path,
        default=DEFAULT_QUERY_DIR,
        help=f"Directory containing feature SQL queries (default: {DEFAULT_QUERY_DIR}).",
    )
    parser.add_argument(
        "--archive-name",
        default="%",
        help="SQL LIKE pattern used to select archive names (default: %%).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help=(
            "Explicit replaceable CSV export. If omitted, write an immutable, "
            f"content-addressed dataset under {DATASET_ROOT}."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.acx.is_file():
        raise FileNotFoundError(f"Campaign index does not exist: {args.acx}")
    if not args.campaign_store.is_dir():
        raise FileNotFoundError(
            f"Campaign store does not exist: {args.campaign_store}"
        )
    if not args.query_dir.is_dir():
        raise FileNotFoundError(
            f"Campaign query directory does not exist: {args.query_dir}"
        )

    from campaign_reader import build_feature_table

    specification = load_feature_spec(args.spec)
    frame = build_feature_table(
        acx_path=args.acx,
        query_dir=args.query_dir,
        campaign_store=args.campaign_store,
        archive_name=args.archive_name,
        feature_specs=specification["features"],
    )
    if frame.empty:
        raise ValueError(
            f"No campaign runs matched archive pattern {args.archive_name!r}"
        )

    acx_path = args.acx.expanduser().resolve()
    spec_path = args.spec.expanduser().resolve()
    query_dir = args.query_dir.expanduser().resolve()
    sql_queries = sorted(
        {
            feature["query"]
            for feature in specification["features"]
            if feature["source"] == "campaign_sql"
        }
    )
    provenance = {
        "source": {
            "campaign_index": str(acx_path),
            "campaign_index_sha256": sha256_file(acx_path),
            "campaign_store": str(args.campaign_store.expanduser().resolve()),
            "archive_pattern": args.archive_name,
            "archives": sorted(frame["archive"].dropna().astype(str).unique()),
        },
        "feature_spec": {
            "path": str(spec_path),
            "sha256": sha256_file(spec_path),
            "schema_version": specification["schema_version"],
            "inputs": [entry["key"] for entry in specification["inputs"]],
            "outputs": [entry["key"] for entry in specification["outputs"]],
        },
        "queries": [
            {
                "path": str((query_dir / query).resolve()),
                "sha256": sha256_file(query_dir / query),
            }
            for query in sql_queries
        ],
        "generator": "RHINO/AI_ready_workflow/3_feature_extraction/build_features.py",
    }

    if args.output is not None:
        dataset_path, manifest_path, digest = persist_explicit_dataset(
            frame, args.output, provenance
        )
        reused = False
    else:
        dataset_path, manifest_path, digest, reused = persist_versioned_dataset(
            frame, DATASET_ROOT, provenance
        )

    print(frame.head())
    print(f"\nRows: {len(frame)}, columns: {len(frame.columns)}")
    missing = frame.isna().sum()
    missing = missing[missing > 0]
    if missing.empty:
        print("Missing values: none")
    else:
        print("Missing values:")
        print(missing.to_string())
    disposition = "Reused" if reused else "Saved"
    print(f"\n{disposition} feature dataset: {dataset_path}")
    print(f"Dataset SHA-256: {digest}")
    print(f"Dataset manifest: {manifest_path}")


if __name__ == "__main__":
    main()
