"""Build a searchable SQLite index from existing HPC Campaign ACA files.

HPC Campaign 0.7 manages individual ``.aca`` archives but does not provide the
``hpc_campaign index`` command assumed by the original workflow.  This module
creates the small ``.acx`` catalog consumed by this repository's SQL feature
queries.  It reads only SQLite/ADIOS metadata; it never modifies an ACA or BP5
dataset.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


INDEX_SCHEMA_VERSION = 1
REQUIRED_ACA_TABLES = {"info", "dataset", "replica", "directory"}


def natural_archive_key(path: Path) -> tuple[str, int, str]:
    """Sort laser2.aca before laser10.aca."""
    match = re.match(r"^(.*?)(\d+)\.aca$", path.name, re.IGNORECASE)
    if match:
        return match.group(1), int(match.group(2)), path.name
    return path.stem, -1, path.name


def discover_archives(campaign_store: Path, archive_prefix: str) -> list[Path]:
    archives = [
        path
        for path in campaign_store.glob(f"{archive_prefix}*.aca")
        if path.is_file()
    ]
    return sorted(archives, key=natural_archive_key)


def validate_archive(connection: sqlite3.Connection, archive_path: Path) -> None:
    tables = {
        row[0]
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        )
    }
    missing = sorted(REQUIRED_ACA_TABLES - tables)
    if missing:
        raise ValueError(
            f"{archive_path} is missing ACA table(s): {', '.join(missing)}"
        )

    info = connection.execute("SELECT id FROM info LIMIT 1").fetchone()
    if info is None or info[0] != "ACA":
        raise ValueError(f"{archive_path} is not an HPC Campaign ACA database")


def dataset_replicas(
    connection: sqlite3.Connection,
    archive_path: Path,
) -> list[tuple[int, str, str, list[Path]]]:
    """Return live ADIOS datasets and their locally resolvable replica paths."""
    validate_archive(connection, archive_path)
    rows = connection.execute(
        """
        SELECT
            d.rowid,
            d.name,
            d.fileformat,
            directory.name,
            replica.name
        FROM dataset AS d
        LEFT JOIN replica ON replica.datasetid = d.rowid
            AND replica.deltime = 0
            AND replica.archiveid = 0
        LEFT JOIN directory ON directory.rowid = replica.dirid
        WHERE d.deltime = 0
        ORDER BY d.rowid, replica.rowid
        """
    ).fetchall()

    datasets: dict[int, tuple[str, str, list[Path]]] = {}
    for dataset_id, name, file_format, directory, replica in rows:
        if dataset_id not in datasets:
            datasets[dataset_id] = (name, file_format, [])
        if directory is None or replica is None:
            continue
        replica_path = Path(replica)
        if not replica_path.is_absolute():
            replica_path = Path(directory) / replica_path
        datasets[dataset_id][2].append(replica_path)

    return [
        (dataset_id, name, file_format, replicas)
        for dataset_id, (name, file_format, replicas) in datasets.items()
        if file_format == "ADIOS"
    ]


def read_adios_attributes(dataset_path: Path) -> list[tuple[str, str]]:
    """Read attribute names and serialized values without loading array data."""
    import adios2

    reader = adios2.FileReader(str(dataset_path))
    try:
        available = reader.available_attributes()
    finally:
        reader.close()

    attributes: list[tuple[str, str]] = []
    for name, metadata in sorted(available.items()):
        value: Any
        if isinstance(metadata, dict) and "Value" in metadata:
            value = metadata["Value"]
        else:
            value = metadata
        if not isinstance(value, str):
            value = json.dumps(value, ensure_ascii=False, sort_keys=True)
        attributes.append((name, value))
    return attributes


def choose_readable_replica(paths: Iterable[Path], dataset_name: str) -> Path:
    candidates = list(paths)
    for path in candidates:
        if path.exists():
            return path
    rendered = ", ".join(str(path) for path in candidates) or "none recorded"
    raise FileNotFoundError(
        f"No readable live replica for dataset {dataset_name!r}; candidates: {rendered}"
    )


def create_schema(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        PRAGMA foreign_keys = ON;

        CREATE TABLE index_info (
            schema_version INTEGER NOT NULL,
            created_utc TEXT NOT NULL,
            generator TEXT NOT NULL
        );

        CREATE TABLE archives (
            archiveid INTEGER PRIMARY KEY,
            name TEXT NOT NULL UNIQUE,
            path TEXT NOT NULL UNIQUE
        );

        CREATE TABLE datasets (
            archiveid INTEGER NOT NULL,
            datasetid INTEGER NOT NULL,
            name TEXT NOT NULL,
            replica_path TEXT NOT NULL,
            PRIMARY KEY (archiveid, datasetid),
            UNIQUE (archiveid, name),
            FOREIGN KEY (archiveid) REFERENCES archives(archiveid)
        );

        CREATE TABLE attributes (
            archiveid INTEGER NOT NULL,
            datasetid INTEGER NOT NULL,
            name TEXT NOT NULL,
            value TEXT NOT NULL,
            PRIMARY KEY (archiveid, datasetid, name),
            FOREIGN KEY (archiveid, datasetid)
                REFERENCES datasets(archiveid, datasetid)
        );

        CREATE INDEX attributes_by_name ON attributes(name);
        CREATE INDEX datasets_by_name ON datasets(name);
        """
    )
    connection.execute(
        "INSERT INTO index_info VALUES (?, ?, ?)",
        (
            INDEX_SCHEMA_VERSION,
            datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "Laser/AI_ready_workflow/2_campaign/create_index.py",
        ),
    )


def build_index(archives: list[Path], index_path: Path) -> tuple[int, int, int]:
    """Create an ACX atomically and return archive/dataset/attribute counts."""
    index_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_fd, temporary_name = tempfile.mkstemp(
        prefix=f".{index_path.name}.", suffix=".tmp", dir=index_path.parent
    )
    os.close(temporary_fd)
    temporary_path = Path(temporary_name)

    dataset_count = 0
    attribute_count = 0
    try:
        output = sqlite3.connect(temporary_path)
        try:
            create_schema(output)
            for archive_id, archive_path in enumerate(archives, start=1):
                print(f"Indexing {archive_path.name}")
                output.execute(
                    "INSERT INTO archives(archiveid, name, path) VALUES (?, ?, ?)",
                    (archive_id, archive_path.name, str(archive_path.resolve())),
                )

                source = sqlite3.connect(f"file:{archive_path}?mode=ro", uri=True)
                try:
                    datasets = dataset_replicas(source, archive_path)
                finally:
                    source.close()

                for dataset_id, name, _file_format, replicas in datasets:
                    replica_path = choose_readable_replica(replicas, name)
                    attributes = read_adios_attributes(replica_path)
                    output.execute(
                        """
                        INSERT INTO datasets(archiveid, datasetid, name, replica_path)
                        VALUES (?, ?, ?, ?)
                        """,
                        (archive_id, dataset_id, name, str(replica_path.resolve())),
                    )
                    output.executemany(
                        """
                        INSERT INTO attributes(archiveid, datasetid, name, value)
                        VALUES (?, ?, ?, ?)
                        """,
                        (
                            (archive_id, dataset_id, attribute_name, value)
                            for attribute_name, value in attributes
                        ),
                    )
                    dataset_count += 1
                    attribute_count += len(attributes)

            output.commit()
            integrity = output.execute("PRAGMA integrity_check").fetchone()
            if integrity != ("ok",):
                raise RuntimeError(f"Index integrity check failed: {integrity}")
        finally:
            output.close()

        os.replace(temporary_path, index_path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise

    return len(archives), dataset_count, attribute_count


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build laser.acx from existing HPC Campaign ACA archives."
    )
    parser.add_argument(
        "--campaign-store",
        type=Path,
        required=True,
        help="Directory containing laser*.aca files.",
    )
    parser.add_argument(
        "--archive-prefix",
        default="laser",
        help="Archive filename prefix (default: laser).",
    )
    parser.add_argument(
        "--index",
        type=Path,
        default=Path("laser.acx"),
        help="Output path or filename relative to the campaign store.",
    )
    args = parser.parse_args()

    campaign_store = args.campaign_store.resolve()
    if not campaign_store.is_dir():
        parser.error(f"campaign store does not exist: {campaign_store}")
    index_path = args.index if args.index.is_absolute() else campaign_store / args.index
    if index_path.suffix != ".acx":
        parser.error(f"index output must end in .acx: {index_path}")

    archives = discover_archives(campaign_store, args.archive_prefix)
    if not archives:
        parser.error(
            f"no {args.archive_prefix}*.aca files found in {campaign_store}"
        )

    archive_count, dataset_count, attribute_count = build_index(archives, index_path)
    print("\nCampaign index created successfully")
    print(f"Index      : {index_path}")
    print(f"Archives   : {archive_count}")
    print(f"Datasets   : {dataset_count}")
    print(f"Attributes : {attribute_count}")


if __name__ == "__main__":
    main()
