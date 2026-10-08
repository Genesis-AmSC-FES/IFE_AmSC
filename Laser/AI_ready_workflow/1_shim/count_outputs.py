"""Count Phoenix run Series and recorded events in a BP5 output directory."""

from __future__ import annotations

import argparse
from pathlib import Path

import openpmd_api as io


def count_outputs(output_dir: Path) -> tuple[int, int]:
    datasets = sorted(output_dir.glob("*.bp5"))
    total_events = 0

    for path in datasets:
        series = io.Series(str(path), io.Access_Type.read_only)
        try:
            key = (
                "run:recordedEventCount"
                if "run:recordedEventCount" in series.attributes
                else "input:num_valid_samples"
            )
            events = int(series.get_attribute(key))
        finally:
            series.close()
        total_events += events
        print(f"{path.name}: {events} recorded event(s)")

    print(f"\nBP5 runs: {len(datasets)}")
    print(f"Total recorded events: {total_events}")
    return len(datasets), total_events


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Count Phoenix BP5 run Series and their recorded events."
    )
    parser.add_argument("output_dir", type=Path, help="Directory containing *.bp5 outputs")
    args = parser.parse_args()

    if not args.output_dir.is_dir():
        parser.error(f"output directory does not exist: {args.output_dir}")
    count_outputs(args.output_dir)


if __name__ == "__main__":
    main()
