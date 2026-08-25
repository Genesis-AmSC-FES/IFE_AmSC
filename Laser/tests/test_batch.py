from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


SHIM_DIR = Path(__file__).resolve().parents[1] / "1_shim"
sys.path.insert(0, str(SHIM_DIR))

import laserWrite_multi  # noqa: E402


class ConvertManyTests(unittest.TestCase):
    def test_skip_existing_does_not_call_converter(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            source = root / "png-42-2026-08-20.h5"
            source.touch()
            output_dir = root / "bp_output"
            existing = output_dir / "png-42-2026-08-20.bp5"
            existing.mkdir(parents=True)

            with patch.object(laserWrite_multi, "laser_to_adios") as converter:
                written = laserWrite_multi.convert_many(
                    source,
                    output_dir,
                    skip_existing=True,
                    verbose=False,
                )

            self.assertEqual(written, [])
            converter.assert_not_called()

    def test_existing_output_is_overwritten_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            source = root / "png-42-2026-08-20.h5"
            source.touch()
            output_dir = root / "bp_output"
            existing = output_dir / "png-42-2026-08-20.bp5"
            existing.mkdir(parents=True)

            with patch.object(
                laserWrite_multi,
                "laser_to_adios",
                return_value=str(existing),
            ) as converter:
                written = laserWrite_multi.convert_many(
                    source,
                    output_dir,
                    verbose=False,
                )

            self.assertEqual(written, [str(existing)])
            converter.assert_called_once()


if __name__ == "__main__":
    unittest.main()
