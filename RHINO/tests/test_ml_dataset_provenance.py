from __future__ import annotations

import hashlib
import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd


RHINO_ROOT = Path(__file__).resolve().parents[1]
FEATURE_DIR = RHINO_ROOT / "AI_ready_workflow" / "3_feature_extraction"
TRAINING_DIR = RHINO_ROOT / "ML" / "surrogate_training"
for module_dir in (str(FEATURE_DIR), str(TRAINING_DIR)):
    if module_dir not in sys.path:
        sys.path.insert(0, module_dir)

from dataset_provenance import persist_explicit_dataset, persist_versioned_dataset
from dataset_tracking import log_mlflow_dataset_inputs, resolve_dataset_manifest


class DatasetProvenanceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.frame = pd.DataFrame(
            {
                "archive": ["rhino1.aca", "rhino1.aca", "rhino2.aca"],
                "input": [1.0, 2.0, 3.0],
                "output": [4.0, 5.0, 6.0],
            }
        )

    def test_identical_feature_tables_reuse_content_addressed_dataset(self) -> None:
        with TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            first = persist_versioned_dataset(self.frame, root, {"source": {}})
            second = persist_versioned_dataset(self.frame, root, {"source": {}})

            self.assertEqual(first[0], second[0])
            self.assertEqual(first[2], second[2])
            self.assertEqual(first[0].parent.name, first[2][:12])
            self.assertFalse(first[3])
            self.assertTrue(second[3])
            manifest = json.loads(first[1].read_text(encoding="utf-8"))
            self.assertEqual(manifest["dataset_id"], first[2][:12])
            self.assertEqual(manifest["sha256"], first[2])
            self.assertEqual(manifest["rows"], 3)

    def test_adjacent_explicit_manifest_is_validated(self) -> None:
        with TemporaryDirectory() as temporary_dir:
            output = Path(temporary_dir) / "features.csv"
            dataset_path, manifest_path, _ = persist_explicit_dataset(
                self.frame, output, {"source": {}}
            )
            resolved_path, manifest = resolve_dataset_manifest(dataset_path, None)

            self.assertEqual(resolved_path, manifest_path)
            self.assertIsNotNone(manifest)
            self.assertEqual(manifest["rows"], 3)

            dataset_path.write_text("changed\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                resolve_dataset_manifest(dataset_path, manifest_path)


class _FakeDataset:
    def __init__(self, frame, source, name, digest=None):
        self.name = name
        self.source = source
        self.digest = digest or hashlib.sha256(
            frame.to_csv(index=False).encode("utf-8")
        ).hexdigest()


class _FakeDataApi:
    @staticmethod
    def from_pandas(frame, *, source, name, digest=None):
        return _FakeDataset(frame, source, name, digest)


class _FakeMlflow:
    data = _FakeDataApi()

    def __init__(self):
        self.inputs = []

    def log_input(self, dataset, *, context):
        self.inputs.append((dataset, context))


class MlflowDatasetInputTests(unittest.TestCase):
    def test_source_training_and_validation_contexts_are_logged(self) -> None:
        frame = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
        fake_mlflow = _FakeMlflow()
        metadata = log_mlflow_dataset_inputs(
            fake_mlflow,
            frame,
            Path("/shared/rhino_features.csv"),
            "full-file-sha256",
            np.asarray([0, 1]),
            np.asarray([2]),
            "rhino-features",
        )

        self.assertEqual(
            [context for _, context in fake_mlflow.inputs],
            ["source", "training", "validation"],
        )
        self.assertEqual(metadata["source"]["sha256"], "full-file-sha256")
        self.assertNotEqual(metadata["source"]["digest"], "full-file-sha256")
        self.assertEqual(metadata["training"]["name"], "rhino-features-training")
        self.assertEqual(
            metadata["source"]["source"],
            "file:///shared/rhino_features.csv",
        )


if __name__ == "__main__":
    unittest.main()
