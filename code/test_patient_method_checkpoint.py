"""Restart integrity checks independent of the expensive patient solves."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import patient_method_checkpoint as checkpoint


class PatientMethodCheckpointTests(unittest.TestCase):
    def signature(self, directory: Path, initial: np.ndarray) -> dict:
        source = directory / "solver.py"
        if not source.exists():
            source.write_text("version = 1\n", encoding="utf-8")
        return checkpoint.input_signature(
            {"method": "graph", "dt": 0.01}, {"initial": initial}, {"solver": source}
        )

    def diagnostics(self, method: str = "graph") -> dict[str, float]:
        return {
            name: (np.nan if method == "screened" and name in checkpoint.SCREENED_UNDEFINED else 0.0)
            for name in checkpoint.DIAGNOSTIC_KEYS
        }

    def test_complete_round_trip_and_exact_input_rejection(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            state = np.array([-1.2, 0.0, 1.1])  # states remain uncensored
            signature = self.signature(root, state)
            checkpoint.save_completed(root, "P1_left_graph", "graph", signature, state, self.diagnostics())
            loaded = checkpoint.load_completed(root, "P1_left_graph", "graph", signature, len(state))
            self.assertIsNotNone(loaded)
            np.testing.assert_array_equal(loaded[0], state)
            changed = self.signature(root, state + 1e-12)
            self.assertIsNone(checkpoint.load_completed(root, "P1_left_graph", "graph", changed, len(state)))
            (root / "solver.py").write_text("version = 2\n", encoding="utf-8")
            changed_source = self.signature(root, state)
            self.assertIsNone(checkpoint.load_completed(root, "P1_left_graph", "graph", changed_source, len(state)))

    def test_screened_missing_diagnostics_use_explicit_json_null(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            state = np.zeros(3)
            signature = self.signature(root, state)
            checkpoint.save_completed(root, "screened", "screened", signature, state, self.diagnostics("screened"))
            self.assertNotIn("NaN", (root / "screened.json").read_text())
            loaded = checkpoint.load_completed(root, "screened", "screened", signature, 3)
            self.assertTrue(np.isnan(loaded[1]["maximum_mass_source_defect"]))

    def test_interrupted_commit_and_corrupt_archive_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            state = np.zeros(3)
            signature = self.signature(root, state)
            checkpoint.save_completed(root, "method", "graph", signature, state, self.diagnostics())
            real_replace = checkpoint.os.replace
            calls = 0
            def interrupted_replace(source, destination):
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise OSError("simulated interruption before JSON commit")
                return real_replace(source, destination)
            with patch.object(checkpoint.os, "replace", side_effect=interrupted_replace):
                with self.assertRaisesRegex(OSError, "interruption"):
                    checkpoint.save_completed(root, "method", "graph", signature, state + 1, self.diagnostics())
            self.assertIsNone(checkpoint.load_completed(root, "method", "graph", signature, 3))
            self.assertFalse(list(root.glob(".method-*")))
            (root / "method.npz").write_bytes(b"corrupt")
            self.assertIsNone(checkpoint.load_completed(root, "method", "graph", signature, 3))

    def test_nonfinite_fields_and_wrong_shape_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            state = np.zeros(3)
            signature = self.signature(root, state)
            with self.assertRaises(ValueError):
                checkpoint.save_completed(root, "method", "graph", signature, np.array([np.nan]), self.diagnostics())
            diagnostics = self.diagnostics()
            diagnostics["maximum_state_residual"] = np.inf
            with self.assertRaises(ValueError):
                checkpoint.save_completed(root, "method", "graph", signature, state, diagnostics)
            checkpoint.save_completed(root, "method", "graph", signature, state, self.diagnostics())
            self.assertIsNone(checkpoint.load_completed(root, "method", "graph", signature, 4))


if __name__ == "__main__":
    unittest.main()
