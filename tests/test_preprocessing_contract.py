"""Dependency-light checks that use the unchanged training preprocessing code."""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from gait_preprocess import prepare_pose
from gait_train_rf import extract_all, select_feature_columns


class PreprocessingContractTest(unittest.TestCase):
    def test_sampled_landmarks_make_training_shaped_inputs(self):
        frames = []
        for index in range(60):
            joints = []
            for joint in range(33):
                joints.append({
                    "x": 0.20 + joint * 0.003 + index * 0.0002,
                    "y": 0.15 + joint * 0.007,
                    "z": 0.0,
                    "visibility": 0.99,
                })
            frames.append({"timestamp": index * 40, "landmarks": joints})

        sequence, duration, ratio = prepare_pose(frames, steps=128, min_visibility=0.5)
        self.assertEqual(sequence.shape, (128, 99))
        self.assertTrue(np.isfinite(sequence).all())
        self.assertAlmostEqual(duration, 2.36, places=2)
        self.assertAlmostEqual(ratio, 1.0)
        raw, names = extract_all(sequence[None], np.array([duration]))
        selected, names = select_feature_columns(raw, names, "no_duration")
        self.assertEqual(selected.shape, (1, 152))
        self.assertNotIn("duration_sec", names)


if __name__ == "__main__":
    unittest.main()
