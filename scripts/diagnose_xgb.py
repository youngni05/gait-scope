"""Compare video-derived XGBoost inputs and decision margin across computers.

Place this file in the existing project's scripts/ directory and run:
    python scripts/diagnose_xgb.py --video test_vedio.mp4

No models are refitted, no raw frames or landmarks are saved.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np

from gait_preprocess import prepare_pose
from gait_train_rf import extract_all, select_feature_columns
from video_pose_inference import extract_video_pose


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description="Diagnose XGBoost gait prediction variability")
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=Path(__file__).resolve().parents[1] / "models" / "xgboost" / "xgb_model.joblib")
    args = parser.parse_args()

    # Read only the trusted model produced by the original training workflow.
    import cv2
    import mediapipe as mp
    import sklearn
    import xgboost

    frames, stats = extract_video_pose(args.video)
    if stats.detection_rate < 0.60 or stats.detected_frames < 30:
        parser.error(f"Insufficient pose detections: {stats.detection_rate:.2%}")
    sequence, duration, visibility = prepare_pose(frames, steps=128, min_visibility=0.5)
    raw, names = extract_all(sequence[None], np.array([duration], dtype=np.float32))
    features, names = select_feature_columns(raw, names, "no_duration")
    bundle = joblib.load(args.model)
    if list(bundle["feature_names"]) != names or bundle["feature_policy"] != "no_duration":
        raise ValueError("XGB model feature schema differs from the original training code")

    xgb_model = bundle["model"]
    probability = float(xgb_model.predict_proba(features)[0, 1])
    predicted_class = int(xgb_model.predict(features)[0])
    float_data = np.asarray(features, dtype="<f4", order="C")
    xgb_prediction = {
        "class_id": predicted_class,
        "probability_abnormal": round(probability, 8),
        "distance_from_0_5": round(abs(probability - 0.5), 8),
        "near_decision_boundary": abs(probability - 0.5) < 0.05,
    }
    results = {
        "video_sha256": sha256_file(args.video),
        "model_sha256": sha256_file(args.model),
        "feature_sha256": hashlib.sha256(float_data.tobytes()).hexdigest(),
        "feature_count": len(names),
        "pose_duration_sec": round(duration, 6),
        "pose_detection_rate": round(stats.detection_rate, 6),
        "joint_visibility_ratio": round(visibility, 6),
        "versions": {
            "numpy": np.__version__,
            "mediapipe": mp.__version__,
            "opencv": cv2.__version__,
            "scikit_learn": sklearn.__version__,
            "xgboost": xgboost.__version__,
        },
        "xgboost_prediction": xgb_prediction,
    }
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
