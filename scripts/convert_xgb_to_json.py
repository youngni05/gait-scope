"""Safely export frozen XGB-A as JSON; promote only after numerical equivalence.

Run from the project root with the actual development video:
    python scripts/convert_xgb_to_json.py --video test_vedio.mp4

This never trains or changes the original model. It refuses to overwrite an
existing verified conversion. The video is processed in memory only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import xgboost as xgb

from gait_preprocess import prepare_pose
from gait_train_rf import extract_all, select_feature_columns
from video_pose_inference import extract_video_pose
from xgb_native_io import NativeBinaryXGB, sha256_file

ROOT = Path(__file__).resolve().parents[1]
MAX_PROBABILITY_DIFF = 1e-6


def assert_equal_predictions(original, native, features, name: str):
    old_prob = original.predict_proba(features)
    new_prob = native.predict_proba(features)
    old_pred = original.predict(features)
    new_pred = native.predict(features)
    if not np.isfinite(old_prob).all() or not np.isfinite(new_prob).all():
        raise ValueError(f"{name}: non-finite probability detected")
    max_diff = float(np.max(np.abs(old_prob - new_prob)))
    if max_diff > MAX_PROBABILITY_DIFF or not np.array_equal(old_pred, new_pred):
        raise ValueError(
            f"{name}: conversion changed predictions (maximum probability difference={max_diff:.9g})"
        )
    return {"rows": int(features.shape[0]), "max_probability_difference": max_diff,
            "class_ids_identical": True,
            "original_probability_abnormal": round(float(old_prob[0, 1]), 8),
            "converted_probability_abnormal": round(float(new_prob[0, 1]), 8)}


def get_video_features(video_path: Path, names_expected: list[str]):
    frames, stats = extract_video_pose(video_path)
    if stats.detected_frames < 30 or stats.detection_rate < 0.60:
        raise ValueError("Video pose quality insufficient for a conversion comparison")
    sequence, duration, _ = prepare_pose(frames, steps=128, min_visibility=0.5)
    raw, names = extract_all(sequence[None], np.array([duration], dtype=np.float32))
    features, names = select_feature_columns(raw, names, "no_duration")
    if names != names_expected or features.shape != (1, 152):
        raise ValueError("Video-derived features differ from XGB-A training schema")
    return features


def main():
    parser = argparse.ArgumentParser(description="Convert and verify frozen XGB-A to native JSON")
    parser.add_argument("--video", type=Path, required=True, help="Same MP4 used in predict_video.py")
    parser.add_argument("--models", type=Path, default=ROOT / "models")
    args = parser.parse_args()

    directory = args.models / "xgboost"
    original_path = directory / "xgb_model.joblib"
    native_path = directory / "xgb_model.json"
    manifest_path = directory / "xgb_conversion_verified.json"
    candidate_path = directory / "xgb_model.candidate.json"
    if native_path.exists() or manifest_path.exists() or candidate_path.exists():
        parser.error("Conversion output exists; refusing overwrite. Inspect files first.")
    if not original_path.is_file():
        parser.error(f"Missing original model: {original_path}")
    if not args.video.is_file():
        parser.error(f"Missing video: {args.video}")

    # Load the trusted original joblib once, without changing it.
    bundle = joblib.load(original_path)
    metadata = {key: bundle[key] for key in ("feature_names", "feature_policy", "label_map")}
    metadata["feature_names"] = list(metadata["feature_names"])
    if (metadata["feature_policy"] != "no_duration"
            or len(metadata["feature_names"]) != 152
            or metadata["label_map"] != {"normal": 0, "abnormal": 1}):
        raise ValueError("Original XGB-A feature schema / labels are unexpected")

    original_model = bundle["model"]
    if not isinstance(original_model, xgb.XGBClassifier):
        raise TypeError("Expected the original XGBClassifier wrapper, not another model type")

    # XGBClassifier.save_model() may fail with `_estimator_type` undefined
    # under XGBoost 3.1.2 + recent scikit-learn. Save the Booster instead.
    original_model.get_booster().save_model(str(candidate_path))
    try:
        converted_booster = xgb.Booster()
        converted_booster.load_model(str(candidate_path))
        converted_model = NativeBinaryXGB(converted_booster)

        # A range of deterministic feature inputs checks serialization of the tree ensemble.
        rng = np.random.default_rng(20261024)
        probes = np.vstack([
            np.zeros((1, 152), dtype=np.float32),
            np.ones((1, 152), dtype=np.float32),
            rng.normal(size=(64, 152)).astype(np.float32),
        ])
        probe_check = assert_equal_predictions(original_model, converted_model, probes, "probe inputs")

        # Mandatory real video comparison ensures the exact demo input is unaffected.
        video_features = get_video_features(args.video, metadata["feature_names"])
        video_check = assert_equal_predictions(original_model, converted_model, video_features, "video")

        result = {
            "schema_version": 1,
            "status": "verified",
            "source_sha256": sha256_file(original_path),
            "native_sha256": sha256_file(candidate_path),
            "video_sha256": sha256_file(args.video),
            "metadata": metadata,
            "checks": {"probe_inputs": probe_check, "video": video_check},
            "xgboost_version_during_conversion": xgb.__version__,
        }
        # Finish writing only after both comparisons have passed.
        candidate_path.rename(native_path)
        manifest_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({"status": "verified", "active_model": str(native_path),
                          "legacy_model_unchanged": str(original_path),
                          "checks": result["checks"]}, ensure_ascii=False, indent=2))
    except Exception:
        # The candidate is intentionally left for diagnosis after failure.
        raise


if __name__ == "__main__":
    main()
