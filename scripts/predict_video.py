"""Frozen four-model gait inference on a newly uploaded walking video.

CLI example (from project root):
    python scripts/predict_video.py --video test_vedio.mp4

This is an exploratory simulated-gait-pattern classifier, NOT a medical tool.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from gait_preprocess import prepare_pose
from gait_train_rf import extract_all, select_feature_columns
from video_pose_inference import extract_video_pose
from xgb_native_io import load_verified_xgb

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODELS = ROOT / "models"
MODEL_LABELS = {0: "정상 패턴", 1: "비정상 패턴"}


def _load_frozen_models(models_dir: Path):
    import joblib
    import torch

    from gait_train_cnn import GaitCNN

    rf = joblib.load(models_dir / "random_forest" / "rf_model.joblib")
    svm = joblib.load(models_dir / "svm" / "svm_model.joblib")
    xgb = load_verified_xgb(models_dir)
    cnn = torch.load(models_dir / "cnn" / "best_cnn.pt", map_location="cpu", weights_only=True)

    for name, bundle in (("RF-B", rf), ("SVM-H", svm), ("XGB-A", xgb)):
        if bundle.get("feature_policy") != "no_duration":
            raise ValueError(f"{name}: unexpected feature policy")
        if len(bundle.get("feature_names", [])) != 152:
            raise ValueError(f"{name}: incorrect feature count")
        if bundle.get("label_map") != {"normal": 0, "abnormal": 1}:
            raise ValueError(f"{name}: invalid label mapping")

    config = cnn.get("model_config", {})
    if config.get("architecture") != "GaitCNN" or config.get("features") != 99:
        raise ValueError("CNN-D: unexpected architecture")
    if cnn["feature_mean"].numel() != 99 or cnn["feature_std"].numel() != 99:
        raise ValueError("CNN-D: incorrect normalization shape")
    cnn_model = GaitCNN(features=99, dropout=float(config["dropout"]))
    cnn_model.load_state_dict(cnn["model_state_dict"])
    cnn_model.eval()
    return rf, svm, xgb, cnn, cnn_model


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _svm_probability(models_dir: Path, features: np.ndarray, feature_names: list[str]):
    """Optional probability from a separately fitted sigmoid calibrator.

    The original SVM label still comes from its unchanged predict() method.
    """
    import joblib

    calibrator_path = models_dir / "svm" / "svm_calibrator.joblib"
    if not calibrator_path.is_file():
        return None
    payload = joblib.load(calibrator_path)
    svm_path = models_dir / "svm" / "svm_model.joblib"
    if payload.get("original_svm_sha256") != _file_hash(svm_path):
        raise ValueError("SVM calibrator was fitted for a different SVM model file")
    if list(payload.get("feature_names", [])) != feature_names:
        raise ValueError("SVM calibrator expects different input features")
    if payload.get("method") != "sigmoid":
        raise ValueError("Unexpected SVM probability calibration method")
    return float(payload["calibrator"].predict_proba(features)[0, 1])


def predict_prepared_pose(
    sequence: np.ndarray,
    duration_sec: float,
    *,
    models_dir: Path = DEFAULT_MODELS,
) -> dict[str, dict]:
    """Predict using the exact feature extractors saved at training time."""
    import torch

    if sequence.shape != (128, 99):
        raise ValueError(f"Expected (128,99), got {sequence.shape}")
    if not np.isfinite(sequence).all():
        raise ValueError("Input contains non-finite pose values")

    rf, svm, xgb, cnn, cnn_model = _load_frozen_models(models_dir)
    raw, names = extract_all(sequence[None], np.array([duration_sec], dtype=np.float32))
    features, names = select_feature_columns(raw, names, "no_duration")
    if features.shape != (1, 152):
        raise ValueError(f"Incorrect engineered features: {features.shape}")
    for name, bundle in (("RF-B", rf), ("SVM-H", svm), ("XGB-A", xgb)):
        if list(bundle["feature_names"]) != names:
            raise ValueError(f"{name}: feature ordering differs from training")

    mean = cnn["feature_mean"].cpu().numpy()
    std = cnn["feature_std"].cpu().numpy()
    if not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std <= 0):
        raise ValueError("Invalid CNN train-fitted normalization values")
    cnn_input = (sequence.astype(np.float32)[None] - mean) / std
    coord_columns = [j for j in range(99) if j % 3 != 2]
    cnn_input[:, :, coord_columns] = np.clip(cnn_input[:, :, coord_columns], -8.0, 8.0)

    with torch.no_grad():
        logits = cnn_model(torch.from_numpy(cnn_input))
        cnn_prediction = int(torch.argmax(logits, dim=1).item())
        cnn_probability = float(torch.softmax(logits, dim=1)[0, 1].item())

    predictions = {
        "RF-B": int(rf["model"].predict(features)[0]),
        "CNN-D": cnn_prediction,
        "SVM-H": int(svm["pipeline"].predict(features)[0]),
        "XGB-A": int(xgb["model"].predict(features)[0]),
    }
    probabilities = {
        "RF-B": float(rf["model"].predict_proba(features)[0, 1]),
        "CNN-D": cnn_probability,
        "SVM-H": _svm_probability(models_dir, features, names),
        "XGB-A": float(xgb["model"].predict_proba(features)[0, 1]),
    }
    return {
        name: {
            "class_id": label,
            "label": MODEL_LABELS[label],
            "probability_abnormal": (
                round(probabilities[name], 6) if probabilities[name] is not None else None
            ),
            "score_note": (
                "sigmoid_calibrated_on_validation_prototype_only" if name == "SVM-H" and probabilities[name] is not None
                else "not_available_until_svm_calibration" if name == "SVM-H"
                else "model_output_not_independently_calibrated"
            ),
        }
        for name, label in predictions.items()
    }


def analyze_video(
    video_path: str | Path,
    *,
    models_dir: Path = DEFAULT_MODELS,
    max_video_sec: float = 60.0,
) -> dict:
    pose_frames, video_stats = extract_video_pose(video_path, max_video_sec=max_video_sec)
    if video_stats.detected_frames < 30:
        raise ValueError(
            f"Too few detected pose frames ({video_stats.detected_frames}); "
            "use a video with one fully visible walking person"
        )
    if video_stats.detection_rate < 0.60:
        raise ValueError(
            f"Pose detection rate is only {video_stats.detection_rate:.1%}; "
            "the recording is not reliable enough for classification"
        )

    # Identical function used to construct train/validation/test tensors.
    sequence, pose_duration, visible_ratio = prepare_pose(
        pose_frames, steps=128, min_visibility=0.5
    )
    result = predict_prepared_pose(sequence, pose_duration, models_dir=models_dir)
    return {
        "status": "ok",
        "predictions": result,
        "quality": {
            "video_duration_sec": round(video_stats.video_duration_sec, 3),
            "detected_pose_duration_sec": round(pose_duration, 3),
            "source_fps": round(video_stats.source_fps, 3),
            "sampled_frames": video_stats.sampled_frames,
            "detected_frames": video_stats.detected_frames,
            "pose_detection_rate": round(video_stats.detection_rate, 4),
            "joint_visibility_ratio": round(visible_ratio, 4),
            "input_shape": [128, 99],
        },
        "notice": (
            "연구용 모사 보행 패턴 분류 결과입니다. 의료 진단 또는 실제 환자에 대한 "
            "성능을 보장하지 않습니다. 방향 전환, 카메라 구도 및 얼굴 가림은 결과에 영향을 줄 수 있습니다."
        ),
    }


def main():
    parser = argparse.ArgumentParser(description="Classify a walking video with four frozen models")
    parser.add_argument("--video", type=Path, required=True, help="MP4 or MOV video file")
    parser.add_argument("--models", type=Path, default=DEFAULT_MODELS)
    parser.add_argument("--max-video-sec", type=float, default=60.0)
    args = parser.parse_args()
    try:
        result = analyze_video(args.video, models_dir=args.models, max_video_sec=args.max_video_sec)
    except (ValueError, FileNotFoundError, RuntimeError, ImportError) as exc:
        parser.exit(1, f"Analysis failed: {exc}\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
