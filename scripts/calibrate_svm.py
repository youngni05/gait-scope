"""Fit a separate sigmoid probability calibrator for the FROZEN SVM-H.

Only the provided validation split is read. The original SVM model is neither
retrained nor overwritten. NEVER point --data to test.npz.

Example (run from the gait_video_inference project directory):
    python scripts/calibrate_svm.py --data D:\\gait\\data\\processed\\gait_step2\\val.npz
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator

from gait_train_rf import extract_all, select_feature_columns

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SVM = ROOT / "models" / "svm" / "svm_model.joblib"
DEFAULT_OUT = ROOT / "models" / "svm" / "svm_calibrator.joblib"


def sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def calibrate(data_path: Path, svm_path: Path, output_path: Path) -> dict:
    if data_path.name.lower() != "val.npz":
        raise ValueError("Only val.npz is accepted. Do not use train.npz or test.npz")
    if not data_path.is_file() or not svm_path.is_file():
        raise FileNotFoundError("Missing validation split or frozen svm_model.joblib")
    if output_path.exists():
        raise FileExistsError(f"Calibration file already exists: {output_path}. Existing files are preserved")

    with np.load(data_path, allow_pickle=False) as data:
        X = data["X"]
        y = data["y"].astype(int)
        durations = data["duration_sec"]
    if X.ndim != 3 or X.shape[1:] != (128, 99):
        raise ValueError(f"Invalid validation pose tensor: {X.shape}")
    counts = np.bincount(y, minlength=2)
    if set(np.unique(y)) != {0, 1} or np.min(counts) < 5:
        raise ValueError("Calibration set must contain at least 5 samples of each binary class")

    raw, feature_names = extract_all(X, durations)
    features, feature_names = select_feature_columns(raw, feature_names, "no_duration")
    saved = joblib.load(svm_path)  # Load only our previously trained, trusted model.
    if saved.get("feature_policy") != "no_duration" or list(saved.get("feature_names", [])) != feature_names:
        raise ValueError("The frozen SVM has a different feature schema")
    if saved.get("label_map") != {"normal": 0, "abnormal": 1}:
        raise ValueError("Unexpected label mapping")

    frozen_svm = saved["pipeline"]
    calibrator = CalibratedClassifierCV(
        estimator=FrozenEstimator(frozen_svm),
        method="sigmoid",
    )
    # This fits ONLY the sigmoid calibrator, not the SVM or StandardScaler.
    calibrator.fit(features, y)
    prob = calibrator.predict_proba(features)
    if prob.shape != (len(y), 2) or not np.isfinite(prob).all():
        raise ValueError("Calibration produced invalid probabilities")

    payload = {
        "calibrator": calibrator,
        "original_svm_sha256": sha256_file(svm_path),
        "feature_names": feature_names,
        "method": "sigmoid",
        "source_split": "val.npz",
        "calibration_records": len(y),
        "normal_records": int(counts[0]),
        "abnormal_records": int(counts[1]),
        "note": "Prototype-only calibration: validation split was previously used for model selection; no independent probability validation.",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(payload, output_path)
    return {
        "saved_to": str(output_path),
        "original_svm_unchanged": True,
        "records": len(y),
        "normal": int(counts[0]),
        "abnormal": int(counts[1]),
        "method": "sigmoid",
        "warning": "Calibration probabilities are not independently validated; do not claim clinical risk or accuracy.",
    }


def main():
    parser = argparse.ArgumentParser(description="Sigmoid-calibrate existing SVM-H without retraining")
    parser.add_argument("--data", type=Path, required=True, help="Path to validation val.npz")
    parser.add_argument("--svm", type=Path, default=DEFAULT_SVM)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    try:
        result = calibrate(args.data, args.svm, args.out)
    except (ValueError, FileNotFoundError, FileExistsError, KeyError) as exc:
        parser.exit(1, f"Calibration failed: {exc}\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
