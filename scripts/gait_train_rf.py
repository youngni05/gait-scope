"""Step 3A: train a pose-only Random Forest baseline (without reading test.npz).

Run from the project folder:
  python -m pip install numpy scikit-learn joblib
  python gait_train_rf.py --data gait_step2 --out gait_step3_rf
  python gait_train_rf.py --data gait_step2 --out gait_step3_rf_no_duration --exclude-duration
  python gait_train_rf.py --data gait_step2 --out gait_step3_rf_no_timing --exclude-timing

Validation is used for model selection. Keep test.npz untouched until Step 4.
"""

import argparse
import csv
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, confusion_matrix,
    f1_score, precision_score, recall_score,
)


JOINTS = {
    "shoulder_L": 11, "shoulder_R": 12,
    "elbow_L": 13, "elbow_R": 14,
    "wrist_L": 15, "wrist_R": 16,
    "hip_L": 23, "hip_R": 24,
    "knee_L": 25, "knee_R": 26,
    "ankle_L": 27, "ankle_R": 28,
    "heel_L": 29, "heel_R": 30,
    "toe_L": 31, "toe_R": 32,
}


def angle_deg(a, b, c):
    """2D angle at b, in degrees, for each frame."""
    ba, bc = a - b, c - b
    cosv = np.sum(ba * bc, axis=1) / np.maximum(
        np.linalg.norm(ba, axis=1) * np.linalg.norm(bc, axis=1), 1e-8
    )
    return np.degrees(np.arccos(np.clip(cosv, -1.0, 1.0)))


def summarize(values, valid, name, result):
    values = np.asarray(values, dtype=float)
    valid = np.asarray(valid, dtype=bool) & np.isfinite(values)
    sample = values[valid]
    if len(sample) < 5:
        # Missingness is separately represented by valid_fraction.
        sample = np.array([0.0])
    result[f"{name}_mean"] = float(np.mean(sample))
    result[f"{name}_std"] = float(np.std(sample))
    result[f"{name}_range80"] = float(np.percentile(sample, 90) - np.percentile(sample, 10))
    result[f"{name}_valid_fraction"] = float(np.mean(valid))


def extract_one(sequence, duration):
    if sequence.ndim != 2 or sequence.shape[1] != 99:
        raise ValueError(f"Expected [time,99], got {sequence.shape}")
    if duration <= 0:
        raise ValueError("Invalid duration")
    pose = sequence.reshape(-1, 33, 3)
    xy, confidence_mask = pose[:, :, :2], pose[:, :, 2] >= 0.5
    result = {"duration_sec": float(duration)}

    for side in ("L", "R"):
        hip, knee, ankle = (JOINTS[f"{part}_{side}"] for part in ("hip", "knee", "ankle"))
        shoulder, elbow, wrist = (JOINTS[f"{part}_{side}"] for part in ("shoulder", "elbow", "wrist"))

        knee_angle = angle_deg(xy[:, hip], xy[:, knee], xy[:, ankle])
        hip_angle = angle_deg(xy[:, shoulder], xy[:, hip], xy[:, knee])
        elbow_angle = angle_deg(xy[:, shoulder], xy[:, elbow], xy[:, wrist])
        for name, arr, ids in (
            ("knee_angle", knee_angle, (hip, knee, ankle)),
            ("hip_angle", hip_angle, (shoulder, hip, knee)),
            ("elbow_angle", elbow_angle, (shoulder, elbow, wrist)),
        ):
            summarize(arr, confidence_mask[:, list(ids)].all(axis=1), f"{side}_{name}", result)

        for part in ("knee", "ankle", "wrist", "toe"):
            j = JOINTS[f"{part}_{side}"]
            for axis, axis_name in enumerate(("x", "y")):
                summarize(xy[:, j, axis], confidence_mask[:, j], f"{side}_{part}_{axis_name}", result)
            # Average 2D landmark speed in torso lengths per second.
            step_seconds = duration / (len(xy) - 1)
            speed = np.linalg.norm(np.diff(xy[:, j], axis=0), axis=1) / step_seconds
            valid_pairs = confidence_mask[1:, j] & confidence_mask[:-1, j]
            summarize(speed, valid_pairs, f"{side}_{part}_speed", result)

    # Pelvic tilt and shoulder tilt (normalized image-coordinate displacement).
    for part in ("hip", "shoulder"):
        left, right = JOINTS[f"{part}_L"], JOINTS[f"{part}_R"]
        pair_valid = confidence_mask[:, [left, right]].all(axis=1)
        summarize(xy[:, left, 1] - xy[:, right, 1], pair_valid, f"{part}_vertical_difference", result)

    # Direct left-right asymmetry summaries (same-side statistics may also help).
    for part in ("knee", "ankle", "wrist"):
        left, right = JOINTS[f"{part}_L"], JOINTS[f"{part}_R"]
        pair_valid = confidence_mask[:, [left, right]].all(axis=1)
        for axis, axis_name in enumerate(("x", "y")):
            delta = np.abs(xy[:, left, axis] - xy[:, right, axis])
            summarize(delta, pair_valid, f"{part}_LR_absdiff_{axis_name}", result)

    return result


def extract_all(X, durations):
    rows = [extract_one(x, float(t)) for x, t in zip(X, durations)]
    names = list(rows[0])
    features = np.array([[row[name] for name in names] for row in rows], dtype=np.float32)
    if not np.isfinite(features).all():
        raise ValueError("Non-finite engineered features")
    return features, names


def select_feature_columns(X, names, policy):
    if policy == "all":
        indices = list(range(len(names)))
    elif policy == "no_duration":
        indices = [i for i, name in enumerate(names) if name != "duration_sec"]
    elif policy == "no_timing":
        # Speed features are divided by duration as well: remove them too.
        indices = [i for i, name in enumerate(names) if name != "duration_sec" and "_speed_" not in name]
    else:
        raise ValueError(f"Unknown feature policy: {policy}")
    return X[:, indices], [names[i] for i in indices]


def load_split(path):
    with np.load(path, allow_pickle=False) as data:
        return {name: data[name] for name in ("X", "y", "duration_sec", "record_key", "original_label")}


def evaluate(y_true, y_pred):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "abnormal_precision": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "abnormal_recall": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "abnormal_f1": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "normal_recall": float(recall_score(y_true, y_pred, pos_label=0, zero_division=0)),
        "confusion_matrix_normal_abnormal": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
    }


def main():
    parser = argparse.ArgumentParser(description="Train pose-only Random Forest on train/validation only")
    parser.add_argument("--data", type=Path, default=Path("gait_step2"))
    parser.add_argument("--out", type=Path, default=Path("gait_step3_rf"))
    feature_choice = parser.add_mutually_exclusive_group()
    feature_choice.add_argument("--exclude-duration", action="store_true", help="Drop duration_sec, retain speed features")
    feature_choice.add_argument("--exclude-timing", action="store_true", help="Drop duration_sec and all time-based speed features")
    args = parser.parse_args()
    policy = "no_timing" if args.exclude_timing else "no_duration" if args.exclude_duration else "all"
    train = load_split(args.data / "train.npz")
    val = load_split(args.data / "val.npz")
    if set(train["record_key"].tolist()) & set(val["record_key"].tolist()):
        raise ValueError("Record overlap detected between train and validation")

    X_train, names = extract_all(train["X"], train["duration_sec"])
    X_val, val_names = extract_all(val["X"], val["duration_sec"])
    assert names == val_names
    X_train, names = select_feature_columns(X_train, names, policy)
    X_val, val_names = select_feature_columns(X_val, val_names, policy)
    assert names == val_names
    y_train, y_val = train["y"], val["y"]

    candidates = [
        {"n_estimators": 200, "max_depth": None, "min_samples_leaf": 1},
        {"n_estimators": 200, "max_depth": None, "min_samples_leaf": 3},
        {"n_estimators": 200, "max_depth": 8, "min_samples_leaf": 2},
        {"n_estimators": 400, "max_depth": 8, "min_samples_leaf": 4},
    ]
    attempts = []
    chosen = None
    # Macro F1 gives the minority Normal class meaningful weight.
    for idx, params in enumerate(candidates):
        model = RandomForestClassifier(**params, class_weight="balanced", random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        scores = evaluate(y_val, preds)
        attempts.append({"candidate": idx + 1, **params, **{k: v for k, v in scores.items() if not isinstance(v, list)}})
        ranking = (scores["macro_f1"], scores["balanced_accuracy"])
        if chosen is None or ranking > chosen["ranking"]:
            chosen = {"model": model, "params": params, "scores": scores, "prediction": preds, "ranking": ranking}

    args.out.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": chosen["model"], "feature_names": names, "feature_policy": policy, "preprocessing": "gait_preprocess.py: hip-centered XY and torso-scaled, 128 frames", "label_map": {"normal": 0, "abnormal": 1}}, args.out / "rf_model.joblib")
    importance = sorted(zip(names, chosen["model"].feature_importances_), key=lambda pair: pair[1], reverse=True)
    with (args.out / "feature_importance.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.writer(stream)
        writer.writerow(["feature", "importance"])
        writer.writerows(importance)
    with (args.out / "candidate_results.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(attempts[0]))
        writer.writeheader()
        writer.writerows(attempts)
    with (args.out / "validation_predictions.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.writer(stream)
        writer.writerow(["record_key", "original_label", "true_binary", "predicted_binary"])
        for key, label, actual, predicted in zip(val["record_key"], val["original_label"], y_val, chosen["prediction"]):
            writer.writerow([key, label, int(actual), int(predicted)])
    result = {
        "model": "RandomForestClassifier", "feature_count": len(names), "feature_policy": policy,
        "train_records": len(y_train), "validation_records": len(y_val),
        "selection_criterion": "maximum validation macro_f1, tie break balanced_accuracy",
        "selected_hyperparameters": {**chosen["params"], "class_weight": "balanced", "random_state": 42},
        "validation_metrics": chosen["scores"],
        "warning": "Test split was not read. Validation is record-level and does not guarantee participant independence.",
        "timing_note": "no_duration still retains speed features computed using duration_sec; no_timing removes both direct duration and those speed features.",
    }
    (args.out / "validation_summary.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"\nSaved outputs to {args.out.resolve()}")


if __name__ == "__main__":
    main()
