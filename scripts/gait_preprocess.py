"""Step 1: inspect and preprocess MediaPipe gait pose records from a ZIP.

Usage: python gait_preprocess.py "Multimodal_Pathological_Gait_Dataset.zip" --out gait_step1
Requires: numpy (python -m pip install numpy)
No smartphone, smartwatch, audio or video files are read.
"""

import argparse
import csv
import json
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np


CLASS_MAP = {
    "normal": ("normal", 0),
    "normal gait": ("normal", 0),
    "control": ("normal", 0),
    "duchenne": ("duchenne", 1),
    "duchenne gait": ("duchenne", 1),
    "hemi": ("hemiparesis", 1),
    "hemiparesis": ("hemiparesis", 1),
    "para": ("paraparesis", 1),
    "paraparesis": ("paraparesis", 1),
    "trendelenburg": ("trendelenburg", 1),
    "trendelenburg gait": ("trendelenburg", 1),
    "foot_drop": ("foot_drop", 1),
    "foot drop": ("foot_drop", 1),
}


def read_json(archive, name):
    return json.loads(archive.read(name).decode("utf-8-sig"))


def normalize_label(value):
    return str(value).strip().lower().replace("-", " ").replace("_", " ")


def prepare_pose(frames, steps, min_visibility):
    """Return 128x(33*3) array: normalized XY, validity mask, and duration."""
    if not isinstance(frames, list) or len(frames) < 30:
        raise ValueError("too_few_frames")
    samples = []
    for frame in frames:
        joints = frame.get("landmarks")
        if not isinstance(joints, list) or len(joints) != 33:
            continue
        try:
            t = float(frame["timestamp"])
            xyz = np.array([[p["x"], p["y"], p.get("visibility", 1.0)] for p in joints], dtype=float)
        except (KeyError, TypeError, ValueError):
            continue
        if np.isfinite(t):
            samples.append((t, xyz))
    if len(samples) < 30:
        raise ValueError("too_few_valid_frames")

    samples.sort(key=lambda item: item[0])
    # Duplicate timestamps: keep the first frame at each time point.
    unique = {}
    for t, data in samples:
        unique.setdefault(t, data)
    ts = np.array(list(unique), dtype=float)
    data = np.stack(list(unique.values()))
    duration = (ts[-1] - ts[0]) / 1000.0
    if duration < 2.0 or duration > 180.0:
        raise ValueError("invalid_duration")

    xy = data[:, :, :2].copy()
    visibility = data[:, :, 2]
    valid = np.isfinite(xy).all(axis=2) & np.isfinite(visibility) & (visibility >= min_visibility)
    visible_ratio = float(valid.mean())
    if visible_ratio < 0.5:
        raise ValueError("low_visibility")

    # Interpolate low-confidence joint positions on the original time axis.
    for joint in range(33):
        indices = np.where(valid[:, joint])[0]
        if len(indices) < 2:
            xy[:, joint, :] = 0.0
            continue
        for axis in range(2):
            xy[:, joint, axis] = np.interp(ts, ts[indices], xy[indices, joint, axis])

    # Center on hip midpoint and scale by robust shoulder-hip torso length.
    hip = (xy[:, 23] + xy[:, 24]) / 2
    shoulder = (xy[:, 11] + xy[:, 12]) / 2
    torso = np.linalg.norm(shoulder - hip, axis=1)
    usable = np.isfinite(torso) & (torso > 0.02)
    if usable.sum() < len(torso) // 2:
        raise ValueError("invalid_torso_scale")
    scale = float(np.median(torso[usable]))
    xy = (xy - hip[:, None, :]) / scale
    if not np.isfinite(xy).all():
        raise ValueError("nonfinite_coordinates")

    target = np.linspace(ts[0], ts[-1], steps)
    xy_out = np.empty((steps, 33, 2), dtype=np.float32)
    mask_out = np.empty((steps, 33), dtype=np.float32)
    for joint in range(33):
        for axis in range(2):
            xy_out[:, joint, axis] = np.interp(target, ts, xy[:, joint, axis])
        # Mask highlights where source landmark confidence was low.
        mask_out[:, joint] = np.interp(target, ts, valid[:, joint].astype(float))

    features = np.concatenate((xy_out, mask_out[..., None]), axis=2)
    return features.reshape(steps, 99), duration, visible_ratio


def main():
    parser = argparse.ArgumentParser(description="Preprocess pose-only gait dataset from ZIP")
    parser.add_argument("zip_path", type=Path)
    parser.add_argument("--out", type=Path, default=Path("gait_step1"))
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--visibility", type=float, default=0.5)
    args = parser.parse_args()
    if args.steps < 16 or not 0 <= args.visibility <= 1:
        parser.error("--steps must be >=16 and --visibility in [0,1]")
    args.out.mkdir(parents=True, exist_ok=True)

    details = []
    processed = []
    y_values, durations, record_keys, original_labels = [], [], [], []
    with zipfile.ZipFile(args.zip_path) as archive:
        names = set(archive.namelist())
        metadata = sorted(n for n in names if n.endswith("/meta_data.json"))
        for name in metadata:
            folder = name.rsplit("/", 1)[0]
            location = next((part for part in folder.split("/") if part.startswith("Location_")), "unknown")
            row = {"folder": folder, "location": location, "original_label": "", "class": "", "binary_label": "", "frames": 0, "duration_sec": "", "visible_ratio": "", "status": ""}
            try:
                meta = read_json(archive, name)
                label = normalize_label(meta.get("gait_pattern", ""))
                row["original_label"] = str(meta.get("gait_pattern", ""))
                if label not in CLASS_MAP:
                    raise ValueError("unknown_label")
                canonical, y = CLASS_MAP[label]
                row["class"] = canonical
                row["binary_label"] = y
                pose_name = folder + "/pose_landmarks.json"
                if pose_name not in names:
                    raise ValueError("pose_missing")
                frames = read_json(archive, pose_name)
                row["frames"] = len(frames) if isinstance(frames, list) else 0
                feat, duration, visibility = prepare_pose(frames, args.steps, args.visibility)
                row["duration_sec"] = round(duration, 3)
                row["visible_ratio"] = round(visibility, 4)
                row["status"] = "included"
                processed.append(feat)
                y_values.append(y)
                durations.append(duration)
                record_keys.append(folder)
                original_labels.append(canonical)
            except (ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
                row["status"] = str(exc)
            details.append(row)

    report_path = args.out / "quality_report.csv"
    with report_path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(details[0]) if details else ["folder"])
        writer.writeheader()
        writer.writerows(details)

    if processed:
        np.savez_compressed(
            args.out / "pose_preprocessed.npz",
            X=np.stack(processed).astype(np.float32),
            y=np.asarray(y_values, dtype=np.int64),
            duration_sec=np.asarray(durations, dtype=np.float32),
            record_key=np.asarray(record_keys),
            original_label=np.asarray(original_labels),
        )
    counts = Counter(row["status"] for row in details)
    labels = Counter(row["class"] for row in details if row["status"] == "included")
    summary = {"total_metadata": len(details), "included": len(processed), "status_counts": dict(counts), "included_class_counts": dict(labels), "shape": list(np.stack(processed).shape) if processed else None, "important": "Record-level preprocessing only. No train/test split or model fitting performed."}
    (args.out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Saved outputs to {args.out.resolve()}")


if __name__ == "__main__":
    main()
