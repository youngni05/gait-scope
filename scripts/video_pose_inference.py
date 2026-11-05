"""Extract MediaPipe Pose landmarks from a real walking video.

The output frame dictionaries intentionally mirror the dataset's
``pose_landmarks.json`` schema. No IMU, audio, or ground-truth labels are used.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable


@dataclass(frozen=True)
class VideoPoseStats:
    source_fps: float
    video_duration_sec: float
    sampled_frames: int
    detected_frames: int
    detection_rate: float
    frame_width: int
    frame_height: int


def extract_video_pose(
    video_path: str | Path,
    *,
    target_fps: float = 25.0,
    max_video_sec: float = 60.0,
    progress_callback: Callable[[int, int | None], None] | None = None,
) -> tuple[list[dict], VideoPoseStats]:
    """Return dataset-shaped pose frames and non-identifying quality stats.

    Only frames with a detected body are returned. Timestamps use the video's
    actual elapsed timeline; skipping missing detections does not compress time.
    """
    import cv2
    import mediapipe as mp

    path = Path(video_path)
    if not path.is_file():
        raise FileNotFoundError(f"Video not found: {path}")
    if target_fps <= 0 or max_video_sec <= 0:
        raise ValueError("target_fps and max_video_sec must be positive")

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError("Cannot open video; provide a readable MP4/MOV file")

    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if not 0 < fps < 240:
            raise ValueError("Video has an invalid or unsupported frame rate")
        if width < 64 or height < 64:
            raise ValueError("Video resolution is too low")
        duration = count / fps if count > 0 else 0.0
        if duration > max_video_sec:
            raise ValueError(
                f"Video is {duration:.1f}s; maximum allowed is {max_video_sec:.1f}s"
            )

        # Do not interpolate or synthesize landmarks before the original
        # gait_preprocess.prepare_pose function handles missing visibility.
        frame_step = max(1, round(fps / target_fps))
        expected_samples = (count + frame_step - 1) // frame_step if count > 0 else None
        samples: list[dict] = []
        processed = 0
        found = 0
        frame_idx = 0
        with mp.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        ) as pose:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if frame_idx % frame_step:
                    frame_idx += 1
                    continue
                timestamp_ms = round((frame_idx / fps) * 1000)
                frame_idx += 1
                processed += 1
                if progress_callback is not None and (processed == 1 or processed % 10 == 0):
                    progress_callback(processed, expected_samples)

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgb.flags.writeable = False
                output = pose.process(rgb)
                if output.pose_landmarks is None:
                    continue
                joints = [
                    {
                        "x": float(point.x),
                        "y": float(point.y),
                        "z": float(point.z),
                        "visibility": float(point.visibility),
                    }
                    for point in output.pose_landmarks.landmark
                ]
                if len(joints) != 33:
                    continue
                found += 1
                samples.append({"timestamp": timestamp_ms, "landmarks": joints})

        if processed == 0:
            raise ValueError("No decodable video frames")
        if progress_callback is not None:
            progress_callback(processed, expected_samples)
        return samples, VideoPoseStats(
            source_fps=fps,
            video_duration_sec=duration or (frame_idx / fps),
            sampled_frames=processed,
            detected_frames=found,
            detection_rate=found / processed,
            frame_width=width,
            frame_height=height,
        )
    finally:
        cap.release()
