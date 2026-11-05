"""Thin Streamlit adapter around the existing, frozen gait model inference."""

from __future__ import annotations

from typing import Callable

import numpy as np

from gait_preprocess import prepare_pose
from predict_video import DEFAULT_MODELS, predict_prepared_pose
from video_pose_inference import extract_video_pose


def run_web_analysis(video_path, callback: Callable[[str, float], None]):
    callback("영상 확인 및 관절 추출 준비", 0.05)

    def pose_progress(done, total):
        if total and total > 0:
            fraction = min(1.0, done / total)
            callback(f"MediaPipe 관절 좌표 추출 중 · {done}/{total}프레임", 0.08 + 0.68 * fraction)
        else:
            callback(f"MediaPipe 관절 좌표 추출 중 · {done}프레임 처리", 0.08)

    frames, stats = extract_video_pose(video_path, max_video_sec=60.0, progress_callback=pose_progress)
    if stats.detected_frames < 30:
        raise ValueError("관절이 충분히 검출되지 않았어요. 전신이 보이는 영상을 다시 업로드해 주세요.")
    if stats.detection_rate < 0.60:
        raise ValueError("관절 검출이 불안정해요. 카메라 구도와 조명을 확인하고 다시 촬영해 주세요.")

    callback("관절 좌표 정규화 및 전처리", 0.80)
    sequence, duration, _ = prepare_pose(frames, steps=128, min_visibility=0.5)
    callback("4개 AI 모델 분석 중", 0.90)
    predictions = predict_prepared_pose(sequence, duration, models_dir=DEFAULT_MODELS)
    callback("분석 완료", 1.0)
    return predictions
