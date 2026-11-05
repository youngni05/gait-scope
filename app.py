"""GaitScope: research-only MP4 gait classification demonstration.

Run from the project root: streamlit run app.py
"""

from __future__ import annotations

import html
import logging
import sys
import tempfile
from pathlib import Path

import streamlit as st
from scripts.ui_metrics import display_class_score, model_predictions_disagree

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "scripts"))

st.set_page_config(page_title="GaitScope | AI 보행 패턴 분석", page_icon="🚶", layout="centered")

st.markdown(
    """
<style>
    .block-container { max-width: 960px; padding-top: 2.3rem; padding-bottom: 3rem; }
    .gs-brand { text-align: center; padding: 1.2rem 0 1.5rem; }
    .gs-brand h1 { font-size: 2.75rem; font-weight: 800; letter-spacing: -.055em; margin: 0; }
    .gs-brand p { color: #8393a8; margin: .4rem 0 0; font-size: 1rem; }
    .gs-eyebrow { font-size: .8rem; letter-spacing: .12em; color: #6b92b8; font-weight: 700; text-align:center; }
    .gs-cardhead { display:flex; justify-content:space-between; align-items:center; gap:.5rem; }
    .gs-model { font-size: 1.05rem; font-weight: 750; }
    .gs-badge { border-radius: 99px; font-size: .77rem; padding: .25rem .65rem; font-weight: 700; }
    .gs-badge.abnormal { background:#fcebe7; color:#a93b2b; }
    .gs-badge.normal { background:#e4f5ea; color:#227347; }
    .gs-percent { font-size: 2.1rem; font-weight: 800; letter-spacing: -.05em; margin: 1rem 0 .1rem; }
    .gs-hint { font-size:.84rem; color:#8190a0; }
    .gs-bar-track { background:rgba(126,145,169,.20); height:9px; border-radius:99px; overflow:hidden; margin:1.15rem 0 1.3rem; }
    .gs-bar-fill { height:100%; border-radius:99px; }
    .gs-footer { text-align:center; font-size:.82rem; color:#8994a0; margin-top:1.6rem; }
    div[data-testid="stFileUploader"] { border: 1.5px dashed #9aaac0; border-radius: 16px; padding: 1rem; }
</style>
""",
    unsafe_allow_html=True,
)


def header(subtitle: str):
    st.markdown(
        '<div class="gs-brand"><h1>🚶 GaitScope</h1>'
        f'<p>{html.escape(subtitle)}</p></div>',
        unsafe_allow_html=True,
    )


def reset():
    for key in ("view", "predictions", "pending_video", "error_message"):
        st.session_state.pop(key, None)
    st.session_state["upload_key"] = st.session_state.get("upload_key", 0) + 1
    st.rerun()


def result_card(title: str, data: dict):
    abnormal = data["class_id"] == 1
    score_label, probability = display_class_score(data)
    with st.container(border=True):
        st.markdown(
            '<div class="gs-cardhead">'
            f'<div class="gs-model">{html.escape(title)}</div>'
            f'<span class="gs-badge {"abnormal" if abnormal else "normal"}">'
            f'{"비정상 패턴" if abnormal else "정상 패턴"}</span></div>',
            unsafe_allow_html=True,
        )
        if probability is None:
            st.markdown("**추정 점수 표시 불가**")
            st.caption("확률 보정 파일을 확인해 주세요.")
        else:
            percentage = 100.0 * probability
            color = "#dc7d6a" if abnormal else "#4ba789"
            st.markdown(
                f'<div class="gs-percent">{percentage:.2f}%</div>'
                f'<div class="gs-hint">{score_label}</div>'
                '<div class="gs-bar-track">'
                f'<div class="gs-bar-fill" style="width:{percentage:.4f}%;background:{color}"></div>'
                '</div>',
                unsafe_allow_html=True,
            )


def render_results():
    header("AI 모델별 보행 패턴 분석 결과")
    st.success("분석이 완료되었습니다.", icon="✅")
    st.subheader("모델별 분석 결과")
    models = (
        ("Random Forest", "RF-B"),
        ("1D-CNN", "CNN-D"),
        ("SVM", "SVM-H"),
        ("XGBoost", "XGB-A"),
    )
    for row in (models[:2], models[2:]):
        left, right = st.columns(2, gap="medium")
        for col, (title, key) in zip((left, right), row):
            with col:
                result_card(title, st.session_state["predictions"][key])

    if model_predictions_disagree(st.session_state["predictions"]):
        st.warning(
            "AI 모델마다 분석 방식이 달라 동일한 영상에서도 서로 다른 결과가 나올 수 있습니다. "
            "모델별 판정을 개별적으로 확인해 주세요.",
            icon="⚠️",
        )
    st.info(
        "이 결과는 연구용 보행 패턴 분류 결과입니다. 표시된 추정 점수는 모델의 정확도나 "
        "의료적 진단 확률을 뜻하지 않으며, 모델별 점수의 보정 조건도 동일하지 않습니다.",
        icon="ℹ️",
    )
    if st.button("다른 영상 분석하기", type="primary", use_container_width=True):
        reset()


def process_upload():
    from scripts.web_backend import run_web_analysis

    header("업로드한 보행 영상을 분석하고 있습니다")
    st.markdown('<div class="gs-eyebrow">AI ANALYSIS IN PROGRESS</div>', unsafe_allow_html=True)
    status_text = st.empty()
    bar = st.progress(0.0)
    with st.status("영상 분석 처리 중", expanded=True) as steps:
        last_step = [None]

        def progress(message: str, fraction: float):
            status_text.write(f"⏳ {message}")
            bar.progress(max(0.0, min(1.0, fraction)))
            stage = message.split(" · ", 1)[0]
            if last_step[0] != stage:
                steps.write(stage)
                last_step[0] = stage

        try:
            with tempfile.TemporaryDirectory(prefix="gaitscope-") as temporary:
                path = Path(temporary) / "uploaded.mp4"
                path.write_bytes(st.session_state["pending_video"]["content"])
                predictions = run_web_analysis(path, progress)
            st.session_state["predictions"] = predictions
            st.session_state.pop("pending_video", None)
            st.session_state.pop("error_message", None)
            st.session_state["view"] = "result"
            steps.update(label="분석 완료", state="complete", expanded=False)
        except Exception as exc:
            logging.exception("GaitScope analysis failed")
            steps.update(label="분석 실패", state="error", expanded=False)
            st.session_state["error_message"] = (
                str(exc) if isinstance(exc, ValueError) and len(str(exc)) < 180
                else "영상 분석 과정에 문제가 발생했습니다. 파일 형식과 촬영 조건을 확인해 주세요."
            )
            st.session_state["view"] = "error"
    st.rerun()


def render_error():
    header("분석을 완료하지 못했어요")
    st.error(st.session_state.get("error_message", "영상 분석 중 오류가 발생했습니다."), icon="⚠️")
    st.write("동일한 영상으로 다시 시도하거나, 다른 영상을 선택할 수 있어요.")
    left, right = st.columns(2)
    with left:
        if st.button("↻ 다시 시도하기", use_container_width=True, type="primary"):
            st.session_state["view"] = "processing"
            st.rerun()
    with right:
        if st.button("← 다른 영상 선택하기", use_container_width=True):
            reset()


def render_upload():
    header("영상 기반 AI 보행 패턴 분석")
    st.markdown("### 보행 영상 업로드")
    st.write("보행 영상을 업로드하고 네 가지 AI 모델의 분석 결과를 확인하세요.")
    uploaded = st.file_uploader(
        "MP4 영상 선택",
        type=["mp4"],
        accept_multiple_files=False,
        key=f"video_{st.session_state.get('upload_key', 0)}",
        help="한 사람의 전신이 나오는 60초 이하 MP4 영상",
    )
    if uploaded is not None:
        st.caption(f"선택된 파일: {uploaded.name} · {uploaded.size / (1024 * 1024):.1f} MB")
        st.video(uploaded)

    with st.expander("촬영 가이드", expanded=True):
        st.markdown(
            """
            1. 휴대폰을 **약 1m 높이**의 삼각대에 고정해 주세요.
            2. 카메라에 **등을 보인 상태**에서 출발해 6m를 직선으로 걸어가세요.
            3. 6m 지점에서 자연스럽게 방향을 전환한 뒤,
            카메라를 바라보며 출발 지점까지 걸어오세요.
            4. 전체 왕복 과정에서 **머리부터 발끝까지 전신이 보이도록**
            촬영해 주세요.
            """
        )
        
    st.caption("권장 영상 길이: 약 15~25초 · 지원 형식: MP4 · 최대 분석 길이: 60초")
    too_big = uploaded is not None and uploaded.size > 120 * 1024 * 1024
    if too_big:
        st.warning("120MB 이하 영상을 선택해 주세요.")
    clicked = st.button(
        "보행 패턴 분석 시작", type="primary", use_container_width=True,
        disabled=uploaded is None or too_big,
    )
    st.markdown(
        '<div class="gs-footer">연구용 시연 시스템 · 의료 진단 목적 사용 불가 · '
        '업로드 영상은 분석을 위한 임시 파일로만 사용</div>',
        unsafe_allow_html=True,
    )
    if clicked:
        st.session_state["pending_video"] = {"content": uploaded.getvalue()}
        st.session_state["view"] = "processing"
        st.rerun()


if st.session_state.get("view") == "result" and "predictions" in st.session_state:
    render_results()
elif st.session_state.get("view") == "processing" and "pending_video" in st.session_state:
    process_upload()
elif st.session_state.get("view") == "error" and "pending_video" in st.session_state:
    render_error()
else:
    render_upload()
