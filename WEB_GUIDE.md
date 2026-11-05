# GaitScope Streamlit 웹 프로토타입

기존 네 모델(RF-B, CNN-D, SVM-H, XGB-A)과 MediaPipe 영상 분석 코드를
그대로 사용하며, **모델 재학습은 하지 않습니다**.

## 필요한 파일

기존 프로젝트 루트(`D:\gait\gait-video-inference\gait_video_inference`)에:

- `app.py` **추가**
- `scripts/web_backend.py` **추가**
- `scripts/ui_metrics.py` **추가** (모델 판정별 점수 표시)
- `scripts/video_pose_inference.py` **교체** (기존과 동일한 추출, 진행 상태 콜백만 추가)
- `requirements.txt` **교체** (호환 버전 고정: Streamlit 1.45.1, NumPy 1.26.4, Protobuf 4.25.9, Pandas 2.2.3)

기존에 동작을 확인한 다음 파일은 그대로 있어야 합니다.

- `scripts/predict_video.py` (JSON XGB 및 SVM 확률 보정 반영 버전)
- `scripts/xgb_native_io.py`
- `models/xgboost/xgb_model.json`, `models/xgboost/xgb_conversion_verified.json`
- `models/svm/svm_calibrator.joblib`
- 그 외 RF-B/CNN-D/SVM-H의 기존 모델 파일

## Windows PowerShell 실행

가상환경 `(gait-video)`가 표시된 기존 VS Code 터미널에서:

```powershell
cd D:\gait\gait-video-inference\gait_video_inference
python -m pip install "streamlit==1.45.1" "pandas==2.2.3" "numpy==1.26.4" "protobuf==4.25.9" --no-cache-dir
python -m pip check
python -m streamlit run app.py
```

터미널에 표시되는 로컬 주소(보통 `http://localhost:8501`)를 브라우저에서 엽니다.
웹 주소를 외부에 공개하는 **서버 배포는 아직 수행하지 않았습니다**.

### 사용

1. MP4 파일을 업로드하고, 원하면 재생해 확인합니다.
2. **보행 패턴 분석 시작**을 누릅니다.
3. MediaPipe 프레임 처리 진행률과 데이터 전처리/모델 추론 단계가 표시됩니다.
4. 분석이 성공하면 결과 화면으로 자동 전환됩니다.
5. 정상 판정은 정상 추정 점수, 비정상 판정은 비정상 추정 점수와 막대를 확인합니다.
6. 모델 판정이 불일치하면 간단한 안내가 표시되며, 스켈레톤 시각화는 생략했습니다.
7. 오류 시 **다시 시도하기**, **다른 영상 선택** 버튼을 사용합니다.
8. **다른 영상 분석하기**로 처음 화면으로 돌아갑니다.

촬영 가이드: 휴대폰을 약 1m 높이에 고정하고, 카메라 정면에서 6m 구간을
왕복하는 모습을 촬영하세요. 한 사람의 전신과 발이 보여야 합니다.

## 개인정보 및 데이터 처리

- 업로드 영상은 로컬 Streamlit 서버에 전송되어 **처리 중 임시 파일로 저장**됩니다.
- 분석 성공과 오류 모두에서 임시 파일이 자동 정리되도록 구현했습니다.
- 오류 발생 시 **같은 영상 재시도**가 가능하도록 영상 바이트를 브라우저 세션에 연결된
  서버 메모리에 일시 보관합니다. 결과가 나오거나 다른 영상을 선택하면 이 복사본을 지웁니다.
- 결과 화면에는 업로드 영상이나 관절 이미지가 표시되지 않습니다.
- 관절 검출률이 60% 미만이면 결과를 보여주지 않습니다(내부 검증).
- 분석 결과는 해당 Streamlit 브라우저 세션 메모리에 남으며
  `다른 영상 분석하기`를 누르면 초기화됩니다.
- 사람의 보행 동작은 식별 가능성을 완전히 배제할 수 없습니다. 본인 또는
  사용 권한이 확인된 영상만 업로드하고, 타인의 민감한 영상을 공개 서버에
  올리지 마세요.

## 연구적 주의

- 실제 환자 진단이 아니라 **모사된 정상/비정상 보행 패턴 분류 연구**입니다.
- 모델 간 확률 출력은 동일한 독립 보정 과정을 거치지 않았습니다.
- 카메라 각도, 영상 길이, 방향 전환 및 발 가림은 결과에 영향을 줄 수 있습니다.
- 실제 사용 환경에 대한 검증 없이 결과를 의학적 판단에 사용하면 안 됩니다.

## 개발 검증

```powershell
python -m unittest discover -s tests -v
```

이 웹 앱은 실행 환경별 MediaPipe/Streamlit 기능을 영은님의 Windows PC에서
추가 확인해야 합니다. 터미널에 오류가 나오면 전체 메시지를 공유해 주세요.
