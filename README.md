# 👁️ 눈동자 추적 자동 스크롤 시스템

사용자의 눈동자 움직임을 인식하여 화면을 자동으로 스크롤하는 시스템입니다.

## 🎯 주요 기능

- **실시간 눈동자 추적**: MediaPipe를 사용한 정확한 얼굴 랜드마크 감지
- **자동 스크롤**: 위/아래 시선 방향에 따른 자동 스크롤
- **가로 스크롤 지원**: 좌/우 시선 방향 감지 (옵션)
- **GUI 설정 패널**: 실시간 설정 조정 가능
- **부드러운 추적**: 노이즈 제거를 위한 스무딩 알고리즘
- **설정 저장**: 사용자 설정 자동 저장

## 🛠️ 기술 스택

| 목적 | 라이브러리/도구 |
|------|----------------|
| 웹캠 사용 | OpenCV |
| 눈동자 추적 | MediaPipe |
| 시선 방향 해석 | 자체 알고리즘 |
| 스크롤 제어 | pyautogui |
| GUI 표시 | Tkinter |

## 📦 설치 방법

### 1. 필요한 패키지 설치

```bash
pip install -r requirements.txt
```

### 2. 시스템 요구사항

- Python 3.7 이상
- 웹캠
- Windows/macOS/Linux 지원

## 🚀 사용 방법

### 기본 버전 실행

```bash
python eye_tracking_scroll.py
```

### 테스트 페이지 열기

```bash
# 브라우저에서 test_page.html 파일을 열어주세요
```

## ⚙️ 설정 옵션

### 기본 설정
- **스크롤 감도**: 스크롤 반응 민감도 조정
- **스크롤 간격**: 연속 스크롤 간의 시간 간격
- **위쪽 임계값**: 위로 스크롤할 시선 위치 (0.0~1.0)
- **아래쪽 임계값**: 아래로 스크롤할 시선 위치 (0.0~1.0)
- **스크롤 양**: 한 번에 스크롤할 픽셀 수
- **부드러운 추적**: 노이즈 제거를 위한 스무딩 정도
- **가로 스크롤**: 좌우 스크롤 기능 활성화/비활성화

## 🎮 사용법

1. **프로그램 실행**: `python advanced_eye_tracker.py`
2. **GUI 설정**: 원하는 설정값 조정 후 "설정 저장" 클릭
3. **추적 시작**: "시작" 버튼 클릭
4. **테스트**: 웹캠 앞에서 위/아래를 바라보며 스크롤 테스트
5. **종료**: "중지" 버튼 또는 'q' 키로 종료

## 📁 프로젝트 구조

```
cursor-tutorial/
├── eye_tracking_scroll.py      # 기본 눈동자 추적 시스템
├── advanced_eye_tracker.py     # 고급 GUI 버전
├── test_page.html             # 테스트용 웹 페이지
├── requirements.txt           # 필요한 패키지 목록
├── README.md                 # 프로젝트 설명서
└── eye_tracker_config.json   # 설정 파일 (자동 생성)
```

## 🔧 주요 클래스 및 함수

### EyeTrackingScroll (기본 버전)
- `__init__()`: 시스템 초기화
- `detect_gaze_direction()`: 시선 방향 감지
- `scroll_based_on_gaze()`: 시선에 따른 스크롤 제어
- `run()`: 메인 실행 루프

## 🎯 시선 감지 알고리즘

1. **얼굴 랜드마크 감지**: MediaPipe Face Mesh 사용
2. **눈동자 영역 추출**: 좌/우 눈 랜드마크 좌표 계산
3. **시선 중심점 계산**: 양쪽 눈의 평균 중심점
4. **정규화**: 화면 크기에 따른 좌표 정규화 (0~1)
5. **스무딩**: 히스토리 기반 가중 평균으로 노이즈 제거
6. **임계값 비교**: 설정된 임계값과 비교하여 스크롤 방향 결정

## 🐛 문제 해결

### 웹캠이 인식되지 않는 경우
```bash
# 웹캠 인덱스 변경
self.cap = cv2.VideoCapture(1)  # 0, 1, 2 등으로 시도
```

### 스크롤이 너무 민감한 경우
- GUI에서 "스크롤 감도" 값을 낮춰주세요
- "스크롤 간격" 값을 높여주세요

### 시선 추적이 부정확한 경우
- 조명을 밝게 해주세요
- 웹캠과의 거리를 적절히 유지해주세요
- "부드러운 추적" 값을 조정해주세요

## 📊 성능 최적화

- **프레임 처리**: 30fps 이상 유지
- **메모리 사용량**: 히스토리 크기 제한 (최대 10개)
- **CPU 사용량**: MediaPipe 최적화된 모델 사용
- **반응성**: 스레드 분리로 GUI 블로킹 방지

## 🔒 보안 및 안전

- 웹캠 데이터는 로컬에서만 처리
- 개인정보 수집하지 않음
- 설정 파일은 로컬에만 저장

## 🤝 기여하기

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다.

## 🙏 감사의 말

- [MediaPipe](https://mediapipe.dev/) - 얼굴 랜드마크 감지
- [OpenCV](https://opencv.org/) - 컴퓨터 비전 라이브러리
- [pyautogui](https://pyautogui.readthedocs.io/) - 자동화 라이브러리

## 📞 문의

프로젝트에 대한 질문이나 제안사항이 있으시면 이슈를 생성해주세요.

---

**주의**: 이 시스템은 테스트 및 연구 목적으로 제작되었습니다. 실제 사용 시에는 안전성을 고려하여 사용해주세요. 