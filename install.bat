@echo off
echo ========================================
echo    눈동자 추적 스크롤 시스템 설치
echo ========================================
echo.

echo Python이 설치되어 있는지 확인 중...
python --version >nul 2>&1
if errorlevel 1 (
    echo Python이 설치되어 있지 않습니다.
    echo https://www.python.org/downloads/ 에서 Python을 다운로드하여 설치해주세요.
    pause
    exit /b 1
)

echo Python 설치 확인 완료!
echo.

echo 필요한 패키지를 설치하는 중...
pip install -r requirements.txt

if errorlevel 1 (
    echo 패키지 설치 중 오류가 발생했습니다.
    echo 인터넷 연결을 확인하고 다시 시도해주세요.
    pause
    exit /b 1
)

echo.
echo ========================================
echo    설치가 완료되었습니다!
echo ========================================
echo.
echo 사용 방법:
echo 1. python advanced_eye_tracker.py
echo 2. GUI에서 설정을 조정하세요
echo 3. "시작" 버튼을 클릭하세요
echo 4. 웹캠 앞에서 위/아래를 바라보세요
echo.
echo 테스트 페이지: test_page.html
echo.
pause 