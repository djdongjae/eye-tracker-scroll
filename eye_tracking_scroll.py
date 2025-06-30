import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import time
from collections import deque

class EyeTrackingScroll:
    def __init__(self):
        # MediaPipe 초기화 (버전 호환성 처리)
        try:
            self.face_mesh = mp.solutions.FaceMesh(
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
        except AttributeError:
            # 구버전 MediaPipe 호환성
            self.face_mesh = mp.solutions.face_mesh.FaceMesh(
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
        
        # 눈동자 랜드마크 인덱스
        self.LEFT_EYE = [362, 385, 387, 263, 373, 380]
        self.RIGHT_EYE = [33, 160, 158, 133, 153, 144]
        
        # 웹캠 초기화
        self.cap = cv2.VideoCapture(0)
        
        # 스크롤 설정
        self.scroll_threshold = 0.1  # 스크롤 감지 임계값
        self.scroll_cooldown = 0.5   # 스크롤 간격 (초)
        self.last_scroll_time = 0
        
        # 눈동자 위치 히스토리 (부드러운 추적을 위해)
        self.eye_history = deque(maxlen=5)
        
        # 화면 크기 가져오기
        self.screen_width, self.screen_height = pyautogui.size()
        
        print("눈동자 추적 스크롤 시스템이 시작되었습니다.")
        print("위쪽을 보면 위로 스크롤, 아래쪽을 보면 아래로 스크롤합니다.")
        print("'q'를 누르면 종료됩니다.")
    
    def calculate_eye_aspect_ratio(self, eye_points):
        """눈의 종횡비(EAR) 계산"""
        A = np.linalg.norm(eye_points[1] - eye_points[5])
        B = np.linalg.norm(eye_points[2] - eye_points[4])
        C = np.linalg.norm(eye_points[0] - eye_points[3])
        
        ear = (A + B) / (2.0 * C)
        return ear
    
    def get_eye_center(self, eye_points):
        """눈의 중심점 계산"""
        center_x = np.mean(eye_points[:, 0])
        center_y = np.mean(eye_points[:, 1])
        return center_x, center_y
    
    def detect_gaze_direction(self, left_eye_center, right_eye_center):
        """시선 방향 감지"""
        # 양쪽 눈의 평균 중심점
        avg_eye_x = (left_eye_center[0] + right_eye_center[0]) / 2
        avg_eye_y = (left_eye_center[1] + right_eye_center[1]) / 2
        
        # 정규화된 좌표 (0~1 범위)
        normalized_x = avg_eye_x / self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        normalized_y = avg_eye_y / self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        
        # 히스토리에 추가
        self.eye_history.append((normalized_x, normalized_y))
        
        # 평균 위치 계산 (부드러운 추적)
        if len(self.eye_history) >= 3:
            avg_x = np.mean([pos[0] for pos in self.eye_history])
            avg_y = np.mean([pos[1] for pos in self.eye_history])
        else:
            avg_x, avg_y = normalized_x, normalized_y
        
        return avg_x, avg_y
    
    def scroll_based_on_gaze(self, gaze_x, gaze_y):
        """시선 방향에 따른 스크롤 제어"""
        current_time = time.time()
        
        # 스크롤 쿨다운 체크
        if current_time - self.last_scroll_time < self.scroll_cooldown:
            return
        
        # 위쪽을 보고 있는 경우 (y < 0.4)
        if gaze_y < 0.4:
            pyautogui.scroll(3)  # 위로 스크롤
            print("↑ 위로 스크롤")
            self.last_scroll_time = current_time
        
        # 아래쪽을 보고 있는 경우 (y > 0.6)
        elif gaze_y > 0.6:
            pyautogui.scroll(-3)  # 아래로 스크롤
            print("↓ 아래로 스크롤")
            self.last_scroll_time = current_time
    
    def run(self):
        """메인 실행 루프"""
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("웹캠에서 프레임을 읽을 수 없습니다.")
                    break
                
                # BGR을 RGB로 변환
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.face_mesh.process(rgb_frame)
                
                if results.multi_face_landmarks:
                    face_landmarks = results.multi_face_landmarks[0]
                    
                    # 눈동자 랜드마크 추출
                    left_eye_points = []
                    right_eye_points = []
                    
                    for idx in self.LEFT_EYE:
                        landmark = face_landmarks.landmark[idx]
                        x = int(landmark.x * frame.shape[1])
                        y = int(landmark.y * frame.shape[0])
                        left_eye_points.append([x, y])
                    
                    for idx in self.RIGHT_EYE:
                        landmark = face_landmarks.landmark[idx]
                        x = int(landmark.x * frame.shape[1])
                        y = int(landmark.y * frame.shape[0])
                        right_eye_points.append([x, y])
                    
                    left_eye_points = np.array(left_eye_points)
                    right_eye_points = np.array(right_eye_points)
                    
                    # 눈 중심점 계산
                    left_eye_center = self.get_eye_center(left_eye_points)
                    right_eye_center = self.get_eye_center(right_eye_points)
                    
                    # 시선 방향 감지
                    gaze_x, gaze_y = self.detect_gaze_direction(left_eye_center, right_eye_center)
                    
                    # 스크롤 제어
                    self.scroll_based_on_gaze(gaze_x, gaze_y)
                    
                    # 눈동자 영역 시각화
                    cv2.polylines(frame, [left_eye_points], True, (0, 255, 0), 2)
                    cv2.polylines(frame, [right_eye_points], True, (0, 255, 0), 2)
                    
                    # 시선 방향 표시
                    cv2.putText(frame, f"Gaze: ({gaze_x:.2f}, {gaze_y:.2f})", 
                              (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    # 스크롤 방향 표시
                    if gaze_y < 0.4:
                        cv2.putText(frame, "SCROLL UP", (10, 60), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    elif gaze_y > 0.6:
                        cv2.putText(frame, "SCROLL DOWN", (10, 60), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                # 프레임 표시
                cv2.imshow('Eye Tracking Scroll', frame)
                
                # 'q' 키로 종료
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
        except KeyboardInterrupt:
            print("\n프로그램이 중단되었습니다.")
        
        finally:
            self.cleanup()
    
    def cleanup(self):
        """리소스 정리"""
        self.cap.release()
        cv2.destroyAllWindows()
        print("눈동자 추적 시스템이 종료되었습니다.")

if __name__ == "__main__":
    # pyautogui 안전장치 비활성화 (테스트용)
    pyautogui.FAILSAFE = False
    
    # 시스템 시작
    eye_tracker = EyeTrackingScroll()
    eye_tracker.run() 