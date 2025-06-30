import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import time
import tkinter as tk
from tkinter import ttk, messagebox
from collections import deque
import threading
import json
import os

class AdvancedEyeTrackingScroll:
    def __init__(self):
        # 설정 파일 경로
        self.config_file = "eye_tracker_config.json"
        
        # 기본 설정값
        self.default_config = {
            "scroll_sensitivity": 0.1,
            "scroll_cooldown": 0.5,
            "up_threshold": 0.4,
            "down_threshold": 0.6,
            "smooth_factor": 0.7,
            "scroll_amount": 3,
            "enable_horizontal_scroll": False,
            "left_threshold": 0.3,
            "right_threshold": 0.7
        }
        
        # 설정 로드
        self.config = self.load_config()
        
        # MediaPipe 초기화
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # 눈동자 랜드마크 인덱스 (더 정확한 추적을 위해 확장)
        self.LEFT_EYE = [362, 385, 387, 263, 373, 380, 381, 382, 381]
        self.RIGHT_EYE = [33, 160, 158, 133, 153, 144, 145, 146, 145]
        
        # 웹캠 초기화
        self.cap = cv2.VideoCapture(0)
        
        # 상태 변수
        self.is_running = False
        self.last_scroll_time = 0
        self.eye_history = deque(maxlen=10)
        self.screen_width, self.screen_height = pyautogui.size()
        
        # GUI 초기화
        self.setup_gui()
        
        print("고급 눈동자 추적 스크롤 시스템이 초기화되었습니다.")
    
    def load_config(self):
        """설정 파일 로드"""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            else:
                return self.default_config.copy()
        except:
            return self.default_config.copy()
    
    def save_config(self):
        """설정 파일 저장"""
        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"설정 저장 실패: {e}")
    
    def setup_gui(self):
        """GUI 설정 패널 생성"""
        self.root = tk.Tk()
        self.root.title("눈동자 추적 스크롤 설정")
        self.root.geometry("400x500")
        self.root.resizable(False, False)
        
        # 메인 프레임
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # 제목
        title_label = ttk.Label(main_frame, text="눈동자 추적 스크롤 설정", 
                               font=("Arial", 14, "bold"))
        title_label.grid(row=0, column=0, columnspan=2, pady=(0, 20))
        
        # 설정 항목들
        row = 1
        
        # 스크롤 감도
        ttk.Label(main_frame, text="스크롤 감도:").grid(row=row, column=0, sticky=tk.W)
        self.sensitivity_var = tk.DoubleVar(value=self.config["scroll_sensitivity"])
        sensitivity_scale = ttk.Scale(main_frame, from_=0.05, to=0.3, 
                                     variable=self.sensitivity_var, orient=tk.HORIZONTAL)
        sensitivity_scale.grid(row=row, column=1, sticky=(tk.W, tk.E), padx=(10, 0))
        row += 1
        
        # 스크롤 쿨다운
        ttk.Label(main_frame, text="스크롤 간격 (초):").grid(row=row, column=0, sticky=tk.W)
        self.cooldown_var = tk.DoubleVar(value=self.config["scroll_cooldown"])
        cooldown_scale = ttk.Scale(main_frame, from_=0.1, to=2.0, 
                                  variable=self.cooldown_var, orient=tk.HORIZONTAL)
        cooldown_scale.grid(row=row, column=1, sticky=(tk.W, tk.E), padx=(10, 0))
        row += 1
        
        # 위쪽 임계값
        ttk.Label(main_frame, text="위쪽 임계값:").grid(row=row, column=0, sticky=tk.W)
        self.up_threshold_var = tk.DoubleVar(value=self.config["up_threshold"])
        up_scale = ttk.Scale(main_frame, from_=0.2, to=0.6, 
                            variable=self.up_threshold_var, orient=tk.HORIZONTAL)
        up_scale.grid(row=row, column=1, sticky=(tk.W, tk.E), padx=(10, 0))
        row += 1
        
        # 아래쪽 임계값
        ttk.Label(main_frame, text="아래쪽 임계값:").grid(row=row, column=0, sticky=tk.W)
        self.down_threshold_var = tk.DoubleVar(value=self.config["down_threshold"])
        down_scale = ttk.Scale(main_frame, from_=0.4, to=0.8, 
                              variable=self.down_threshold_var, orient=tk.HORIZONTAL)
        down_scale.grid(row=row, column=1, sticky=(tk.W, tk.E), padx=(10, 0))
        row += 1
        
        # 스크롤 양
        ttk.Label(main_frame, text="스크롤 양:").grid(row=row, column=0, sticky=tk.W)
        self.scroll_amount_var = tk.IntVar(value=self.config["scroll_amount"])
        scroll_spinbox = ttk.Spinbox(main_frame, from_=1, to=10, 
                                    variable=self.scroll_amount_var, width=10)
        scroll_spinbox.grid(row=row, column=1, sticky=tk.W, padx=(10, 0))
        row += 1
        
        # 부드러운 추적
        ttk.Label(main_frame, text="부드러운 추적:").grid(row=row, column=0, sticky=tk.W)
        self.smooth_var = tk.DoubleVar(value=self.config["smooth_factor"])
        smooth_scale = ttk.Scale(main_frame, from_=0.1, to=0.9, 
                                variable=self.smooth_var, orient=tk.HORIZONTAL)
        smooth_scale.grid(row=row, column=1, sticky=(tk.W, tk.E), padx=(10, 0))
        row += 1
        
        # 가로 스크롤 활성화
        self.horizontal_var = tk.BooleanVar(value=self.config["enable_horizontal_scroll"])
        horizontal_check = ttk.Checkbutton(main_frame, text="가로 스크롤 활성화", 
                                          variable=self.horizontal_var)
        horizontal_check.grid(row=row, column=0, columnspan=2, sticky=tk.W, pady=(10, 0))
        row += 1
        
        # 상태 표시
        self.status_var = tk.StringVar(value="대기 중")
        status_label = ttk.Label(main_frame, textvariable=self.status_var, 
                                font=("Arial", 10, "bold"))
        status_label.grid(row=row, column=0, columnspan=2, pady=(20, 0))
        row += 1
        
        # 버튼들
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=row, column=0, columnspan=2, pady=(20, 0))
        
        self.start_button = ttk.Button(button_frame, text="시작", command=self.start_tracking)
        self.start_button.pack(side=tk.LEFT, padx=(0, 10))
        
        self.stop_button = ttk.Button(button_frame, text="중지", command=self.stop_tracking, 
                                     state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT, padx=(0, 10))
        
        ttk.Button(button_frame, text="설정 저장", command=self.save_settings).pack(side=tk.LEFT)
        
        # 그리드 가중치 설정
        main_frame.columnconfigure(1, weight=1)
    
    def save_settings(self):
        """설정 저장"""
        self.config.update({
            "scroll_sensitivity": self.sensitivity_var.get(),
            "scroll_cooldown": self.cooldown_var.get(),
            "up_threshold": self.up_threshold_var.get(),
            "down_threshold": self.down_threshold_var.get(),
            "smooth_factor": self.smooth_var.get(),
            "scroll_amount": self.scroll_amount_var.get(),
            "enable_horizontal_scroll": self.horizontal_var.get()
        })
        self.save_config()
        messagebox.showinfo("알림", "설정이 저장되었습니다.")
    
    def start_tracking(self):
        """추적 시작"""
        if not self.is_running:
            self.is_running = True
            self.start_button.config(state=tk.DISABLED)
            self.stop_button.config(state=tk.NORMAL)
            self.status_var.set("추적 중...")
            
            # 별도 스레드에서 추적 실행
            self.tracking_thread = threading.Thread(target=self.run_tracking)
            self.tracking_thread.daemon = True
            self.tracking_thread.start()
    
    def stop_tracking(self):
        """추적 중지"""
        self.is_running = False
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.status_var.set("대기 중")
    
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
        """시선 방향 감지 (개선된 버전)"""
        # 양쪽 눈의 평균 중심점
        avg_eye_x = (left_eye_center[0] + right_eye_center[0]) / 2
        avg_eye_y = (left_eye_center[1] + right_eye_center[1]) / 2
        
        # 정규화된 좌표 (0~1 범위)
        normalized_x = avg_eye_x / self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        normalized_y = avg_eye_y / self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        
        # 히스토리에 추가
        self.eye_history.append((normalized_x, normalized_y))
        
        # 부드러운 추적을 위한 가중 평균 계산
        if len(self.eye_history) >= 3:
            weights = np.exp(np.linspace(-1, 0, len(self.eye_history)))
            weights = weights / np.sum(weights)
            
            x_positions = [pos[0] for pos in self.eye_history]
            y_positions = [pos[1] for pos in self.eye_history]
            
            avg_x = np.average(x_positions, weights=weights)
            avg_y = np.average(y_positions, weights=weights)
        else:
            avg_x, avg_y = normalized_x, normalized_y
        
        return avg_x, avg_y
    
    def scroll_based_on_gaze(self, gaze_x, gaze_y):
        """시선 방향에 따른 스크롤 제어 (개선된 버전)"""
        current_time = time.time()
        
        # 스크롤 쿨다운 체크
        if current_time - self.last_scroll_time < self.config["scroll_cooldown"]:
            return
        
        scroll_amount = self.config["scroll_amount"]
        
        # 세로 스크롤
        if gaze_y < self.config["up_threshold"]:
            pyautogui.scroll(scroll_amount)
            print(f"↑ 위로 스크롤 (y={gaze_y:.2f})")
            self.last_scroll_time = current_time
        
        elif gaze_y > self.config["down_threshold"]:
            pyautogui.scroll(-scroll_amount)
            print(f"↓ 아래로 스크롤 (y={gaze_y:.2f})")
            self.last_scroll_time = current_time
        
        # 가로 스크롤 (옵션)
        if self.config["enable_horizontal_scroll"]:
            if gaze_x < self.config["left_threshold"]:
                pyautogui.hscroll(-scroll_amount)
                print(f"← 왼쪽으로 스크롤 (x={gaze_x:.2f})")
                self.last_scroll_time = current_time
            
            elif gaze_x > self.config["right_threshold"]:
                pyautogui.hscroll(scroll_amount)
                print(f"→ 오른쪽으로 스크롤 (x={gaze_x:.2f})")
                self.last_scroll_time = current_time
    
    def run_tracking(self):
        """추적 실행 루프"""
        try:
            while self.is_running:
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
                    if gaze_y < self.config["up_threshold"]:
                        cv2.putText(frame, "SCROLL UP", (10, 60), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    elif gaze_y > self.config["down_threshold"]:
                        cv2.putText(frame, "SCROLL DOWN", (10, 60), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    
                    # 가로 스크롤 표시
                    if self.config["enable_horizontal_scroll"]:
                        if gaze_x < self.config["left_threshold"]:
                            cv2.putText(frame, "SCROLL LEFT", (10, 90), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                        elif gaze_x > self.config["right_threshold"]:
                            cv2.putText(frame, "SCROLL RIGHT", (10, 90), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                
                # 프레임 표시
                cv2.imshow('Advanced Eye Tracking Scroll', frame)
                
                # 'q' 키로 종료
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
        except Exception as e:
            print(f"추적 중 오류 발생: {e}")
        
        finally:
            self.stop_tracking()
    
    def run(self):
        """GUI 실행"""
        try:
            self.root.mainloop()
        except KeyboardInterrupt:
            print("\n프로그램이 중단되었습니다.")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """리소스 정리"""
        self.is_running = False
        if hasattr(self, 'cap'):
            self.cap.release()
        cv2.destroyAllWindows()
        print("눈동자 추적 시스템이 종료되었습니다.")

if __name__ == "__main__":
    # pyautogui 안전장치 비활성화 (테스트용)
    pyautogui.FAILSAFE = False
    
    # 시스템 시작
    eye_tracker = AdvancedEyeTrackingScroll()
    eye_tracker.run() 