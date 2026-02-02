import os
import cv2
import csv
import numpy as np
import time
from collections import deque
from ultralytics import YOLO
from utils import score, detect_down, detect_up, in_hoop_region, clean_hoop_pos, clean_ball_pos

# ==============================================================================
# ⚙️ TURBO SETTINGS
# ==============================================================================
VIDEO_PATH = "../HoopVids/2025/2.mp4"
MODEL_PATH = "../models/Rishit.onnx"
SESSION_NAME = "Turbo_Session"
RENDER_EVERY_N_FRAMES = 3  # Calculate every frame, but only DRAW every 3rd (Speed hack)


# ==============================================================================

class TurboVideoAnalysis:
    def __init__(self, video_path, model_path, session_name):
        # Initialize with specific GPU device if possible
        self.model = YOLO(model_path, task="detect")

        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            print(f"❌ Error: Could not open {video_path}")
            return

        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Logic Vars
        self.ball_pos = []
        self.hoop_pos = []
        self.makes = 0
        self.attempts = 0
        self.up = self.down = self.peak = False
        self.trajectory = deque(maxlen=30)

        # UI Setup
        self.window_name = "NBA 2K25 TURBO ANALYSIS"
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 1280, 720)

        self.frame_index = 0
        self.paused = False
        self.run()

    def run(self):
        print(f"=== STARTING TURBO ANALYSIS ({self.total_frames} Frames) ===")
        start_time = time.time()

        while self.cap.isOpened():
            if not self.paused:
                ret, frame = self.cap.read()
                if not ret: break

                # 1. PROCESS EVERY FRAME (Physics need high frequency)
                self.process_logic(frame)

                # 2. RENDER ONLY OCCASIONALLY (Huge speedup)
                if self.frame_index % RENDER_EVERY_N_FRAMES == 0:
                    self.draw_debug(frame)
                    cv2.imshow(self.window_name, frame)

                    # Check for quit only when rendering
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                    elif key == ord(' '):
                        self.paused = not self.paused

                self.frame_index += 1
            else:
                # When paused, loop the waitKey so interface doesn't freeze
                if cv2.waitKey(100) & 0xFF == ord(' '):
                    self.paused = not self.paused

        end_time = time.time()
        total_time = end_time - start_time
        print(f"\n✅ Analysis Complete in {total_time:.2f}s")
        print(f"   -> Average Speed: {self.total_frames / total_time:.2f} FPS")

        self.cap.release()
        cv2.destroyAllWindows()

    def process_logic(self, frame):
        # RECT=TRUE is the magic sauce here.
        # It fits inference to 1280x736 instead of 1280x1280
        results = self.model(
            frame,
            stream=True,
            verbose=False,
            imgsz=1280,
            rect=True,  # <--- 40% FASTER
            half=False,
            conf=0.50
        )

        for r in results:
            boxes = r.boxes.cpu().numpy()
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = box.conf[0]
                cls = int(box.cls[0])
                label = "Ball" if cls == 1 else "Ring"  # Assuming 0=Ring, 1=Ball based on your class_names

                center = ((x1 + x2) // 2, (y1 + y2) // 2)
                w, h = x2 - x1, y2 - y1

                if label == "Ball":
                    self.ball_pos.append((center, self.frame_index, w, h, conf))
                    self.trajectory.append(center)
                elif label == "Ring":
                    self.hoop_pos.append((center, self.frame_index, w, h, conf))

        # Physics Pipeline
        self.ball_pos = clean_ball_pos(self.ball_pos, self.frame_index)
        if self.hoop_pos: self.hoop_pos = clean_hoop_pos(self.hoop_pos)

        if self.hoop_pos and self.ball_pos:
            if not self.up: self.up = detect_up(self.ball_pos, self.hoop_pos)
            if self.up and not self.down: self.down = detect_down(self.ball_pos, self.hoop_pos)

            # Simple Peak
            if self.up and not self.peak and len(self.ball_pos) > 2:
                if self.ball_pos[-1][0][1] > self.ball_pos[-2][0][1]: self.peak = True

            if self.up and self.down and self.peak:
                self.attempts += 1
                result = "SCORE" if score(self.ball_pos, self.hoop_pos) else "MISS"
                if result == "SCORE": self.makes += 1
                print(f"[{self.frame_index}] Shot: {result} ({self.makes}/{self.attempts})")

                # Reset
                self.up = self.down = self.peak = False
                self.trajectory.clear()

    def draw_debug(self, frame):
        # Only called when we actually show the frame

        # Draw Trajectory
        if len(self.trajectory) > 1:
            pts = np.array(list(self.trajectory), np.int32)
            cv2.polylines(frame, [pts], False, (0, 255, 0), 2)

        # Draw Text
        state_text = f"UP:{self.up} P:{self.peak} DN:{self.down}"
        cv2.putText(frame, state_text, (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(frame, f"SCORE: {self.makes}/{self.attempts}", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)


if __name__ == "__main__":
    TurboVideoAnalysis(VIDEO_PATH, MODEL_PATH, SESSION_NAME)