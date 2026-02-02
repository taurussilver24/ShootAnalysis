import os
import cv2
import csv
import time
import numpy as np
from ultralytics import YOLO
from utils import score, detect_down, detect_up, in_hoop_region, clean_hoop_pos, clean_ball_pos

# ==============================================================================
# ⚙️ MINIMAL SETTINGS
# ==============================================================================
VIDEO_PATH = "../HoopVids/Done_Requested/SpaceJam.mp4"
MODEL_PATH = "../models/Rishit.onnx"
SESSION_NAME = "Minimal_Session"

# OPTIMIZATION
# 1 = Smooth playback (slower), 3 = Turbo (choppy but fast)
# Note: The script automatically switches to 1 during a flash so you see the result!
NORMAL_SKIP_RATE = 2
HISTORY_LEN = 60  # Keep 1 sec of physics history


# ==============================================================================

class MinimalShotDetector:
    def __init__(self, video_path, model_path, session_name):
        self.model = YOLO(model_path, task="detect")
        self.cap = cv2.VideoCapture(video_path)

        if not self.cap.isOpened():
            print(f"❌ Error: Could not open {video_path}")
            return

        # Setup Stats
        self.ball_pos = []
        self.hoop_pos = []
        self.makes = 0
        self.attempts = 0
        self.up = self.down = self.peak = False

        # Flash Logic
        self.fade_frames = 15
        self.fade_counter = 0
        self.overlay_color = (0, 0, 0)
        self.overlay_buffer = None  # Pre-allocate memory

        # Window
        self.window_name = "MINIMAL DETECTOR"
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 1280, 720)

        self.frame_idx = 0
        self.paused = False
        self.run()

    def run(self):
        print("=== MINIMAL MODE STARTED ===")
        print("   -> Only Boxes & Flashes")

        while self.cap.isOpened():
            if not self.paused:
                ret, frame = self.cap.read()
                if not ret: break

                # 1. LOGIC (Always Run)
                self.process_logic(frame)

                # 2. RENDER DECISION
                # We draw if:
                # a) It's a "Render Frame" (based on skip rate)
                # b) A Flash is happening (We NEVER skip flash frames)
                should_render = (self.frame_idx % NORMAL_SKIP_RATE == 0) or (self.fade_counter > 0)

                if should_render:
                    self.draw_visuals(frame)
                    cv2.imshow(self.window_name, frame)

                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                    elif key == ord(' '):
                        self.paused = not self.paused

                self.frame_idx += 1
            else:
                if cv2.waitKey(100) & 0xFF == ord(' '): self.paused = not self.paused

        self.cap.release()
        cv2.destroyAllWindows()

    def process_logic(self, frame):
        # Truncate History (Speed)
        if len(self.ball_pos) > HISTORY_LEN: self.ball_pos = self.ball_pos[-HISTORY_LEN:]
        if len(self.hoop_pos) > HISTORY_LEN: self.hoop_pos = self.hoop_pos[-HISTORY_LEN:]

        # Inference
        results = self.model(
            frame,
            stream=True,
            verbose=False,
            imgsz=1280,
            rect=True,  # Speed Optimization
            half=False,
            conf=0.50
        )

        for r in results:
            boxes = r.boxes.cpu().numpy()
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = box.conf[0]
                cls = int(box.cls[0])

                center = ((x1 + x2) // 2, (y1 + y2) // 2)
                w, h = x2 - x1, y2 - y1

                # Just store data, don't draw yet
                if cls == 1:  # Ball
                    self.ball_pos.append((center, self.frame_idx, w, h, conf))
                elif cls == 0:  # Ring
                    self.hoop_pos.append((center, self.frame_idx, w, h, conf))

                    # Store box coordinates for drawing later
                    # We sneak this into the list so draw_visuals can find it
                    # (Quick hack to avoid a separate list)
                    self.hoop_pos[-1] = (center, self.frame_idx, w, h, conf, (x1, y1, x2, y2))

                if cls == 1:  # Update ball box too
                    self.ball_pos[-1] = (center, self.frame_idx, w, h, conf, (x1, y1, x2, y2))

        # Physics Pipeline
        self.ball_pos = clean_ball_pos(self.ball_pos, self.frame_idx)
        if self.hoop_pos: self.hoop_pos = clean_hoop_pos(self.hoop_pos)

        if self.hoop_pos and self.ball_pos:
            if not self.up: self.up = detect_up(self.ball_pos, self.hoop_pos)
            if self.up and not self.down: self.down = detect_down(self.ball_pos, self.hoop_pos)

            if self.up and not self.peak and len(self.ball_pos) > 2:
                if self.ball_pos[-1][0][1] > self.ball_pos[-2][0][1]: self.peak = True

            if self.up and self.down and self.peak:
                self.attempts += 1
                if score(self.ball_pos, self.hoop_pos):
                    self.makes += 1
                    self.overlay_color = (0, 255, 0)  # GREEN
                    print(f"[{self.frame_idx}] SCORE! ({self.makes}/{self.attempts})")
                else:
                    self.overlay_color = (0, 0, 255)  # RED
                    print(f"[{self.frame_idx}] MISS. ({self.makes}/{self.attempts})")

                # Trigger Flash
                self.fade_counter = self.fade_frames

                # Reset
                self.up = self.down = self.peak = False

    def draw_visuals(self, frame):
        # 1. Draw Boxes (Only latest frame)
        # We look at the last item in our lists to get current coords
        if self.ball_pos:
            data = self.ball_pos[-1]
            if len(data) >= 6 and data[1] == self.frame_idx:  # Check if updated this frame
                x1, y1, x2, y2 = data[5]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Red Box

        if self.hoop_pos:
            data = self.hoop_pos[-1]
            if len(data) >= 6 and data[1] == self.frame_idx:
                x1, y1, x2, y2 = data[5]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Blue Box

        # 2. Draw Flash
        if self.fade_counter > 0:
            # Create overlay only once if needed (Optimization)
            if self.overlay_buffer is None or self.overlay_buffer.shape != frame.shape:
                self.overlay_buffer = np.zeros_like(frame)

            # Fill buffer with current color
            self.overlay_buffer[:] = self.overlay_color

            # Calculate alpha
            alpha = 0.3 * (self.fade_counter / self.fade_frames)

            # Blend
            cv2.addWeighted(frame, 1 - alpha, self.overlay_buffer, alpha, 0, dst=frame)
            self.fade_counter -= 1


if __name__ == "__main__":
    MinimalShotDetector(VIDEO_PATH, MODEL_PATH, SESSION_NAME)