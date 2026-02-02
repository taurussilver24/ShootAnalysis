import os
import cv2
import dxcam
import time
import subprocess
import csv
import numpy as np
import win32gui
import ctypes
from ultralytics import YOLO
from utils import score, detect_down, detect_up, in_hoop_region, clean_hoop_pos, clean_ball_pos

# ==============================================================================
# ⚙️ FACTORY SETTINGS
# ==============================================================================
MONITOR_INDEX = 0  # Primary Monitor
MATCH_DURATION_MINUTES = 25  # Restart trigger
RESTART_EXE = "restart_routine.exe"
GAME_WINDOW_TITLE = "NBA 2K25"  # Must match the window title exactly!
# ==============================================================================

# 🔧 FORCE DPI AWARENESS (Fixes the "Zoomed in" / "Weird Crop" bug)
try:
    ctypes.windll.shcore.SetProcessDpiAwareness(1)
except Exception:
    ctypes.windll.user32.SetProcessDPIAware()


class DataFactory:
    def __init__(self, model_path, session_name, model_name):
        self.model = YOLO(model_path, task="detect")
        self.class_names = ['Ring', 'Ball']

        # 1. Camera Init
        try:
            self.camera = dxcam.create(output_idx=MONITOR_INDEX, output_color="BGR")
            self.camera.start(target_fps=60, video_mode=True)
        except Exception as e:
            print(f"❌ Camera Error: {e}")
            return

        # 2. Stats & Logic
        self.ball_pos = []
        self.hoop_pos = []
        self.makes = 0
        self.attempts = 0
        self.up = self.down = self.peak = False
        self.up_frame = self.down_frame = 0

        # 3. Timers
        self.start_time = time.time()
        self.last_shot_time = time.time()
        self.session_count = 1
        self.frame_count = 0
        self.is_restarting = False

        # 4. Logging & UI
        self.session_name = session_name
        self.model_name = model_name
        self.setup_logging()

        # UI Setup
        self.window_name = "NBA 2K25 TRACKER"
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 1280, 720)  # Big view for you

        self.fade_frames = 20
        self.fade_counter = 0
        self.overlay_color = (0, 0, 0)

        self.paused = False
        self.run()

    def get_game_rect(self):
        """Finds the NBA 2K25 window coordinates dynamically."""
        hwnd = win32gui.FindWindow(None, GAME_WINDOW_TITLE)
        if hwnd:
            # Returns (left, top, right, bottom)
            rect = win32gui.GetWindowRect(hwnd)
            return rect
        return None

    def setup_logging(self):
        results_dir = os.path.join('../Results', self.session_name)
        os.makedirs(results_dir, exist_ok=True)
        csv_path = os.path.join(results_dir, f'{self.model_name}_Match{self.session_count}.csv')
        self.csv_file = open(csv_path, mode='w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(["Shot", "Result", "Ball", "Hoop", "Score", "Time"])

    def check_triggers(self):
        if self.is_restarting: return
        elapsed_min = (time.time() - self.start_time) / 60
        if elapsed_min >= MATCH_DURATION_MINUTES:
            print(f"\n[TRIGGER] Timer hit {MATCH_DURATION_MINUTES}m. Restarting...")
            self.trigger_restart()

    def trigger_restart(self):
        self.is_restarting = True
        self.csv_file.close()
        print(f"🔄 RESTARTING (Session #{self.session_count})")

        if os.path.exists(RESTART_EXE):
            subprocess.run([RESTART_EXE], check=True)
            print("   -> Sequence sent. Waiting 45s for load...")
            time.sleep(45)
            self.reset_match()
        else:
            print(f"❌ Missing {RESTART_EXE}!")
            self.is_restarting = False

    def reset_match(self):
        self.session_count += 1
        self.makes = 0
        self.attempts = 0
        self.ball_pos = []
        self.hoop_pos = []
        self.start_time = time.time()
        self.is_restarting = False
        self.setup_logging()
        print("✅ RESUMING.")

    def run(self):
        print(f"=== TRACKING WINDOW: '{GAME_WINDOW_TITLE}' ===")
        print("   -> Move the game anywhere! The bot will follow.")
        print("   -> Press 'R' to test restart.")

        while True:
            if not self.is_restarting and not self.paused:
                # 1. Grab Frame (Full Monitor)
                full_frame = self.camera.get_latest_frame()
                if full_frame is None: continue

                # 2. Find Window Position
                rect = self.get_game_rect()

                if rect:
                    x1, y1, x2, y2 = rect
                    # Clamp coordinates to monitor bounds (prevent crashes if window drags off-screen)
                    h, w, _ = full_frame.shape
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w, x2), min(h, y2)

                    # Only process if window is valid size
                    if x2 > x1 and y2 > y1:
                        game_frame = full_frame[y1:y2, x1:x2]
                        self.process_frame(game_frame)

                        # Display
                        elapsed = int(time.time() - self.start_time)
                        cv2.putText(game_frame, f"{elapsed // 60}:{elapsed % 60:02d}", (20, 50),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        cv2.imshow(self.window_name, game_frame)
                    else:
                        print("Window minimized or off-screen...", end='\r')
                else:
                    print(f"Waiting for '{GAME_WINDOW_TITLE}' window...", end='\r')

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord(' '):
                self.paused = not self.paused
            elif key == ord('r'):
                self.trigger_restart()

        self.camera.stop()
        cv2.destroyAllWindows()

    def process_frame(self, frame):
        # 1. INFERENCE TWEAKS
        # imgsz=1280: Keeps small objects (ball/hoop) visible
        # conf=0.50: Catches faster/blurrier balls
        results = self.model(
            frame,
            stream=True,
            verbose=False,
            imgsz=1280,  # <--- CRITICAL CHANGE (Was 640)
            half=False,
            conf=0.50  # <--- LOWERED THRESHOLD (Was 0.75)
        )

        detected_anything = False

        for r in results:
            boxes = r.boxes.cpu().numpy()
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = box.conf[0]
                cls = int(box.cls[0])
                label = self.class_names[cls]

                detected_anything = True

                # DRAW EVERYTHING (Debug Visuals)
                # Ball = Red, Hoop = Blue
                color = (0, 0, 255) if label == "Ball" else (255, 0, 0)

                # Draw Box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                # Draw Label (So you know confidence)
                cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                # Data Collection Logic
                center = ((x1 + x2) // 2, (y1 + y2) // 2)
                if label == "Ball":
                    self.ball_pos.append((center, self.frame_count, 0, 0, conf))
                elif label == "Ring":
                    self.hoop_pos.append((center, self.frame_count, 0, 0, conf))

        # Console Debug (Spammy but useful for 5 seconds)
        # if detected_anything:
        #    print(f"Frame {self.frame_count}: Objects Detected!")

        self.clean_motion(frame)
        self.shot_detection()
        self.display_score(frame)
        self.frame_count += 1

    # Utils
    def clean_motion(self, frame):
        self.ball_pos = clean_ball_pos(self.ball_pos, self.frame_count)
        if self.hoop_pos: self.hoop_pos = clean_hoop_pos(self.hoop_pos)

    def detect_peak(self, ball_pos):
        if len(ball_pos) < 3: return False
        return ball_pos[-2][0][1] > ball_pos[-3][0][1] and ball_pos[-2][0][1] > ball_pos[-1][0][1]

    def shot_detection(self):
        if self.hoop_pos and self.ball_pos:
            if not self.up:
                self.up = detect_up(self.ball_pos, self.hoop_pos)
                if self.up: self.up_frame = self.ball_pos[-1][1]

            if self.up and not self.down:
                self.down = detect_down(self.ball_pos, self.hoop_pos)
                if self.down: self.down_frame = self.ball_pos[-1][1]

            if self.up and not self.peak:
                self.peak = self.detect_peak(self.ball_pos)
                if self.peak: self.peak_frame = self.ball_pos[-1][1]

            if self.up and self.down and self.up_frame < self.down_frame:
                self.attempts += 1
                self.up = self.down = self.peak = False

                if score(self.ball_pos, self.hoop_pos):
                    self.makes += 1
                    self.overlay_color = (0, 255, 0)
                    res_str = "Successful"
                else:
                    self.overlay_color = (0, 0, 255)
                    res_str = "Failed"

                print(f"Shot: {self.makes}/{self.attempts} ({res_str})")

                self.fade_counter = self.fade_frames
                try:
                    self.csv_writer.writerow([
                        self.attempts, res_str,
                        self.ball_pos[-1][0], self.hoop_pos[-1][0],
                        f"{self.makes}/{self.attempts}", self.frame_count / 60
                    ])
                except ValueError:
                    pass

    def display_score(self, frame):
        text = f"{self.makes} / {self.attempts}"
        cv2.putText(frame, text, (50, 125), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 6)
        cv2.putText(frame, text, (50, 125), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 0), 3)

        if self.fade_counter > 0:
            alpha = 0.2 * (self.fade_counter / self.fade_frames)
            overlay = np.full_like(frame, self.overlay_color)
            cv2.addWeighted(frame, 1 - alpha, overlay, alpha, 0, dst=frame)
            self.fade_counter -= 1


if __name__ == "__main__":
    DataFactory("../models/Rishit.onnx", "Factory_Output", "Rishit_V1")