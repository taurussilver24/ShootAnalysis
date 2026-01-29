import os
import cv2
import dxcam
import numpy as np
import csv
import time
import subprocess
from ultralytics import YOLO
from utils import score, detect_down, detect_up, in_hoop_region, clean_hoop_pos, clean_ball_pos


class DataFactoryShotDetector:
    def __init__(self, model_path, session_name, model_name, auto_restart=True):
        # --- FIX 1: Set half=False to prevent ONNX Crash ---
        self.model = YOLO(model_path, task="detect")

        # Camera Setup
        self.camera = dxcam.create(output_color="BGR", output_idx=0)
        self.camera.start(target_fps=60, video_mode=True)
        self.target_width, self.target_height = 1920, 1080

        # Stats Variables
        self.ball_pos = []
        self.hoop_pos = []
        self.makes = 0
        self.attempts = 0
        self.up = self.down = self.peak = False
        self.up_frame = self.down_frame = 0
        self.frame_count = 0
        self.class_names = ['Ring', 'Ball']

        # --- RESTART CONFIGURATION ---
        self.auto_restart = auto_restart
        # This EXE handles the keyboard inputs
        self.restart_exe_path = "restart_routine.exe"
        self.last_shot_time = time.time()
        # If no shots for 5 mins, assume game over & restart
        self.idle_timeout = 300
        self.cooldown_active = False

        # Session & Logging
        self.session_name = session_name
        self.setup_logging(session_name, model_name)

        # UI
        self.window_name = f"DATA FACTORY | {model_name}"
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        self.paused = False

        self.run()

    def setup_logging(self, session_name, model_name):
        results_dir = os.path.join('Results', session_name)
        os.makedirs(results_dir, exist_ok=True)
        csv_path = os.path.join(results_dir, f'{model_name}_shot_results.csv')
        self.csv_file = open(csv_path, mode='w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(["Session", "Result", "Ball", "Hoop", "Score", "Time"])

    def trigger_restart(self):
        """Python calls the AHK Worker to do the heavy lifting."""
        if not self.auto_restart or self.cooldown_active:
            return

        print("\n[RESTART] Triggering Worker (AHK)...")
        self.cooldown_active = True

        # Check if Worker exists
        if not os.path.exists(self.restart_exe_path):
            print(f"❌ ERROR: Missing '{self.restart_exe_path}'")
            print("   -> Did you compile the AHK script yet?")
            self.cooldown_active = False
            return

        try:
            # RUN THE WORKER
            subprocess.run([self.restart_exe_path], check=True)

            print("[RESTART] Worker finished. Waiting 30s for loading...")
            time.sleep(30)  # Wait for game to load

            # Reset Stats for new game
            self.makes = 0
            self.attempts = 0
            self.ball_pos = []
            self.hoop_pos = []
            self.last_shot_time = time.time()
            self.cooldown_active = False
            print("[RESTART] Resuming Detection.")

        except Exception as e:
            print(f"❌ Worker Failed: {e}")
            self.cooldown_active = False

    def check_idle_timeout(self):
        if not self.auto_restart: return

        # If no shots for X seconds, assume game ended
        if (time.time() - self.last_shot_time) > self.idle_timeout:
            print(f"[IDLE] No shots for {self.idle_timeout}s. Restarting...")
            self.trigger_restart()

    def run(self):
        print("=== NBA 2K25 DATA FACTORY STARTED ===")
        print("   -> Press 'R' to test the restart sequence.")

        while True:
            if not self.paused:
                frame = self.camera.get_latest_frame()
                if frame is None: continue

                # Resize Logic
                if frame.shape[:2] != (self.target_height, self.target_width):
                    frame = cv2.resize(frame, (self.target_width, self.target_height))

                self.process_frame(frame)

                cv2.imshow(self.window_name, frame)
                self.check_idle_timeout()

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord(' '):
                self.paused = not self.paused
            elif key == ord('r'):
                self.trigger_restart()  # Manual Test

        self.camera.stop()
        cv2.destroyAllWindows()

    def process_frame(self, frame):
        # --- FIX 1: half=False is mandatory for CPU ---
        results = self.model(
            frame,
            stream=True,
            verbose=False,
            imgsz=(736, 1280),
            half=False,
            device='0',
            conf=0.60
        )

        # Basic Visualization Loop
        for r in results:
            boxes = r.boxes.cpu().numpy()
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls = int(box.cls[0])
                conf = box.conf[0]

                if conf > 0.60:
                    label = self.class_names[cls]
                    color = (0, 0, 255) if label == "Ball" else (255, 0, 0)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                    if label == "Ball":
                        self.last_shot_time = time.time()  # Reset idle timer

        self.frame_count += 1


if __name__ == "__main__":
    detector = DataFactoryShotDetector(
        model_path="../models/Rishit.onnx",
        session_name="Factory_Session_1",
        model_name="Rishit_V1",
        auto_restart=True
    )