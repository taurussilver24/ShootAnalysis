import os
import math
import csv
import numpy as np
import cv2
import mss
import psutil
import win32gui
import win32process
from ultralytics import YOLO
import cvzone
from utils import score, detect_down, detect_up, in_hoop_region, clean_hoop_pos, clean_ball_pos

class ShotDetector:
    def __init__(self, model_path, process_name, model_name):
        # Load YOLO model
        self.model = YOLO(model_path)
        self.class_names = ['Ring', 'Ball']

        # Try to get the window rect for the process name, retry a few times
        self.monitor = self.get_window_rect_by_process_name(process_name)
        if self.monitor is None:
            raise RuntimeError(f"Could not find visible window for process '{process_name}'")

        self.fps = 30  # Assumed FPS for timing calculations
        self.total_frames = float('inf')  # No fixed frames, live capture

        self.ball_pos = []
        self.hoop_pos = []

        self.frame_count = 0
        self.frame = None

        self.makes = 0
        self.attempts = 0

        # Shot detection states
        self.up = False
        self.down = False
        self.peak = False
        self.up_frame = 0
        self.down_frame = 0

        # Overlay fade effect parameters
        self.fade_frames = 20
        self.fade_counter = 0
        self.overlay_color = (0, 0, 0)

        # Prepare results directory and CSV file for logging
        results_dir = os.path.join('Results', model_name)
        os.makedirs(results_dir, exist_ok=True)
        self.csv_file = open(os.path.join(results_dir, f'{model_name}_shot_results.csv'), mode='w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(["Shot Taken", "Result", "Ball Coordinates",
                                  "Hoop Coordinates", "Current Score", "Video Timing (seconds)"])

        # Window name for OpenCV display
        self.window_name = f"MODEL: {os.path.basename(model_path)} | PROCESS: {process_name}"
        cv2.namedWindow(self.window_name)
        cv2.createTrackbar('Time (s)', self.window_name, 0, 100, self.on_time_slider_change)  # dummy slider for UI

        self.paused = False

        print("Press 'q' to quit, SPACE to pause/resume.")
        self.run()

    def get_window_rect_by_process_name(self, process_name, max_attempts=5):
        """
        Finds the rectangle (left, top, width, height) of the first visible window
        belonging to the given process name. Retries a few times if not found immediately.
        """
        for attempt in range(max_attempts):
            # Get all PIDs matching process name
            pids = [p.pid for p in psutil.process_iter(['name']) if p.info['name'] and p.info['name'].lower() == process_name.lower()]
            if not pids:
                print(f"[Attempt {attempt + 1}] No process found with name '{process_name}'. Retrying...")
                cv2.waitKey(1000)
                continue

            hwnds = []
            def enum_windows_callback(hwnd, hwnds):
                try:
                    _, pid = win32process.GetWindowThreadProcessId(hwnd)
                    if pid in pids and win32gui.IsWindowVisible(hwnd):
                        hwnds.append(hwnd)
                except Exception:
                    pass
                return True

            win32gui.EnumWindows(enum_windows_callback, hwnds)
            if hwnds:
                hwnd = hwnds[0]
                left, top, right, bottom = win32gui.GetWindowRect(hwnd)
                width, height = right - left, bottom - top
                print(f"Found window for '{process_name}' at {(left, top, width, height)}")
                return {"left": left, "top": top, "width": width, "height": height}

            print(f"[Attempt {attempt + 1}] No visible window found for '{process_name}'. Retrying...")
            cv2.waitKey(1000)

        return None

    def on_time_slider_change(self, pos):
        # Dummy callback for trackbar, no real seeking in screen capture
        pass

    def run(self):
        with mss.mss() as sct:
            while True:
                if not self.paused:
                    try:
                        # Capture the specified window region from screen
                        screenshot = np.array(sct.grab(self.monitor))
                        self.frame = cv2.cvtColor(screenshot, cv2.COLOR_BGRA2BGR)
                    except Exception as e:
                        print(f"Screen capture error: {e}")
                        break

                # Resize frame for consistent processing/display
                self.frame = cv2.resize(self.frame, (1280, 720))

                # Run YOLO model inference on frame
                results = self.model(self.frame, stream=True, verbose=False)

                # Process detections
                for r in results:
                    boxes = r.boxes
                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0]
                        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                        w, h = x2 - x1, y2 - y1
                        conf = math.ceil(box.conf[0] * 100) / 100
                        if conf < 0.55:
                            continue

                        cls = int(box.cls[0])
                        current_class = self.class_names[cls]
                        center = (x1 + w // 2, y1 + h // 2)
                        color = (0, 0, 255) if current_class == "Ball" else (255, 0, 0)

                        # Draw detection box and label
                        cv2.rectangle(self.frame, (x1, y1), (x2, y2), color, 1)
                        label = f"{current_class} {conf:.2f}"
                        (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                        cv2.rectangle(self.frame, (x1, y1 - text_h - 10), (x1 + text_w, y1), (255, 255, 255), cv2.FILLED)
                        cv2.putText(self.frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                        # Append ball and hoop positions with confidence filtering
                        if (current_class == "Ball" and conf > 0.3) or (in_hoop_region(center, self.hoop_pos) and conf > 0.15):
                            self.ball_pos.append((center, self.frame_count, w, h, conf))
                            cvzone.cornerRect(self.frame, (x1, y1, w, h))

                        if current_class == "Ring" and conf > 0.3:
                            self.hoop_pos.append((center, self.frame_count, w, h, conf))
                            cvzone.cornerRect(self.frame, (x1, y1, w, h))

                self.clean_motion()
                self.shot_detection()
                self.display_score()

                self.frame_count += 1

                # Display frame
                cv2.imshow(self.window_name, self.frame)

                key = cv2.waitKey(30) & 0xFF  # ~30 FPS update
                if key == ord('q'):
                    print("Quitting...")
                    break
                elif key == ord(' '):
                    self.paused = not self.paused
                    print("Paused" if self.paused else "Resumed")

        # Cleanup
        cv2.destroyAllWindows()
        self.csv_file.close()

    def clean_motion(self):
        # Remove stale positions, smooth hoop pos
        self.ball_pos = clean_ball_pos(self.ball_pos, self.frame_count)
        if self.hoop_pos:
            self.hoop_pos = clean_hoop_pos(self.hoop_pos)
            # Draw last hoop position on frame
            cv2.circle(self.frame, self.hoop_pos[-1][0], 2, (128, 128, 0), 2)

    def detect_peak(self, ball_pos):
        if len(ball_pos) < 3:
            return False
        return ball_pos[-2][0][1] > ball_pos[-3][0][1] and ball_pos[-2][0][1] > ball_pos[-1][0][1]

    def shot_detection(self):
        if not self.hoop_pos or not self.ball_pos:
            return

        # Detect upward ball movement
        if not self.up:
            self.up = detect_up(self.ball_pos, self.hoop_pos)
            if self.up:
                self.up_frame = self.ball_pos[-1][1]

        # Detect downward ball movement
        if self.up and not self.down:
            self.down = detect_down(self.ball_pos, self.hoop_pos)
            if self.down:
                self.down_frame = self.ball_pos[-1][1]

        # Detect peak frame in ball trajectory
        if self.up and not self.peak:
            self.peak = self.detect_peak(self.ball_pos)
            if self.peak:
                self.peak_frame = self.ball_pos[-1][1]

        # When shot attempt is complete
        if self.up and self.down and self.up_frame < self.down_frame:
            self.attempts += 1
            self.up = False
            self.down = False

            if score(self.ball_pos, self.hoop_pos):
                self.makes += 1
                self.overlay_color = (0, 255, 0)  # Green overlay for success
                self.fade_counter = self.fade_frames
                result = "Successful"
            else:
                self.overlay_color = (0, 0, 255)  # Red overlay for fail
                self.fade_counter = self.fade_frames
                result = "Failed"

            ball_center = self.ball_pos[-1][0]
            hoop_center = self.hoop_pos[-1][0]
            current_score = f"{self.makes} / {self.attempts}"
            video_timing_seconds = self.frame_count / self.fps

            print(f"Shot {self.attempts} detected: {result}")
            self.csv_writer.writerow([self.attempts, result, ball_center,
                                      hoop_center, current_score, video_timing_seconds])

    def display_score(self):
        text = f"{self.makes} / {self.attempts}"
        cv2.putText(self.frame, text, (50, 125), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 6)
        cv2.putText(self.frame, text, (50, 125), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 0), 3)

        # Overlay fade effect for shot result feedback
        if self.fade_counter > 0:
            alpha = 0.2 * (self.fade_counter / self.fade_frames)
            overlay = np.full_like(self.frame, self.overlay_color)
            self.frame = cv2.addWeighted(self.frame, 1 - alpha, overlay, alpha, 0)
            self.fade_counter -= 1


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Shot detection using YOLO and screen capture by process name")
    parser.add_argument('--model', type=str, required=True, default="models/Rishit.pt", help="Path to YOLO model file")
    parser.add_argument('--process', type=str, required=True, default="nba2k11.exe", help="Process name of target window")
    args = parser.parse_args()

    ShotDetector(model_path=args.model, process_name=args.process, model_name=os.path.splitext(os.path.basename(args.model))[0])
