import os
import cv2
import cvzone
import numpy as np
import csv
import time
from ultralytics import YOLO
# Assuming your 'utils.py' file is in the same directory or accessible
from utils import score, detect_down, detect_up, in_hoop_region, clean_hoop_pos, clean_ball_pos
import argparse


class ShotDetector:
    def __init__(self, model_name, video_name):
        """
        Initializes the shot detector using only model and video filenames.
        Paths are constructed automatically.
        """
        # --- Path Construction ---
        self.model_path = os.path.join("models", model_name)
        self.video_path = os.path.join("HoopVids", video_name)
        video_basename = os.path.splitext(video_name)[0]
        model_basename = os.path.splitext(model_name)[0]

        # Model initialization
        self.model = YOLO(self.model_path, task="detect")
        self.class_names = ['Ring', 'Ball']

        # --- Define Model Input Size ---
        self.model_input_width = 1280
        self.model_input_height = 736

        # Video capture setup
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            raise IOError(f"Error opening video file: {self.video_path}")

        # Define Original Video Size
        self.target_width = 1920
        self.target_height = 1080
        self.video_fps = self.cap.get(cv2.CAP_PROP_FPS)
        if self.video_fps == 0: self.video_fps = 60

        # --- FPS Counter Initialization ---
        self.fps_start_time = time.time()
        self.fps_frame_count = 0
        self.display_fps = 0

        # Detection buffers and state
        self.frame_count = 0
        self.ball_pos = []
        self.hoop_pos = []
        self.makes = 0
        self.attempts = 0
        self.up = False
        self.down = False
        self.peak = False

        # UI elements
        self.fade_frames = 20
        self.fade_counter = 0
        self.overlay_color = (0, 0, 0)

        # --- Results Logging Setup ---
        results_dir = os.path.join('Results', video_basename)
        os.makedirs(results_dir, exist_ok=True)
        csv_path = os.path.join(results_dir, f'{model_basename}_shot_results.csv')
        self.csv_file = open(csv_path, mode='w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow([
            "Shot Taken", "Result", "Ball Coordinates",
            "Hoop Coordinates", "Current Score", "Video Timing (seconds)"
        ])

        # Window setup
        self.window_name = f"MODEL: {model_name} | VIDEO: {video_name}"
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 1280, 720)
        cv2.createTrackbar('Pause', self.window_name, 0, 1, self.on_pause_trackbar_change)
        self.paused = False

        self.run()

    def on_pause_trackbar_change(self, pos):
        self.paused = bool(pos)

    def run(self):
        while self.cap.isOpened():
            if not self.paused:
                ret, frame = self.cap.read()
                if not ret:
                    print("End of video reached.")
                    break

                if frame.shape[1] != self.target_width or frame.shape[0] != self.target_height:
                    frame = cv2.resize(frame, (self.target_width, self.target_height))

                self.process_frame(frame)

                self.update_fps()
                self.draw_debug_info(frame)

            cv2.imshow(self.window_name, frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord(' '):
                self.paused = not self.paused
                cv2.setTrackbarPos('Pause', self.window_name, int(self.paused))

        self.cleanup()

    def draw_debug_info(self, frame):
        """Draws FPS and resolution text on the top-left of the frame."""
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.9
        color = (0, 255, 0)
        thickness = 2
        bg_color = (0, 0, 0)
        cv2.putText(frame, f"Processing FPS: {self.display_fps:.2f}", (15, 35), font, font_scale, bg_color,
                    thickness + 1, cv2.LINE_AA)
        cv2.putText(frame, f"Processing FPS: {self.display_fps:.2f}", (15, 35), font, font_scale, color, thickness,
                    cv2.LINE_AA)
        cv2.putText(frame, f"Display: {self.target_width}x{self.target_height}", (15, 70), font, font_scale, bg_color,
                    thickness + 1, cv2.LINE_AA)
        cv2.putText(frame, f"Display: {self.target_width}x{self.target_height}", (15, 70), font, font_scale, color,
                    thickness, cv2.LINE_AA)

    def update_fps(self):
        """Calculates the processing frames per second."""
        self.fps_frame_count += 1
        elapsed_time = time.time() - self.fps_start_time
        if elapsed_time > 1.0:
            self.display_fps = self.fps_frame_count / elapsed_time
            self.fps_frame_count = 0
            self.fps_start_time = time.time()

    def process_frame(self, frame):
        model_frame = cv2.resize(frame, (self.model_input_width, self.model_input_height))

        # --- MODIFICATION 1: Lower confidence to catch all potential detections ---
        results = self.model(
            model_frame, stream=True, verbose=False,
            imgsz=(self.model_input_height, self.model_input_width),
            half=True, device='0', conf=0.15
        )

        scale_x = self.target_width / self.model_input_width
        scale_y = self.target_height / self.model_input_height

        # --- START OF MODIFIED LOGIC ---

        # 1. Collect all raw detections from the current frame into a list
        frame_detections = []
        for r in results:
            boxes = r.boxes.cpu().numpy()
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                frame_detections.append({
                    "x1_orig": int(x1 * scale_x), "y1_orig": int(y1 * scale_y),
                    "x2_orig": int(x2 * scale_x), "y2_orig": int(y2 * scale_y),
                    "w_orig": int((x2 - x1) * scale_x), "h_orig": int((y2 - y1) * scale_y),
                    "conf": round(box.conf[0].item(), 2),
                    "class_name": self.class_names[int(box.cls[0])]
                })

        # 2. Filter rings and balls based on our refined rules
        valid_rings = [d for d in frame_detections if d['class_name'] == 'Ring' and d['conf'] > 0.6]
        valid_balls = [
            d for d in frame_detections if d['class_name'] == 'Ball' and
                                           (d['conf'] > 0.5 or (d['conf'] > 0.15 and in_hoop_region(
                                               (d['x1_orig'] + d['w_orig'] // 2, d['y1_orig'] + d['h_orig'] // 2),
                                               self.hoop_pos)))
        ]

        # 3. Find the single BEST ring and ball (if they exist)
        best_ring = max(valid_rings, key=lambda d: d['conf']) if valid_rings else None
        best_ball = max(valid_balls, key=lambda d: d['conf']) if valid_balls else None

        # 4. Use only the best detections for drawing and analysis
        if best_ring:
            cv2.rectangle(frame, (best_ring['x1_orig'], best_ring['y1_orig']),
                          (best_ring['x2_orig'], best_ring['y2_orig']), (255, 0, 0), 2)
            center = (best_ring['x1_orig'] + best_ring['w_orig'] // 2, best_ring['y1_orig'] + best_ring['h_orig'] // 2)
            self.hoop_pos.append(
                (center, self.frame_count, best_ring['w_orig'], best_ring['h_orig'], best_ring['conf']))

        if best_ball:
            cv2.rectangle(frame, (best_ball['x1_orig'], best_ball['y1_orig']),
                          (best_ball['x2_orig'], best_ball['y2_orig']), (0, 0, 255), 2)
            center = (best_ball['x1_orig'] + best_ball['w_orig'] // 2, best_ball['y1_orig'] + best_ball['h_orig'] // 2)
            self.ball_pos.append(
                (center, self.frame_count, best_ball['w_orig'], best_ball['h_orig'], best_ball['conf']))

        # --- END OF MODIFIED LOGIC ---

        self.clean_motion(frame)
        self.shot_detection()
        self.display_score(frame)
        self.frame_count += 1

    def clean_motion(self, frame):
        self.ball_pos = clean_ball_pos(self.ball_pos, self.frame_count)
        if self.hoop_pos:
            self.hoop_pos = clean_hoop_pos(self.hoop_pos)
            cv2.circle(frame, self.hoop_pos[-1][0], 5, (0, 255, 255), -1)

    def shot_detection(self):
        if not self.hoop_pos or not self.ball_pos: return

        if not self.up:
            self.up = detect_up(self.ball_pos, self.hoop_pos)
            if self.up: self.up_frame = self.ball_pos[-1][1]

        if self.up and not self.down:
            self.down = detect_down(self.ball_pos, self.hoop_pos)
            if self.down: self.down_frame = self.ball_pos[-1][1]

        if self.up and self.down and hasattr(self, 'up_frame') and hasattr(self,
                                                                           'down_frame') and self.up_frame < self.down_frame:
            self.attempts += 1
            result = "Failed"
            if score(self.ball_pos, self.hoop_pos):
                self.makes += 1
                self.overlay_color = (0, 255, 0)
                result = "Successful"
            else:
                self.overlay_color = (0, 0, 255)

            self.fade_counter = self.fade_frames
            ball_center = self.ball_pos[-1][0]
            hoop_center = self.hoop_pos[-1][0]
            score_text = f"{self.makes} / {self.attempts}"
            timestamp = self.frame_count / self.video_fps
            print(f"Shot #{self.attempts} detected at {timestamp:.2f}s. Result: {result}")
            self.csv_writer.writerow([
                self.attempts, result, ball_center,
                hoop_center, score_text, f"{timestamp:.2f}"
            ])
            self.up = self.down = False
            # We don't need to clean ball_pos here since it's handled in clean_motion

    def display_score(self, frame):
        text = f"{self.makes} / {self.attempts}"
        cv2.putText(frame, text, (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 6, cv2.LINE_AA)
        cv2.putText(frame, text, (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 0), 3, cv2.LINE_AA)

        if self.fade_counter > 0:
            alpha = 0.3 * (self.fade_counter / self.fade_frames)
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (self.target_width, self.target_height), self.overlay_color, -1)
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
            self.fade_counter -= 1

    def cleanup(self):
        self.cap.release()
        cv2.destroyAllWindows()
        self.csv_file.close()
        print("Processing finished and resources released.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Basketball shot detector for prerecorded videos. Assumes videos are in 'HoopVids/' and models are in 'models/'."
    )
    parser.add_argument('--model', type=str, required=True,
                        help="Filename of the YOLO model (e.g., 'Rishit.onnx')")
    parser.add_argument('--video', type=str, required=True,
                        help="Filename of the input video (e.g., 'my_gameplay.mp4')")
    args = parser.parse_args()

    try:
        detector = ShotDetector(
            model_name=args.model,
            video_name=args.video
        )
    except (IOError, Exception) as e:
        print(f"An error occurred: {e}")