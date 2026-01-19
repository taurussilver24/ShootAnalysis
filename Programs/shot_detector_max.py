import os
import cv2
import numpy as np
import csv
import time
import argparse
from ultralytics import YOLO
# Make sure you have a 'utils.py' file with your helper functions
from utils import score, detect_down, detect_up, in_hoop_region, clean_hoop_pos, clean_ball_pos


class ShotDetector:
    def __init__(self, model_name, video_name, display_fps=90):
        """
        Initializes a single-threaded shot detector, prioritizing analysis
        accuracy and real-time synchronization over maximum throughput.
        Includes a robust Pause/Play feature.
        """
        # --- CONFIGURATION ---
        self.display_fps = display_fps
        self.frame_display_delay = 1.0 / self.display_fps
        print(f"Running in Single-Threaded Mode for Maximum Accuracy.")
        print(f"Loop capped at: {self.display_fps} FPS. Press 'P' to Pause/Play, 'Q' to Quit.")

        # --- PATH AND MODEL SETUP ---
        self.model_path = os.path.join("models", model_name)
        self.video_path = os.path.join("HoopVids", video_name)
        self.model = YOLO(self.model_path, task="detect")
        self.class_names = ['Ring', 'Ball']
        self.model_input_width = 1280
        self.model_input_height = 736

        # --- VIDEO CAPTURE SETUP ---
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            raise IOError(f"Error opening video file: {self.video_path}")
        self.target_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.target_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.video_fps = self.cap.get(cv2.CAP_PROP_FPS)
        if self.video_fps == 0: self.video_fps = 60

        # --- STATE AND TRACKING ---
        self.paused = False  # <--- ADDED: Pause state
        self.processing_fps_start_time = time.time()
        self.processing_frame_count = 0
        self.display_processing_fps = 0
        self.ball_pos, self.hoop_pos = [], []
        self.makes, self.attempts = 0, 0
        self.up, self.down = False, False
        self.fade_frames, self.fade_counter, self.overlay_color = 20, 0, (0, 0, 0)

        # --- LOGGING SETUP ---
        video_basename = os.path.splitext(video_name)[0]
        model_basename = os.path.splitext(model_name)[0]
        results_dir = os.path.join('Results', video_basename)
        os.makedirs(results_dir, exist_ok=True)
        csv_path = os.path.join(results_dir, f'{model_basename}_shot_results.csv')
        self.csv_file = open(csv_path, mode='w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(
            ["Shot Taken", "Result", "Ball Coordinates", "Hoop Coordinates", "Current Score", "Video Timing (seconds)"])

        # Window setup
        self.window_name = f"Shot Detector | {os.path.basename(self.model_path)}"
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 1280, 720)

        self.run()

    def run(self):
        """
        Main single-threaded loop with pause/play functionality.
        """
        frame_index = 0
        while self.cap.isOpened():
            loop_start_time = time.time()

            # --- PAUSE LOGIC ---
            if not self.paused:
                ret, frame = self.cap.read()
                if not ret:
                    print("End of video reached.")
                    break

                self.process_frame(frame, frame_index)
                frame_index += 1

            # Drawing and display happen whether paused or not, to show the current frame
            self.update_fps()
            self.draw_debug_info(frame)  # Use the last valid frame when paused
            cv2.imshow(self.window_name, frame)

            # --- KEY HANDLING ---
            # Calculate wait time but ensure it's at least 1ms
            elapsed_time = time.time() - loop_start_time
            wait_time = self.frame_display_delay - elapsed_time
            wait_ms = max(1, int(wait_time * 1000))  # Ensures waitKey is never 0

            key = cv2.waitKey(wait_ms) & 0xFF
            if key == ord('q'):
                break
            if key == ord('p'):
                self.paused = not self.paused
                print("Paused" if self.paused else "Resumed")

        self.cleanup()

    def process_frame(self, frame, frame_index):
        """Processes a single frame for detection and analysis."""
        model_frame = cv2.resize(frame, (self.model_input_width, self.model_input_height))
        results = self.model(model_frame, stream=False, verbose=False, device='0', conf=0.75)[0]

        scale_x = self.target_width / self.model_input_width
        scale_y = self.target_height / self.model_input_height

        boxes = results.boxes.cpu().numpy()
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1_orig, y1_orig = int(x1 * scale_x), int(y1 * scale_y)
            x2_orig, y2_orig = int(x2 * scale_x), int(y2 * scale_y)
            w_orig, h_orig = x2_orig - x1_orig, y2_orig - y1_orig
            conf, cls = round(box.conf[0].item(), 2), int(box.cls[0])
            current_class = self.class_names[cls]
            center_orig = (x1_orig + w_orig // 2, y1_orig + h_orig // 2)
            color = (0, 0, 255) if current_class == "Ball" else (255, 0, 0)
            cv2.rectangle(frame, (x1_orig, y1_orig), (x2_orig, y2_orig), color, 2)
            if current_class == "Ball":
                self.ball_pos.append((center_orig, frame_index, w_orig, h_orig, conf))
            elif current_class == "Ring":
                self.hoop_pos.append((center_orig, frame_index, w_orig, h_orig, conf))
        self.clean_motion(frame, frame_index)
        self.shot_detection(frame_index)
        self.display_score(frame)
        self.processing_frame_count += 1

    def draw_debug_info(self, frame):
        font, font_scale, thickness = cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2
        color, bg_color = (0, 255, 0), (0, 0, 0)
        fps_text = f"Processing FPS: {self.display_processing_fps:.2f}"
        cv2.putText(frame, fps_text, (15, 35), font, font_scale, bg_color, thickness + 1, cv2.LINE_AA)
        cv2.putText(frame, fps_text, (15, 35), font, font_scale, color, thickness, cv2.LINE_AA)
        res_text = f"Display: {self.target_width}x{self.target_height} @ {self.display_fps} FPS Cap"
        cv2.putText(frame, res_text, (15, 70), font, font_scale, bg_color, thickness + 1, cv2.LINE_AA)
        cv2.putText(frame, res_text, (15, 70), font, font_scale, color, thickness, cv2.LINE_AA)
        if self.paused:
            pause_text = "PAUSED"
            text_size, _ = cv2.getTextSize(pause_text, font, 2, 3)
            text_x = (self.target_width - text_size[0]) // 2
            text_y = (self.target_height + text_size[1]) // 2
            cv2.putText(frame, pause_text, (text_x, text_y), font, 2, (0, 0, 255), 3, cv2.LINE_AA)

    def update_fps(self):
        # This will only update when not paused
        if not self.paused:
            elapsed_time = time.time() - self.processing_fps_start_time
            if elapsed_time > 1.0:
                self.display_processing_fps = self.processing_frame_count / elapsed_time
                self.processing_frame_count = 0
                self.processing_fps_start_time = time.time()

    def clean_motion(self, frame, frame_index):
        # Your clean_motion logic
        pass

    def shot_detection(self, frame_index):
        # Your shot_detection logic
        pass

    def display_score(self, frame):
        # Your display_score logic
        pass

    def cleanup(self):
        self.cap.release()
        cv2.destroyAllWindows()
        if self.csv_file: self.csv_file.close()
        print("Processing finished and resources released.")


if __name__ == "__main__":
    # Create a dummy utils.py if it doesn't exist
    if not os.path.exists('utils.py'):
        with open('utils.py', 'w') as f:
            f.write("""
def score(ball, hoop): return True
def detect_up(ball, hoop): return len(ball) > 5
def detect_down(ball, hoop): return len(ball) > 10
def in_hoop_region(center, hoop_pos): return True
def clean_hoop_pos(pos): return pos
def clean_ball_pos(pos, count): return pos
""")
    parser = argparse.ArgumentParser(
        description="Single-threaded shot detector focused on accuracy and real-time feel.")
    parser.add_argument('--model', type=str, required=True, help="Filename of the YOLO model (e.g., 'Rishit.onnx')")
    parser.add_argument('--video', type=str, required=True, help="Filename of the input video (e.g., '1.mp4')")
    parser.add_argument('--display_fps', type=int, default=90,
                        help="Target FPS for video playback and processing loop.")
    args = parser.parse_args()

    try:
        detector = ShotDetector(
            model_name=args.model,
            video_name=args.video,
            display_fps=args.display_fps
        )
    except (IOError, Exception) as e:
        print(f"An error occurred: {e}")