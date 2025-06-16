import os
import cv2
import numpy as np
import csv
import time
import argparse
import threading
import queue
from ultralytics import YOLO
# Make sure you have a 'utils.py' file with your helper functions
from utils import score, detect_down, detect_up, in_hoop_region, clean_hoop_pos, clean_ball_pos


class ShotDetector:
    def __init__(self, model_name, video_name, batch_size=8, display_fps=60):
        """
        Initializes the shot detector with a Producer-Consumer model for
        high-throughput processing and smooth, paced real-time display.
        """
        self.batch_size = batch_size
        self.display_fps = display_fps
        self.frame_display_delay = 1.0 / self.display_fps
        print(f"Goal: Restore Tracking Accuracy. Using small batch size: {self.batch_size}")
        print(f"Display capped at: {self.display_fps} FPS")

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
        self.processing_fps_start_time = time.time()
        self.processing_frame_count = 0
        self.display_processing_fps = 0
        self.global_frame_index = 0
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

        # --- THREADING AND QUEUE SETUP ---
        self.processed_frames_queue = queue.Queue(maxsize=self.batch_size * 2)
        self.stop_event = threading.Event()

        self.run()

    def producer(self):
        """BACKGROUND THREAD: Reads video, processes batches at max speed."""
        frame_batch = []
        while not self.stop_event.is_set():
            ret, frame = self.cap.read()
            if not ret:
                if frame_batch:
                    self.process_frame_batch(frame_batch)
                    for f in frame_batch: self.processed_frames_queue.put(f)
                break
            frame_batch.append(frame)
            if len(frame_batch) == self.batch_size:
                self.process_frame_batch(frame_batch)
                for f in frame_batch: self.processed_frames_queue.put(f)
                frame_batch = []
        self.processed_frames_queue.put(None)
        print("Producer thread finished.")

    def run(self):
        """MAIN THREAD: Starts producer, consumes frames for smooth display."""
        producer_thread = threading.Thread(target=self.producer, daemon=True)
        producer_thread.start()

        self.window_name = f"MODEL: {os.path.basename(self.model_path)} | BATCH: {self.batch_size}"
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 1280, 720)

        while not self.stop_event.is_set():
            loop_start_time = time.time()
            try:
                frame = self.processed_frames_queue.get(timeout=1)
                if frame is None: break
                self.draw_debug_info(frame)
                cv2.imshow(self.window_name, frame)

                elapsed_time = time.time() - loop_start_time
                wait_time = self.frame_display_delay - elapsed_time
                wait_ms = int(wait_time * 1000) if wait_time > 0 else 1
                if cv2.waitKey(wait_ms) & 0xFF == ord('q'):
                    break
            except queue.Empty:
                if not producer_thread.is_alive(): break

        self.cleanup()

    def process_frame_batch(self, frame_batch):
        model_frames = [cv2.resize(f, (self.model_input_width, self.model_input_height)) for f in frame_batch]
        results_batch = self.model(model_frames, stream=False, verbose=False, device='0', conf=0.75)

        scale_x = self.target_width / self.model_input_width
        scale_y = self.target_height / self.model_input_height

        for i, results in enumerate(results_batch):
            original_frame = frame_batch[i]
            self.analyze_and_draw_results(original_frame, results, scale_x, scale_y)
            self.processing_frame_count += 1

    def analyze_and_draw_results(self, frame, results, scale_x, scale_y):
        """Analyzes results for a single frame and draws on it."""
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
                self.ball_pos.append((center_orig, self.global_frame_index, w_orig, h_orig, conf))
            elif current_class == "Ring":
                self.hoop_pos.append((center_orig, self.global_frame_index, w_orig, h_orig, conf))

        self.clean_motion(frame)
        self.shot_detection()
        self.display_score(frame)
        self.global_frame_index += 1

    def draw_debug_info(self, frame):
        font, font_scale, thickness = cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2
        color, bg_color = (0, 255, 0), (0, 0, 0)

        elapsed_time = time.time() - self.processing_fps_start_time
        if elapsed_time > 1.0:
            self.display_processing_fps = self.processing_frame_count / elapsed_time
            self.processing_frame_count = 0
            self.processing_fps_start_time = time.time()

        fps_text = f"Processing FPS: {self.display_processing_fps:.2f}"
        cv2.putText(frame, fps_text, (15, 35), font, font_scale, bg_color, thickness + 1, cv2.LINE_AA)
        cv2.putText(frame, fps_text, (15, 35), font, font_scale, color, thickness, cv2.LINE_AA)

        res_text = f"Display: {self.target_width}x{self.target_height} @ {self.display_fps} FPS"
        cv2.putText(frame, res_text, (15, 70), font, font_scale, bg_color, thickness + 1, cv2.LINE_AA)
        cv2.putText(frame, res_text, (15, 70), font, font_scale, color, thickness, cv2.LINE_AA)

    def clean_motion(self, frame):
        self.ball_pos = clean_ball_pos(self.ball_pos, self.global_frame_index)
        if self.hoop_pos:
            self.hoop_pos = clean_hoop_pos(self.hoop_pos)
            if self.hoop_pos: cv2.circle(frame, self.hoop_pos[-1][0], 5, (0, 255, 255), -1)

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
                self.makes += 1;
                self.overlay_color = (0, 255, 0);
                result = "Successful"
            else:
                self.overlay_color = (0, 0, 255)
            self.fade_counter = self.fade_frames
            timestamp = self.global_frame_index / self.video_fps
            print(f"Shot #{self.attempts} detected at {timestamp:.2f}s. Result: {result}")
            self.csv_writer.writerow(
                [self.attempts, result, self.ball_pos[-1][0], self.hoop_pos[-1][0], f"{self.makes} / {self.attempts}",
                 f"{timestamp:.2f}"])
            self.up = self.down = False
            self.ball_pos = [p for p in self.ball_pos if p[1] > self.down_frame]

    def display_score(self, frame):
        text = f"{self.makes} / {self.attempts}"
        cv2.putText(frame, text, (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 6, cv2.LINE_AA)
        cv2.putText(frame, text, (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 0), 3, cv2.LINE_AA)
        if self.fade_counter > 0:
            alpha = 0.3 * (self.fade_counter / self.fade_frames);
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (self.target_width, self.target_height), self.overlay_color, -1)
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
            self.fade_counter -= 1

    def cleanup(self):
        self.stop_event.set()
        self.cap.release()
        cv2.destroyAllWindows()
        if self.csv_file: self.csv_file.close()
        print("Processing finished and resources released.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Hybrid (threaded) basketball shot detector for high performance and smooth display.")
    parser.add_argument('--model', type=str, required=True, help="Filename of the YOLO model (e.g., 'Rishit.onnx')")
    parser.add_argument('--video', type=str, required=True, help="Filename of the input video (e.g., '1.mp4')")
    parser.add_argument('--batch_size', type=int, default=8,
                        help="Frames to process at once. Smaller values (4-8) improve tracking accuracy.")
    parser.add_argument('--display_fps', type=int, default=60, help="Target FPS for video playback.")
    args = parser.parse_args()

    try:
        detector = ShotDetector(
            model_name=args.model,
            video_name=args.video,
            batch_size=args.batch_size,
            display_fps=args.display_fps
        )
    except (IOError, Exception) as e:
        print(f"An error occurred: {e}")