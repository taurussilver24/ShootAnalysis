# ultimate_analyzer.py (v2 - With Data Sanitization)
import os
import cv2
import csv
import argparse
from ultralytics import YOLO
from tqdm import tqdm
import numpy as np
# Make sure you have a 'utils.py' file with your actual helper functions
from utils import score, detect_down, detect_up, in_hoop_region, clean_hoop_pos, clean_ball_pos


class UltimateAnalyzer:
    def __init__(self, model_name, video_name, batch_size=64):
        self.batch_size = batch_size
        print(f"Initializing Ultimate Analyzer with batch size: {self.batch_size}")

        self.model_path = os.path.join("models", model_name)
        self.video_path = os.path.join("HoopVids", video_name)
        self.model = YOLO(self.model_path, task="detect")
        self.class_names = ['Ring', 'Ball']

        self.model_input_width = 1280
        self.model_input_height = 736

        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            raise IOError(f"Error opening video file: {self.video_path}")

        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.video_fps = self.cap.get(cv2.CAP_PROP_FPS)
        if self.video_fps == 0: self.video_fps = 60

        self.original_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.original_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Analysis state
        self.ball_pos, self.hoop_pos = [], []
        self.makes, self.attempts = 0, 0
        self.up, self.down = False, False

        # --- CSV LOGGING SETUP (FOR SIMPLE SHOT RESULTS) ---
        video_basename = os.path.splitext(video_name)[0]
        model_basename = os.path.splitext(model_name)[0]
        results_dir = os.path.join('Results', video_basename)
        os.makedirs(results_dir, exist_ok=True)
        self.csv_path = os.path.join(results_dir, f'{model_basename}_shot_results.csv')

        self.csv_file = open(self.csv_path, mode='w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow([
            "Shot Taken", "Result", "Ball Coordinates", "Hoop Coordinates", "Current Score", "Video Timing (seconds)"
        ])

    def run_analysis(self):
        frame_batch = []
        start_frame_index = 0

        with tqdm(total=self.total_frames, desc="Analyzing Video") as self.pbar:
            while self.cap.isOpened():
                ret, frame = self.cap.read()
                if not ret:
                    if frame_batch: self.process_batch(frame_batch, start_frame_index)
                    break
                frame_batch.append(frame)
                if len(frame_batch) == self.batch_size:
                    self.process_batch(frame_batch, start_frame_index)
                    start_frame_index += len(frame_batch)
                    frame_batch = []
        self.cleanup()

    def process_batch(self, frame_batch, start_frame_index):
        model_frames = [cv2.resize(f, (self.model_input_width, self.model_input_height)) for f in frame_batch]
        results_batch = self.model(model_frames, stream=False, verbose=False, device='0', conf=0.10)

        scale_x = self.original_width / self.model_input_width
        scale_y = self.original_height / self.model_input_height

        for i, results in enumerate(results_batch):
            current_frame_index = start_frame_index + i

            boxes = results.boxes.cpu().numpy()
            for box in boxes:
                x1_orig, y1_orig, x2_orig, y2_orig = (
                            box.xyxy[0] * np.array([scale_x, scale_y, scale_x, scale_y])).astype(int)
                w_orig, h_orig = x2_orig - x1_orig, y2_orig - y1_orig
                conf, cls = round(box.conf[0].item(), 2), int(box.cls[0])
                current_class = self.class_names[cls]
                center_orig = ((x1_orig + x2_orig) // 2, (y1_orig + y2_orig) // 2)

                if current_class == "Ring" and conf > 0.75:
                    self.hoop_pos.append((center_orig, current_frame_index, w_orig, h_orig, conf))
                if (current_class == "Ball" and conf > 0.75) or \
                        (current_class == "Ball" and in_hoop_region(center_orig, self.hoop_pos) and conf > 0.15):
                    self.ball_pos.append((center_orig, current_frame_index, w_orig, h_orig, conf))

            self.ball_pos = clean_ball_pos(self.ball_pos, current_frame_index)
            self.hoop_pos = clean_hoop_pos(self.hoop_pos)

            self.detect_shot_event()

        self.pbar.update(len(frame_batch))

    def detect_shot_event(self):
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
                result = "Successful"

            score_text = f"{self.makes}/{self.attempts}"
            timestamp = self.down_frame / self.video_fps
            print(f"Shot #{self.attempts} detected at frame ~{self.down_frame}. Result: {result}")

            # --- THE FINAL FIX: Sanitize data before writing to CSV ---
            # Unpack the tuples and convert any numpy types to standard python types
            ball_data = self.ball_pos[-1]
            b_center, b_frame, b_w, b_h, b_conf = ball_data
            clean_ball_tuple = ((int(b_center[0]), int(b_center[1])), int(b_frame), int(b_w), int(b_h), float(b_conf))

            hoop_data = self.hoop_pos[-1]
            h_center, h_frame, h_w, h_h, h_conf = hoop_data
            clean_hoop_tuple = ((int(h_center[0]), int(h_center[1])), int(h_frame), int(h_w), int(h_h), float(h_conf))

            # Write the clean, standard data
            self.csv_writer.writerow([
                self.attempts, result, clean_ball_tuple, clean_hoop_tuple, score_text, f"{timestamp:.2f}"
            ])

            self.up = self.down = False
            self.ball_pos = [p for p in self.ball_pos if p[1] > self.down_frame]

    def cleanup(self):
        self.cap.release()
        self.csv_file.close()
        print(f"\nAnalysis complete. Simple shot data saved to {self.csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Ultimate Headless Analyzer: Fast processing with simple shot log output.")
    parser.add_argument('--model', type=str, required=True, help="YOLO model file.")
    parser.add_argument('--video', type=str, required=True, help="Input video file.")
    parser.add_argument('--batch_size', type=int, default=64, help="Batch size for processing for maximum speed.")
    args = parser.parse_args()



    analyzer = UltimateAnalyzer(args.model, args.video, args.batch_size)
    analyzer.run_analysis()