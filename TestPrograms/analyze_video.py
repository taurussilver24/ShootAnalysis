# analyze_video.py (With High-Confidence & Contextual Low-Confidence Rules)
import os
import cv2
import csv
import argparse
from ultralytics import YOLO
from tqdm import tqdm
# Make sure you have a 'utils.py' file with your actual helper functions
from utils import score, detect_down, detect_up, in_hoop_region, clean_hoop_pos, clean_ball_pos


class HeadlessAnalyzer:
    def __init__(self, model_name, video_name, batch_size):
        self.batch_size = batch_size
        print(f"Initializing Headless Analyzer with batch size: {self.batch_size}")

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
        self.up_frame = None
        self.down_frame = None

        # --- CSV LOGGING SETUP ---
        video_basename = os.path.splitext(video_name)[0]
        model_basename = os.path.splitext(model_name)[0]
        results_dir = os.path.join('Render_Results', video_basename)
        os.makedirs(results_dir, exist_ok=True)
        self.csv_path = os.path.join(results_dir, f'{model_basename}_analysis_data.csv')

        self.csv_file = open(self.csv_path, mode='w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow([
            "frame_number", "class_name", "x1", "y1", "x2", "y2", "confidence",
            "shot_attempt_id", "shot_result", "current_score", "apex_frame"
        ])

    def run_analysis(self):
        frame_batch = []
        start_frame_index = 0

        with tqdm(total=self.total_frames, desc="Analyzing Video") as self.pbar:
            while self.cap.isOpened():
                ret, frame = self.cap.read()
                if not ret:
                    if frame_batch:
                        self.process_batch(frame_batch, start_frame_index)
                    break
                frame_batch.append(frame)
                if len(frame_batch) == self.batch_size:
                    self.process_batch(frame_batch, start_frame_index)
                    start_frame_index += len(frame_batch)
                    frame_batch = []
        self.cleanup()

    def process_batch(self, frame_batch, start_frame_index):
        # Step 1: High-speed inference
        # --- MODIFIED: Lower confidence to see potential low-conf balls near hoop ---
        model_frames = [cv2.resize(f, (self.model_input_width, self.model_input_height)) for f in frame_batch]
        results_batch = self.model(model_frames, stream=False, verbose=False, device='0', conf=0.15)

        # Step 2: Collect all raw detections
        all_detections = []
        scale_x = self.original_width / self.model_input_width
        scale_y = self.original_height / self.model_input_height

        for i, results in enumerate(results_batch):
            current_frame_index = start_frame_index + i
            frame_detections = []
            boxes = results.boxes.cpu().numpy()
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1_orig, y1_orig = int(x1 * scale_x), int(y1 * scale_y)
                x2_orig, y2_orig = int(x2 * scale_x), int(y2 * scale_y)
                w_orig, h_orig = x2_orig - x1_orig, y2_orig - y1_orig
                conf, cls = round(box.conf[0].item(), 2), int(box.cls[0])
                class_name = self.class_names[cls]
                center_orig = (x1_orig + w_orig // 2, y1_orig + h_orig // 2)

                frame_detections.append({
                    "frame_number": current_frame_index, "class_name": class_name,
                    "x1": x1_orig, "y1": y1_orig, "x2": x2_orig, "y2": y2_orig,
                    "confidence": conf, "center": center_orig, "w": w_orig, "h": h_orig
                })
            all_detections.append(frame_detections)

        # Step 3: Loop through frames sequentially for stateful analysis
        for i, frame_detections in enumerate(all_detections):
            current_frame_index = start_frame_index + i

            # --- MODIFIED LOGIC TO RE-INTRODUCE CONTEXTUAL RULE ---
            # 1. Filter rings with a strict confidence threshold
            valid_rings = [d for d in frame_detections if d['class_name'] == 'Ring' and d['confidence'] > 0.6]

            # 2. Filter balls with the hybrid confidence rule
            valid_balls = [
                d for d in frame_detections if d['class_name'] == 'Ball' and
                (d['confidence'] > 0.5 or (d['confidence'] > 0.15 and in_hoop_region(d['center'], self.hoop_pos)))
            ]

            # 3. Find the single BEST ring, if any exist, and add to history
            if valid_rings:
                best_ring = max(valid_rings, key=lambda d: d['confidence'])
                self.hoop_pos.append((best_ring['center'], current_frame_index, best_ring['w'], best_ring['h'], best_ring['confidence']))

            # 4. Find the single BEST ball, if any exist, and add to history
            if valid_balls:
                best_ball = max(valid_balls, key=lambda d: d['confidence'])
                self.ball_pos.append((best_ball['center'], current_frame_index, best_ball['w'], best_ball['h'], best_ball['confidence']))
            # --- END OF MODIFIED LOGIC ---

            self.ball_pos = clean_ball_pos(self.ball_pos, current_frame_index)
            self.hoop_pos = clean_hoop_pos(self.hoop_pos)

            is_attempt, shot_result, score_text, apex_frame = self.detect_shot_event()

            # Log all original (low-confidence) detections to the CSV for review
            if not frame_detections and is_attempt:
                self.csv_writer.writerow([current_frame_index, "", "", "", "", "", "", self.attempts, shot_result, score_text, apex_frame if is_attempt else ""])
            else:
                for det in frame_detections:
                    self.csv_writer.writerow([
                        det['frame_number'], det['class_name'],
                        det['x1'], det['y1'], det['x2'], det['y2'], det['confidence'],
                        self.attempts if is_attempt else "",
                        shot_result if is_attempt else "",
                        score_text if is_attempt else "",
                        apex_frame if is_attempt else ""
                    ])

        self.pbar.update(len(frame_batch))

    def detect_shot_event(self):
        if not self.hoop_pos or not self.ball_pos:
            return False, "", "", None

        if not self.up:
            self.up = detect_up(self.ball_pos, self.hoop_pos)
            if self.up:
                self.up_frame = self.ball_pos[-1][1]
        if self.up and not self.down:
            self.down = detect_down(self.ball_pos, self.hoop_pos)
            if self.down:
                self.down_frame = self.ball_pos[-1][1]

        if self.up and self.down and self.up_frame and self.down_frame and self.up_frame < self.down_frame:
            self.attempts += 1
            result = "Failed"
            if score(self.ball_pos, self.hoop_pos):
                self.makes += 1
                result = "Successful"
            score_text = f"{self.makes}/{self.attempts}"
            print(f"Shot #{self.attempts} detected at frame ~{self.down_frame}. Result: {result}")

            apex_frame_for_log = self.up_frame
            shot_end_frame = self.down_frame

            self.up = self.down = False
            self.up_frame = self.down_frame = None
            self.ball_pos = [p for p in self.ball_pos if p[1] > shot_end_frame]

            return True, result, score_text, apex_frame_for_log

        return False, "", "", None

    def cleanup(self):
        self.cap.release()
        self.csv_file.close()
        print(f"\nAnalysis complete. Data saved to {self.csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Headless Video Analyzer Engine.")
    parser.add_argument('--model', type=str, required=True, help="YOLO model file.")
    parser.add_argument('--video', type=str, required=True, help="Input video file.")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size for processing.")
    args = parser.parse_args()

    analyzer = HeadlessAnalyzer(args.model, args.video, args.batch_size)
    analyzer.run_analysis()