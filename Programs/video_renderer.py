import os
import cv2
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO
from utils import score, detect_down, detect_up, in_hoop_region, clean_hoop_pos, clean_ball_pos

# ==============================================================================
# ⚙️ ULTIMATE RENDER SETTINGS
# ==============================================================================
INPUT_VIDEO = "../HoopVids/Done_Requested/SpaceJam.mp4"
OUTPUT_VIDEO = "Ultimate_Analysis.mp4"
MODEL_PATH = "../models/Rishit.onnx"
BATCH_SIZE = 24  # High batch size for RTX 4080


# ==============================================================================

class UltimateRenderer:
    def __init__(self, model_path, input_video, output_video, batch_size):
        self.batch_size = batch_size
        self.model = YOLO(model_path, task="detect")
        self.class_names = ['Ring', 'Ball']

        # Video Setup
        self.cap = cv2.VideoCapture(input_video)
        if not self.cap.isOpened(): raise IOError(f"❌ Error opening {input_video}")

        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Output Setup
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.writer = cv2.VideoWriter(output_video, fourcc, self.fps, (self.width, self.height))

        # Logic State
        self.ball_pos = []
        self.hoop_pos = []
        self.makes = 0
        self.attempts = 0
        self.up = self.down = False

        # Visual State
        self.fade_frames = 15
        self.fade_counter = 0
        self.overlay_color = (0, 0, 0)
        self.overlay_buffer = np.zeros((self.height, self.width, 3), dtype=np.uint8)

    def run(self):
        print(f"=== 🚀 ULTIMATE BATCH RENDERER ===")
        print(f"   GPU: RTX 4080 | Batch Size: {self.batch_size}")

        frame_batch = []  # Raw frames for writing
        resize_batch = []  # Resized frames for AI

        pbar = tqdm(total=self.total_frames, unit="frames")

        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                if frame_batch: self.process_batch(resize_batch, frame_batch)
                break

            # Prepare Batch
            frame_batch.append(frame)
            resize_batch.append(cv2.resize(frame, (1280, 736)))  # AI Resolution

            if len(frame_batch) == self.batch_size:
                self.process_batch(resize_batch, frame_batch)
                pbar.update(len(frame_batch))
                frame_batch = []
                resize_batch = []

        pbar.close()
        self.cleanup()

    def process_batch(self, ai_frames, raw_frames):
        # 1. GPU INFERENCE (Massive Speedup)
        # We use a low confidence threshold (0.15) so we can filter smartly later
        results = self.model(ai_frames, stream=False, verbose=False, device='0', conf=0.15)

        # Scaling Factors (AI -> 1440p/1080p)
        sx = self.width / 1280
        sy = self.height / 736

        # 2. SEQUENTIAL PROCESSING (Physics must be done in order)
        for i, r in enumerate(results):
            # A. Extract Data
            current_frame = raw_frames[i]
            boxes = r.boxes.cpu().numpy()

            frame_balls = []
            frame_rings = []

            # Parse Boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                conf = box.conf[0]
                cls = int(box.cls[0])
                label = self.class_names[cls]

                # Scale to Original Resolution
                x1, x2 = int(x1 * sx), int(x2 * sx)
                y1, y2 = int(y1 * sy), int(y2 * sy)
                w, h = x2 - x1, y2 - y1
                center = ((x1 + x2) // 2, (y1 + y2) // 2)

                item = {'c': center, 'box': (x1, y1, x2, y2), 'conf': conf, 'w': w, 'h': h}

                if label == 'Ball':
                    frame_balls.append(item)
                elif label == 'Ring':
                    frame_rings.append(item)

            # B. Smart Filtering (The "Better" Logic)
            best_ball = None
            best_ring = None

            # Ring: Strict (Must be > 0.60)
            valid_rings = [x for x in frame_rings if x['conf'] > 0.60]
            if valid_rings:
                best_ring = max(valid_rings, key=lambda x: x['conf'])
                self.hoop_pos.append((best_ring['c'], 0, best_ring['w'], best_ring['h'], best_ring['conf']))

            # Ball: Hybrid (Must be > 0.50 OR > 0.15 if near hoop)
            valid_balls = [
                x for x in frame_balls
                if x['conf'] > 0.50 or (x['conf'] > 0.15 and in_hoop_region(x['c'], self.hoop_pos))
            ]
            if valid_balls:
                best_ball = max(valid_balls, key=lambda x: x['conf'])
                self.ball_pos.append((best_ball['c'], 0, best_ball['w'], best_ball['h'], best_ball['conf']))

            # C. Physics & Scoring
            self.ball_pos = clean_ball_pos(self.ball_pos, 0)
            if self.hoop_pos: self.hoop_pos = clean_hoop_pos(self.hoop_pos)

            if self.hoop_pos and self.ball_pos:
                if not self.up: self.up = detect_up(self.ball_pos, self.hoop_pos)
                if self.up and not self.down: self.down = detect_down(self.ball_pos, self.hoop_pos)

                if self.up and self.down:
                    self.attempts += 1
                    if score(self.ball_pos, self.hoop_pos):
                        self.makes += 1
                        self.overlay_color = (0, 255, 0)
                    else:
                        self.overlay_color = (0, 0, 255)
                    self.fade_counter = self.fade_frames
                    self.up = self.down = False

            # D. Draw Visuals
            # Boxes
            if best_ball:
                b = best_ball['box']
                cv2.rectangle(current_frame, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)
            if best_ring:
                b = best_ring['box']
                cv2.rectangle(current_frame, (b[0], b[1]), (b[2], b[3]), (255, 0, 0), 2)

            # Flash
            if self.fade_counter > 0:
                self.overlay_buffer[:] = self.overlay_color
                alpha = 0.3 * (self.fade_counter / self.fade_frames)
                cv2.addWeighted(current_frame, 1 - alpha, self.overlay_buffer, alpha, 0, dst=current_frame)
                self.fade_counter -= 1

            # Score
            cv2.putText(current_frame, f"SCORE: {self.makes}/{self.attempts}", (30, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

            # E. Write
            self.writer.write(current_frame)

    def cleanup(self):
        self.cap.release()
        self.writer.release()
        print("\n✅ DONE!")


if __name__ == "__main__":
    UltimateRenderer(MODEL_PATH, INPUT_VIDEO, OUTPUT_VIDEO, BATCH_SIZE).run()