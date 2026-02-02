import os
import cv2
import glob
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO
from utils import score, detect_down, detect_up, in_hoop_region, clean_hoop_pos, clean_ball_pos

# ==============================================================================
# ⚙️ DUAL FACTORY SETTINGS
# ==============================================================================
INPUT_FOLDER = r"C:\Users\Rishit\Videos\NBA2K_Raw"
OUTPUT_ROOT = "NBA2K_Thesis_Master_Dataset"
MODEL_PATH = "../models/Rishit.onnx"

# SAMPLING RATES
GENERAL_SAVE_RATE = 45  # Save 1 frame every ~0.75s (Dribbling/Walking)
SHOT_SAVE_RATE = 3  # Save 1 frame every ~0.05s (High density during shots)

CONF_STRICT = 0.60  # Ring
CONF_RELAXED = 0.15  # Ball
BATCH_SIZE = 24  # High batch for RTX 4080 / RX 6800


# ==============================================================================

class DualDatasetGenerator:
    def __init__(self, input_folder, output_root, model_path):
        self.input_folder = input_folder
        self.model = YOLO(model_path, task="detect")
        self.class_names = {0: 'Ring', 1: 'Ball'}

        # Setup Dual Directories
        self.path_gen_img = os.path.join(output_root, "Dataset_1_General", "images")
        self.path_gen_lbl = os.path.join(output_root, "Dataset_1_General", "labels")
        self.path_shot_img = os.path.join(output_root, "Dataset_2_Shots", "images")
        self.path_shot_lbl = os.path.join(output_root, "Dataset_2_Shots", "labels")

        for p in [self.path_gen_img, self.path_gen_lbl, self.path_shot_img, self.path_shot_lbl]:
            os.makedirs(p, exist_ok=True)

        self.stats_general = 0
        self.stats_shots = 0

    def run(self):
        video_files = glob.glob(os.path.join(self.input_folder, "*.mp4"))
        print(f"=== 🏭 DUAL DATASET FACTORY STARTED ===")
        print(f"   📂 Input: {len(video_files)} videos")
        print(f"   💾 General: {self.path_gen_img}")
        print(f"   💾 Shots:   {self.path_shot_img}")

        for video_path in video_files:
            try:
                self.process_video(video_path)
            except Exception as e:
                print(f"\n❌ Error processing {os.path.basename(video_path)}: {e}")
                continue

        print(f"\n✅ GENERATION COMPLETE!")
        print(f"   General Images: {self.stats_general}")
        print(f"   Shot Images:    {self.stats_shots}")

    def process_video(self, video_path):
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"\nProcessing: {video_name} ({total_frames} frames)")

        # Buffers
        frame_batch = []  # Resized frames for AI
        raw_batch = []  # Original frames for saving
        indices_batch = []  # Frame numbers

        # State History (Needed for Physics)
        self.ball_pos = []
        self.hoop_pos = []
        self.up = self.down = False

        pbar = tqdm(total=total_frames, unit="frames")
        frame_idx = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                if frame_batch: self.process_batch(frame_batch, raw_batch, indices_batch, video_name)
                break

            # AI Input (Resize for speed)
            ai_frame = cv2.resize(frame, (1280, 736))

            frame_batch.append(ai_frame)
            raw_batch.append(frame)
            indices_batch.append(frame_idx)

            if len(frame_batch) == BATCH_SIZE:
                self.process_batch(frame_batch, raw_batch, indices_batch, video_name)
                frame_batch = []
                raw_batch = []
                indices_batch = []

            frame_idx += 1
            pbar.update(1)

        cap.release()
        pbar.close()

    def process_batch(self, ai_frames, raw_frames, indices, video_name):
        # 1. INFERENCE (Every frame)
        results = self.model(ai_frames, stream=False, verbose=False, device='0', conf=CONF_RELAXED)

        for i, r in enumerate(results):
            current_idx = indices[i]

            # --- DATA EXTRACTION ---
            label_lines = []
            boxes = r.boxes.cpu().numpy()

            balls = []
            rings = []

            for box in boxes:
                x, y, w, h = box.xywhn[0]
                conf = box.conf[0]
                cls = int(box.cls[0])

                if cls == 0:
                    rings.append(
                        {'line': f"{cls} {x:.6f} {y:.6f} {w:.6f} {h:.6f}", 'conf': conf, 'pos': (x, y), 'size': (w, h)})
                elif cls == 1:
                    balls.append({'line': f"{cls} {x:.6f} {y:.6f} {w:.6f} {h:.6f}", 'conf': conf, 'pos': (x, y)})

            # --- PHYSICS UPDATE ---
            # 1. Update Hoop History
            current_rings = [r for r in rings if r['conf'] > CONF_STRICT]
            if current_rings:
                for r in current_rings:
                    c_px = (r['pos'][0] * 1280, r['pos'][1] * 736)
                    w_px, h_px = r['size'][0] * 1280, r['size'][1] * 736
                    self.hoop_pos.append((c_px, current_idx, w_px, h_px, r['conf']))

            # 2. Update Ball History (Best Ball Only)
            best_ball = None
            if balls:
                # Get highest confidence ball that makes sense
                candidates = []
                for b in balls:
                    b_px = (b['pos'][0] * 1280, b['pos'][1] * 736)
                    # Context Check
                    if b['conf'] > 0.50:
                        candidates.append(b)
                    elif self.hoop_pos and in_hoop_region(b_px, self.hoop_pos):
                        candidates.append(b)

                if candidates:
                    best_ball = max(candidates, key=lambda x: x['conf'])
                    b_px = (best_ball['pos'][0] * 1280, best_ball['pos'][1] * 736)
                    self.ball_pos.append((b_px, current_idx, 0, 0, best_ball['conf']))

            # 3. Clean History
            self.ball_pos = clean_ball_pos(self.ball_pos, current_idx)
            if self.hoop_pos: self.hoop_pos = clean_hoop_pos(self.hoop_pos)

            # 4. Check Shot State
            is_shooting = False
            if self.hoop_pos and self.ball_pos:
                if not self.up: self.up = detect_up(self.ball_pos, self.hoop_pos)
                if self.up: is_shooting = True  # Ball is going up!

                if self.up and not self.down: self.down = detect_down(self.ball_pos, self.hoop_pos)
                if self.down: is_shooting = True  # Ball is going down!

                # Reset logic (End of shot)
                if self.up and self.down:
                    # We keep is_shooting = True for this frame, then reset
                    if score(self.ball_pos, self.hoop_pos) or not score(self.ball_pos, self.hoop_pos):
                        self.up = self.down = False

            # --- SAVING LOGIC (The Dual Sorter) ---
            found_ring = len(current_rings) > 0
            found_ball = best_ball is not None

            # Prepare Label Text
            final_labels = []
            if found_ring:
                for r in current_rings: final_labels.append(r['line'])
            if found_ball:
                final_labels.append(best_ball['line'])

            # DECISION TREE
            save_target = None

            if found_ring and found_ball:
                if is_shooting:
                    # Condition: Active Shot -> Save frequently
                    if current_idx % SHOT_SAVE_RATE == 0:
                        save_target = "SHOT"
                else:
                    # Condition: Just dribbling -> Save rarely
                    if current_idx % GENERAL_SAVE_RATE == 0:
                        save_target = "GENERAL"

            # WRITE TO DISK
            if save_target:
                filename = f"{video_name}_{current_idx:06d}"

                if save_target == "SHOT":
                    img_path = os.path.join(self.path_shot_img, f"{filename}.jpg")
                    lbl_path = os.path.join(self.path_shot_lbl, f"{filename}.txt")
                    self.stats_shots += 1
                else:
                    img_path = os.path.join(self.path_gen_img, f"{filename}.jpg")
                    lbl_path = os.path.join(self.path_gen_lbl, f"{filename}.txt")
                    self.stats_general += 1

                # Save (Use raw frame for quality, but resize if you want smaller dataset size)
                # We save the AI-resized frame (1280x736) to match the normalized labels perfectly
                # and keep dataset size manageable.
                cv2.imwrite(img_path, ai_frames[i])

                with open(lbl_path, 'w') as f:
                    f.write('\n'.join(final_labels))


if __name__ == "__main__":
    DualDatasetGenerator(INPUT_FOLDER, OUTPUT_ROOT, MODEL_PATH).run()