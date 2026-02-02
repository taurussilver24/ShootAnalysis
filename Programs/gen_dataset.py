# INPUT_FOLDER = r"C:\Users\Rishit\PycharmProjects\ShootAnalysis\HoopVids\NBA2k"  # Folder with your MP4s


import os
import cv2
import glob
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO
from utils import in_hoop_region, clean_hoop_pos

# ==============================================================================
# ⚙️ DATASET FACTORY SETTINGS
# ==============================================================================
INPUT_FOLDER = r"C:\Users\Rishit\PycharmProjects\ShootAnalysis\HoopVids\NBA2k"  # Folder with your MP4s

OUTPUT_DATASET = "NBA2K_Thesis_Dataset"
MODEL_PATH = "../models/Rishit.onnx"

SAVE_EVERY_N_FRAMES = 15
CONF_STRICT = 0.60
CONF_RELAXED = 0.15
BATCH_SIZE = 16


# ==============================================================================

class DatasetGenerator:
    def __init__(self, input_folder, output_root, model_path):
        self.input_folder = input_folder
        self.model = YOLO(model_path, task="detect")
        self.class_names = {0: 'Ring', 1: 'Ball'}

        self.images_dir = os.path.join(output_root, "images")
        self.labels_dir = os.path.join(output_root, "labels")
        os.makedirs(self.images_dir, exist_ok=True)
        os.makedirs(self.labels_dir, exist_ok=True)

        self.total_images_saved = 0

    def run(self):
        video_files = glob.glob(os.path.join(self.input_folder, "*.mp4"))
        print(f"=== 🏭 DATASET FACTORY STARTED ===")
        print(f"   📂 Input: {len(video_files)} videos found")
        print(f"   💾 Output: {self.images_dir}")

        for video_path in video_files:
            try:
                self.process_video(video_path)
            except Exception as e:
                print(f"\n❌ Error processing {os.path.basename(video_path)}: {e}")
                continue

        print(f"\n✅ GENERATION COMPLETE!")
        print(f"   Total Images: {self.total_images_saved}")

    def process_video(self, video_path):
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"\nProcessing: {video_name} ({total_frames} frames)")

        batch_frames = []
        batch_indices = []

        # Stores full tuple: (center_xy, frame_idx, width, height, conf)
        known_hoop_pos = []

        pbar = tqdm(total=total_frames, unit="frames")
        frame_idx = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                if batch_frames: self.process_batch(batch_frames, batch_indices, video_name, known_hoop_pos)
                break

            if frame_idx % SAVE_EVERY_N_FRAMES == 0:
                ai_frame = cv2.resize(frame, (1280, 736))
                batch_frames.append(ai_frame)
                batch_indices.append(frame_idx)

                if len(batch_frames) == BATCH_SIZE:
                    self.process_batch(batch_frames, batch_indices, video_name, known_hoop_pos)
                    batch_frames = []
                    batch_indices = []

            frame_idx += 1
            pbar.update(1)

        cap.release()
        pbar.close()

    def process_batch(self, frames, indices, video_name, known_hoop_pos):
        results = self.model(frames, stream=False, verbose=False, device='0', conf=CONF_RELAXED)

        for i, r in enumerate(results):
            frame_idx = indices[i]
            frame_img = frames[i]

            label_lines = []
            boxes = r.boxes.cpu().numpy()

            balls = []
            rings = []

            for box in boxes:
                x, y, w, h = box.xywhn[0]
                conf = box.conf[0]
                cls = int(box.cls[0])

                # Store normalized sizes for label, but pixels for logic logic
                if cls == 0:  # Ring
                    rings.append({
                        'line': f"{cls} {x:.6f} {y:.6f} {w:.6f} {h:.6f}",
                        'conf': conf,
                        'pos': (x, y),
                        'size': (w, h)  # Needed for pixel conversion
                    })
                elif cls == 1:  # Ball
                    balls.append({
                        'line': f"{cls} {x:.6f} {y:.6f} {w:.6f} {h:.6f}",
                        'conf': conf,
                        'pos': (x, y)
                    })

            # --- LOGIC FIX ---

            # 1. Update Known Ring Positions (With WIDTH/HEIGHT now!)
            current_rings = [r for r in rings if r['conf'] > CONF_STRICT]
            if current_rings:
                for r in current_rings:
                    label_lines.append(r['line'])

                    # Convert normalized to pixels for utils.py compatibility
                    center_px = (r['pos'][0] * 1280, r['pos'][1] * 736)
                    w_px = r['size'][0] * 1280
                    h_px = r['size'][1] * 736

                    # APPEND FULL TUPLE (Fixes the IndexError)
                    # Structure: ( (x,y), frame, width, height, conf )
                    known_hoop_pos.append((center_px, frame_idx, w_px, h_px, r['conf']))

            # 2. Filter Balls
            for b in balls:
                is_high_conf = b['conf'] > 0.50

                b_pixel = (b['pos'][0] * 1280, b['pos'][1] * 736)

                # Safe check: in_hoop_region usually needs a populated list
                is_near_hoop = False
                if known_hoop_pos:
                    try:
                        is_near_hoop = in_hoop_region(b_pixel, known_hoop_pos)
                    except Exception:
                        is_near_hoop = False  # Fallback if utils logic fails strictly

                if is_high_conf or (b['conf'] > CONF_RELAXED and is_near_hoop):
                    label_lines.append(b['line'])

            # --- SAVE ---
            if label_lines:
                filename = f"{video_name}_{frame_idx:06d}"
                img_path = os.path.join(self.images_dir, f"{filename}.jpg")
                cv2.imwrite(img_path, frame_img)

                lbl_path = os.path.join(self.labels_dir, f"{filename}.txt")
                with open(lbl_path, 'w') as f:
                    f.write('\n'.join(label_lines))

                self.total_images_saved += 1


if __name__ == "__main__":
    DatasetGenerator(INPUT_FOLDER, OUTPUT_DATASET, MODEL_PATH).run()