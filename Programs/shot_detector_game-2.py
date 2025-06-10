import os
import cv2
import dxcam
import cvzone
import numpy as np
import csv
from ultralytics import YOLO
from utils import score, detect_down, detect_up, in_hoop_region, clean_hoop_pos, clean_ball_pos


class ShotDetector:
    def __init__(self, model_path, video_name, model_name):
        # Model initialization with accuracy-focused settings
        self.model = YOLO(model_path)
        self.model.fuse()
        self.class_names = ['Ring', 'Ball']

        # Screen capture setup - ensure native 720p capture
        self.target_width, self.target_height = 1280, 720
        self.capture_region = (0, 0, self.target_width, self.target_height)
        self.camera = dxcam.create(
            output_color="BGR",
            region=self.capture_region,
            output_idx=0  # Primary monitor
        )
        self.camera.start(target_fps=60, video_mode=True)

        # Performance tracking
        self.fps = 60
        self.frame_count = 0
        self.last_time = cv2.getTickCount()

        # Detection buffers
        self.ball_pos = []
        self.hoop_pos = []

        # Shot tracking
        self.makes = 0
        self.attempts = 0
        self.up = False
        self.down = False
        self.peak = False
        self.up_frame = 0
        self.down_frame = 0

        # UI elements
        self.fade_frames = 20
        self.fade_counter = 0
        self.overlay_color = (0, 0, 0)

        # Results logging
        results_dir = os.path.join('Results', video_name)
        os.makedirs(results_dir, exist_ok=True)
        csv_path = os.path.join(results_dir, f'{model_name}_shot_results.csv')
        self.csv_file = open(csv_path, mode='w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow([
            "Shot Taken", "Result", "Ball Coordinates",
            "Hoop Coordinates", "Current Score", "Video Timing (seconds)"
        ])

        # Window setup
        self.window_name = f"MODEL: {os.path.basename(model_path)}  Screen Capture"
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.createTrackbar('Pause', self.window_name, 0, 1, self.on_pause_trackbar_change)
        self.paused = False

        # Start main loop
        self.run()

    def on_pause_trackbar_change(self, pos):
        self.paused = bool(pos)

    def run(self):
        while True:
            if not self.paused:
                try:
                    frame = self.camera.get_latest_frame()
                    if frame is None:
                        continue

                    # Verify and enforce 720p resolution
                    if frame.shape[0] != self.target_height or frame.shape[1] != self.target_width:
                        frame = cv2.resize(frame, (self.target_width, self.target_height))

                    # Process frame at full resolution
                    self.process_frame(frame)

                    # Performance monitoring
                    current_time = cv2.getTickCount()
                    time_elapsed = (current_time - self.last_time) / cv2.getTickFrequency()
                    fps = 1.0 / time_elapsed
                    self.last_time = current_time

                    # Display FPS for debugging
                    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                except Exception as e:
                    print(f"Error: {e}")
                    break

                # Display frame
                cv2.imshow(self.window_name, frame)

            # Key handling
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord(' '):
                self.paused = not self.paused
                cv2.setTrackbarPos('Pause', self.window_name, int(self.paused))

        # Cleanup
        cv2.destroyAllWindows()
        self.csv_file.close()
        self.camera.stop()

    def process_frame(self, frame):
        # Run detection at FULL 720p resolution
        results = self.model(
            frame,  # Feed the full resolution frame
            stream=True,
            verbose=False,
            imgsz=(720, 1280),  # Explicit 720p input size
            half=True,  # FP16 acceleration if available
            device='0',  # Use GPU
            conf=0.75  # Slightly lower confidence threshold for better detection
        )

        # Process detections
        for r in results:
            boxes = r.boxes.cpu().numpy()
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)  # No scaling needed
                w, h = x2 - x1, y2 - y1
                conf = round(box.conf[0].item(), 2)

                if conf > 0.75:
                    cls = int(box.cls[0])
                    current_class = self.class_names[cls]
                    center = (x1 + w // 2, y1 + h // 2)
                    color = (0, 0, 255) if current_class == "Ball" else (255, 0, 0)

                    # Draw bounding box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)

                    # Draw label
                    label = f"{current_class} {conf:.2f}"
                    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    cv2.rectangle(frame, (x1, y1 - th - 5), (x1 + tw, y1), color, -1)
                    cv2.putText(frame, label, (x1, y1 - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                    # Track positions
                    if (current_class == "Ball" and conf > 0.75) or (
                            in_hoop_region(center, self.hoop_pos) and conf > 0.15):
                        self.ball_pos.append((center, self.frame_count, w, h, conf))
                        cvzone.cornerRect(frame, (x1, y1, w, h), colorR=color)

                    if current_class == "Ring" and conf > 0.75:
                        self.hoop_pos.append((center, self.frame_count, w, h, conf))
                        cvzone.cornerRect(frame, (x1, y1, w, h), colorR=color)

        # Clean and analyze motion
        self.clean_motion(frame)
        self.shot_detection()
        self.display_score(frame)
        self.frame_count += 1

    def clean_motion(self, frame):
        self.ball_pos = clean_ball_pos(self.ball_pos, self.frame_count)
        if self.hoop_pos:
            self.hoop_pos = clean_hoop_pos(self.hoop_pos)
            cv2.circle(frame, self.hoop_pos[-1][0], 2, (128, 128, 0), 2)

    def detect_peak(self, ball_pos):
        if len(ball_pos) < 3:
            return False
        return ball_pos[-2][0][1] > ball_pos[-3][0][1] and ball_pos[-2][0][1] > ball_pos[-1][0][1]

    def shot_detection(self):
        if self.hoop_pos and self.ball_pos:
            if not self.up:
                self.up = detect_up(self.ball_pos, self.hoop_pos)
                if self.up:
                    self.up_frame = self.ball_pos[-1][1]

            if self.up and not self.down:
                self.down = detect_down(self.ball_pos, self.hoop_pos)
                if self.down:
                    self.down_frame = self.ball_pos[-1][1]

            if self.up and not self.peak:
                self.peak = self.detect_peak(self.ball_pos)
                if self.peak:
                    self.peak_frame = self.ball_pos[-1][1]

            if self.up and self.down and self.up_frame < self.down_frame:
                self.attempts += 1
                self.up = self.down = self.peak = False

                if score(self.ball_pos, self.hoop_pos):
                    self.makes += 1
                    self.overlay_color = (0, 255, 0)
                    result = "Successful"
                else:
                    self.overlay_color = (0, 0, 255)
                    result = "Failed"

                self.fade_counter = self.fade_frames
                ball_center = self.ball_pos[-1][0]
                hoop_center = self.hoop_pos[-1][0]
                score_text = f"{self.makes} / {self.attempts}"
                timestamp = self.frame_count / self.fps
                print(f"Shot detected: {self.attempts}, Result: {result}")
                self.csv_writer.writerow([
                    self.attempts, result, ball_center,
                    hoop_center, score_text, timestamp
                ])

    def display_score(self, frame):
        text = f"{self.makes} / {self.attempts}"
        cv2.putText(frame, text, (50, 125),
                    cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 6)
        cv2.putText(frame, text, (50, 125),
                    cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 0), 3)

        if self.fade_counter > 0:
            alpha = 0.2 * (self.fade_counter / self.fade_frames)
            overlay = np.full_like(frame, self.overlay_color)
            frame = cv2.addWeighted(frame, 1 - alpha, overlay, alpha, 0)
            self.fade_counter -= 1


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Optimized basketball shot detector")
    parser.add_argument('--model', type=str, default="models/Rishit.pt",
                        help="YOLO model path")
    parser.add_argument('--name', type=str, default="NBA2K25.exe",
                        help="Session name for results folder")
    args = parser.parse_args()

    detector = ShotDetector(
        model_path=args.model,
        video_name=args.name,
        model_name=os.path.splitext(os.path.basename(args.model))[0]
    )