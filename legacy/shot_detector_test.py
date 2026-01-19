import os
from ultralytics import YOLO
import cv2
import cvzone
import math
import numpy as np
import csv
from utils import score, detect_down, detect_up, in_hoop_region, clean_hoop_pos, clean_ball_pos

class ShotDetector:
    def __init__(self, model1_path, model2_path, video_path, video_name, model_name):
        # Load YOLO models for each class
        self.model_ball = YOLO(model1_path)  # Model for Ball (Class 2)
        self.model_ring = YOLO(model2_path)  # Model for Ring (Class 1)
        self.class_names = ['Ring', 'Ball']

        # Initialize video capture
        self.cap = cv2.VideoCapture(video_path)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)  # Frames per second

        self.ball_pos = []  # List of tuples: ((x_pos, y_pos), frame_count, width, height, confidence)
        self.hoop_pos = []  # List of tuples: ((x_pos, y_pos), frame_count, width, height, confidence)

        self.frame_count = 0
        self.frame = None

        self.makes = 0
        self.attempts = 0

        # Shot detection flags
        self.up = False
        self.down = False
        self.peak = False
        self.up_frame = 0
        self.down_frame = 0

        # Fade effect for makes/misses
        self.fade_frames = 20
        self.fade_counter = 0
        self.overlay_color = (0, 0, 0)

        # Create results directory and CSV file
        results_dir = 'Results/' + video_name
        os.makedirs(results_dir, exist_ok=True)
        self.csv_file = open(results_dir + '/' + model_name + '_shot_results.csv', mode='w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(["Shot Taken", "Result", "Ball Coordinates",
                                  "Hoop Coordinates", "Current Score", "Video Timing (seconds)"])

        # Calculate total video time in seconds
        self.total_time_seconds = self.total_frames / self.fps

        # Create window with trackbar
        self.window_name = "MODEL: " + model1_path + " & " + model2_path + "  VIDEO: " + video_path
        cv2.namedWindow(self.window_name)
        cv2.createTrackbar('Time (s)', self.window_name, 0, int(self.total_time_seconds), self.on_time_slider_change)
        self.paused = False

        self.run()

    def on_time_slider_change(self, pos):
        # 秒単位の時間をフレーム番号に変換
        frame_number = int(pos * self.fps)
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        self.frame_count = frame_number

    def run(self):
        while True:
            if not self.paused:
                ret, self.frame = self.cap.read()
                if not ret:
                    break  # End of video or error

            self.frame = cv2.resize(self.frame, (1280, 720))

            # Run inference for both models
            results_ring = self.model_ring(self.frame, stream=True, verbose=False)  # Detect Ring (Class 1)
            results_ball = self.model_ball(self.frame, stream=True, verbose=False)  # Detect Ball (Class 2)

            # Process Ring detections
            for r in results_ring:
                boxes = r.boxes
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    w, h = x2 - x1, y2 - y1
                    conf = math.ceil((box.conf[0] * 100)) / 100
                    cls = int(box.cls[0])

                    if conf > 0.55 and self.class_names[cls] == "Ring":
                        center = (int(x1 + w / 2), int(y1 + h / 2))
                        self.hoop_pos.append((center, self.frame_count, w, h, conf))
                        cv2.rectangle(self.frame, (x1, y1), (x2, y2), (255, 0, 0), 1)  # Blue for Ring
                        label = f"Ring {conf:.2f}"
                        cv2.putText(self.frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            # Process Ball detections
            for r in results_ball:
                boxes = r.boxes
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    w, h = x2 - x1, y2 - y1
                    conf = math.ceil((box.conf[0] * 100)) / 100
                    cls = int(box.cls[0])

                    if conf > 0.55 and self.class_names[cls] == "Ball":
                        center = (int(x1 + w / 2), int(y1 + h / 2))
                        self.ball_pos.append((center, self.frame_count, w, h, conf))
                        cv2.rectangle(self.frame, (x1, y1), (x2, y2), (0, 0, 255), 1)  # Red for Ball
                        label = f"Ball {conf:.2f}"
                        cv2.putText(self.frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            # Clean motion and detect shots
            self.clean_motion()
            self.shot_detection()
            self.frame_count += 1

            # Update time slider
            current_time_seconds = self.frame_count / self.fps
            cv2.setTrackbarPos('Time (s)', self.window_name, int(current_time_seconds))

            # Display frame
            cv2.imshow(self.window_name, self.frame)

            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord(' '):
                self.paused = not self.paused

        self.cap.release()
        cv2.destroyAllWindows()
        self.csv_file.close()

    def clean_motion(self):
        # ボール位置データをクリーンアップするが、サークルは描画しない
        self.ball_pos = clean_ball_pos(self.ball_pos, self.frame_count)

        # フープ位置データをクリーンアップし、現在のフープ中心を表示
        if len(self.hoop_pos) > 0:
            self.hoop_pos = clean_hoop_pos(self.hoop_pos)
            cv2.circle(self.frame, self.hoop_pos[-1][0], 2, (128, 128, 0), 2)

    def detect_peak(self, ball_pos):
        # ボールの軌道のピークを検出する関数
        if len(ball_pos) < 3:
            return False
        # ボールの現在位置が前後の位置よりも高い場合にピークとする
        return ball_pos[-2][0][1] > ball_pos[-3][0][1] and ball_pos[-2][0][1] > ball_pos[-1][0][1]

    def shot_detection(self):
        if len(self.hoop_pos) > 0 and len(self.ball_pos) > 0:
            # ボールが 'up' 領域にあるか検出
            if not self.up:
                self.up = detect_up(self.ball_pos, self.hoop_pos)
                if self.up:
                    self.up_frame = self.ball_pos[-1][1]

            # ボールが 'up' 領域にあり、その後 'down' 領域にあるか検出
            if self.up and not self.down:
                self.down = detect_down(self.ball_pos, self.hoop_pos)
                if self.down:
                    self.down_frame = self.ball_pos[-1][1]

            # ピークの検出（オプション）
            if self.up and not self.peak:
                self.peak = self.detect_peak(self.ball_pos)
                if self.peak:
                    self.peak_frame = self.ball_pos[-1][1]

            # ボールが 'up' 領域から 'down' 領域に移動した場合、ショットとみなす
            if self.up and self.down and self.up_frame < self.down_frame:
                self.attempts += 1
                self.up = False
                self.down = False

                # メイクの場合、緑のオーバーレイを表示
                if score(self.ball_pos, self.hoop_pos):
                    self.makes += 1
                    self.overlay_color = (0, 255, 0)
                    self.fade_counter = self.fade_frames
                    result = "Successful"
                # ミスの場合、赤のオーバーレイを表示
                else:
                    self.overlay_color = (0, 0, 255)
                    self.fade_counter = self.fade_frames
                    result = "Failed"

                # 結果とビデオタイミング（秒単位）を記録
                ball_center = self.ball_pos[-1][0]
                hoop_center = self.hoop_pos[-1][0]
                current_score = f"{self.makes} / {self.attempts}"
                video_timing_seconds = self.frame_count / self.fps
                print(f"Shot detected: {self.attempts}, Result: {result}")
                self.csv_writer.writerow([self.attempts, result, ball_center,
                                          hoop_center, current_score, video_timing_seconds])

    def display_score(self):
        # テキストを追加
        text = str(self.makes) + " / " + str(self.attempts)
        cv2.putText(self.frame, text, (50, 125), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 6)
        cv2.putText(self.frame, text, (50, 125), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 0), 3)

        # ショット後に色を徐々にフェードアウト
        if self.fade_counter > 0:
            alpha = 0.2 * (self.fade_counter / self.fade_frames)
            self.frame = cv2.addWeighted(self.frame, 1 - alpha, np.full_like(self.frame, self.overlay_color), alpha, 0)
            self.fade_counter -= 1

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="YOLO8を使用し、ボールとリングの検出")
    parser.add_argument('--modelB', type=str, default="RokkenV2.pt", help="YOLOモデル1のパス")
    parser.add_argument('--modelR', type=str, default="Rishit.pt", help="YOLOモデル2のパス")
    parser.add_argument('--video', type=str, default="HoopVids/Done_Requested/DNvsTW.mp4", help="動画のパス")
    args = parser.parse_args()

    ShotDetector(model1_path="models/" + args.modelB,model2_path= "models/" + args.modelR, video_path="HoopVids/Done_Requested/" + args.video,
                 video_name=args.video, model_name="Both")
