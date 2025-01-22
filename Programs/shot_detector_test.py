import os
from ultralytics import YOLO
import cv2
import cvzone
import math
import numpy as np
import csv
from utils import score, detect_down, detect_up, in_hoop_region, clean_hoop_pos, clean_ball_pos


class ShotDetector:
    def __init__(self, ball_model_path,hoop_model_path, video_path, video_name):
        # Load separate YOLO models for ball and hoop
        self.ball_model = YOLO(ball_model_path)
        self.hoop_model = YOLO(hoop_model_path)
        self.class_names_ball = ['Ball']
        self.class_names_hoop = ['Hoop']

        # Video initialization (same as before)
        self.cap = cv2.VideoCapture(video_path)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)

        # Initialization of detection results
        self.ball_pos = []
        self.hoop_pos = []
        self.frame_count = 0
        self.frame = None
        self.makes = 0
        self.attempts = 0

        # Detection states
        self.up = False
        self.down = False
        self.peak = False
        self.up_frame = 0
        self.down_frame = 0

        # CSV logging setup
        results_dir = 'Results/' + video_name
        os.makedirs(results_dir, exist_ok=True)
        self.csv_file = open(results_dir + '/shot_results.csv', mode='w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(["Shot Taken", "Result", "Ball Coordinates",
                                  "Hoop Coordinates", "Current Score", "Video Timing (seconds)"])

        # Video timing and display setup
        self.total_time_seconds = self.total_frames / self.fps
        self.window_name = "BALL MODEL: " + ball_model_path + "  HOOP MODEL: " + hoop_model_path + "  VIDEO: " + video_path
        cv2.namedWindow(self.window_name)
        cv2.createTrackbar('Time (s)', self.window_name, 0, int(self.total_time_seconds), self.on_time_slider_change)
        self.paused = False

        self.run()

    def on_time_slider_change(self, pos):
        frame_number = int(pos * self.fps)
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        self.frame_count = frame_number

    def run(self):
        while True:
            if not self.paused:
                ret, self.frame = self.cap.read()
                if not ret:
                    break

            self.frame = cv2.resize(self.frame, (1280, 720))

            # Run detection using the ball model
            ball_results = self.ball_model(self.frame, stream=True)
            for r in ball_results:
                boxes = r.boxes
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    w, h = x2 - x1, y2 - y1
                    conf = math.ceil((box.conf[0] * 100)) / 100

                    if conf > 0.75:
                        cls = int(box.cls[0])
                        if cls >= len(self.class_names_ball):
                            print(f"Invalid class ID {cls} for ball model.")
                            continue
                        current_class = self.class_names_ball[cls]
                        center = (x1 + w // 2, y1 + h // 2)

                        # Draw for ball with model label
                        model_label = "Ball Model"
                        cv2.rectangle(self.frame, (x1, y1), (x2, y2), (0, 0, 255), 1)
                        label = f"{model_label}: {current_class} {conf:.2f}"
                        cv2.putText(self.frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                        self.ball_pos.append((center, self.frame_count, w, h, conf))

            # Run detection using the hoop model
            hoop_results = self.hoop_model(self.frame, stream=True)
            for r in hoop_results:
                boxes = r.boxes
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    w, h = x2 - x1, y2 - y1
                    conf = math.ceil((box.conf[0] * 100)) / 100

                    if conf > 0.75:
                        cls = int(box.cls[0])
                        if cls >= len(self.class_names_hoop):
                            print(f"Invalid class ID {cls} for hoop model.")
                            continue
                        current_class = self.class_names_hoop[cls]
                        center = (x1 + w // 2, y1 + h // 2)

                        # Draw for hoop with model label
                        model_label = "Hoop Model"
                        cv2.rectangle(self.frame, (x1, y1), (x2, y2), (255, 0, 0), 1)
                        label = f"{model_label}: {current_class} {conf:.2f}"
                        cv2.putText(self.frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                        self.hoop_pos.append((center, self.frame_count, w, h, conf))

            self.clean_motion()
            self.shot_detection()
            self.frame_count += 1
            current_time_seconds = self.frame_count / self.fps
            cv2.setTrackbarPos('Time (s)', self.window_name, int(current_time_seconds))
            cv2.imshow(self.window_name, self.frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord(' '):
                self.paused = not self.paused

        self.cap.release()
        cv2.destroyAllWindows()
        self.csv_file.close()

    # (Rest of the methods remain unchanged)

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
    parser.add_argument('--model', type=str, default="Yolo-Weights/best6.pt", help="YOLOモデルのパス")
    parser.add_argument('--video', type=str, default="HoopVids/Done_Requested/DNvsTW.mp4", help="動画のパス")
    args = parser.parse_args()

    ShotDetector(ball_model_path= "Yolo-Weights/" + args.model ,hoop_model_path="Yolo-Weights/bestRokken_8m.pt" , video_path="HoopVids/Done_Requested/" + args.video,
                 video_name=args.video)
