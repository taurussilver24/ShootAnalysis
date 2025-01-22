import os
from ultralytics import YOLO
import cv2
import cvzone
import math
import numpy as np
import csv
from utils import score, detect_down, detect_up, in_hoop_region, clean_hoop_pos, clean_ball_pos

class ShotDetector:
    def __init__(self, model_path, video_path, video_name):
        # main.pyから作成されたYOLOモデルをロード - 相対パスに変更
        self.model = YOLO(model_path)
        self.class_names = ['Ring', 'Ball']

        # ビデオを使用 - テキストをビデオパスに置き換え
        self.cap = cv2.VideoCapture(video_path)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)  # フレーム毎秒

        self.ball_pos = []  # タプルの配列 ((x_pos, y_pos), フレームカウント, 幅, 高さ, 信頼度)
        self.hoop_pos = []  # タプルの配列 ((x_pos, y_pos), フレームカウント, 幅, 高さ, 信頼度)

        self.frame_count = 0
        self.frame = None

        self.makes = 0
        self.attempts = 0

        # ショットを検出するために使用（上部および下部の領域）
        self.up = False
        self.down = False
        self.peak = False
        self.up_frame = 0
        self.down_frame = 0

        # メイク/ミス後の緑と赤の色を使用
        self.fade_frames = 20
        self.fade_counter = 0
        self.overlay_color = (0, 0, 0)

        # CSVファイルをライトモードで開き、ヘッダーを書き込む
        results_dir = 'Results/' + video_name
        os.makedirs(results_dir, exist_ok=True)
        self.csv_file = open(results_dir + '/shot_results.csv', mode='w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(["Shot Taken", "Result", "Ball Coordinates",
                                  "Hoop Coordinates", "Current Score", "Video Timing (seconds)"])

        # ビデオの総時間を秒で計算
        self.total_time_seconds = self.total_frames / self.fps

        # ビデオとスライダーを表示するためのウィンドウを作成
        self.window_name = "MODEL: " + model_path + "  VIDEO: " + video_path
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
                    # ビデオの終了またはエラーが発生
                    break

            self.frame = cv2.resize(self.frame, (1280, 720))
            results = self.model(self.frame, stream=True)

            for r in results:
                boxes = r.boxes
                for box in boxes:
                    # バウンディングボックス
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    w, h = x2 - x1, y2 - y1

                    # 信頼度
                    conf = math.ceil((box.conf[0] * 100)) / 100

                    # 信頼度が0.75より大きい場合のみ続行
                    if conf > 0.75:
                        # クラス名
                        cls = int(box.cls[0])
                        current_class = self.class_names[cls]

                        center = (int(x1 + w / 2), int(y1 + h / 2))

                        # 異なるクラスの色を定義
                        if current_class == "Ball":
                            color = (0, 0, 255)  # ボールの赤
                        else:
                            color = (255, 0, 0)  # フープの青

                        # バウンディングボックスとラベルを描画
                        cv2.rectangle(self.frame, (x1, y1), (x2, y2), color, 1)
                        label = f"{current_class} {conf:.2f}"

                        # テキストサイズの判定
                        (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)

                        # レクタングルの座標
                        rect_x1 = x1
                        rect_y1 = y1 - text_height - 10  # テキストが正しく配置されていない場合はこの値を調整
                        rect_x2 = x1 + text_width
                        rect_y2 = y1

                        # レクタングルを描く
                        cv2.rectangle(self.frame, (rect_x1, rect_y1), (rect_x2, rect_y2), (255, 255, 255), cv2.FILLED)
                        cv2.putText(self.frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                        # 高信頼度またはフープ近くの場合にのみボールポイントを作成
                        if (current_class == "Ball" and conf > 0.3) or \
                                (in_hoop_region(center, self.hoop_pos) and conf > 0.15):
                            self.ball_pos.append((center, self.frame_count, w, h, conf))
                            cvzone.cornerRect(self.frame, (x1, y1, w, h))

                        # 高信頼度の場合にフープポイントを作成
                        if current_class == "Hoop" and conf > 0.3:
                            self.hoop_pos.append((center, self.frame_count, w, h, conf))
                            cvzone.cornerRect(self.frame, (x1, y1, w, h))

            self.clean_motion()
            self.shot_detection()
            self.frame_count += 1

            # 時間スライダー位置を更新
            current_time_seconds = self.frame_count / self.fps
            cv2.setTrackbarPos('Time (s)', self.window_name, int(current_time_seconds))

            cv2.imshow(self.window_name, self.frame)

            # 'q'がクリックされた場合に閉じる
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

    ShotDetector(model_path="models/" + args.model, video_path="HoopVids/Done_Requested/" + args.video,
                 video_name=args.video)
