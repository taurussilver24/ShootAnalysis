import os
from ultralytics import YOLO
import cv2
import mss
import cvzone
import math
import numpy as np
import csv
from utils import score, detect_down, detect_up, in_hoop_region, clean_hoop_pos, clean_ball_pos


class ShotDetector:
    def __init__(self, model_path, video_name, model_name):
        # Load YOLO model
        self.model = YOLO(model_path)
        self.class_names = ['Ring', 'Ball']

        # Setup screen capture
        self.sct = mss.mss()
        # Define the screen region you want to capture (adjust these coordinates)
        self.monitor = {"top": 100, "left": 100, "width": 2560, "height": 1440}

        self.fps = 30  # approximate FPS for timing (screen capture might vary)
        self.frame_count = 0
        self.frame = None

        self.ball_pos = []  # ((x, y), frame_count, w, h, conf)
        self.hoop_pos = []  # ((x, y), frame_count, w, h, conf)

        self.makes = 0
        self.attempts = 0

        # Shot detection flags and frame tracking
        self.up = False
        self.down = False
        self.peak = False
        self.up_frame = 0
        self.down_frame = 0

        # For fading overlay colors on shot results
        self.fade_frames = 20
        self.fade_counter = 0
        self.overlay_color = (0, 0, 0)

        # Setup results folder & CSV output
        results_dir = 'Results/' + video_name
        os.makedirs(results_dir, exist_ok=True)
        self.csv_file = open(results_dir + '/' + model_name + '_shot_results.csv', mode='w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(["Shot Taken", "Result", "Ball Coordinates",
                                  "Hoop Coordinates", "Current Score", "Video Timing (seconds)"])

        # Setup display window and trackbar for time slider (though without video, this might be limited)
        self.window_name = f"MODEL: {model_path}  Screen Capture"
        cv2.namedWindow(self.window_name)
        cv2.createTrackbar('Pause', self.window_name, 0, 1, self.on_pause_trackbar_change)
        self.paused = False

        self.run()

    def on_pause_trackbar_change(self, pos):
        self.paused = bool(pos)

    def run(self):
        while True:
            if not self.paused:
                try:
                    screenshot = np.array(self.sct.grab(self.monitor))
                    self.frame = cv2.cvtColor(screenshot, cv2.COLOR_BGRA2BGR)
                except Exception as e:
                    print("Screen capture failed:", e)
                    break  # Exit the loop on failure

                self.frame = cv2.resize(self.frame, (1280, 720))

                results = self.model(self.frame, stream=True, verbose=False)

                for r in results:
                    boxes = r.boxes
                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0]
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                        w, h = x2 - x1, y2 - y1
                        conf = math.ceil((box.conf[0] * 100)) / 100
                        if conf > 0.55:
                            cls = int(box.cls[0])
                            current_class = self.class_names[cls]
                            center = (int(x1 + w / 2), int(y1 + h / 2))

                            color = (0, 0, 255) if current_class == "Ball" else (255, 0, 0)

                            cv2.rectangle(self.frame, (x1, y1), (x2, y2), color, 1)
                            label = f"{current_class} {conf:.2f}"
                            (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                                                                  2)
                            rect_x1, rect_y1 = x1, y1 - text_height - 10
                            rect_x2, rect_y2 = x1 + text_width, y1
                            cv2.rectangle(self.frame, (rect_x1, rect_y1), (rect_x2, rect_y2), (255, 255, 255),
                                          cv2.FILLED)
                            cv2.putText(self.frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                            if (current_class == "Ball" and conf > 0.3) or (
                                    in_hoop_region(center, self.hoop_pos) and conf > 0.15):
                                self.ball_pos.append((center, self.frame_count, w, h, conf))
                                cvzone.cornerRect(self.frame, (x1, y1, w, h))

                            if current_class == "Ring" and conf > 0.3:
                                self.hoop_pos.append((center, self.frame_count, w, h, conf))
                                cvzone.cornerRect(self.frame, (x1, y1, w, h))

                self.clean_motion()
                self.shot_detection()
                self.display_score()

                self.frame_count += 1

            cv2.imshow(self.window_name, self.frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord(' '):
                self.paused = not self.paused
                cv2.setTrackbarPos('Pause', self.window_name, int(self.paused))

        cv2.destroyAllWindows()
        self.csv_file.close()

    def clean_motion(self):
        self.ball_pos = clean_ball_pos(self.ball_pos, self.frame_count)

        if len(self.hoop_pos) > 0:
            self.hoop_pos = clean_hoop_pos(self.hoop_pos)
            cv2.circle(self.frame, self.hoop_pos[-1][0], 2, (128, 128, 0), 2)

    def detect_peak(self, ball_pos):
        if len(ball_pos) < 3:
            return False
        return ball_pos[-2][0][1] > ball_pos[-3][0][1] and ball_pos[-2][0][1] > ball_pos[-1][0][1]

    def shot_detection(self):
        if len(self.hoop_pos) > 0 and len(self.ball_pos) > 0:
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
                self.up = False
                self.down = False

                if score(self.ball_pos, self.hoop_pos):
                    self.makes += 1
                    self.overlay_color = (0, 255, 0)
                    self.fade_counter = self.fade_frames
                    result = "Successful"
                else:
                    self.overlay_color = (0, 0, 255)
                    self.fade_counter = self.fade_frames
                    result = "Failed"

                ball_center = self.ball_pos[-1][0]
                hoop_center = self.hoop_pos[-1][0]
                current_score = f"{self.makes} / {self.attempts}"
                video_timing_seconds = self.frame_count / self.fps
                print(f"Shot detected: {self.attempts}, Result: {result}")
                self.csv_writer.writerow([self.attempts, result, ball_center,
                                          hoop_center, current_score, video_timing_seconds])

    def display_score(self):
        text = f"{self.makes} / {self.attempts}"
        cv2.putText(self.frame, text, (50, 125), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 6)
        cv2.putText(self.frame, text, (50, 125), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 0), 3)

        if self.fade_counter > 0:
            alpha = 0.2 * (self.fade_counter / self.fade_frames)
            overlay = np.full_like(self.frame, self.overlay_color)
            self.frame = cv2.addWeighted(self.frame, 1 - alpha, overlay, alpha, 0)
            self.fade_counter -= 1


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="YOLO-based basketball shot detector with screen capture")
    parser.add_argument('--model', type=str, default="models/Rishit.pt", help="YOLO model path")
    parser.add_argument('--name', type=str, default="NBA2K25.exe", help="Session name for results folder")
    args = parser.parse_args()
    model_path = "models/" + args.model

    ShotDetector(model_path=args.model, video_name=args.name,
                 model_name=os.path.splitext(os.path.basename(args.model))[0])
