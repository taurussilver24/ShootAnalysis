import os
from ultralytics import YOLO
import cv2
import cvzone
import math
import numpy as np
import csv
from utils import score, detect_down, detect_up, in_hoop_region, clean_hoop_pos, clean_ball_pos

class ShotDetector:
    def __init__(self, ball_model_path, hoop_model_path, video_path, video_name):
        # Load YOLO models for ball and hoop
        self.ball_model = YOLO(ball_model_path)
        self.hoop_model = YOLO(hoop_model_path)
        self.class_names = ['Ring', 'Ball']

        # Video capture
        self.cap = cv2.VideoCapture(video_path)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)  # Frames per second

        self.ball_pos = []  # Array of tuples ((x_pos, y_pos), frame_count, width, height, confidence)
        self.hoop_pos = []  # Array of tuples ((x_pos, y_pos), frame_count, width, height, confidence)

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

        # Fade effect for make/miss
        self.fade_frames = 20
        self.fade_counter = 0
        self.overlay_color = (0, 0, 0)

        # CSV file for results
        results_dir = 'Results/' + video_name
        os.makedirs(results_dir, exist_ok=True)
        self.csv_file = open(results_dir + '/shot_results.csv', mode='w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(["Shot Taken", "Result", "Ball Coordinates",
                                  "Hoop Coordinates", "Current Score", "Video Timing (seconds)"])

        # Total video time in seconds
        self.total_time_seconds = self.total_frames / self.fps

        # Create window for video and slider
        self.window_name = "MODEL: Ball - " + ball_model_path + " Hoop - " + hoop_model_path + "  VIDEO: " + video_path
        cv2.namedWindow(self.window_name)
        cv2.createTrackbar('Time (s)', self.window_name, 0, int(self.total_time_seconds), self.on_time_slider_change)
        self.paused = False

        self.run()

    def on_time_slider_change(self, pos):
        # Convert time in seconds to frame number
        frame_number = int(pos * self.fps)
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        self.frame_count = frame_number

    def run(self):
        while True:
            if not self.paused:
                ret, self.frame = self.cap.read()
                if not ret:
                    # End of video or error
                    break

            self.frame = cv2.resize(self.frame, (1280, 720))

            # Detect ball using ball model
            ball_results = self.ball_model(self.frame, stream=True)
            for r in ball_results:
                boxes = r.boxes
                for box in boxes:
                    self.process_box(box, "Ball")

            # Detect hoop using hoop model
            hoop_results = self.hoop_model(self.frame, stream=True)
            for r in hoop_results:
                boxes = r.boxes
                for box in boxes:
                    self.process_box(box, "Ring")

            self.clean_motion()
            self.shot_detection()
            self.frame_count += 1

            # Update time slider position
            current_time_seconds = self.frame_count / self.fps
            cv2.setTrackbarPos('Time (s)', self.window_name, int(current_time_seconds))

            cv2.imshow(self.window_name, self.frame)

            # Close if 'q' is pressed
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord(' '):
                self.paused = not self.paused

        self.cap.release()
        cv2.destroyAllWindows()
        self.csv_file.close()

    def process_box(self, box, class_name):
        # Bounding box
        x1, y1, x2, y2 = box.xyxy[0]
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        w, h = x2 - x1, y2 - y1

        # Confidence
        conf = math.ceil((box.conf[0] * 100)) / 100

        # Only proceed if confidence is greater than 0.75
        if conf > 0.75:
            center = (int(x1 + w / 2), int(y1 + h / 2))

            # Define different colors for different classes
            if class_name == "Ball":
                color = (0, 0, 255)  # Red for ball
            else:
                color = (255, 0, 0)  # Blue for hoop

            # Draw bounding box and label
            cv2.rectangle(self.frame, (x1, y1), (x2, y2), color, 1)
            label = f"{class_name} {conf:.2f}"

            # Determine text size
            (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)

            # Rectangle coordinates
            rect_x1 = x1
            rect_y1 = y1 - text_height - 10  # Adjust this value if text is not placed correctly
            rect_x2 = x1 + text_width
            rect_y2 = y1

            # Draw rectangle
            cv2.rectangle(self.frame, (rect_x1, rect_y1), (rect_x2, rect_y2), (255, 255, 255), cv2.FILLED)
            cv2.putText(self.frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Create ball points only if high confidence or near hoop
            if (class_name == "Ball" and conf > 0.3) or \
                    (in_hoop_region(center, self.hoop_pos) and conf > 0.15):
                self.ball_pos.append((center, self.frame_count, w, h, conf))
                cvzone.cornerRect(self.frame, (x1, y1, w, h))

            # Create hoop points only if high confidence
            if class_name == "Ring" and conf > 0.3:
                self.hoop_pos.append((center, self.frame_count, w, h, conf))
                cvzone.cornerRect(self.frame, (x1, y1, w, h))

    def clean_motion(self):
        # Clean up ball position data
        self.ball_pos = clean_ball_pos(self.ball_pos, self.frame_count)

        # Clean up hoop position data and display current hoop center
        if len(self.hoop_pos) > 0:
            self.hoop_pos = clean_hoop_pos(self.hoop_pos)
            cv2.circle(self.frame, self.hoop_pos[-1][0], 2, (128, 128, 0), 2)

    def detect_peak(self, ball_pos):
        # Detect peak in ball trajectory
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
                self.csv_writer.writerow([self.attempts, result, ball_center,
                                          hoop_center, current_score, video_timing_seconds])

    def display_score(self):
        text = str(self.makes) + " / " + str(self.attempts)
        cv2.putText(self.frame, text, (50, 125), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 6)
        cv2.putText(self.frame, text, (50, 125), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 0), 3)

        if self.fade_counter > 0:
            alpha = 0.2 * (self.fade_counter / self.fade_frames)
            self.frame = cv2.addWeighted(self.frame, 1 - alpha, np.full_like(self.frame, self.overlay_color), alpha, 0)
            self.fade_counter -= 1

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Detect ball and hoop using YOLO models")
    parser.add_argument('--ball_model', type=str, default="models/RokkenV2.pt", help="Path to ball YOLO model")
    parser.add_argument('--hoop_model', type=str, default="models/Rishit.pt", help="Path to hoop YOLO model")
    parser.add_argument('--video', type=str, default="HoopVids/Done_Requested/DNvsTW.mp4", help="Path to video")
    args = parser.parse_args()

    ShotDetector(ball_model_path=args.ball_model, hoop_model_path=args.hoop_model,
                 video_path="HoopVids/Done_Requested/" + args.video, video_name=os.path.basename(args.video))