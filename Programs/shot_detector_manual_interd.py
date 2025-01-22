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
        # Load YOLO model
        self.model = YOLO(model_path)
        self.class_names = ['Ball', 'Hoop']

        # Use video
        self.cap = cv2.VideoCapture(video_path)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)  # Frames per second

        self.ball_pos = []  # Array of tuples ((x_pos, y_pos), frame count, width, height, confidence)
        self.hoop_pos = []  # Array of tuples ((x_pos, y_pos), frame count, width, height, confidence)

        self.frame_count = 0
        self.frame = None

        self.makes = 0
        self.attempts = 0

        # Used to detect shots (upper and lower regions)
        self.up = False
        self.down = False
        self.up_frame = 0
        self.down_frame = 0

        # Use green and red colors after make/miss
        self.fade_frames = 20
        self.fade_counter = 0
        self.overlay_color = (0, 0, 0)

        # Open CSV file in write mode and write the header
        results_dir = 'Results/' + video_name
        os.makedirs(results_dir, exist_ok=True)
        self.csv_file = open('Results/' + video_name + '/shot_results_ground.csv', mode='w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(["Shot Taken", "Result", "Ball Coordinates", "Hoop Coordinates", "Current Score", "Video Timing (seconds)"])

        # Calculate the total time of the video in seconds
        self.total_time_seconds = self.total_frames / self.fps

        # Create window to display video and slider
        self.window_name = "MODEL: " + model_path + "  VIDEO: " + video_path
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
                    # End of video or error occurred
                    break

            self.frame = cv2.resize(self.frame, (1280, 720))
            results = self.model(self.frame, stream=True)

            for r in results:
                boxes = r.boxes
                for box in boxes:
                    # Bounding box
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    w, h = x2 - x1, y2 - y1

                    # Confidence
                    conf = math.ceil((box.conf[0] * 100)) / 100

                    # Continue only if confidence is greater than 0.75
                    if conf > 0.75:
                        # Class name
                        cls = int(box.cls[0])
                        current_class = self.class_names[cls]

                        center = (int(x1 + w / 2), int(y1 + h / 2))

                        # Define colors for different classes
                        if current_class == "Ball":
                            color = (0, 0, 255)  # Red for ball
                        else:
                            color = (255, 0, 0)  # Blue for hoop

                        # Draw bounding box and label
                        cv2.rectangle(self.frame, (x1, y1), (x2, y2), color, 1)
                        label = f"{current_class} {conf:.2f}"

                        # Determine text size
                        (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)

                        # Rectangle coordinates
                        rect_x1 = x1
                        rect_y1 = y1 - text_height - 10  # Adjust this value if the text is not placed correctly
                        rect_x2 = x1 + text_width
                        rect_y2 = y1

                        # Draw rectangle
                        cv2.rectangle(self.frame, (rect_x1, rect_y1), (rect_x2, rect_y2), (255, 255, 255), cv2.FILLED)
                        cv2.putText(self.frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                        # Create ball points only if confidence is high or near hoop
                        if (current_class == "Hoop" and conf > 0.7) or (in_hoop_region(center, self.hoop_pos) and conf > 0.15):
                            self.ball_pos.append((center, self.frame_count, w, h, conf))
                            cvzone.cornerRect(self.frame, (x1, y1, w, h))

                        # Create hoop points only if confidence is high
                        if current_class == "Ball" and conf > 0.7:
                            self.hoop_pos.append((center, self.frame_count, w, h, conf))
                            cvzone.cornerRect(self.frame, (x1, y1, w, h))

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
            elif key == ord('s'):
                self.add_csv_entry("Successful")
            elif key == ord('f'):
                self.add_csv_entry("Failed")

        self.cap.release()
        cv2.destroyAllWindows()
        self.csv_file.close()

    def clean_motion(self):
        # Clean up ball position data but do not draw circles
        self.ball_pos = clean_ball_pos(self.ball_pos, self.frame_count)

        # Clean up hoop position data and display current hoop center
        if len(self.hoop_pos) > 1:
            self.hoop_pos = clean_hoop_pos(self.hoop_pos)
            cv2.circle(self.frame, self.hoop_pos[-1][0], 2, (128, 128, 0), 2)

    def shot_detection(self):
        if len(self.hoop_pos) > 0 and len(self.ball_pos) > 0:
            # Detect if ball is in 'up' and 'down' regions - ball can only be in 'down' after being in 'up'
            if not self.up:
                self.up = detect_up(self.ball_pos, self.hoop_pos)
                if self.up:
                    self.up_frame = self.ball_pos[-1][1]

            if self.up and not self.down:
                self.down = detect_down(self.ball_pos, self.hoop_pos)
                if self.down:
                    self.down_frame = self.ball_pos[-1][1]

            # If ball moves from 'up' to 'down' in that order, increase attempts and reset
            if self.frame_count % 10 == 0:
                if self.up and self.down and self.up_frame < self.down_frame:
                    self.attempts += 1
                    self.up = False
                    self.down = False

                    # Display green overlay if make
                    if score(self.ball_pos, self.hoop_pos):
                        self.makes += 1
                        self.overlay_color = (0, 255, 0)
                        self.fade_counter = self.fade_frames
                    # Display red overlay if miss
                    else:
                        self.overlay_color = (0, 0, 255)
                        self.fade_counter = self.fade_frames

    def add_csv_entry(self, result):
        self.attempts += 1
        if result == "Successful":
            self.makes += 1
            self.overlay_color = (0, 255, 0)
        else:
            self.overlay_color = (0, 0, 255)
        self.fade_counter = self.fade_frames

        ball_center = self.ball_pos[-1][0] if self.ball_pos else (0, 0)
        hoop_center = self.hoop_pos[-1][0] if self.hoop_pos else (0, 0)
        current_score = f"{self.makes} / {self.attempts}"
        video_timing_seconds = self.frame_count / self.fps
        self.csv_writer.writerow([self.attempts, result, ball_center, hoop_center, current_score, video_timing_seconds])

    def display_score(self):
        # Add text
        text = str(self.makes) + " / " + str(self.attempts)
        cv2.putText(self.frame, text, (50, 125), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 6)
        cv2.putText(self.frame, text, (50, 125), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 0), 3)

        # Gradually fade out the color after a shot
        if self.fade_counter > 0:
            alpha = 0.2 * (self.fade_counter / self.fade_frames)
            self.frame = cv2.addWeighted(self.frame, 1 - alpha, np.full_like(self.frame, self.overlay_color), alpha, 0)
            self.fade_counter -= 1


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Detect ball and hoop using YOLO8")
    parser.add_argument('--model', type=str, default="Yolo-Weights/best6.pt", help="Path to YOLO model")
    parser.add_argument('--video', type=str, default="HoopVids/DNvsTW.mp4", help="Path to video")
    args = parser.parse_args()

    ShotDetector(model_path="Yolo-Weights/" + args.model, video_path="HoopVids/" + args.video, video_name=args.video)
