# render_video.py (With Screen Flash on Shot Outcome)
import os
import cv2
import pandas as pd
import argparse
from tqdm import tqdm
import numpy as np  # <--- Import numpy for the overlay


def render_video_with_overlays(video_path, csv_path, output_path, preview=False):
    print("Loading analysis data...")
    try:
        df = pd.read_csv(csv_path)
        frames_with_data = {frame: group for frame, group in df.groupby('frame_number')}
    except FileNotFoundError:
        print(f"Error: Analysis data file not found at {csv_path}")
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file at {video_path}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    print(f"Starting video rendering process... Output will be saved to {output_path}")

    if preview:
        cv2.namedWindow("Renderer Preview", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Renderer Preview", 1280, 720)

    # --- ADDED: State variables for the flash effect ---
    flash_counter = 0
    flash_color = (0, 0, 0)  # Default to black (no flash)
    FADE_FRAMES = 20  # How many frames the flash will last

    for frame_num in tqdm(range(total_frames), desc="Rendering Video"):
        ret, frame = cap.read()
        if not ret:
            break

        if frame_num in frames_with_data:
            frame_data = frames_with_data[frame_num]
            shot_info_drawn = False
            for _, row in frame_data.iterrows():
                if pd.notna(row['class_name']):
                    x1, y1, x2, y2 = int(row['x1']), int(row['y1']), int(row['x2']), int(row['y2'])
                    label = f"{row['class_name']} {row['confidence']:.2f}"
                    color = (0, 0, 255) if row['class_name'] == 'Ball' else (255, 0, 0)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

                if not shot_info_drawn and pd.notna(row['shot_attempt_id']) and row['shot_attempt_id'] != '':
                    shot_result = row['shot_result']
                    result_text = f"Shot: {shot_result}"
                    score_text = f"Score: {row['current_score']}"

                    font, font_scale, thickness = cv2.FONT_HERSHEY_SIMPLEX, 2, 3
                    text_color = (0, 255, 255)
                    bg_color = (0, 0, 0)
                    cv2.putText(frame, result_text, (50, 100), font, font_scale, bg_color, thickness + 2, cv2.LINE_AA)
                    cv2.putText(frame, result_text, (50, 100), font, font_scale, text_color, thickness, cv2.LINE_AA)
                    cv2.putText(frame, score_text, (50, 180), font, font_scale, bg_color, thickness + 2, cv2.LINE_AA)
                    cv2.putText(frame, score_text, (50, 180), font, font_scale, text_color, thickness, cv2.LINE_AA)

                    # --- ADDED: Trigger the flash effect ---
                    if shot_result == "Successful":
                        flash_color = (0, 255, 0)  # Green for make
                    else:
                        flash_color = (0, 0, 255)  # Red for miss
                    flash_counter = FADE_FRAMES  # Start the countdown

                    shot_info_drawn = True

        # --- ADDED: Apply the flash overlay if the counter is active ---
        if flash_counter > 0:
            # Create a solid color image of the same size as the frame
            overlay = np.full(frame.shape, flash_color, dtype=np.uint8)

            # Calculate alpha for a fade-out effect (starts at 0.3 transparency)
            alpha = 0.3 * (flash_counter / FADE_FRAMES)

            # Blend the frame with the overlay
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

            # Decrement the flash counter
            flash_counter -= 1

        # Write the (potentially flashed) frame to the output video
        out.write(frame)

        if preview:
            cv2.imshow("Renderer Preview", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    print("Rendering complete.")
    cap.release()
    out.release()
    if preview:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Video Renderer with Analysis Overlays.")
    parser.add_argument('--video', type=str, required=True, help="Path to the original input video.")
    parser.add_argument('--data', type=str, required=True, help="Path to the analysis CSV data file.")
    parser.add_argument('--output', type=str, required=True, help="Path for the output video file.")
    parser.add_argument('--preview', action='store_true', help="Show a preview window while rendering (slower).")
    args = parser.parse_args()

    render_video_with_overlays(args.video, args.data, args.output, args.preview)