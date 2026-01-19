# render_video.py (Final version with all features)
import os
import cv2
import pandas as pd
import argparse
from tqdm import tqdm
import numpy as np
import uuid  # For random filenames


# --- Re-usable helper function for YOLO export ---
def export_frame_for_yolo(frame, frame_number, frame_data, width, height, export_path, reason, video_basename):
    """Saves a frame and its corresponding YOLO label file with a random name."""
    # Use UUID to generate a random, unique filename
    filename_base = str(uuid.uuid4())

    image_filepath = os.path.join(export_path, 'images', f"{filename_base}.jpg")
    label_filepath = os.path.join(export_path, 'labels', f"{filename_base}.txt")

    cv2.imwrite(image_filepath, frame)

    with open(label_filepath, 'w') as label_file:
        for _, detection_row in frame_data.iterrows():
            if pd.notna(detection_row['class_name']):
                # Convert bounding box to YOLO format
                x1, y1, x2, y2 = int(detection_row['x1']), int(detection_row['y1']), int(detection_row['x2']), int(
                    detection_row['y2'])
                dw = 1. / width;
                dh = 1. / height
                x_center = (x1 + x2) / 2.0;
                y_center = (y1 + y2) / 2.0
                w = x2 - x1;
                h = y2 - y1
                x_center_norm = x_center * dw;
                y_center_norm = y_center * dh
                w_norm = w * dw;
                h_norm = h * dh
                class_index = 0 if detection_row['class_name'] == 'Ring' else 1
                yolo_line = f"{class_index} {x_center_norm:.6f} {y_center_norm:.6f} {w_norm:.6f} {h_norm:.6f}"
                label_file.write(yolo_line + '\n')


def render_video_with_overlays(video_path, csv_path, output_path, preview=False, export_yolo_path=None,
                               low_conf_threshold=0.5):
    print("Loading analysis data...")
    try:
        df = pd.read_csv(csv_path, dtype={'apex_frame': 'Int64'})
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

    exported_frames = set()

    if export_yolo_path:
        os.makedirs(os.path.join(export_yolo_path, 'images'), exist_ok=True)
        os.makedirs(os.path.join(export_yolo_path, 'labels'), exist_ok=True)
        print(f"YOLO export enabled. Saving data to: {export_yolo_path}")

    print("Starting video rendering process...")
    video_basename = os.path.splitext(os.path.basename(video_path))[0]

    all_video_frames = {}
    print("Pre-loading video frames into memory for efficient export...")
    for i in tqdm(range(total_frames), desc="Pre-loading frames"):
        ret, frame = cap.read()
        if not ret: break
        all_video_frames[i] = frame
    cap.release()

    flash_counter = 0
    flash_color = (0, 0, 0)
    FADE_FRAMES = 20

    # --- ADDED: State variable to hold the current score for persistent display ---
    persistent_score_text = "0/0"

    for frame_num in tqdm(range(total_frames), desc="Rendering Video"):
        frame = all_video_frames.get(frame_num)
        if frame is None: continue
        original_frame = frame.copy()

        if frame_num in frames_with_data:
            frame_data = frames_with_data[frame_num]
            shot_info_drawn = False

            # --- ADDED: Check for score updates and update the state variable ---
            latest_score_in_frame = frame_data[frame_data['current_score'].notna()]['current_score'].iloc[-1] if \
            frame_data['current_score'].notna().any() else None
            if latest_score_in_frame:
                persistent_score_text = latest_score_in_frame

            # --- Data export and drawing logic ---
            if export_yolo_path:
                for _, row in frame_data.iterrows():
                    if pd.notna(row['confidence']) and row['confidence'] < low_conf_threshold:
                        if frame_num not in exported_frames:
                            export_frame_for_yolo(original_frame, frame_num, frame_data, width, height,
                                                  export_yolo_path, f"low_conf_{row['confidence']:.2f}", video_basename)
                            exported_frames.add(frame_num)
                        break

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

                    # This large text only appears temporarily when a shot is detected
                    cv2.putText(frame, result_text, (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 5, cv2.LINE_AA)
                    cv2.putText(frame, result_text, (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 3,
                                cv2.LINE_AA)

                    if shot_result == "Successful":
                        flash_color = (0, 255, 0)
                    else:
                        flash_color = (0, 0, 255)
                    flash_counter = FADE_FRAMES
                    shot_info_drawn = True

                    if export_yolo_path:
                        if frame_num not in exported_frames:
                            export_frame_for_yolo(original_frame, frame_num, frame_data, width, height,
                                                  export_yolo_path, f"shot_outcome_{shot_result}", video_basename)
                            exported_frames.add(frame_num)
                        if pd.notna(row['apex_frame']):
                            apex_frame_num = int(row['apex_frame'])
                            if apex_frame_num in all_video_frames and apex_frame_num not in exported_frames:
                                apex_frame_original = all_video_frames[apex_frame_num]
                                apex_frame_data = frames_with_data.get(apex_frame_num)
                                if apex_frame_data is not None:
                                    export_frame_for_yolo(apex_frame_original, apex_frame_num, apex_frame_data, width,
                                                          height, export_yolo_path, "shot_apex", video_basename)
                                    exported_frames.add(apex_frame_num)

        # --- ADDED: Draw the persistent score counter on EVERY frame ---
        score_display_text = f"Score: {persistent_score_text}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.2
        thickness = 2
        text_color = (255, 255, 255)
        bg_color = (0, 0, 0)
        (text_width, text_height), baseline = cv2.getTextSize(score_display_text, font, font_scale, thickness)

        # Draw a semi-transparent background rectangle for readability
        overlay = frame.copy()
        cv2.rectangle(overlay, (20, 20), (30 + text_width, 30 + text_height + baseline), bg_color, -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        # Draw the score text on top of the background
        cv2.putText(frame, score_display_text, (30, 30 + text_height), font, font_scale, text_color, thickness,
                    cv2.LINE_AA)

        # Apply flash effect if active
        if flash_counter > 0:
            overlay = np.full(frame.shape, flash_color, dtype=np.uint8)
            alpha = 0.3 * (flash_counter / FADE_FRAMES)
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
            flash_counter -= 1

        out.write(frame)
        if preview:
            cv2.imshow("Renderer Preview", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): break

    print("Rendering complete.")
    out.release()
    if preview: cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Video Renderer with Advanced YOLO Export.")
    parser.add_argument('--video', type=str, required=True, help="Path to the original input video.")
    parser.add_argument('--data', type=str, required=True, help="Path to the analysis CSV data file.")
    parser.add_argument('--output', type=str, required=True, help="Path for the output video file.")
    parser.add_argument('--preview', action='store_true', help="Show a preview window while rendering (slower).")
    parser.add_argument('--export_yolo', type=str, default=None,
                        help="Path to export YOLO dataset files for key moments.")
    parser.add_argument('--low_conf_threshold', type=float, default=0.50,
                        help="Confidence threshold below which to export a frame for retraining.")
    args = parser.parse_args()

    render_video_with_overlays(args.video, args.data, args.output, args.preview, args.export_yolo,
                               args.low_conf_threshold)