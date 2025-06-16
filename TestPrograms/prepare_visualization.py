# prepare_visualization.py (v4 - Half-Court with Path Smoothing)
import pandas as pd
import json
import argparse
import os
import numpy as np
from scipy.signal import savgol_filter


def prepare_data_for_half_court_view(csv_path, output_path):
    print(f"Loading data from {csv_path}...")
    try:
        df = pd.read_csv(csv_path).dropna(subset=['class_name'])
    except FileNotFoundError:
        print(f"Error: CSV file not found at {csv_path}")
        return

    # --- 1. Calculate a Single, Stable Hoop Position ---
    hoop_df = df[df['class_name'] == 'Ring'].copy()
    if hoop_df.empty:
        print("Error: No hoop detections found in the data.")
        return

    avg_hoop = {
        "x": int(hoop_df[['x1', 'x2']].mean().mean()),
        "y": int(hoop_df[['y1', 'y2']].mean().mean()),
        "width": int((hoop_df['x2'] - hoop_df['x1']).mean())
    }
    print(f"Calculated stable hoop position: {avg_hoop}")

    # --- 2. Isolate Shot Segments and Calculate Relative/Smoothed Paths ---
    shot_segments = []
    shots_df = df.dropna(subset=['shot_attempt_id']).drop_duplicates('shot_attempt_id', keep='first')

    ball_df = df[df['class_name'] == 'Ball'].copy()
    ball_df = ball_df.sort_values('confidence', ascending=False).drop_duplicates('frame_number').sort_values(
        'frame_number')

    for _, shot in shots_df.iterrows():
        shot_id = int(shot['shot_attempt_id'])
        end_frame = int(shot['frame_number'])
        start_frame = end_frame - 45  # Look back 45 frames for the shot motion

        segment_df = ball_df[ball_df['frame_number'].between(start_frame, end_frame)]
        if len(segment_df) < 5: continue  # Need at least 5 points to smooth

        raw_path = []
        for _, ball_row in segment_df.iterrows():
            ball_center_x = (ball_row['x1'] + ball_row['x2']) / 2
            ball_center_y = (ball_row['y1'] + ball_row['y2']) / 2
            raw_path.append({
                "x": int(ball_center_x - avg_hoop['x']),
                "y": int(ball_center_y - avg_hoop['y'])
            })

        # --- PATH SMOOTHING LOGIC ---
        x_coords = [p['x'] for p in raw_path]
        y_coords = [p['y'] for p in raw_path]

        # Savitzky-Golay filter needs a window length (must be odd) and polynomial order
        window_length = min(5, len(x_coords))
        if window_length % 2 == 0: window_length -= 1  # Ensure odd length

        if window_length > 2:  # Polyorder must be less than window_length
            x_smooth = savgol_filter(x_coords, window_length, 2)  # polyorder 2
            y_smooth = savgol_filter(y_coords, window_length, 2)

            smoothed_path = [{"x": int(x), "y": int(y)} for x, y in zip(x_smooth, y_smooth)]
        else:
            smoothed_path = raw_path  # Not enough points to smooth

        shot_segments.append({
            "id": shot_id,
            "result": shot['shot_result'],
            "raw_path": raw_path,
            "smoothed_path": smoothed_path
        })
    print(f"Processed and smoothed {len(shot_segments)} shot segments.")

    final_data = {"hoop": avg_hoop, "shots": shot_segments}
    with open(output_path, 'w') as f:
        json.dump(final_data, f, indent=2)
    print(f"Successfully created smoothed half-court view data at {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare analysis data for 2D half-court visualization.")
    parser.add_argument('--data', type=str, required=True, help="Path to the input analysis CSV file.")
    parser.add_argument('--output', type=str, required=True, help="Path for the output JSON data file.")
    args = parser.parse_args()
    prepare_data_for_half_court_view(args.data, args.output)