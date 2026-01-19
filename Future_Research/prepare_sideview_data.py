# prepare_sideview_data.py (v6 - Advanced Outlier Rejection)
import pandas as pd
import json
import argparse
import os
import numpy as np
from scipy.signal import savgol_filter


def remove_path_outliers(path):
    """
    Cleans a trajectory path by removing points that represent a sharp,
    unrealistic change in direction.
    """
    if len(path) < 3:
        return path

    cleaned_path = [path[0]]
    for i in range(1, len(path) - 1):
        # Get the vectors for the incoming and outgoing paths from the current point
        p_prev = np.array([path[i - 1]['x'], path[i - 1]['y']])
        p_curr = np.array([path[i]['x'], path[i]['y']])
        p_next = np.array([path[i + 1]['x'], path[i + 1]['y']])

        v_in = p_curr - p_prev
        v_out = p_next - p_curr

        # Normalize the vectors
        norm_in = np.linalg.norm(v_in)
        norm_out = np.linalg.norm(v_out)

        if norm_in > 0 and norm_out > 0:
            v_in_norm = v_in / norm_in
            v_out_norm = v_out / norm_out

            # Calculate the dot product to find the cosine of the angle
            dot_product = np.dot(v_in_norm, v_out_norm)

            # Clamp the dot product to avoid math errors with acos
            dot_product = np.clip(dot_product, -1.0, 1.0)

            # Calculate the angle in degrees
            angle = np.degrees(np.arccos(dot_product))

            # A basketball in a shot arc should not suddenly reverse direction.
            # We keep the point only if the angle of trajectory change is not sharp.
            # An angle close to 180 means it's continuing straight.
            # An angle close to 0 means it has reversed direction.
            if angle > 90:  # Allow for gentle curves, reject sharp turns/reversals
                cleaned_path.append(path[i])

    cleaned_path.append(path[-1])  # Always keep the last point
    return cleaned_path


def prepare_data_for_side_view(csv_path, output_path):
    print(f"Loading data from {csv_path}...")
    try:
        df = pd.read_csv(csv_path).dropna(subset=['class_name'])
    except FileNotFoundError:
        print(f"Error: CSV file not found at {csv_path}")
        return

    hoop_df = df[df['class_name'] == 'Ring'].copy()
    ball_df = df[df['class_name'] == 'Ball'].copy()

    if hoop_df.empty or ball_df.empty:
        print("Error: Not enough ball or hoop data.")
        return

    shots_df = df.dropna(subset=['shot_attempt_id']).drop_duplicates('shot_attempt_id', keep='first')
    ball_df = ball_df.sort_values('confidence', ascending=False).drop_duplicates('frame_number').sort_values(
        'frame_number')

    shot_segments = []
    print(f"Found {len(shots_df)} shot attempts to process...")

    for _, shot in shots_df.iterrows():
        shot_id = int(shot['shot_attempt_id'])
        end_frame = int(shot['frame_number'])
        start_frame = end_frame - 60

        potential_hoops = hoop_df[hoop_df['frame_number'].between(start_frame, end_frame)]
        if potential_hoops.empty:
            anchor_hoop_row = hoop_df.iloc[np.argmin(np.abs(hoop_df['frame_number'] - start_frame))]
        else:
            anchor_hoop_row = potential_hoops.loc[potential_hoops['confidence'].idxmax()]

        anchor_hoop_x = (anchor_hoop_row['x1'] + anchor_hoop_row['x2']) / 2
        anchor_hoop_y = (anchor_hoop_row['y1'] + anchor_hoop_row['y2']) / 2

        segment_df = ball_df[ball_df['frame_number'].between(start_frame, end_frame)]
        if len(segment_df) < 7: continue  # Need more points for robust cleaning

        raw_path = []
        for _, ball_row in segment_df.iterrows():
            ball_center_x = (ball_row['x1'] + ball_row['x2']) / 2
            ball_center_y = (ball_row['y1'] + ball_row['y2']) / 2
            raw_path.append({"x": int(ball_center_x - anchor_hoop_x), "y": int(ball_center_y - anchor_hoop_y)})

        # --- NEW TWO-STAGE CLEANING ---
        # 1. Remove sharp directional outliers
        cleaned_path = remove_path_outliers(raw_path)

        # 2. Smooth the now-cleaner path
        if len(cleaned_path) < 5:
            smoothed_path = cleaned_path
        else:
            x_coords = [p['x'] for p in cleaned_path]
            y_coords = [p['y'] for p in cleaned_path]
            window_length = min(5, len(x_coords))
            if window_length % 2 == 0: window_length -= 1
            if window_length > 2:
                x_smooth = savgol_filter(x_coords, window_length, 2)
                y_smooth = savgol_filter(y_coords, window_length, 2)
                smoothed_path = [{"x": int(x), "y": int(y)} for x, y in zip(x_smooth, y_smooth)]
            else:
                smoothed_path = cleaned_path

        shot_segments.append({
            "id": shot_id, "result": shot['shot_result'],
            "raw_path": raw_path, "smoothed_path": smoothed_path
        })

    print(f"Processed and smoothed {len(shot_segments)} shot segments.")

    avg_hoop_width = int((hoop_df['x2'] - hoop_df['x1']).mean())
    final_data = {"hoop": {"width": avg_hoop_width}, "shots": shot_segments}

    with open(output_path, 'w') as f:
        json.dump(final_data, f, indent=2)
    print(f"Successfully created side-view visualization data at {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare analysis data for 2D side-view shot visualization.")
    parser.add_argument('--data', type=str, required=True, help="Path to the input analysis CSV file.")
    parser.add_argument('--output', type=str, required=True, help="Path for the output JSON data file.")
    args = parser.parse_args()
    prepare_data_for_side_view(args.data, args.output)