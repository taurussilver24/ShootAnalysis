# prepare_simple_viz.py (v3 - With Robust Regex Parsing)
import pandas as pd
import json
import argparse
import os
import re  # <--- Import the regular expressions module


def parse_coord_string_regex(coord_str):
    """
    Parses the coordinate tuple string from the CSV using a robust regex.
    """
    if not isinstance(coord_str, str):
        return None

    # This regex is designed to find the numbers inside the nested tuple format
    # Format: ((centerX, centerY), frame, width, height, confidence)
    pattern = r"\(\((\d+), (\d+)\), (\d+), (\d+), (\d+), ([\d.]+)\)"

    match = re.search(pattern, coord_str)

    if match:
        # Extract all the number groups found by the regex
        groups = match.groups()
        center_x = int(groups[0])
        center_y = int(groups[1])
        width = int(groups[3])
        height = int(groups[4])
        return {'center': (center_x, center_y), 'w': width, 'h': height}
    else:
        # If the regex doesn't find a match, the format is unexpected
        print(f"  -> Warning: Could not parse coordinate string with regex: '{coord_str}'")
        return None


def prepare_data_for_simple_view(csv_path, output_path):
    print(f"Loading shot results from {csv_path}...")
    try:
        df = pd.read_csv(csv_path)
        print(f"Successfully loaded CSV with {len(df)} rows.")
    except FileNotFoundError:
        print(f"Error: CSV file not found at {csv_path}")
        return

    shots_data = []
    for index, row in df.iterrows():
        print(f"\n--- Processing Row {index + 1} ---")

        shot_id = row['Shot Taken']
        result = row['Result']
        print(f"  Shot ID: {shot_id}, Result: {result}")

        # Parse using the new, robust regex function
        ball_info = parse_coord_string_regex(row['Ball Coordinates'])
        hoop_info = parse_coord_string_regex(row['Hoop Coordinates'])

        if ball_info and hoop_info:
            rel_x = ball_info['center'][0] - hoop_info['center'][0]
            rel_y = ball_info['center'][1] - hoop_info['center'][1]

            shots_data.append({
                "id": shot_id,
                "result": result,
                "ball_relative_pos": {"x": rel_x, "y": rel_y},
                "ball_size": ball_info['w'],
                "hoop_size": hoop_info['w']
            })
            print(f"  -> Successfully processed and added shot #{shot_id} to visualization data.")
        else:
            print(f"  -> Skipping shot #{shot_id} due to parsing errors.")

    with open(output_path, 'w') as f:
        json.dump({"shots": shots_data}, f, indent=2)

    print("\n-------------------------")
    print(f"Processing complete.")
    print(f"Successfully created visualization data for {len(shots_data)} shots at {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare shot data for simple 2.5D visualization.")
    parser.add_argument('--data', type=str, required=True, help="Path to the input shot_results.csv file.")
    parser.add_argument('--output', type=str, required=True, help="Path for the output JSON data file.")
    args = parser.parse_args()
    prepare_data_for_simple_view(args.data, args.output)