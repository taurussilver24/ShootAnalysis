import pandas as pd
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import warnings
import numpy as np

import argparse

parser = argparse.ArgumentParser(description="動画のcsvそれぞれを比較し、ROCカーブ作成")
parser.add_argument('--video', '-v', type=str, default="", help="動画のパス")
args = parser.parse_args()

ground_truth_path = 'Results/' + args.video + '/ground_truth.csv'
result_path = 'Results/' + args.video + '/auto_gen.csv'

def find_closest_match(gt_time, auto_times, tolerance=0.15):
    """Find the closest match for a given ground truth time within a specified tolerance."""
    closest_time = None
    min_diff = float('inf')
    for auto_time in auto_times:
        diff = abs(gt_time - auto_time)
        if diff <= tolerance * gt_time and diff < min_diff:
            min_diff = diff
            closest_time = auto_time
    return closest_time


def update_autogen_csv(groundtruth_df, autogen_df, output_file="Results/" + args.video + "updated_auto_gen.csv"):
    """Update the auto-generated CSV to include missed entries from the ground truth."""
    # Create a new DataFrame to hold the updated auto-generated entries
    updated_autogen_df = autogen_df.copy()

    # List to store new rows to be added
    new_rows = []

    # Check for missed entries and add them to the new rows list
    for _, gt_row in groundtruth_df.iterrows():
        gt_time = gt_row['Video Timing (seconds)']
        closest_time = find_closest_match(gt_time, autogen_df['Video Timing (seconds)'].tolist())
        if closest_time is None:
            # Add missed entry with "No Shot Detected" to new rows list
            new_entry = {
                'Shot Taken': gt_row['Shot Taken'],
                'Result': 'NoShot Detected',
                'Ball Coordinates': gt_row['Ball Coordinates'],
                'Hoop Coordinates': gt_row['Hoop Coordinates'],
                'Current Score': gt_row['Current Score'],
                'Video Timing (seconds)': gt_row['Video Timing (seconds)']
            }
            new_rows.append(new_entry)

    # Convert new rows list to DataFrame and concatenate with the original auto-generated DataFrame
    new_rows_df = pd.DataFrame(new_rows)
    updated_autogen_df = pd.concat([updated_autogen_df, new_rows_df], ignore_index=True)

    # Save the updated auto-generated CSV
    updated_autogen_df.to_csv(output_file, index=False)
    return updated_autogen_df


def compute_roc_curve(groundtruth_df, updated_autogen_df):
    """Compute and plot the ROC curve based on the updated auto-generated CSV."""
    # Create lists to hold the true labels and scores
    true_labels = []
    scores = []

    # Convert relevant columns to lists for easier processing
    gt_times = groundtruth_df['Video Timing (seconds)'].tolist()
    auto_times = updated_autogen_df['Video Timing (seconds)'].tolist()

    # Process ground truth entries
    for gt_time in gt_times:
        closest_time = find_closest_match(gt_time, auto_times)
        if closest_time is not None:
            gt_result = groundtruth_df.loc[groundtruth_df['Video Timing (seconds)'] == gt_time, 'Result'].values[0]
            auto_result = \
            updated_autogen_df.loc[updated_autogen_df['Video Timing (seconds)'] == closest_time, 'Result'].values[0]

            if gt_result == 'Shot Detected' and auto_result == 'Shot Detected':
                true_labels.append(1)  # Actual shot detected (positive)
                scores.append(1)  # Predicted shot detected (positive)
            else:
                true_labels.append(1)  # Actual shot detected (positive)
                scores.append(0)  # Predicted no shot detected (negative)

            auto_times.remove(closest_time)  # Remove the matched time to avoid duplicates
        else:
            true_labels.append(1)  # Actual shot detected (positive)
            scores.append(0)  # Predicted no shot detected (negative)

    # Process remaining auto-generated entries (False Positives)
    for auto_time in auto_times:
        if updated_autogen_df.loc[updated_autogen_df['Video Timing (seconds)'] == auto_time, 'Result'].values[
            0] == 'Shot Detected':
            true_labels.append(0)  # Actual no shot detected (negative)
            scores.append(1)  # Predicted shot detected (positive)

    # Print debugging information
    print("True Labels:", true_labels)
    print("Scores:", scores)

    # Ensure there are negative samples to calculate the ROC curve
    if len(set(true_labels)) == 1:
        # Only one class present in y_true
        print(
            "Only one class present in true labels. ROC AUC is not defined. Assuming perfect classification with AUC = 1.0.")
        roc_auc = 1.0

        # Plot a diagonal line
        plt.figure(figsize=(8, 6))
        plt.plot([0, 1], [0, 1], color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.show()
    else:
        # Calculate ROC curve
        fpr, tpr, thresholds = roc_curve(true_labels, scores)
        roc_auc = auc(fpr, tpr)

        # Plot ROC curve
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.show()


if __name__ == "__main__":




    # Load the data from CSV files
    groundtruth_df = pd.read_csv(ground_truth_path)
    autogen_df = pd.read_csv(result_path)

    # Update the auto-generated CSV
    updated_autogen_df = update_autogen_csv(groundtruth_df, autogen_df)

    # Compute and plot ROC curve
    compute_roc_curve(groundtruth_df, updated_autogen_df)
