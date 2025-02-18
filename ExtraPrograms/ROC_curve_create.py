import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import os
import argparse

# Global variables
parser = argparse.ArgumentParser(description="動画のcsvそれぞれを比較し、ROCカーブ作成")
parser.add_argument('--video', '-v', type=str, default="", help="動画のパス")
args = parser.parse_args()
video_path = args.video

def read_csv(file_path):
    """Read CSV file into a DataFrame"""
    return pd.read_csv(file_path)

def ensure_directory_exists(directory):
    """Ensure that a directory exists, create if it does not"""
    if not os.path.exists(directory):
        os.makedirs(directory)

def match_entries(ground_truth_df, result_df, tolerance=1):
    """Match entries based on video timing with a given tolerance of ±1 second"""
    matches = []
    for _, gt_row in ground_truth_df.iterrows():
        matched = False
        for _, res_row in result_df.iterrows():
            if abs(gt_row['Video Timing (seconds)'] - res_row['Video Timing (seconds)']) <= tolerance:
                matches.append((gt_row, res_row))
                matched = True
                break
        if not matched:
            matches.append((gt_row, None))  # No match found in result_df
    return matches

def handle_missed_entries(ground_truth_df, result_df, tolerance=1):
    """Handle missed entries in the auto-generated CSV"""
    matches = match_entries(ground_truth_df, result_df, tolerance)

    # List to store new rows for the auto-gen CSV
    new_rows = []

    for gt_row, res_row in matches:
        if res_row is None:
            # Missed entry in the result CSV, add it as "No Shot Detected"
            new_row = gt_row.copy()
            new_row['Result'] = 'No Shot Detected'
            new_rows.append(new_row)

    # Create a DataFrame for the new rows and append to the result DataFrame
    if new_rows:
        new_entries_df = pd.DataFrame(new_rows)
        updated_result_df = pd.concat([result_df, new_entries_df], ignore_index=True)
        updated_result_df.to_csv("Results/" + video_path + '/updated_result.csv', index=False)
    else:
        updated_result_df = result_df.copy()

    return updated_result_df, matches

def calculate_metrics(matches, result_df):
    """Calculate TP, TN, FP, FN based on matched entries"""
    tp = fp = tn = fn = 0
    y_true = []
    y_scores = []

    for gt_row, res_row in matches:
        y_true.append(1 if gt_row['Result'] == 'Successful' else 0)
        if res_row is not None:
            if gt_row['Result'] == 'Successful':
                if res_row['Result'] == 'Successful':
                    tp += 1
                    y_scores.append(1)  # True Positive
                else:
                    fp += 1
                    y_scores.append(0)  # False Positive
            else:
                if res_row['Result'] == 'Failed':
                    tn += 1
                    y_scores.append(0)  # True Negative
                else:
                    fn += 1
                    y_scores.append(1)  # False Negative
        else:
            if gt_row['Result'] == 'Successful':
                fn += 1
                y_scores.append(0)  # False Negative
            else:
                tn += 1
                y_scores.append(0)  # True Negative

    return tp, tn, fp, fn, y_true, y_scores

def print_metrics(tp, tn, fp, fn):
    """Print metrics including recall and false positive rate"""
    total_positive = tp + fn
    total_negative = tn + fp

    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0

    print(f"True Positives (TP): {tp}")
    print(f"True Negatives (TN): {tn}")
    print(f"False Positives (FP): {fp}")
    print(f"False Negatives (FN): {fn}")
    print(f"Recall: {recall:.2f}")
    print(f"False Positive Rate: {false_positive_rate:.2f}")

def plot_roc_curve(y_true, y_scores):
    """Plot ROC curve"""
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

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

def main(video_path):
    ground_truth_path = 'Results/' + video_path + '/shot_results_ground.csv'
    result_path = 'Results/' + video_path + '/shot_results.csv'

    ground_truth_df = read_csv(ground_truth_path)
    result_df = read_csv(result_path)

    # Handle missed entries and update the result DataFrame
    updated_result_df, matches = handle_missed_entries(ground_truth_df, result_df)

    # Calculate metrics using the updated result DataFrame
    tp, tn, fp, fn, y_true, y_scores = calculate_metrics(matches, updated_result_df)
    print_metrics(tp, tn, fp, fn)
    plot_roc_curve(y_true, y_scores)

if __name__ == "__main__":
    main(video_path)
