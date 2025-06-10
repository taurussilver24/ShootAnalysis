import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import os
import argparse

# Global variables
parser = argparse.ArgumentParser(description="動画のcsvそれぞれを比較し、混同行列を表示")
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

def calculate_metrics(matches):
    """Calculate TP, TN, FP, FN and binary labels for confusion matrix"""
    tp = fp = tn = fn = 0
    y_true = []
    y_pred = []

    for gt_row, res_row in matches:
        true_label = 1 if gt_row['Result'] == 'Successful' else 0
        pred_label = 0  # Default if no match or invalid result

        if res_row is not None:
            if res_row['Result'] == 'Successful':
                pred_label = 1
            elif res_row['Result'] == 'Failed':
                pred_label = 0
            else:
                pred_label = 0  # Treat unknown/missed detection as negative
        else:
            pred_label = 0

        y_true.append(true_label)
        y_pred.append(pred_label)

        # Count metrics
        if true_label == 1 and pred_label == 1:
            tp += 1
        elif true_label == 0 and pred_label == 0:
            tn += 1
        elif true_label == 0 and pred_label == 1:
            fp += 1
        elif true_label == 1 and pred_label == 0:
            fn += 1

    return tp, tn, fp, fn, y_true, y_pred

def print_metrics(tp, tn, fp, fn):
    """Print metrics including recall and false positive rate"""
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0

    print(f"True Positives (TP): {tp}")
    print(f"True Negatives (TN): {tn}")
    print(f"False Positives (FP): {fp}")
    print(f"False Negatives (FN): {fn}")
    print(f"Recall: {recall:.2f}")
    print(f"False Positive Rate: {false_positive_rate:.2f}")

def plot_confusion_matrix(y_true, y_pred, labels=["Failed", "Successful"]):
    """Plot and save confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.grid(False)
    plt.show()

    save_path = f"Results/{video_path}/confusion_matrix.png"
    ensure_directory_exists(os.path.dirname(save_path))
    plt.savefig(save_path)
    print(f"Confusion matrix saved to: {save_path}")

def main(video_path):
    ground_truth_path = 'Results/' + video_path + '/shot_results_ground.csv'
    result_path = 'Results/' + video_path + '/shot_results.csv'

    ground_truth_df = read_csv(ground_truth_path)
    result_df = read_csv(result_path)

    # Handle missed entries and update the result DataFrame
    updated_result_df, matches = handle_missed_entries(ground_truth_df, result_df)

    # Calculate metrics and predictions
    tp, tn, fp, fn, y_true, y_pred = calculate_metrics(matches)

    # Print metrics
    print_metrics(tp, tn, fp, fn)

    # Plot and save confusion matrix
    plot_confusion_matrix(y_true, y_pred)

if __name__ == "__main__":
    main(video_path)
