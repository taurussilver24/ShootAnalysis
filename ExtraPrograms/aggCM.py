import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
import os

RESULTS_ROOT = "Results"  # Root folder containing one subfolder per video
TOLERANCE = 1  # Matching tolerance in seconds


def read_csv(file_path):
    """Read a CSV file into a DataFrame."""
    return pd.read_csv(file_path)


def match_entries(gt_df, res_df, tolerance=1):
    """Match GT and result rows within the timing tolerance."""
    matches = []
    for _, gt_row in gt_df.iterrows():
        matched = False
        for _, res_row in res_df.iterrows():
            if abs(gt_row['Video Timing (seconds)'] - res_row['Video Timing (seconds)']) <= tolerance:
                matches.append((gt_row, res_row))
                matched = True
                break
        if not matched:
            matches.append((gt_row, None))
    return matches


def calculate_metrics(matches):
    """Compute confusion matrix values."""
    tp = fp = tn = fn = 0
    for gt_row, res_row in matches:
        gt_result = gt_row['Result']
        pred_result = res_row['Result'] if res_row is not None else "No Shot Detected"

        if gt_result == "Successful":
            if pred_result == "Successful":
                tp += 1
            else:
                fn += 1
        else:  # GT was Failed
            if pred_result == "Successful":
                fp += 1
            else:
                tn += 1
    return tp, tn, fp, fn


def aggregate_all_videos(results_root):
    total_tp = total_tn = total_fp = total_fn = 0

    for video_folder in os.listdir(results_root):
        video_path = os.path.join(results_root, video_folder)
        gt_path = os.path.join(video_path, "shot_results_ground.csv")
        pred_path = os.path.join(video_path, "shot_results.csv")

        if not os.path.exists(gt_path) or not os.path.exists(pred_path):
            print(f"Missing files for video: {video_folder}")
            continue

        gt_df = read_csv(gt_path)
        res_df = read_csv(pred_path)
        matches = match_entries(gt_df, res_df, tolerance=TOLERANCE)
        tp, tn, fp, fn = calculate_metrics(matches)

        print(f"[{video_folder}] TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}")
        total_tp += tp
        total_tn += tn
        total_fp += fp
        total_fn += fn

    return total_tp, total_tn, total_fp, total_fn


def plot_global_confusion_matrix(tp, tn, fp, fn):
    cm = np.array([[tn, fp], [fn, tp]])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Failed", "Successful"])
    disp.plot(cmap="Blues")
    plt.title("Global Confusion Matrix (All Videos)")
    plt.grid(False)
    plt.show()



if __name__ == "__main__":
    tp, tn, fp, fn = aggregate_all_videos(RESULTS_ROOT)
    print("\n=== Global Totals ===")
    print(f"TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}")
    plot_global_confusion_matrix(tp, tn, fp, fn)
