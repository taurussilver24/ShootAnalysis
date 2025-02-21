import csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import auc


def read_csv(file_path):
    """Reads CSV file and extracts 'Result' and 'Video Timing (seconds)'."""
    shots, timings = [], []
    with open(file_path, mode='r') as file:
        csv_reader = csv.DictReader(file)
        for row in csv_reader:
            shots.append(row['Result'])
            timings.append(float(row['Video Timing (seconds)']))
    return shots, timings


def compute_tp_tn_fp_fn(shots1, timings1, shots2, timings2, margin=0.7):
    """Computes TP, TN, FP, FN based on timing margin."""
    TP, TN, FP, FN = 0, 0, 0, 0
    used_indices = set()

    for j, t1 in enumerate(timings1):
        best_match_index = None
        best_time_diff = float("inf")

        for i, t2 in enumerate(timings2):
            if i in used_indices:
                continue
            time_diff = abs(t2 - t1)
            if time_diff <= margin and time_diff < best_time_diff:
                best_match_index = i
                best_time_diff = time_diff

        if best_match_index is not None:
            used_indices.add(best_match_index)
            if shots1[j] == "Successful" and shots2[best_match_index] == "Successful":
                TP += 1
            elif shots1[j] == "Failed" and shots2[best_match_index] == "Failed":
                TN += 1
            elif shots1[j] == "Failed" and shots2[best_match_index] == "Successful":
                FP += 1
            elif shots1[j] == "Successful" and shots2[best_match_index] == "Failed":
                FN += 1
        else:
            if shots1[j] == "Successful":
                FN += 1
            else:
                TN += 1  # If no match and ground truth was Failed, assume TN

    FP += len(timings2) - len(used_indices)  # Extra detections with no match
    return TP, TN, FP, FN


def plot_roc_curve(file1, detected_files):
    """Plots ROC curve using multiple detected files."""
    TPRs, FPRs = [], []

    # Read ground truth
    shots1, timings1 = read_csv(file1)

    for file in detected_files:
        shots2, timings2 = read_csv(file)
        TP, TN, FP, FN = compute_tp_tn_fp_fn(shots1, timings1, shots2, timings2)

        TPR = TP / (TP + FN) if (TP + FN) > 0 else 0  # True Positive Rate (Recall)
        FPR = FP / (FP + TN) if (FP + TN) > 0 else 0  # False Positive Rate

        print(f"{file}: TP={TP}, TN={TN}, FP={FP}, FN={FN}, TPR={TPR:.2f}, FPR={FPR:.2f}")
        TPRs.append(TPR)
        FPRs.append(FPR)

    # Sort FPR and TPR to avoid issues with auc calculation
    sorted_FPRs, sorted_TPRs = zip(*sorted(zip(FPRs, TPRs)))

    # Compute AUC
    roc_auc = auc(sorted_FPRs, sorted_TPRs)

    # Plot ROC Curve
    plt.figure(figsize=(8, 6))
    plt.plot(sorted_FPRs, sorted_TPRs, marker='o', linestyle='-', label=f"ROC Curve (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')  # Random classifier line
    plt.xlabel("False Positive Rate (FPR)")
    plt.ylabel("True Positive Rate (TPR / Recall)")
    plt.title("ROC Curve for Shot Detection")
    plt.legend()
    plt.grid()
    plt.show()



def main():
    file1 = "Results/DNvsTW.mp4/CSVtest/ground.csv"  # Ground truth
    detected_files = [
        "Results/DNvsTW.mp4/CSVtest/YOLO.csv",
        "Results/DNvsTW.mp4/CSVtest/YOLO2.csv",
        "Results/DNvsTW.mp4/CSVtest/YOLO3.csv"
    ]

    plot_roc_curve(file1, detected_files)


if __name__ == "__main__":
    main()
