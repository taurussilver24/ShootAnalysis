import csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc


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

    # Compare shots
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


def plot_roc_curve(shots1, timings1, shots2, timings2):
    """Plots ROC curve by varying the margin."""
    margins = np.linspace(0.1, 1.5, 20)  # Vary margin from 0.1s to 1.5s
    TPRs, FPRs = [], []

    for margin in margins:
        TP, TN, FP, FN = compute_tp_tn_fp_fn(shots1, timings1, shots2, timings2, margin)

        TPR = TP / (TP + FN) if (TP + FN) > 0 else 0  # True Positive Rate (Recall)
        FPR = FP / (FP + TN) if (FP + TN) > 0 else 0  # False Positive Rate

        TPRs.append(TPR)
        FPRs.append(FPR)

    # Compute AUC
    roc_auc = auc(FPRs, TPRs)

    # Plot ROC Curve
    plt.figure(figsize=(8, 6))
    plt.plot(FPRs, TPRs, marker='o', linestyle='-', label=f"ROC Curve (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')  # Random classifier line
    plt.xlabel("False Positive Rate (FPR)")
    plt.ylabel("True Positive Rate (TPR / Recall)")
    plt.title("ROC Curve for Shot Detection")
    plt.legend()
    plt.grid()
    plt.show()


def main():
    file1 = "Results/DNvsTW.mp4/CSVtest/ground.csv"  # Ground truth
    file2 = "Results/DNvsTW.mp4/CSVtest/YOLO.csv"  # Detected results

    shots1, timings1 = read_csv(file1)
    shots2, timings2 = read_csv(file2)

    TP, TN, FP, FN = compute_tp_tn_fp_fn(shots1, timings1, shots2, timings2)

    TPR = TP / (TP + FN) if (TP + FN) > 0 else 0  # True Positive Rate
    FPR = FP / (FP + TN) if (FP + TN) > 0 else 0  # False Positive Rate

    print(f"\nTrue Positives (TP): {TP}")
    print(f"True Negatives (TN): {TN}")
    print(f"False Positives (FP): {FP}")
    print(f"False Negatives (FN): {FN}")

    print(f"\nTrue Positive Rate (Recall): {TPR:.2f}")
    print(f"False Positive Rate (FPR): {FPR:.2f}")

    # Plot ROC curve
    plot_roc_curve(shots1, timings1, shots2, timings2)


if __name__ == "__main__":
    main()
