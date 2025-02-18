import pandas as pd
import numpy as np


# Function to generate a CSV file with the specified format
def generate_csv(filename, num_samples):
    np.random.seed(42)  # For reproducibility

    # Generate random data
    data = {
        'Shot Taken': list(range(1, num_samples + 1)),
        'Actual Result': np.random.choice(['Successful', 'Failed'], size=num_samples),
        'Result': np.random.choice(['Successful', 'Failed'], size=num_samples),
        'Ball Coordinates': [(np.random.randint(500, 1100), np.random.randint(200, 310)) for _ in range(num_samples)],
        'Hoop Coordinates': [(np.random.randint(500, 1100), np.random.randint(200, 310)) for _ in range(num_samples)],
        'Current Score': [f"{i} / {i + 1}" for i in range(1, num_samples + 1)],
        'Video Timing (seconds)': np.random.uniform(100, 600, size=num_samples)
    }

    # Create DataFrame
    df = pd.DataFrame(data)

    # Write DataFrame to CSV
    df.to_csv(filename, index=False)

    print(f"CSV file '{filename}' created successfully with {num_samples} samples.")


# Generate CSV with specified format
generate_csv('test2.csv', 120)  # Adjust num_samples as needed

# Load data from CSV into pandas DataFrame
# df = pd.read_csv('high_auc_data.csv')
#
# # Calculate ROC curve
# fpr, tpr, thresholds = roc_curve(df['true_labels'], df['scores'])
# roc_auc = auc(fpr, tpr)
#
# # Plot ROC curve
# plt.figure(figsize=(8, 6))
# plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
# plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Receiver Operating Characteristic (ROC) Curve')
# plt.legend(loc="lower right")
# plt.grid(True)
# plt.show()
