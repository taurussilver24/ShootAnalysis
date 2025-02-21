import csv


def read_csv(file_path):
    """Reads CSV file and extracts 'Result' and 'Video Timing (seconds)'."""
    shots, timings = [], []
    with open(file_path, mode='r') as file:
        csv_reader = csv.DictReader(file)
        for row in csv_reader:
            shots.append(row['Result'])
            timings.append(float(row['Video Timing (seconds)']))
    return shots, timings


def compare_shots_with_timing(shots1, timings1, shots2, timings2, margin=0.7):
    """
    Compares shots2 against shots1 based on timing within a margin.
    - Matches a shot from file2 to the closest available shot in file1 within the margin.
    - Ensures every shot from file1 is matched only once.
    """
    matched_shots = 0
    used_indices = set()  # Track matched indices from shots2

    for j, t1 in enumerate(timings1):
        best_match_index = None
        best_time_diff = float("inf")

        for i, t2 in enumerate(timings2):
            if i in used_indices:  # Skip already matched shots
                continue
            time_diff = abs(t2 - t1)
            if time_diff <= margin and time_diff < best_time_diff:
                best_match_index = i
                best_time_diff = time_diff

        if best_match_index is not None:
            used_indices.add(best_match_index)
            if shots1[j] == shots2[best_match_index]:  # Compare the 'Result' values
                matched_shots += 1

    percentage_matching = (matched_shots / len(shots1)) * 100
    return matched_shots, percentage_matching


def compare_timings(timings1, timings2, margin=0.7):
    """Calculates the average timing difference within the ±margin."""
    total_difference = 0
    count = 0
    used_indices = set()

    for t1 in timings1:
        best_match = None
        best_diff = float("inf")

        for i, t2 in enumerate(timings2):
            if i in used_indices:
                continue
            time_diff = abs(t2 - t1)
            if time_diff <= margin and time_diff < best_diff:
                best_match = t2
                best_diff = time_diff

        if best_match is not None:
            used_indices.add(i)
            total_difference += best_diff
            count += 1

    average_difference = total_difference / count if count > 0 else 0
    return average_difference


def main():
    file1 = "Results/DNvsTW.mp4/CSVtest/ground.csv"  # Ground truth
    file2 = "Results/DNvsTW.mp4/CSVtest/YOLO.csv"  # Detected results

    shots1, timings1 = read_csv(file1)
    shots2, timings2 = read_csv(file2)

    array1_size = len(shots1)
    array2_size = len(shots2)
    percentage_detected = (array2_size / array1_size) * 100

    print(f"\nPercentage of shots detected: {percentage_detected:.2f}%")

    matching_shots, percentage_matching = compare_shots_with_timing(shots1, timings1, shots2, timings2)
    print(f"\nNumber of matching shots within margin: {matching_shots} out of {array1_size}")
    print(f"Percentage of matching shots: {percentage_matching:.2f}%")

    average_difference = compare_timings(timings1, timings2)
    print(f"\nAverage timing difference within margin: {average_difference:.6f} seconds")


if __name__ == "__main__":
    main()
