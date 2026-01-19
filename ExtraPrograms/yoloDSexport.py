import os
import argparse
import random
import shutil
from tqdm import tqdm
import cv2


def package_yolo_dataset(images_dir, labels_dir, output_zip_path, split_ratio=0.8):
    """
    Takes folders of images and labels and packages them into a full YOLO dataset archive.
    """
    # --- 1. Input Validation and File Discovery ---
    print("Finding and matching image and label files...")
    if not os.path.isdir(images_dir):
        print(f"Error: Images directory not found at '{images_dir}'")
        return
    if not os.path.isdir(labels_dir):
        print(f"Error: Labels directory not found at '{labels_dir}'")
        return

    image_files = {os.path.splitext(f)[0] for f in os.listdir(images_dir) if
                   f.lower().endswith(('.png', '.jpg', '.jpeg'))}
    label_files = {os.path.splitext(f)[0] for f in os.listdir(labels_dir) if f.lower().endswith('.txt')}

    # Find the common files that have both an image and a label
    common_files = sorted(list(image_files.intersection(label_files)))

    if not common_files:
        print("Error: No matching image/label pairs found. Check your file names.")
        return

    print(f"Found {len(common_files)} matching image/label pairs.")

    # --- 2. Setup Temporary Build Directory ---
    build_dir = "temp_yolo_package"
    if os.path.exists(build_dir):
        shutil.rmtree(build_dir)

    os.makedirs(os.path.join(build_dir, "obj_train_data"), exist_ok=True)
    os.makedirs(os.path.join(build_dir, "obj_valid_data"), exist_ok=True)

    # --- 3. Split Data into Training and Validation Sets ---
    random.shuffle(common_files)
    split_index = int(len(common_files) * split_ratio)
    train_files = common_files[:split_index]
    valid_files = common_files[split_index:]
    print(f"Splitting into {len(train_files)} training files and {len(valid_files)} validation files.")

    # --- 4. Copy Files and Create Path Lists ---
    subsets = {
        "train": (train_files, "obj_train_data"),
        "valid": (valid_files, "obj_valid_data")
    }
    path_lists = {"train": [], "valid": []}

    for subset_name, (file_list, data_folder) in subsets.items():
        print(f"\nProcessing '{subset_name}' subset...")
        for base_filename in tqdm(file_list, desc=f"Copying {subset_name} files"):
            # Find the original image extension (.jpg, .png, etc.)
            original_image_name = next(f for f in os.listdir(images_dir) if os.path.splitext(f)[0] == base_filename)

            # Define source and destination paths
            src_image_path = os.path.join(images_dir, original_image_name)
            src_label_path = os.path.join(labels_dir, f"{base_filename}.txt")

            # We will standardize on .jpg for the dataset
            dst_image_path = os.path.join(build_dir, data_folder, f"{base_filename}.jpg")
            dst_label_path = os.path.join(build_dir, data_folder, f"{base_filename}.txt")

            # Copy files, converting image to jpg if necessary
            shutil.copy2(src_label_path, dst_label_path)
            if os.path.splitext(src_image_path)[1].lower() != '.jpg':
                img = cv2.imread(src_image_path)
                cv2.imwrite(dst_image_path, img)
            else:
                shutil.copy2(src_image_path, dst_image_path)

            # Add relative path to the list for train.txt/valid.txt
            path_lists[subset_name].append(os.path.join(data_folder, f"{base_filename}.jpg").replace("\\", "/"))

    # --- 5. Create Metadata Files ---
    print("\nCreating metadata files...")
    # obj.names
    with open(os.path.join(build_dir, "obj.names"), 'w') as f:
        f.write("Ring\n")
        f.write("Ball\n")

    # train.txt
    with open(os.path.join(build_dir, "train.txt"), 'w') as f:
        f.write("\n".join(path_lists["train"]))

    # valid.txt
    with open(os.path.join(build_dir, "valid.txt"), 'w') as f:
        f.write("\n".join(path_lists["valid"]))

    # obj.data
    with open(os.path.join(build_dir, "obj.data"), 'w') as f:
        f.write("classes = 2\n")
        f.write("names = obj.names\n")
        f.write("train = train.txt\n")
        f.write("valid = valid.txt\n")
        f.write("backup = backup/\n")

    # --- 6. Create ZIP Archive and Clean Up ---
    print(f"Creating ZIP archive at {output_zip_path}...")
    archive_base_name = os.path.splitext(output_zip_path)[0]
    shutil.make_archive(archive_base_name, 'zip', build_dir)
    shutil.rmtree(build_dir)

    print("\nDone! YOLO dataset archive created successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Package image and label folders into a full YOLO dataset archive.")
    parser.add_argument('--images_dir', type=str, required=True,
                        help="Path to the folder containing your source images.")
    parser.add_argument('--labels_dir', type=str, required=True,
                        help="Path to the folder containing your source YOLO .txt labels.")
    parser.add_argument('--output_zip', type=str, required=True,
                        help="Path for the final output ZIP file (e.g., 'final_dataset.zip').")
    parser.add_argument('--split_ratio', type=float, default=0.8,
                        help="Train/validation split ratio (default: 0.8 for 80/20 split).")
    args = parser.parse_args()

    package_yolo_dataset(args.images_dir, args.labels_dir, args.output_zip, args.split_ratio)