import os
import random
import shutil

# Input directories containing all images and labels
images_dir = r"C:\Users\..."
labels_dir = r"C:\Users\..."

# Output directories for split datasets
output_dirs = {
    "train": {"images": "dataset_kalo/train/images", "labels": "dataset_kalo/train/labels"},
    "val": {"images": "dataset_kalo/val/images", "labels": "dataset_kalo/val/labels"},
    "test": {"images": "dataset_kalo/test/images", "labels": "dataset_kalo/test/labels"},
}

# Create output directories
for split, paths in output_dirs.items():
    os.makedirs(paths["images"], exist_ok=True)
    os.makedirs(paths["labels"], exist_ok=True)

# Get all images and shuffle
images = [f for f in os.listdir(images_dir) if f.endswith('.jpg')]
random.seed(42)
random.shuffle(images)

# Split ratios
train_ratio = 0.7
val_ratio = 0.2
test_ratio = 0.1

# Calculate split indices
total_images = len(images)
train_end = int(total_images * train_ratio)
val_end = train_end + int(total_images * val_ratio)

# Perform split
train_images = images[:train_end]
val_images = images[train_end:val_end]
test_images = images[val_end:]

# Helper function to move files
def move_files(split_name, image_list):
    for img in image_list:
        base_name = os.path.splitext(img)[0]
        # Move image file
        shutil.copy(os.path.join(images_dir, img), os.path.join(output_dirs[split_name]["images"], img))
        # Move corresponding label file
        label_file = f"{base_name}.txt"
        if os.path.exists(os.path.join(labels_dir, label_file)):
            shutil.copy(os.path.join(labels_dir, label_file), os.path.join(output_dirs[split_name]["labels"], label_file))

# Move files to respective directories
move_files("train", train_images)
move_files("val", val_images)
move_files("test", test_images)

print(f"Dataset split completed! Total images: {total_images}")
print(f"Train: {len(train_images)}, Val: {len(val_images)}, Test: {len(test_images)}")






