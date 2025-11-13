import os
import shutil
import random
from pathlib import Path

# Config
image_dir = r'C:\Users\...\IMGS-640'
annotation_dir = r'C:\Users\...\xmls'  # Set to None if not using annotations
output_base = r'C:\Users\...\data640'
split_ratio = {
    'train': 0.7,
    'val': 0.1,
    'test': 0.2
}
image_exts = ['.jpg', '.png', '.jpeg', ]  # supported image extensions

# Get all image files
image_files = [f for f in os.listdir(image_dir) if Path(f).suffix.lower() in image_exts]
image_files.sort()
random.shuffle(image_files)

# Split dataset
n_total = len(image_files)
n_train = int(split_ratio['train'] * n_total)
n_val = int(split_ratio['val'] * n_total)
n_test = n_total - n_train - n_val

splits = {
    'train': image_files[:n_train],
    'val': image_files[n_train:n_train + n_val],
    'test': image_files[n_train + n_val:]
}

# Create output folders and copy files
for split, files in splits.items():
    image_out_dir = os.path.join(output_base, split, 'images')
    os.makedirs(image_out_dir, exist_ok=True)

    if annotation_dir:
        annotation_out_dir = os.path.join(output_base, split, 'annotations')
        os.makedirs(annotation_out_dir, exist_ok=True)

    for file in files:
        # Copy image
        src_img = os.path.join(image_dir, file)
        dst_img = os.path.join(image_out_dir, file)
        shutil.copy2(src_img, dst_img)

        # Copy annotation
        if annotation_dir:
            stem = Path(file).stem
            # Support XML or JSON annotations
            for ext in ['.xml', '.json']:
                ann_file = stem + ext
                src_ann = os.path.join(annotation_dir, ann_file)
                if os.path.exists(src_ann):
                    dst_ann = os.path.join(annotation_out_dir, ann_file)
                    shutil.copy2(src_ann, dst_ann)
                    break

print("✅ Dataset split complete.")
