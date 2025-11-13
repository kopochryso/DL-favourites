import os 
import cv2
import numpy as np
from PIL import Image

# === CONFIGURATION ===
IMG_DIR = r'C:\Users\...'  # Input image directory
LABEL_DIR = r'C:\Users\...'  # YOLO-style .txt label directory
OUTPUT_IMG_DIR = r'C:\Users\...'
OUTPUT_LABEL_DIR = r'C:\Users\...'

CROP_SIZE = 640
OVERLAP = 0.5  # 50% overlap
EXTENSIONS = ['.jpg', '.JPG']
RETAINED_THRESHOLD = 0.9  # 🆕 Only keep boxes with 90% of original area inside crop

# === UTILITIES ===
os.makedirs(OUTPUT_IMG_DIR, exist_ok=True)
os.makedirs(OUTPUT_LABEL_DIR, exist_ok=True)

def read_yolo_labels(label_path, img_w, img_h):
    boxes = []
    if not os.path.exists(label_path):
        return boxes
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            cls, x_center, y_center, w, h = map(float, parts)
            x_center *= img_w
            y_center *= img_h
            w *= img_w
            h *= img_h
            xmin = int(x_center - w / 2)
            ymin = int(y_center - h / 2)
            xmax = int(x_center + w / 2)
            ymax = int(y_center + h / 2)
            boxes.append((int(cls), xmin, ymin, xmax, ymax))
    return boxes

def save_crop_and_label(crop_img, crop_boxes, crop_index, base_filename, crop_coords):
    crop_filename = f"{base_filename}_crop{crop_index}.jpg"
    crop_path = os.path.join(OUTPUT_IMG_DIR, crop_filename)
    
    # Convert from NumPy BGR to PIL RGB
    Image.fromarray(cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)).save(crop_path)

    # Save corresponding YOLO label
    label_filename = crop_filename.replace('.jpg', '.txt')
    label_path = os.path.join(OUTPUT_LABEL_DIR, label_filename)
    xmin_crop, ymin_crop = crop_coords
    with open(label_path, 'w') as f:
        for cls, xmin, ymin, xmax, ymax in crop_boxes:
            new_xmin = max(0, xmin - xmin_crop)
            new_ymin = max(0, ymin - ymin_crop)
            new_xmax = min(CROP_SIZE, xmax - xmin_crop)
            new_ymax = min(CROP_SIZE, ymax - ymin_crop)
            
            # Reject tiny boxes
            if new_xmax - new_xmin < 2 or new_ymax - new_ymin < 2:
                continue

            # Convert to YOLO format
            cx = (new_xmin + new_xmax) / 2 / CROP_SIZE
            cy = (new_ymin + new_ymax) / 2 / CROP_SIZE
            w = (new_xmax - new_xmin) / CROP_SIZE
            h = (new_ymax - new_ymin) / CROP_SIZE
            f.write(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")

def crop_image_and_labels(image_path, boxes, base_filename):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Failed to read image: {image_path}")
        return

    h, w = img.shape[:2]
    step = int(CROP_SIZE * (1 - OVERLAP))
    crop_index = 0

    for y in range(0, h - CROP_SIZE + 1, step):
        for x in range(0, w - CROP_SIZE + 1, step):
            crop = img[y:y+CROP_SIZE, x:x+CROP_SIZE]
            crop_boxes = []
            for cls, xmin, ymin, xmax, ymax in boxes:
                # Calculate intersection with crop
                ixmin = max(xmin, x)
                iymin = max(ymin, y)
                ixmax = min(xmax, x + CROP_SIZE)
                iymax = min(ymax, y + CROP_SIZE)

                if ixmax <= ixmin or iymax <= iymin:
                    continue  # No overlap

                # 🆕 Retention logic: how much of the original bbox is in the crop
                inter_area = (ixmax - ixmin) * (iymax - iymin)
                orig_area = (xmax - xmin) * (ymax - ymin)
                retained_ratio = inter_area / orig_area

                if retained_ratio < RETAINED_THRESHOLD:
                    continue  # Too little of the object retained

                crop_boxes.append((cls, xmin, ymin, xmax, ymax))

            if crop_boxes:
                save_crop_and_label(crop, crop_boxes, crop_index, base_filename, (x, y))
                crop_index += 1

# === MAIN PROCESS ===
image_files = [f for f in os.listdir(IMG_DIR) if os.path.splitext(f)[1] in EXTENSIONS]
for img_file in image_files:
    base_filename = os.path.splitext(img_file)[0]
    img_path = os.path.join(IMG_DIR, img_file)
    label_path = os.path.join(LABEL_DIR, base_filename + '.txt')

    img = cv2.imread(img_path)
    if img is None:
        continue
    h, w = img.shape[:2]
    boxes = read_yolo_labels(label_path, w, h)
    if not boxes:
        continue  # Skip images with no labels
    crop_image_and_labels(img_path, boxes, base_filename)

print("✅ Cropping complete. Output in:", OUTPUT_IMG_DIR)
