import os
import json
from PIL import Image

# Define your class mapping (background = 0, then the weeds)
CLASS_MAPPING = {
    0: "AMARE",
    1: "CHEAL",
    2: "CYPSS",
    3: "POROL"
}
# COCO format requires background class, so shift indices
CLASS_MAPPING_COCO = {i+1: name for i, name in CLASS_MAPPING.items()}  
CLASS_MAPPING_COCO[0] = "Background"  # Add background as class 0

# Paths
YOLO_LABELS_DIR = r"C:\Users\...\labels"  # Folder containing .txt labels
IMAGE_DIR = r"C:\Users\...\images"  # Folder containing images
OUTPUT_JSON = "converted_coco-test.json"  # Output COCO JSON

# Initialize COCO dictionary
coco_format = {
    "images": [],
    "annotations": [],
    "categories": [{"id": k, "name": v} for k, v in CLASS_MAPPING_COCO.items()]
}

annotation_id = 0  # ID counter for annotations

# Loop through each YOLO label file
for label_file in os.listdir(YOLO_LABELS_DIR):
    if not label_file.endswith(".txt"):
        continue
    
    image_name = label_file.replace(".txt", ".jpg")  # Assuming images are .jpg
    image_path = os.path.join(IMAGE_DIR, image_name)

    # Check if image exists
    if not os.path.exists(image_path):
        print(f"Warning: {image_path} not found, skipping.")
        continue

    # Get image dimensions
    with Image.open(image_path) as img:
        img_width, img_height = img.size

    # Add image entry
    image_id = len(coco_format["images"]) + 1
    coco_format["images"].append({
        "id": image_id,
        "file_name": image_name,
        "width": img_width,
        "height": img_height
    })

    # Read YOLO label file
    with open(os.path.join(YOLO_LABELS_DIR, label_file), "r") as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split()
        if len(parts) != 5:
            continue  # Skip invalid lines

        class_id, x_center, y_center, width, height = map(float, parts)

        # Convert class_id (YOLO 0-indexed, COCO expects background as 0)
        coco_class_id = int(class_id) + 1  # Shift by +1 to reserve 0 for background

        # Convert from YOLO normalized format to absolute pixel format
        x_min = int((x_center - width / 2) * img_width)
        y_min = int((y_center - height / 2) * img_height)
        x_max = int((x_center + width / 2) * img_width)
        y_max = int((y_center + height / 2) * img_height)

        # Ensure bounding boxes are within image dimensions
        x_min, y_min = max(0, x_min), max(0, y_min)
        x_max, y_max = min(img_width, x_max), min(img_height, y_max)

        # Add annotation
        coco_format["annotations"].append({
            "id": annotation_id,
            "image_id": image_id,
            "category_id": coco_class_id,
            "bbox": [x_min, y_min, x_max - x_min, y_max - y_min],  # COCO format (x, y, width, height)
            "area": (x_max - x_min) * (y_max - y_min),
            "iscrowd": 0
        })
        annotation_id += 1

# Save COCO JSON
with open(OUTPUT_JSON, "w") as f:
    json.dump(coco_format, f, indent=4)

print(f"Conversion completed. COCO JSON saved at {OUTPUT_JSON}")
