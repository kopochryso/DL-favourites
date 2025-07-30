import os
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt

# Path to the directory containing your XML annotation files
xml_directory = r'C:\Users\chryk\Downloads\InnoPP\resized\MPOULO\backup-fullsize\NEA'

# Function to parse XML and extract bounding box aspect ratios
def extract_aspect_ratios(xml_directory):
    aspect_ratios = []
    
    for filename in os.listdir(xml_directory):
        if filename.endswith(".xml"):
            file_path = os.path.join(xml_directory, filename)
            tree = ET.parse(file_path)
            root = tree.getroot()
            
            for object_elem in root.findall('object'):
                bbox = object_elem.find('bndbox')
                if bbox is not None:
                    xmin = int(bbox.find('xmin').text)
                    ymin = int(bbox.find('ymin').text)
                    xmax = int(bbox.find('xmax').text)
                    ymax = int(bbox.find('ymax').text)
                    
                    width = xmax - xmin
                    height = ymax - ymin
                    
                    if height != 0:
                        aspect_ratio = width / height
                        aspect_ratios.append(aspect_ratio)
    
    return aspect_ratios

# Function to plot a histogram of aspect ratios and suggest anchor sizes/ratios
def analyze_aspect_ratios(aspect_ratios):
    # Plot a histogram of aspect ratios
    plt.hist(aspect_ratios, bins=50, edgecolor='black')
    plt.title('Distribution of Aspect Ratios')
    plt.xlabel('Aspect Ratio (Width/Height)')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()

    # Calculate some basic statistics for aspect ratios
    min_ratio = min(aspect_ratios)
    max_ratio = max(aspect_ratios)
    mean_ratio = sum(aspect_ratios) / len(aspect_ratios)
    
    print(f"Min Aspect Ratio: {min_ratio}")
    print(f"Max Aspect Ratio: {max_ratio}")
    print(f"Mean Aspect Ratio: {mean_ratio}")

    # Suggest anchor ratios based on distribution
    if mean_ratio < 1:
        print("Suggested Anchor Ratios: (0.5, 1.0, 2.0) or (0.25, 0.5, 1.0)")
    elif mean_ratio > 1:
        print("Suggested Anchor Ratios: (1.0, 2.0, 4.0) or (1.0, 2.0, 3.0)")
    else:
        print("Suggested Anchor Ratios: (0.5, 1.0, 2.0)")

# Main function
def main():
    aspect_ratios = extract_aspect_ratios(xml_directory)
    
    if aspect_ratios:
        analyze_aspect_ratios(aspect_ratios)
    else:
        print("No bounding boxes found in the XML files.")

# Run the main function
if __name__ == "__main__":
    main()
