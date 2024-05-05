import os
import cv2
import numpy as np
from PIL import Image
from concurrent.futures import ThreadPoolExecutor

def load_image_and_annotation(filename):
    # Construct the paths to the image and annotation directories
    script_dir = os.path.dirname(os.path.realpath(__file__))
    parent_dir = os.path.dirname(script_dir)
    data_dir = os.path.join(parent_dir, "Dataset", "ORIGA 200 Images")
    image_dir = data_dir

    image_path = os.path.join(image_dir, filename)
    base_filename = os.path.splitext(filename)[0]

    # Read image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Read annotation if it's a .png file
    if filename.endswith(".png"):
        annotation_path = os.path.join(image_dir, filename.replace(".png", ".tif"))
        annotation = Image.open(annotation_path)
        annotation = (np.array(annotation) > 0).astype(np.uint8)
    else:
        annotation = None

    return base_filename, image, annotation

def crop_image(args):
    base_filename, image, annotation = args

    # Find the bounding box of the non-zero area in the annotation
    coords_annotation = np.array(np.nonzero(annotation))
    top_left_annotation = np.min(coords_annotation, axis=1)
    bottom_right_annotation = np.max(coords_annotation, axis=1)

    # Crop the image using the bounding box from the annotation
    cropped_image = image[top_left_annotation[0]:bottom_right_annotation[0], top_left_annotation[1]:bottom_right_annotation[1]]

    return base_filename, cropped_image

# Get the directory of the current script
script_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(script_dir)
data_dir = os.path.join(parent_dir, "Dataset", "ORIGA 200 Images")
image_dir = data_dir

# Load images and annotations in parallel
with ThreadPoolExecutor() as executor:
    results = list(executor.map(load_image_and_annotation, os.listdir(image_dir)))

# Separate images and annotations
images = [(base_filename, image) for base_filename, image, annotation in results]
annotations = {base_filename: annotation for base_filename, image, annotation in results if annotation is not None}

# Crop images in parallel
with ThreadPoolExecutor() as executor:
    cropped_images = list(executor.map(crop_image, [(base_filename, image, annotations[base_filename]) for base_filename, image in images if base_filename in annotations]))

# Save images
output_dir = os.path.join(parent_dir, "Dataset", "ProcessedImages")
os.makedirs(output_dir, exist_ok=True)

for i in range(0, len(cropped_images), 2):
    _, unlabeled_image = cropped_images[i]
    _, labeled_image = cropped_images[i+1]
    cv2.imwrite(os.path.join(output_dir, f"{i//2+1}.png"), cv2.cvtColor(unlabeled_image, cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(output_dir, f"{i//2+1}.tif"), cv2.cvtColor(labeled_image, cv2.COLOR_RGB2BGR))