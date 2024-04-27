import os
import cv2
import numpy as np
from PIL import Image

def load_images_and_annotations():
    images = []
    annotations = []

    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.realpath(__file__))

    # Construct the paths to the image and annotation directories
    parent_dir = os.path.dirname(script_dir)
    data_dir = os.path.join(parent_dir, "Dataset", "ORIGA 200 Images")  # Modified path to match the provided structure
    image_dir = data_dir  # Images and annotations are directly under ORIGA 200 Images

    # Iterate through image files
    for filename in os.listdir(image_dir):
        if filename.endswith((".png", ".tif")):  # Modified file extensions to match the provided structure
            image_path = os.path.join(image_dir, filename)
            if filename.endswith(".png"):  # Modified file extension to match the provided structure
                annotation_path = os.path.join(image_dir, filename.replace(".png", ".tif"))
            else:
                annotation_path = os.path.join(image_dir, filename.replace(".tif", ".png"))

            # Read image
            image = cv2.imread(image_path)
            # Convert image to RGB format
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Read annotation (highlighted area)
            annotation = Image.open(annotation_path)
            # Convert annotation to binary mask (e.g., 0 for background, 1 for highlighted area)
            annotation = (np.array(annotation) > 0).astype(np.uint8)

            images.append(image)
            annotations.append(annotation)

    return images, annotations

def preprocess_images(images, annotations, desired_width=256, desired_height=256):
    preprocessed_images = []
    preprocessed_annotations = []

    for image, annotation in zip(images, annotations):
        # Resize images and annotations to a consistent size
        resized_image = cv2.resize(image, (desired_width, desired_height))
        resized_annotation = cv2.resize(annotation, (desired_width, desired_height))

        # Normalize image pixel values to range [0, 1]
        normalized_image = resized_image / 255.0

        preprocessed_images.append(normalized_image)
        preprocessed_annotations.append(resized_annotation)

    return preprocessed_images, preprocessed_annotations

# Load images and annotations
images, annotations = load_images_and_annotations()

# Preprocess images and annotations
preprocessed_images, preprocessed_annotations = preprocess_images(images, annotations)
