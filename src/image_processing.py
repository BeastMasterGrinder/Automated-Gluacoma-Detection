import os
import cv2
import numpy as np
from PIL import Image
from concurrent.futures import ThreadPoolExecutor

#Create ProcessedImages folder in Dataset folder
#Create Images and Masks folder in ProcessedImages folder
if not os.path.exists("Dataset/ProcessedImages/Images"):
    os.makedirs("Dataset/ProcessedImages/Images")
if not os.path.exists("Dataset/ProcessedImages/DiscMasks"):
    os.makedirs("Dataset/ProcessedImages/DiscMasks")
if not os.path.exists("Dataset/ProcessedImages/CupMasks"):
    os.makedirs("Dataset/ProcessedImages/CupMasks")
    
def load_image_and_annotation(filename):
    # Construct the paths to the image and annotation directories
    script_dir = os.path.dirname(os.path.realpath(__file__))
    parent_dir = os.path.dirname(script_dir)
    data_dir = os.path.join(parent_dir, "Dataset", "ORIGA 100 Images")
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

    # Add a margin to the bounding box
    margin = 10
    top_left_annotation = np.maximum(top_left_annotation - margin, 0)
    bottom_right_annotation = np.minimum(bottom_right_annotation + margin, np.array(image.shape[:2]) - 1)

    # Crop the image using the bounding box from the annotation
    cropped_image = image[top_left_annotation[0]:bottom_right_annotation[0], top_left_annotation[1]:bottom_right_annotation[1]]

    return base_filename, cropped_image


def enhance_image(image):
    # Convert the image to LAB color space
    image_lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)

    # Split the LAB image into L, A and B channels
    l_channel, a_channel, b_channel = cv2.split(image_lab)

    # Apply CLAHE to L channel
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    l_channel_clahe = clahe.apply(l_channel)

    # Merge the CLAHE enhanced L channel with the original A and B channel
    enhanced_image = cv2.merge((l_channel_clahe, a_channel, b_channel))

    # Convert the image back to RGB color space
    enhanced_image = cv2.cvtColor(enhanced_image, cv2.COLOR_LAB2RGB)

    return enhanced_image


def separate_and_save_cup(base_filename, labeled_image, output_dir):
    # Create a new image where the cup pixels are preserved and all other pixels are set to black
    cup_image = np.where(labeled_image == 255, 255, 0).astype(np.uint8)

    # Save the cup image
    cup_filename = f"{base_filename}-cup.tif"
    cv2.imwrite(os.path.join(output_dir, cup_filename), cup_image)

# Get the directory of the current script
script_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(script_dir)
data_dir = os.path.join(parent_dir, "Dataset", "ORIGA 100 Images")
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

# Save images and masks
output_dir_images = os.path.join(parent_dir, "Dataset", "ProcessedImages", "Images")
output_dir_disc_masks = os.path.join(parent_dir, "Dataset", "ProcessedImages", "DiscMasks")
output_dir_cups_masks = os.path.join(parent_dir, "Dataset", "ProcessedImages", "CupMasks")
os.makedirs(output_dir_images, exist_ok=True)
os.makedirs(output_dir_disc_masks, exist_ok=True)
os.makedirs(output_dir_cups_masks, exist_ok=True)




for i in range(0, len(cropped_images), 2):
    base_filename, unlabeled_image = cropped_images[i]
    highcontrast_image = enhance_image(unlabeled_image)
    
    _, labeled_image = cropped_images[i+1]
    cv2.imwrite(os.path.join(output_dir_images, f"{base_filename}.png"), cv2.cvtColor(highcontrast_image, cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(output_dir_disc_masks, f"{base_filename}.tif"), cv2.cvtColor(labeled_image, cv2.COLOR_RGB2BGR))
    separate_and_save_cup(base_filename, labeled_image, output_dir_cups_masks)