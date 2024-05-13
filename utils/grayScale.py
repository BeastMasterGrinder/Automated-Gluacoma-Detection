import cv2
import os


def print_image_values(image_path):
    # Read the image
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

    # Get the non-zero values
    non_zero_values = image[image != 0]

    # Convert to set and print
    unique_values = set(non_zero_values)
    print(unique_values)



print_image_values('Dataset/ProcessedImages/Masks/1.tif')