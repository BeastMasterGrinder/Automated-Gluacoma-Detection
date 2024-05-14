import cv2
import numpy as np

def calculate_cdr(disc_image_path, cup_image_path):
    # Load the images in grayscale
    disc_image = cv2.imread(disc_image_path, cv2.IMREAD_GRAYSCALE)
    cup_image = cv2.imread(cup_image_path, cv2.IMREAD_GRAYSCALE)

    # Threshold the images to binary images
    _, disc_image = cv2.threshold(disc_image, 127, 255, cv2.THRESH_BINARY)
    _, cup_image = cv2.threshold(cup_image, 127, 255, cv2.THRESH_BINARY)

    # Calculate the area of the disc and the cup
    disc_area = np.sum(disc_image == 255)
    cup_area = np.sum(cup_image == 255)

    print(disc_area, cup_area)
    # Calculate the Cup to Disc Ratio (CDR)
    cdr = cup_area / disc_area

    return cdr

# disc_path = 'Dataset/ProcessedImages/DiscMasks/1.tif'
# cup_path = 'Dataset/ProcessedImages/CupMasks/1-cup.tif'

# cdr = calculate_cdr(disc_path, cup_path)

# print(cdr)