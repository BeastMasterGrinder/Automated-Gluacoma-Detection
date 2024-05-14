import cv2
import numpy as np
import torch
from torchvision import transforms

class Preprocessor:
    def __init__(self, resize_shape=(512, 512)):
        self.resize_shape = resize_shape

        # Define transformations for normalization
        self.normalize_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def preprocess_image(self, image_path):
        # Load the image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Unable to load image from path: {image_path}")

        # Convert image to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Resize image
        image = cv2.resize(image, self.resize_shape)

        # Convert image to tensor and normalize
        image = self.normalize_transform(image)

        # Add batch dimension
        image = image.unsqueeze(0)

        return image
