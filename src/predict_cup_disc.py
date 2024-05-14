import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import albumentations as A
from segmentation_models_pytorch import Unet
import cv2
from utils.grayScale import print_image_values


def load_model(model_path):
    # Load the model architecture
    model = Unet(
        encoder_name="resnet50",
        encoder_weights="imagenet",
        in_channels=3,
        classes=1)
    # Load the model state dictionary
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    # model.load_state_dict(torch.load(model_path))
    
    return model

def predict_cup_and_disc(image_path, disc_model, cup_model):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Move the models to the appropriate device
    disc_model = disc_model.to(device)
    cup_model = cup_model.to(device)

    # Ensure the models are in evaluation mode
    disc_model.eval()
    cup_model.eval()

    # Load the image
    image = Image.open(image_path)

    # Define the transformations
    transform = A.Compose([
        A.Resize(512, 512),
    ], p=1)

    # Apply the transformations
    transformed = transform(image=np.array(image))

    # Extract transformed image
    image = transformed['image']
    
    # Convert image to tensor and move to device
    image = transforms.ToTensor()(image).unsqueeze(0).to(device)

    # Get the predictions
    with torch.no_grad():
        disc_pred = disc_model(image)
        cup_pred = cup_model(image)

    # Apply sigmoid and convert to numpy arrays
    disc_pred = torch.sigmoid(disc_pred).squeeze(0).squeeze(0).cpu().numpy()
    cup_pred = torch.sigmoid(cup_pred).squeeze(0).squeeze(0).cpu().numpy()
    
    # Convert predictions to binary masks
    disc_pred = np.where(disc_pred < 0.5, 0, 1).astype(np.int16)
    cup_pred = np.where(cup_pred < 0.5, 0, 1).astype(np.int16)

    # Convert predictions to PIL images
    disc_pred_img = Image.fromarray(np.uint8(disc_pred * 255), 'L')
    cup_pred_img = Image.fromarray(np.uint8(cup_pred * 255), 'L')

    return disc_pred_img, cup_pred_img
