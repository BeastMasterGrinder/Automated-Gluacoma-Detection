import os
from glob import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch
from torch.utils.data import DataLoader
import albumentations as A
# !pip install segmentation_models_pytorch

import segmentation_models_pytorch as smp
from torch.optim import lr_scheduler
from PIL import Image
from tqdm import tqdm
from albumentations.pytorch import ToTensorV2 as ToTensor
from torchvision.transforms import v2
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, f1_score, jaccard_score

# Step 1: Correct the path
images_path = 'Dataset/ProcessedImages/Images'

# Step 2: Check if images are in the directory
if not os.path.exists(images_path) or not os.listdir(images_path):
    print(f"No images found in {images_path}")
    exit(1)

def pad_to_32(x):
    h, w = x.shape[-2:]
    h_pad = 0 if h % 32 == 0 else 32 - h % 32
    w_pad = 0 if w % 32 == 0 else 32 - w % 32
    return F.pad(x, (0, w_pad, 0, h_pad))

# Define DataFrame for all data
all_img_paths = sorted(glob(os.path.join('Dataset/ProcessedImages/Images', '*.png')))
all_disc_mask_paths = sorted(glob(os.path.join('Dataset/ProcessedImages/DiscMasks', '*.tif')))
all_cup_mask_paths = sorted(glob(os.path.join('Dataset/ProcessedImages/CupMasks', '*-cup.tif')))

# Split the data into training and testing sets for disc model
train_img_paths, test_img_paths, train_disc_mask_paths, test_disc_mask_paths = train_test_split(
    all_img_paths, all_disc_mask_paths, test_size=0.2, random_state=42)

# Split the data into training and testing sets for cup model
train_img_paths_cup, test_img_paths_cup, train_cup_mask_paths, test_cup_mask_paths = train_test_split(
    all_img_paths, all_cup_mask_paths, test_size=0.2, random_state=42)

# Define DataFrame for disc training data
disc_train_df = pd.DataFrame({"images": train_img_paths, "masks": train_disc_mask_paths})

# Define DataFrame for disc test data
disc_test_df = pd.DataFrame({"images": test_img_paths, "masks": test_disc_mask_paths})

# Define DataFrame for cup training data
cup_train_df = pd.DataFrame({"images": train_img_paths_cup, "masks": train_cup_mask_paths})

# Define DataFrame for cup test data
cup_test_df = pd.DataFrame({"images": test_img_paths_cup, "masks": test_cup_mask_paths})

# Define transformations for training and test data
train_transforms = A.Compose([
    A.Resize(576, 576),
    A.RandomCrop(height=512, width=512, always_apply=True),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.01, scale_limit=(-0.04,0.04), rotate_limit=(-15,15), p=0.5),
])

test_transforms = A.Compose([
    A.Resize(512, 512),
])

# Define custom Dataset class
class MyDataset(torch.utils.data.Dataset):
    def __init__(self, dataframe, transforms_=None):
        self.df = dataframe
        self.transforms_ = transforms_
        self.pre_normalize = v2.Compose([v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        self.resize = [512, 512]
        self.class_size = 2
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        img = cv2.cvtColor(cv2.imread(self.df.iloc[index]['images']), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.df.iloc[index]['masks'],cv2.IMREAD_GRAYSCALE)
        mask = np.where(mask<127, 0, 1).astype(np.int16)
        aug = self.transforms_(image=img, mask=mask)            
        img, mask = aug['image'], aug['mask']
        img = img/255
        img = torch.tensor(img, dtype=torch.float).permute(2, 0, 1)
        target = torch.tensor(mask, dtype=torch.long)
        sample = {'x': img, 'y': target}
        return sample

# Define Dataset for disc model
disc_train_dataset = MyDataset(disc_train_df, train_transforms)
disc_val_dataset = MyDataset(disc_test_df, test_transforms)

# Define Dataset for cup model
cup_train_dataset = MyDataset(cup_train_df, train_transforms)
cup_val_dataset = MyDataset(cup_test_df, test_transforms)

# Set batch size
BATCH_SIZE = 4

# Create data loaders for disc model
disc_train_loader = DataLoader(disc_train_dataset, batch_size=BATCH_SIZE, shuffle=True)
disc_val_loader = DataLoader(disc_val_dataset, batch_size=BATCH_SIZE)

# Create data loaders for cup model
cup_train_loader = DataLoader(cup_train_dataset, batch_size=BATCH_SIZE, shuffle=True)
cup_val_loader = DataLoader(cup_val_dataset, batch_size=BATCH_SIZE)

# Define disc model
disc_model = smp.Unet(
    encoder_name="resnet50",
    encoder_weights="imagenet",
    in_channels=3,
    classes=1,
)
disc_model.cuda()

# Define cup model
cup_model = smp.Unet(
    encoder_name="resnet50",
    encoder_weights="imagenet",
    in_channels=3,
    classes=1,
)
cup_model.cuda()

# Define loss function for both models
loss_fn = smp.losses.DiceLoss(mode="binary")

# Define optimizer for disc model
disc_optimizer = torch.optim.Adam(disc_model.parameters(), lr=0.001)

# Define optimizer for cup model
cup_optimizer = torch.optim.Adam(cup_model.parameters(), lr=0.001)

# Define learning rate scheduler for disc model
disc_scheduler = lr_scheduler.StepLR(disc_optimizer, step_size=100, gamma=0.1)

# Define learning rate scheduler for cup model
cup_scheduler = lr_scheduler.StepLR(cup_optimizer, step_size=100, gamma=0.1)

# Define number of epochs
EPOCHS = 50

# Define paths for saving/loading model checkpoints
disc_checkpoint_path = 'models/segmentationDISC.pth'
cup_checkpoint_path = 'models/segmentationCUP.pth'

disc_train_losses = []
cup_train_losses = []

# Training loop for disc model
for epoch in tqdm(range(EPOCHS)):
    disc_model.train()
    disc_epoch_loss = 0.0
    for batch in disc_train_loader:
        inputs, targets = batch['x'].cuda(), batch['y'].cuda()
        disc_optimizer.zero_grad()
        outputs = disc_model(inputs)
        loss = loss_fn(outputs, targets)
        loss.backward()
        disc_optimizer.step()
        disc_epoch_loss += loss.item() * inputs.size(0)
    disc_epoch_loss /= len(disc_train_loader.dataset)
    disc_train_losses.append(disc_epoch_loss)
    disc_scheduler.step()
    # Save model checkpoint
    torch.save(disc_model.state_dict(), disc_checkpoint_path)

# Load disc model checkpoint
disc_model.load_state_dict(torch.load(disc_checkpoint_path))

# Training loop for cup model
for epoch in tqdm(range(EPOCHS)):
    cup_model.train()
    cup_epoch_loss = 0.0
    for batch in cup_train_loader:
        inputs, targets = batch['x'].cuda(), batch['y'].cuda()
        cup_optimizer.zero_grad()
        outputs = cup_model(inputs)
        loss = loss_fn(outputs, targets)
        loss.backward()
        cup_optimizer.step()
        cup_epoch_loss += loss.item() * inputs.size(0)
    cup_epoch_loss /= len(cup_train_loader.dataset)
    cup_train_losses.append(cup_epoch_loss)
    cup_scheduler.step()
    # Save model checkpoint
    torch.save(cup_model.state_dict(), cup_checkpoint_path)

# Load cup model checkpoint
cup_model.load_state_dict(torch.load(cup_checkpoint_path))




# Visualize predictions for both disc and cup models side by side
show_imgs = 4
random_list = np.random.choice(len(disc_val_dataset), show_imgs, replace=False)

# Compute precision, F1 score, and IOU score for disc model
def compute_metrics(model, data_loader):
    model.eval()
    predictions = []
    targets = []
    for batch in data_loader:
        inputs, labels = batch['x'].cuda(), batch['y'].cuda()
        with torch.no_grad():
            outputs = model(inputs)
        predictions.extend(torch.sigmoid(outputs).cpu().numpy())
        targets.extend(labels.cpu().numpy())
    predictions = np.vstack(predictions)
    targets = np.vstack(targets)
    predictions = np.where(predictions < 0.5, 0, 1)
    
    precision = precision_score(targets.flatten(), predictions.flatten())
    f1 = f1_score(targets.flatten(), predictions.flatten())
    iou = jaccard_score(targets.flatten(), predictions.flatten())
    
    return precision, f1, iou

disc_precision, disc_f1, disc_iou = compute_metrics(disc_model, disc_val_loader)
cup_precision, cup_f1, cup_iou = compute_metrics(cup_model, cup_val_loader)

print("Disc Model Metrics:")
print(f"Precision: {disc_precision}")
print(f"F1 Score: {disc_f1}")
print(f"IoU Score: {disc_iou}")

print("Cup Model Metrics:")
print(f"Precision: {cup_precision}")
print(f"F1 Score: {cup_f1}")
print(f"IoU Score: {cup_iou}")


plt.figure(figsize=(10, 5))
plt.plot(range(EPOCHS), disc_train_losses, label='Disc Training Loss')
plt.plot(range(EPOCHS), cup_train_losses, label='Cup Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Loss Curves')
plt.legend()
plt.show()

for i in range(show_imgs):
    idx = random_list[i]
    sample = disc_val_dataset[idx]
    
    # Make prediction using disc model
    disc_pred = disc_model(sample['x'].to('cuda', dtype=torch.float32).unsqueeze(0))
    disc_pred = torch.sigmoid(disc_pred).squeeze(0).squeeze(0)
    disc_pred = disc_pred.data.cpu().numpy()
    disc_pred = np.where(disc_pred < 0.5, 0, 1).astype(np.int16)
    disc_pred_img = Image.fromarray(np.uint8(disc_pred), 'L')
    
    # Make prediction using cup model
    cup_pred = cup_model(sample['x'].to('cuda', dtype=torch.float32).unsqueeze(0))
    cup_pred = torch.sigmoid(cup_pred).squeeze(0).squeeze(0)
    cup_pred = cup_pred.data.cpu().numpy()
    cup_pred = np.where(cup_pred < 0.5, 0, 1).astype(np.int16)
    cup_pred_img = Image.fromarray(np.uint8(cup_pred), 'L')

    img_view = sample['x'].permute(1, 2, 0).cpu().numpy()
    img_view = (img_view * 255).astype(np.uint8)
    img_view = Image.fromarray(img_view, 'RGB')
    
    disc_mask_view = disc_pred_img
    cup_mask_view = cup_pred_img
    
    f, axarr = plt.subplots(1, 4, figsize=(15, 5))
    axarr[0].imshow(img_view)
    axarr[0].set_title('Input')
    axarr[1].imshow(disc_mask_view)
    axarr[1].set_title('Disc Prediction')
    axarr[2].imshow(cup_mask_view)
    axarr[2].set_title('Cup Prediction')
    axarr[3].imshow(sample['y'], cmap='gray')
    axarr[3].set_title('Ground Truth')
    plt.show()
