import os
from glob import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch
from torch.utils.data import DataLoader
import albumentations as A
!pip install segmentation_models_pytorch
import segmentation_models_pytorch as smp
from torch.optim import lr_scheduler
from PIL import Image
from tqdm import tqdm
from albumentations.pytorch import ToTensorV2 as ToTensor
from torchvision.transforms import v2
import torch.nn.functional as F
from sklearn.model_selection import train_test_split


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

# Set your root directory

# Define DataFrame for all data
all_img_paths = sorted(glob(os.path.join('Dataset/ProcessedImages/Images', '*.png')))
all_mask_paths = sorted(glob(os.path.join('Dataset/ProcessedImages/Masks', '*.tif')))

# Split the data into training and testing sets
train_img_paths, test_img_paths, train_mask_paths, test_mask_paths = train_test_split(
    all_img_paths, all_mask_paths, test_size=0.2, random_state=42)

# Define DataFrame for training data
train_df = pd.DataFrame({"images": train_img_paths, "masks": train_mask_paths})

# Define DataFrame for test data
test_df = pd.DataFrame({"images": test_img_paths, "masks": test_mask_paths})


print(train_df.head())
print(test_df.head())

#Show the Images
show_imgs = 4
idx = np.random.choice(len(train_df), show_imgs, replace=False)
fig, axes = plt.subplots(show_imgs*2//4, 4, figsize=(15, 8))
axes = axes.flatten()
for i, ax in enumerate(axes):
    new_i = i//2
    if i % 2 ==0 :
        full_path = train_df.loc[idx[new_i]]['images']
        basename = os.path.basename(full_path) 
    else:
        full_path = train_df.loc[idx[new_i]]['masks']
        basename = os.path.basename(full_path) + ' -mask' 
    ax.imshow(plt.imread(full_path))
    ax.set_title(basename)
    ax.set_axis_off()

# Define transformations for training and test data
train_transforms = A.Compose([
    A.Resize(576, 576),
    A.RandomCrop(height=512, width=512, always_apply=True),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.01, scale_limit=(-0.04,0.04), rotate_limit=(-15,15), p=0.5),
    # A.Normalize(p=1.0)
])

test_transforms = A.Compose([
    A.Resize(512, 512),
    # ToTensor(),
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
        # img = self.pre_normalize(img)
        img = torch.tensor(img, dtype=torch.float).permute(2, 0, 1)
        target = torch.tensor(mask, dtype=torch.long)
        # img = pad_to_32(img)
        # target = pad_to_32(target)
        sample = {'x': img, 'y': target}
        return sample


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


# Create train and test datasets
train_dataset = MyDataset(train_df, train_transforms)
val_dataset = MyDataset(test_df, test_transforms)

# Set batch size
BATCH_SIZE = 8

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

print(f'len train: {len(train_df)}')
print(f'len val: {len(test_df)}')

# Define model
model = smp.Unet(
    encoder_name="resnet50",
    encoder_weights="imagenet",
    in_channels=3,
    classes=1,
)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.cuda()  # Move the model to the GPU

print(model(torch.randn((1,3,512,512)).cuda()).shape)


#Training and Evaluation Functions
def train(dataloader, model, loss_fn, optimizer, lr_scheduler):
    size = len(dataloader.dataset) # number of samples
    num_batches = len(dataloader) # batches per epoch
    model.train() # to training mode.
    epoch_loss = 0
    epoch_iou_score = 0
    for batch_i, batch in enumerate(dataloader):
        x, y = batch['x'].to(device), batch['y'].to(device) # move data to GPU
        optimizer.zero_grad()
        pred = model(x)
        loss = loss_fn(pred, y)

        loss.backward() # backpropagation to compute gradients
        optimizer.step() # update model params

        epoch_loss += loss.item() # tensor -> python value
        pred = pred.squeeze(dim=1)
        pred = torch.sigmoid(pred)
        y = y.round().long()
        tp, fp, fn, tn = smp.metrics.get_stats(pred, y, mode='binary', threshold=0.5)
        iou_score = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro").item()
        # print(f'train_acc: {acc}')
        epoch_iou_score += iou_score
        lr_scheduler.step()
    # return avg loss of epoch, acc of epoch
    return epoch_loss/num_batches, epoch_iou_score/num_batches



def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset) # number of samples
    num_batches = len(dataloader) # batches per epoch

    model.eval() # model to test mode.
    epoch_loss = 0
    epoch_iou_score = 0
    # No gradient for test data
    with torch.no_grad():
        for batch_i, batch in enumerate(dataloader):
            x, y = batch['x'].to(device), batch['y'].to(device) # move data to GPU

            # Compute prediction loss
            pred = model(x)
            loss = loss_fn(pred, y)

            # write to logs
            epoch_loss += loss.item()
            pred = pred.squeeze(dim=1)
            pred = torch.sigmoid(pred)
            y = y.round().long()
            tp, fp, fn, tn = smp.metrics.get_stats(pred, y, mode='binary', threshold=0.5)
            iou_score = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro").item()
            # print(f'val_acc: {acc}')
            epoch_iou_score += iou_score
            # size += y.shape[0]
    return epoch_loss/num_batches, epoch_iou_score/num_batches




# Define loss function
loss_fn = smp.losses.DiceLoss(mode="binary")

# Define optimizer
learning_rate = 0.001
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Define learning rate scheduler
step_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)

# Define number of epochs
EPOCHS = 50

# Define logging dictionary
logs = {'train_loss': [], 'val_loss': [], 'train_iou_score': [], 'val_iou_score': []}


best_loss = np.inf
# Training loop
for epoch in tqdm(range(EPOCHS)):
    train_loss, train_iou_score = train(train_loader, model, loss_fn, optimizer, step_lr_scheduler)
    val_loss, val_iou_score = test(val_loader, model, loss_fn)
    logs['train_loss'].append(train_loss)
    logs['val_loss'].append(val_loss)
    logs['train_iou_score'].append(train_iou_score)
    logs['val_iou_score'].append(val_iou_score)
    
    print(f'EPOCH: {str(epoch+1).zfill(3)} \
    train_loss: {train_loss:.4f}, val_loss: {val_loss:.4f} \
    train_iou_score: {train_iou_score:.3f}, val_iou_score: {val_iou_score:.3f} \
    lr: {optimizer.param_groups[0]["lr"]}')

    # Save model checkpoint
    torch.save(model.state_dict(), "./models/New_segmentation.pth")
    
    # Check for improvement and save best model
    if val_loss < best_loss:
        counter = 0
        best_loss = val_loss
        torch.save(model.state_dict(), "./models/New_segmentation.pth")
    else:
        counter += 1
    if counter >= 5:
        print("Earlystop!")
        break

# Plot training and validation metrics
plt.figure(figsize=(15, 5))
plt.subplot(1, 2, 1)
plt.plot(logs['train_loss'], label='Train_Loss')
plt.plot(logs['val_loss'], label='Validation_Loss')
plt.title('Train_Loss & Validation_Loss', fontsize=20)
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(logs['train_iou_score'], label='Train_Iou_Score')
plt.plot(logs['val_iou_score'], label='Validation_Iou_Score')
plt.title('Train_Iou_score & Validation_Iou_score', fontsize=20)
plt.legend()

# Define test Dataset class
class TestDataset(torch.utils.data.Dataset):
    def __init__(self, dataframe, transforms_=None):
        self.df = dataframe
        self.transforms_ = transforms_
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        img = cv2.cvtColor(cv2.imread(self.df.iloc[index]['images']), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.df.iloc[index]['masks'], cv2.IMREAD_GRAYSCALE)
        aug = self.transforms_(image=img, mask=mask)            
        img, mask = aug['image'], aug['mask']
        img_view = np.copy(img)
        img = img / 255
        img = torch.tensor(img, dtype=torch.float).permute(2, 0, 1)
        mask_view = np.copy(mask)
        mask = np.where(mask < 127, 0, 1).astype(np.int16)
        target = torch.tensor(mask, dtype=torch.long)
        sample = {'x': img, 'y': target, 'img_view': img_view, 'mask_view': mask_view}
        return sample

# Create test dataset and DataLoader
test_dataset = TestDataset(test_df, test_transforms)
test_loader = DataLoader(test_dataset, batch_size=4)

# Load best model checkpoint
model.load_state_dict(torch.load('./models/New_segmentation.pth'))
model.to(device)

def get_metrics(model, dataloder, threshold):
    IoU_score, precision, f1_score, recall, acc= 0, 0, 0, 0, 0
    batchs = 0
    model.eval()
    with torch.no_grad():
        for batch_i, batch in enumerate(dataloder):
            x, y = batch['x'].to(device), batch['y'].to(device) # move data to GPU
            pred = model(x)
            pred = pred.squeeze(dim=1)
            pred = torch.sigmoid(pred)
            y = y.round().long()
            tp, fp, fn, tn = smp.metrics.get_stats(pred, y, mode='binary', threshold=threshold)
            batch_iou_score = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro").item()
            batch_acc = smp.metrics.accuracy(tp, fp, fn, tn, reduction="micro").item()
            batch_f1_score = smp.metrics.f1_score(tp, fp, fn, tn, reduction="micro").item()
            batch_recall = smp.metrics.recall(tp, fp, fn, tn, reduction="micro").item()
            batch_precision = smp.metrics.precision(tp, fp, fn, tn, reduction="micro").item()
            IoU_score += batch_iou_score
            acc += batch_acc
            f1_score += batch_f1_score
            recall += batch_recall
            precision += batch_precision
            batchs += 1
    IoU_score = round(IoU_score/batchs, 3)
    precision = round(precision/batchs, 3)
    f1_score = round(f1_score/batchs, 3)
    recall = round(recall/batchs, 3)
    acc = round(acc/batchs, 3)
    sample = {'iou':IoU_score, 'pre':precision, 'fi':f1_score, 're':recall, 'acc':acc}
    return sample

# Define threshold list
threshold_list = [0.3, 0.4, 0.5, 0.6, 0.7]

# Evaluate model on test set with different thresholds
for threshold in threshold_list:
    sample = get_metrics(model, test_loader, threshold)
    print(f"Threshold: {str(threshold)} \
    IoU Score: {sample['iou']:.3f} \
    Precision: {sample['pre']:.3f} \
    F1 Score: {sample['fi']:.3f} \
    Recall: {sample['re']:.3f} \
    Accuracy: {sample['acc']:.3f}")

# Visualize predictions
show_imgs = 4
random_list = np.random.choice(len(test_dataset), show_imgs, replace=False)

for i in range(show_imgs):
    idx = random_list[i]
    sample = test_dataset[idx]
    pred = model(sample['x'].to('cuda', dtype=torch.float32).unsqueeze(0))
    pred = torch.sigmoid(pred).squeeze(0).squeeze(0)
    pred = pred.data.cpu().numpy()
    pred = np.where(pred < 0.5, 0, 1).astype(np.int16)
    pred_img = Image.fromarray(np.uint8(pred), 'L')

    img_view = sample['img_view']
    img_view = Image.fromarray(img_view, 'RGB')
    
    mask_view = sample['mask_view']
    mask_view = Image.fromarray(mask_view, 'L')
                
    f, axarr = plt.subplots(1, 3) 
    axarr[0].imshow(img_view)
    axarr[0].set_title('Input')
    axarr[1].imshow(pred_img)
    axarr[1].set_title('Prediction')
    axarr[2].imshow(mask_view)
    axarr[2].set_title('Ground Truth')
    plt.show()
