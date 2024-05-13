# Automated Gluacoma Detection

## Setup

This project requires certain dependencies to run. These dependencies are listed in the `requirements.txt` file.

To set up the project:

1. Clone the repository to your local machine.
2. Navigate to the project root directory in your terminal.

If you are on a Windows machine, run the following command:

```cmd
setup.bat
```
If you are not on a Windows machine (like Linux or MacOS), run the following command:

```cmd
sh setup.sh
```
## Image Processing

The `image_processing.py` script performs several tasks:

1. **Load Images and Annotations:** The script loads images and their corresponding annotations from the "ORIGA 200 Images" directory. The images are loaded in RGB format using OpenCV, and the annotations are loaded as binary masks using PIL.

2. **Crop Images:** The script crops each image to the bounding box of the non-zero area in its corresponding annotation. This is done to focus on the area of interest in each image.

3. **Save Processed Images:** The cropped images are saved in the "ProcessedImages" directory. The images are saved as .png files and the annotations are saved as .tif files. The filenames are numbered starting from 1. The unlabeled images (original images) are at even indices and the labeled images (annotations) are at odd indices.

To run the script, navigate to the project root directory in your terminal and run the following command:

```cmd
python src/image_processing.py
```

### Evaluation of Machine Learning Models
The code aims to perform semantic segmentation on retinal fundus images to detect the optic disc and optic cup.

### Key Components
1. **Data Preparation:**
   - Reads image and mask paths from the specified directories.
   - Splits the data into training and testing sets for both disc and cup models.

2. **Data Augmentation:**
   - Utilizes Albumentations library for image augmentation.
   - Randomly crops, flips, rotates, and scales the images for training.

3. **Model Architecture:**
   - Utilizes the UNet architecture implemented in the segmentation_models_pytorch library.
   - ResNet50 is used as the encoder, pre-trained on ImageNet.
   - Separate models are defined for detecting the optic disc and optic cup.

4. **Training:**
   - Implements custom Dataset and DataLoader classes for loading and processing data.
   - Defines loss function as Dice Loss for binary segmentation.
   - Utilizes Adam optimizer with a learning rate scheduler for training both models.
   - Trains each model for a specified number of epochs while saving checkpoints.

5. **Evaluation:**
   - Computes precision, F1 score, and IoU score for both disc and cup models on the validation set.
   - Visualizes training loss curves for monitoring model training.
   - Displays sample predictions alongside ground truth images for qualitative evaluation.

### Dependencies
- Python libraries: os, glob, pandas, numpy, matplotlib, cv2, torch, torchvision, albumentations, segmentation_models_pytorch, PIL, tqdm, sklearn.

### Usage
- Ensure that the dataset is organized as specified in the code (images and masks in separate directories).
- Adjust hyperparameters such as batch size, learning rate, and number of epochs as needed.
- Run the code to train and evaluate the disc and cup segmentation models.

### Results
- Model performance metrics (precision, F1 score, IoU score) are printed for both models.
- Training loss curves are plotted to visualize model convergence.
- Sample predictions are displayed alongside ground truth masks for qualitative assessment.

### Output
- The code generates visualizations, prints performance metrics, and saves model checkpoints for future use.

### Further Steps
- Fine-tune hyperparameters or experiment with different architectures for potential performance improvements.
- Utilize trained models for segmentation tasks in real-world applications.
