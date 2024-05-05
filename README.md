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
