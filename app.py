import os
import cv2
import streamlit as st
from PIL import Image
import numpy as np
# import your_functions  # Import your pre-made functions for processing and segmenting images
from src.predict_cup_disc import predict_cup_and_disc, load_model
from src.cdr_calculation import calculate_cdr

# def process_image(image):
#     # Process the input image (e.g., resize, normalize, etc.)
#     processed_image = your_functions.process_image(image)
#     return processed_image

def segment_disc_and_cup(image):
    # Segment the optic disc and optic cup from the input image
    disc_image, cup_image = predict_cup_and_disc(image)
    return disc_image, cup_image

def classify_glaucoma(cdr):
    # Classify whether the user has glaucoma based on the calculated CDR
    if cdr > 0.3:  # Define your threshold for CDR indicating glaucoma
        return "Likely has glaucoma"
    else:
        return "Unlikely to have glaucoma"

# Main Streamlit app
def main():
    st.title("Glaucoma Detection App")

    uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg"])

    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        # Save the uploaded image to the Input directory
        image_path = 'uploaded_image.png'
        image.save(image_path)

        # Load the models
        disc_model_path = 'models/segmentationDISC.pth'
        cup_model_path = 'models/segmentationCUP.pth'
        disc_model = load_model(disc_model_path)
        cup_model = load_model(cup_model_path)

        # Segment the optic disc and optic cup
        disc_pred_img, cup_pred_img = predict_cup_and_disc(image_path, disc_model, cup_model)

        # Save the segmented images to the Output directory
        disc_pred_img.save('disc.tif')
        cup_pred_img.save('cup.tif')

        # Calculate the cup-to-disc ratio (CDR)
        cdr = calculate_cdr('disc.tif', 'cup.tif')

        # Classify glaucoma
        glaucoma_status = classify_glaucoma(cdr)

        # Display segmented disc and cup images, and glaucoma status
        st.write("Segmented Optic Disc Image:")
        st.image(disc_pred_img, use_column_width=True)

        st.write("Segmented Optic Cup Image:")
        st.image(cup_pred_img, use_column_width=True)

        st.write(f"Cup-to-Disc Ratio (CDR): {cdr}")
        st.write(f"Glaucoma Status: {glaucoma_status}")

        if st.button("Submit"):
            # Perform any additional actions upon submission (e.g., save results, send notification)
            st.write("Image submitted!")

if __name__ == "__main__":
    main()
