import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.models import Model
from tensorflow.keras.layers import concatenate

# Define constants
input_shape = (256, 256, 3)  # Input shape for the model
epochs = 20
batch_size = 16

# Data loading and preprocessing
def load_data(data_dir):
    images = []
    masks = []

    for filename in os.listdir(data_dir):
        if filename.endswith(".png"):
            image_path = os.path.join(data_dir, filename)
            mask_path = os.path.join(data_dir, filename.replace(".png", ".tif"))

            # Read and preprocess image
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (input_shape[0], input_shape[1])) / 255.0

            # Read and preprocess mask
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            mask = cv2.resize(mask, (input_shape[0], input_shape[1]))
            mask = (mask > 0).astype(np.uint8)  # Convert to binary mask

            images.append(image)
            masks.append(mask)

    images = np.array(images)
    masks = np.array(masks)

    # Split data into train, validation, and test sets
    X_train, X_val, y_train, y_val = train_test_split(images, masks, test_size=0.2, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_val, y_val, test_size=0.5, random_state=42)

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)

# Model architecture
def create_unet_model(input_shape):
    # Define encoder
    inputs = Input(input_shape)
    conv1 = Conv2D(64, 3, activation='relu', padding='same')(inputs)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same')(pool1)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same')(pool2)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same')(pool3)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    conv5 = Conv2D(1024, 3, activation='relu', padding='same')(pool4)
    
    # Decoder
    upconv6 = Conv2D(512, 2, activation='relu', padding='same')(UpSampling2D(size=(2, 2))(conv5))
    merge6 = concatenate([conv4, upconv6], axis=3)
    conv6 = Conv2D(512, 3, activation='relu', padding='same')(merge6)
    upconv7 = Conv2D(256, 2, activation='relu', padding='same')(UpSampling2D(size=(2, 2))(conv6))
    merge7 = concatenate([conv3, upconv7], axis=3)
    conv7 = Conv2D(256, 3, activation='relu', padding='same')(merge7)
    upconv8 = Conv2D(128, 2, activation='relu', padding='same')(UpSampling2D(size=(2, 2))(conv7))
    merge8 = concatenate([conv2, upconv8], axis=3)
    conv8 = Conv2D(128, 3, activation='relu', padding='same')(merge8)
    upconv9 = Conv2D(64, 2, activation='relu', padding='same')(UpSampling2D(size=(2, 2))(conv8))
    merge9 = concatenate([conv1, upconv9], axis=3)
    conv9 = Conv2D(64, 3, activation='relu', padding='same')(merge9)
    
    # Output layer
    outputs = Conv2D(1, 1, activation='sigmoid')(conv9)

    model = Model(inputs=inputs, outputs=outputs)
    return model

# Training process
def train_model(model, train_data, val_data):
    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Define callbacks
    checkpoint = ModelCheckpoint("optic_disc_segmentation_model.keras", monitor='val_loss', save_best_only=True, verbose=1)

    # Train the model
    history = model.fit(train_data[0], train_data[1], validation_data=val_data, epochs=epochs, batch_size=batch_size, callbacks=[checkpoint])

    # Plot training history
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

# Evaluation process
def evaluate_model(model, test_data):
    # Evaluate the model
    loss, accuracy = model.evaluate(test_data[0], test_data[1])
    return loss, accuracy

# Visualization functions
def visualize_results(images, masks, predictions):
    # Visualize sample results
    for i in range(5):
        plt.figure(figsize=(15, 5))
        plt.subplot(1, 3, 1)
        plt.imshow(images[i])
        plt.title('Original Image')
        plt.axis('off')

        plt.subplot(1, 3, 2)
        plt.imshow(masks[i], cmap='gray')
        plt.title('Ground Truth Mask')
        plt.axis('off')

        plt.subplot(1, 3, 3)
        plt.imshow(predictions[i], cmap='gray')
        plt.title('Predicted Mask')
        plt.axis('off')

        plt.show()

# Main function
def main():
    # Define the directory containing the dataset
    data_dir = "Dataset/ORIGA 200 Images/"

    # Load data
    train_data, val_data, test_data = load_data(data_dir)

    # Create and train U-Net model
    model = create_unet_model(input_shape)
    train_model(model, train_data, val_data)

    # Evaluate model
    loss, accuracy = evaluate_model(model, test_data)
    print("Test Loss:", loss)
    print("Test Accuracy:", accuracy)

    # Visualize results
    predictions = model.predict(test_data[0])
    visualize_results(test_data[0], test_data[1], predictions)

if __name__ == "__main__":
    main()
