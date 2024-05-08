import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import Xception
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, Conv2D, UpSampling2D, Reshape
from tensorflow.python.client import device_lib
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import MeanIoU

# Define constants
input_shape = (224, 224, 3)  # Input shape for the model
batch_size = 3

# Data loading and preprocessing
def load_data(data_dir):
    images = []
    masks = []

    for filename in os.listdir(data_dir):
        if filename.endswith(".png"):
            image_path = os.path.join(data_dir, filename)
            mask_path = os.path.join(data_dir, filename.replace('.png', '.tif'))

            # Read and preprocess image
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (input_shape[0], input_shape[1]))

            # Read and preprocess mask
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            mask = cv2.resize(mask, (input_shape[0], input_shape[1]))

            images.append(image)
            masks.append(mask)

    images = np.array(images) / 255.0
    masks = np.array(masks) / 255.0

    # Split data into train, validation, and test sets
    X_train, X_val, y_train, y_val = train_test_split(images, masks, test_size=0.2, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_val, y_val, test_size=0.5, random_state=42)

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)

# Model architecture
def create_segmentation_model(input_shape):
    base_model = Xception(weights='imagenet', include_top=False, input_shape=input_shape)
    for layer in base_model.layers:
        layer.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(input_shape[0] * input_shape[1], activation='sigmoid')(x)
    x = Reshape((input_shape[0], input_shape[1]))(x)
    
    model = Model(inputs=base_model.input, outputs=x)
    
    return model

# Data augmentation
def augment_data(train_data):
    datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.3,
        height_shift_range=0.3,
        shear_range=0.3,
        zoom_range=0.3,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode='reflect'
    )
    datagen.fit(train_data[0])
    return datagen

# Train model with data augmentation
def train_model_with_augmentation(model, train_data, val_data, datagen):
    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy', MeanIoU(num_classes=2)])

    checkpoint = ModelCheckpoint("./models/segmentation_model.keras", monitor='val_loss', save_best_only=True, verbose=1)
    early_stopping = EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True)

    history = model.fit(datagen.flow(train_data[0], train_data[1], batch_size=batch_size),
                        validation_data=val_data,
                        steps_per_epoch=len(train_data[0]) // batch_size,
                        epochs=200,
                        callbacks=[checkpoint, early_stopping])
    return history

# Evaluate model
def evaluate_model(model, test_data):
    loss, accuracy, iou = model.evaluate(test_data[0], test_data[1])
    return loss, accuracy, iou

# Visualize results
def visualize_results(images, masks, predictions):
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
    data_dir = "Dataset/ProcessedImages/"
    train_data, val_data, test_data = load_data(data_dir)

    model = create_segmentation_model(input_shape)
    datagen = augment_data(train_data)
    train_history = train_model_with_augmentation(model, train_data, val_data, datagen)

    loss, accuracy, iou = evaluate_model(model, test_data)
    print("Test Loss:", loss)
    print("Test Accuracy:", accuracy)
    print("Intersection over Union (IoU):", iou)

    predictions = model.predict(test_data[0])
    visualize_results(test_data[0], test_data[1], predictions)

if __name__ == "__main__":
    main()
