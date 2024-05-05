import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.models import Model
from tensorflow.keras.layers import concatenate
from tensorflow.python.client import device_lib

from tensorflow.keras.layers import Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.regularizers import l2



# Define constants
input_shape = (256, 256, 3)  # Input shape for the model
epochs = 5
batch_size = 6

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

    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_regularizer=l2(0.001))(inputs)
    conv1 = BatchNormalization()(conv1)
    conv1 = Dropout(0.2)(conv1)
    
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    
    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_regularizer=l2(0.001))(pool1)
    conv2 = BatchNormalization()(conv2)
    conv2 = Dropout(0.2)(conv2)
    
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    
    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_regularizer=l2(0.001))(pool2)
    conv3 = BatchNormalization()(conv3)
    conv3 = Dropout(0.2)(conv3)
    
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    
    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_regularizer=l2(0.001))(pool3)
    conv4 = BatchNormalization()(conv4)
    conv4 = Dropout(0.2)(conv4)
    
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    
    conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_regularizer=l2(0.001))(pool4)
    conv5 = BatchNormalization()(conv5)
    conv5 = Dropout(0.2)(conv5)
    
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
    
    conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_regularizer=l2(0.001))(merge9)
    conv9 = BatchNormalization()(conv9)
    conv9 = Dropout(0.2)(conv9)
    
    # Output layer
    outputs = Conv2D(1, 1, activation='sigmoid')(conv9)

    model = Model(inputs=inputs, outputs=outputs)
    return model


def augment_data(X_train, y_train):
    # Add a channels dimension to the data if it's missing
    if X_train.ndim == 3:
        X_train = np.expand_dims(X_train, axis=-1)
    if y_train.ndim == 3:
        y_train = np.expand_dims(y_train, axis=-1)

    data_gen_args = dict(rotation_range=0.2,
                         width_shift_range=0.05,
                         height_shift_range=0.05,
                         shear_range=0.05,
                         zoom_range=0.05,
                         horizontal_flip=True,
                         fill_mode='nearest')
    image_datagen = ImageDataGenerator(**data_gen_args)
    mask_datagen = ImageDataGenerator(**data_gen_args)
    image_datagen.fit(X_train, augment=True)
    mask_datagen.fit(y_train, augment=True)
    image_generator = image_datagen.flow(X_train, batch_size=batch_size, seed=1)
    mask_generator = mask_datagen.flow(y_train, batch_size=batch_size, seed=1)
    
    train_generator = list(zip(image_generator, mask_generator))
    return train_generator


# Training process
def train_model(model, train_data, val_data):
    # Compile the model
    X_train, y_train = zip(*train_data)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Define callbacks
    checkpoint = ModelCheckpoint("optic_disc_segmentation_model.keras", monitor='val_loss', save_best_only=True, verbose=1)
    early_stopping = EarlyStopping(monitor='val_accuracy', patience=3, restore_best_weights=True)

    # Train the model
    history = model.fit(np.array(X_train), np.array(y_train), validation_data=val_data, epochs=epochs, batch_size=batch_size, callbacks=[checkpoint, early_stopping])

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
        

def get_available_devices():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos]



# Main function
def main():
    # Define the directory containing the dataset
    data_dir = "Dataset/ORIGA 200 Images/"

    # Load data
    train_data, val_data, test_data = load_data(data_dir)

    # Augment data
    train_generator = augment_data(*train_data)

    # Create and train U-Net model
    model = create_unet_model(input_shape)
    print(get_available_devices())
    train_model(model, train_generator, val_data)

    # Evaluate model
    loss, accuracy = evaluate_model(model, test_data)
    print("Test Loss:", loss)
    print("Test Accuracy:", accuracy)

    # Visualize results
    predictions = model.predict(test_data[0])
    visualize_results(test_data[0], test_data[1], predictions)
    


if __name__ == "__main__":
    main()
