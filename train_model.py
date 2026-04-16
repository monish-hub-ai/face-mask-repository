import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D, Dropout, Flatten, Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import numpy as np
import os

# Hyperparameters
INIT_LR = 1e-4
EPOCHS = 10  # Set to a reasonable number for training on CPU/standard setup
BS = 32
DIRECTORY = r"c:/PROJECTS/face-mask-detection/dataset"

def build_and_train_model():
    # 1. Data Augmentation and Preprocessing Pipeline
    # Tensors are standardized to have shapes [batch_size, 224, 224, 3] and scaled by 1.0/255.0
    print("[INFO] Loading and augmenting images...")
    aug = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        zoom_range=0.15,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.15,
        horizontal_flip=True,
        fill_mode="nearest",
        validation_split=0.2 # 80-20 split for train/validation
    )

    train_generator = aug.flow_from_directory(
        DIRECTORY,
        target_size=(224, 224),
        batch_size=BS,
        class_mode="categorical",
        subset="training"
    )

    val_generator = aug.flow_from_directory(
        DIRECTORY,
        target_size=(224, 224),
        batch_size=BS,
        class_mode="categorical",
        subset="validation"
    )

    print(f"[INFO] Classes found: {train_generator.class_indices}")

    # 2. Setup MobileNetV2 Architecture
    print("[INFO] Compiling MobileNetV2 model...")
    # Load MobileNetV2 pre-trained on ImageNet, excluding the top dense layers
    baseModel = MobileNetV2(weights="imagenet", include_top=False, input_tensor=Input(shape=(224, 224, 3)))

    # Construct the classification head
    # Tensor shape from baseModel: (None, 7, 7, 1280)
    headModel = baseModel.output
    # Average pooling downsamples to (None, 1, 1, 1280)
    headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
    # Flatten the tensor to a 1D vector -> (None, 1280)
    headModel = Flatten(name="flatten")(headModel)
    # Fully connected layer -> (None, 128)
    headModel = Dense(128, activation="relu")(headModel)
    headModel = Dropout(0.5)(headModel)
    # Output layer for the 2 classes (Mask, without Mask) -> (None, 2)
    headModel = Dense(2, activation="softmax")(headModel)

    # Attach the head to the base model
    model = Model(inputs=baseModel.input, outputs=headModel)

    # Freeze the layers in the base model so they won't be updated during the first training process
    for layer in baseModel.layers:
        layer.trainable = False

    # Compile our model
    opt = Adam(learning_rate=INIT_LR)
    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

    # 3. Train the Model
    print("[INFO] Training head...")
    
    # Handle potentially small validation set scenarios safely
    steps_per_epoch = max(1, train_generator.samples // BS)
    validation_steps = max(1, val_generator.samples // BS)
    
    H = model.fit(
        train_generator,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_generator,
        validation_steps=validation_steps,
        epochs=EPOCHS
    )

    # 4. Save the Model
    print("[INFO] Saving mask detector model...")
    model.save("mask_detector.h5")

    # Plot the training loss and accuracy
    print("[INFO] Plotting evaluation metrics...")
    N = EPOCHS
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
    plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
    plt.title("Training Loss and Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")
    plt.savefig("plot.png")

if __name__ == "__main__":
    build_and_train_model()
