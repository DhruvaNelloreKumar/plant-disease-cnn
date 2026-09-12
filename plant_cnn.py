import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os

# Dataset location
dataset_path = r"C:\Users\Dhruva\.cache\kagglehub\datasets\emmarex\plantdisease\versions\1\PlantVillage"

# Five selected classes
classes = [
    "Tomato_Bacterial_spot",
    "Tomato_Early_blight",
    "Tomato_healthy",
    "Tomato_Late_blight",
    "Tomato_Leaf_Mold"
]

# Parameters
IMG_SIZE = (128, 128)
BATCH_SIZE = 32
SEED = 42

print("Selected classes:")
for c in classes:
    print(c)

# Create training dataset (70%)
train_ds = tf.keras.utils.image_dataset_from_directory(
    dataset_path,
    labels="inferred",
    label_mode="int",
    class_names=classes,
    validation_split=0.30,
    subset="training",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

# Create temporary 30% dataset
temp_ds = tf.keras.utils.image_dataset_from_directory(
    dataset_path,
    labels="inferred",
    label_mode="int",
    class_names=classes,
    validation_split=0.30,
    subset="validation",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

# Number of batches in temporary dataset
temp_batches = tf.data.experimental.cardinality(temp_ds).numpy()

# Split temporary dataset into validation and test
val_batches = temp_batches // 2

val_ds = temp_ds.take(val_batches)
test_ds = temp_ds.skip(val_batches)

print("\nDataset sizes:")
print("Training batches:", tf.data.experimental.cardinality(train_ds).numpy())
print("Validation batches:", tf.data.experimental.cardinality(val_ds).numpy())
print("Testing batches:", tf.data.experimental.cardinality(test_ds).numpy())

# Normalization
normalization_layer = layers.Rescaling(1.0 / 255)

# Data augmentation
data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1)
])

train_ds = train_ds.map(
    lambda x, y: (data_augmentation(normalization_layer(x)), y)
)

val_ds = val_ds.map(
    lambda x, y: (normalization_layer(x), y)
)

test_ds = test_ds.map(
    lambda x, y: (normalization_layer(x), y)
)

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)
test_ds = test_ds.prefetch(AUTOTUNE)

model = keras.Sequential([

    # Input
    layers.Input(shape=(128, 128, 3)),

    # Convolution Block 1
    layers.Conv2D(32, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),

    # Convolution Block 2
    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),

    # Convolution Block 3
    layers.Conv2D(128, (3, 3), activation="relu"),

    # Reduce feature maps
    layers.GlobalAveragePooling2D(),

    # Fully connected layer
    layers.Dense(64, activation="relu"),

    # Output layer
    layers.Dense(5, activation="softmax")
])

model.summary()

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

print("Model compiled successfully!")

early_stopping = keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True
)

print("Early stopping configured!")

#  history = model.fit(
#  train_ds,
#  validation_data=val_ds,
#  epochs=20,
# callbacks=[early_stopping]
# )

# Load the saved trained model
model = keras.models.load_model("plant_disease_model.keras")

print("Model loaded successfully!")

# Evaluate on test set
test_loss, test_accuracy = model.evaluate(test_ds)

print("\nTest Loss:", test_loss)
print("Test Accuracy:", test_accuracy)

import numpy as np
from sklearn.metrics import confusion_matrix, classification_report

# Get true labels and predictions
y_true = []
y_pred = []

for images, labels in test_ds:
    predictions = model.predict(images, verbose=0)

    y_true.extend(labels.numpy())
    y_pred.extend(np.argmax(predictions, axis=1))

# Convert to numpy arrays
y_true = np.array(y_true)
y_pred = np.array(y_pred)

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)

print("\nConfusion Matrix:")
print(cm)

# Classification report
print("\nClassification Report:")
print(classification_report(
    y_true,
    y_pred,
    target_names=classes
))

import matplotlib.pyplot as plt

# Training results from the completed training run
epochs = list(range(1, 15))

train_accuracy = [
    0.3815, 0.6849, 0.7518, 0.7850,
    0.8172, 0.8449, 0.8456, 0.8711,
    0.8720, 0.8783, 0.8750, 0.8850,
    0.8924, 0.9005
]

val_accuracy = [
    0.4965, 0.7309, 0.6311, 0.7951,
    0.7040, 0.8333, 0.7587, 0.8056,
    0.8672, 0.7318, 0.8819, 0.8733,
    0.8273, 0.8594
]

train_loss = [
    1.4174, 0.8274, 0.6553, 0.5558,
    0.4998, 0.4291, 0.4185, 0.3632,
    0.3639, 0.3396, 0.3389, 0.3162,
    0.3001, 0.2753
]

val_loss = [
    1.1721, 0.7234, 0.9387, 0.5298,
    0.7411, 0.4524, 0.5742, 0.4952,
    0.3718, 0.7112, 0.3246, 0.3544,
    0.4347, 0.3704
]

# Accuracy graph
plt.figure()
plt.plot(epochs, train_accuracy, label="Training Accuracy")
plt.plot(epochs, val_accuracy, label="Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training and Validation Accuracy")
plt.legend()
plt.show()

# Loss graph
plt.figure()
plt.plot(epochs, train_loss, label="Training Loss")
plt.plot(epochs, val_loss, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss")
plt.legend()
plt.show()
