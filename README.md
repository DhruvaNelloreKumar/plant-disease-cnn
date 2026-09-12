Plant Disease Classification using CNN
Project Overview

This project implements a Convolutional Neural Network (CNN) from scratch for classifying plant leaf images into five different disease/health classes.

The model was developed using TensorFlow and Keras and was evaluated using accuracy, loss, confusion matrix, precision, recall, and F1-score.

Dataset

The dataset used is the PlantVillage dataset obtained through Kaggle.

Five classes were selected:

Tomato_Bacterial_spot
Tomato_Early_blight
Tomato_healthy
Tomato_Late_blight
Tomato_Leaf_Mold

Images were resized to 128 × 128 pixels and processed as RGB images.

The dataset was divided into training, validation, and testing sets.

CNN Architecture

The CNN was built from scratch without using transfer learning or a pretrained model.

The architecture consists of:

Conv2D — 32 filters
MaxPooling2D
Conv2D — 64 filters
MaxPooling2D
Conv2D — 128 filters
Global Average Pooling
Dense — 64 neurons
Output Dense layer — 5 neurons

The model contains:

101,829 trainable parameters

Training

The model was trained using the training dataset and monitored using the validation dataset.

Early stopping was used to prevent unnecessary training when validation performance stopped improving.

Results

The final model achieved approximately:

Test Accuracy: 88.22%

Test Loss: 0.3382

Classification Performance
Class	Precision	Recall	F1-score
Tomato_Bacterial_spot	0.97	0.86	0.91
Tomato_Early_blight	0.69	0.94	0.80
Tomato_healthy	0.89	1.00	0.94
Tomato_Late_blight	0.90	0.80	0.85
Tomato_Leaf_Mold	0.91	0.84	0.87

Overall accuracy: 0.88

Confusion Matrix

The confusion matrix was used to compare the predicted classes with the actual classes and identify which plant conditions were most frequently confused by the model.

The largest confusion occurred between some of the visually similar tomato disease classes.

Files
plant_cnn.py — Python implementation of the CNN model, dataset preparation, training, and evaluation.
Technologies Used
Python
TensorFlow
Keras
NumPy
Matplotlib
Scikit-learn
KaggleHub
Conclusion

The CNN successfully classified five plant leaf classes with a test accuracy of approximately 88.22%.

The results show that the model can learn useful visual features from plant leaf images, while some visually similar disease classes remain more difficult to distinguish.
