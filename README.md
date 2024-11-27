# 🐶🐱 Dog vs Cat Classifier

![Python](https://img.shields.io/badge/python-3.12-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

A convolutional neural network (CNN) project that classifies images as either dogs or cats. This project leverages deep learning techniques to achieve high accuracy in binary image classification tasks.

---

## 📖 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Dataset](#dataset)
- [Dataset Folder Structure](#dataset-folder-structure)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Building the CNN Model](#building-the-cnn-model)
- [Usage](#usage)
- [License](#license)

---

## 🌟 Overview

This project implements a CNN-based deep learning model to classify images into two categories: **dogs** and **cats**. It uses TensorFlow and Keras libraries for training the model on an image dataset and evaluating its performance.

---

## ✨ Features

- Binary classification (Dog vs Cat).
- Pre-trained models for feature extraction.
- Visualization of training progress (accuracy and loss curves).
- Supports batch prediction for multiple images.
- Exportable trained model for deployment.

---

## 📊 Dataset

The dataset used for this project consists of labeled images of dogs and cats.

- Source: [Kaggle - Dogs vs. Cats Dataset](https://www.kaggle.com/c/dogs-vs-cats)
- Format: JPEG images.
- Dataset split:
  - Training: 80%
  - Validation: 10%
  - Testing: 10%

---

## 🗂 Dataset Folder Structure

Organize the dataset into the following structure:

```
data/
├── train/
│   ├── dogs/
│   │   ├── dog1.jpg
│   │   ├── dog2.jpg
│   └── cats/
│       ├── cat1.jpg
│       ├── cat2.jpg
├── validation/
│   ├── dogs/
│   └── cats/
└── test/
    ├── dogs/
    └── cats/
```

---

## 🏗 Project Structure

The project is organized as follows:

```
dog-cat-classifier/
├── data/                # Dataset files
├── notebooks/           # Jupyter notebooks for experiments
│   └── cnn_dog_cat_class.ipynb
├── src/                 # Source code
│   ├── train.py         # Model training script
│   ├── predict.py       # Prediction script
│   └── utils.py         # Utility functions
├── models/              # Trained models
│   └── dog_cat_model.h5
├── docs/                # Documentation files
└── README.md            # Project documentation
```

---

## ⚙️ Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/Dhruvsorathiya25/deep-learning-dog-cat-classifier.git
   cd dog-cat-classifier
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Download the dataset and place it in the `data/` directory.

---

## 🏗 Building the CNN Model

This project uses a convolutional neural network (CNN) to classify images. The architecture includes convolutional layers, max-pooling layers, and fully connected layers.

### Model Architecture:

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    MaxPooling2D(pool_size=(2, 2)),

    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),

    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),

    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')  # Binary classification
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
```

### Training the Model:

```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1./255,
                                   rotation_range=20,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)

validation_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    'data/train/',
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary')

validation_generator = validation_datagen.flow_from_directory(
    'data/validation/',
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary')

history = model.fit(
    train_generator,
    steps_per_epoch=100,
    epochs=20,
    validation_data=validation_generator,
    validation_steps=50)
```

---

## 🚀 Usage

1. **Run the Jupyter Notebook**:
   Open the notebook `cnn_dog_cat_class.ipynb` in Google Colab or locally and execute the cells step-by-step.

2. **Train the Model**:
   Run the training pipeline using:

   ```bash
   python src/train.py
   ```

3. **Make Predictions**:
   Use the trained model to predict:
   ```bash
   python src/predict.py --image_path /path/to/image.jpg
   ```

---

## 📝 License

This project is licensed under the [MIT License](LICENSE).

---
