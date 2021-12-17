# -*- coding:utf-8 -*-
"""
This is the first trial for the coursework. 
Date: 21/11/2021
"""
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

from imutils import paths
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import cv2

# Load the images directories
path = "C:/Users/Rory1/Desktop/MEng/AMLS/Coursework/dataset/image"
print(os.listdir(path))

image_paths = list(paths.list_images(path))
print(len(image_paths))

#
images = []
labels = []

for image_path in image_paths:
    label = image_path.split(os.path.sep)[-2]
    image = cv2.imread(image_path)
    image = cv2.resize(image, (224, 224))

    images.append(image)
    labels.append(label)

# Plot an image
def plot_image(image):
    plt.imshow(image)

plot_image(images[0])

# Convert into numpy arrays
images = np.array(images) / 255.0
labels = np.array(labels)

# Perform One-hot encoding
label_binarizer = LabelBinarizer()
labels = label_binarizer.fit_transform(labels)
labels = to_categorical(labels)

print(labels[0])
