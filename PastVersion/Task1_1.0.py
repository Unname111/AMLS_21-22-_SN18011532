
"""
import  necessary libraries
"""

import torch.utils.data as data
# It automatically converts NumPy arrays and Python numerical values into PyTorch Tensors.
# https://pytorch.org/docs/stable/data.html#:~:text=It%20automatically%20converts%20NumPy%20arrays,not%20be%20converted%20into%20Tensors).

import PIL.Image as Image
# adds support for opening, manipulating, and saving many different image file formats.
# https://gethowstuff.com/python-pillow-pil-tutorial-examples/

import os
# The OS module in Python provides functions for creating and removing a directory (folder), fetching its contents,
# changing and identifying the current directory, etc.
# https://www.tutorialsteacher.com/python/os-module#:~:text=The%20OS%20module%20in%20Python,identifying%20the%20current%20directory%2C%20etc.

import pandas as pd
# providing data structures designed to make working with “relational” or “labeled” data both easy and intuitive.
# https://pandas.pydata.org/pandas-docs/stable/getting_started/overview.html

import matplotlib.pyplot as plt
# a collection of functions that make matplotlib work like MATLAB. Each pyplot function makes some change to a
# figure: e.g., creates a figure, creates a plotting area in a figure, plots some lines in a plotting area,
# decorates the plot with labels, etc.
# https://matplotlib.org/stable/tutorials/introductory/pyplot.html

# import numpy
import numpy as np
# contains multidimensional array and matrix data structures.
# It provides ndarray, a homogeneous n-dimensional array object, with methods to efficiently operate on it.
# https://numpy.org/doc/stable/user/absolute_beginners.html


# from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA
# Linear dimensionality reduction using Singular Value Decomposition of the data to project it to a lower dimensional
# space.
# http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html

from sklearn.model_selection import train_test_split
# Split arrays or matrices into random train and test subsets
# https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html

from sklearn.tree import DecisionTreeClassifier
# A decision tree classifier.
# https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html

from sklearn.model_selection import GridSearchCV
# Exhaustive search over specified parameter values for an estimator.
# https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html

"""
load the data 
"""
class Imagedata(data.Dataset):
# An abstract class representing a dataset.

    def __init__(self, ):
    # This method called when an object is created from the class and it allow the class to initialize the attributes of a class.
    # https://micropyramid.com/blog/understand-self-and-__init__-method-in-python-class/
        path = "./dataset/image"
        ImageList = os.listdir("./dataset/image") # Return a list containing the names of the files in the directory.
        self.imgs = [os.path.join(path, i) for i in ImageList] # 11111 get the path for each image
        # concatenates various path components with exactly one directory separator ('/') following each non-empty
        # part except the last path component.
        self.transforms = None

    def __getitem__(self, i):
    # used for list indexing, dictionary lookup, or accessing ranges of values
    # https://www.geeksforgeeks.org/__getitem__-in-python/
        ImagePath = self.imgs[i]
        PILimage = Image.open(ImagePath)
        PILimage = PILimage.convert('L') #translate the image to black and white
        height = PILimage.size[1]
        width = PILimage.size[0]
        PILimage = PILimage.resize((int(width * 0.1), int(height * 0.1)), Image.ANTIALIAS)
        # Use the resize method to make the images smaller
        if self.transforms:
            PILimage = self.transforms(PILimage)
        else:
            PILimage = np.array(PILimage)
        return PILimage
    #11111 decide if the images need transformations
    # Transforms this image. This method creates a new image with the given size, and the same mode as the original,
    # and copies data to the new image using the given transform.

    def __len__(self):
        return len(self.imgs)
    #11111  return length, for the dataloader in the later stage

dataset = Imagedata()

"""
load the label
"""
ImageLabel = pd.read_csv("./dataset/label.csv")
BinaryLabel = [1 if label == "no_tumor"
               else 0 for label in ImageLabel["label"].values]

"""
show the examples
"""
"""
def show_image(index,flag):
    plt.figure(figsize=(6,6))
    plt.imshow(dataset_1[index])
    plt.xticks([])
    plt.yticks([])
    plt.title("{}".format(flag))
    plt.show()
show_image(0,"no_tumor")
show_image(1,"tumor")
"""

def show_image(index, title):
    plt.figure(figsize=(5, 5))
    plt.imshow(dataset[index])
    plt.title("{}".format(title))
    plt.xticks([]) # disable x-axis
    plt.yticks([]) # disable y-axis
    plt.show()
IndexNoTumor = [index for index, i in enumerate(BinaryLabel) if i == 1] #generate index for an image lablled no tumor
show_image(np.random.choice(IndexNoTumor), "No Tumor")
IndexTumor = [index for index, i in enumerate(BinaryLabel) if i == 0] # generate index for an image lablled tumor
show_image(np.random.choice(IndexTumor), "Tumor")

"""
reshape the image
"""
BinaryData = np.zeros((3000, dataset[0].reshape(-1).shape[0]))
for index, value in enumerate(dataset):
    BinaryData[index, :] = value.reshape(-1)
# 11111
# When using a -1, the dimension corresponding to the -1 will be the product of the dimensions of the original array
# divided by the product of the dimensions given to reshape so as to maintain the same number of elements.
# https://stackoverflow.com/questions/18691084/what-does-1-mean-in-numpy-reshape


"""
Using PCA technology to select the best dimensionality reduction
It is important to reduce the dimensionality of the data before decision tree, to allow shorter training time and remove 
the unuseful features. 
https://dorukkilitcioglu.com/2018/08/11/pca-decision-tree.html
"""
