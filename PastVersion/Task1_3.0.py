# -*- coding:utf-8 -*-
"""
This is the second trial of the code.
The code cannot run properly today. Today mainly focus on the understanding of SVM and Kernel trick.
Date: 05/12/2021
"""
from pandas.core.common import random_state

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
class image_data(data.Dataset):
# An abstract class representing a dataset.

    def __init__(self, ):
    # This method called when an object is created from the class and it allow the class to initialize the attributes of a class.
    # https://micropyramid.com/blog/understand-self-and-__init__-method-in-python-class/
        path = "./dataset/image"
        image_list = os.listdir("./dataset/image") # Return a list containing the names of the files in the directory.
        self.imgs = [os.path.join(path, i) for i in image_list] # 11111 get the path for each image
        # concatenates various path components with exactly one directory separator ('/') following each non-empty
        # part except the last path component.
        self.transforms = None

    def __getitem__(self, i):
    # used for list indexing, dictionary lookup, or accessing ranges of values
    # https://www.geeksforgeeks.org/__getitem__-in-python/
        image_path = self.imgs[i]
        pil_image = Image.open(image_path)
        pil_image = pil_image.convert('L') #translate the image to black and white
        height = pil_image.size[1]
        width = pil_image.size[0]
        pil_image = pil_image.resize((int(width * 0.1), int(height * 0.1)), Image.ANTIALIAS)
        # Use the resize method to make the images smaller
        if self.transforms:
            pil_image = self.transforms(pil_image)
        else:
            pil_image = np.array(pil_image)
        return pil_image
    #11111 decide if the images need transformations
    # Transforms this image. This method creates a new image with the given size, and the same mode as the original,
    # and copies data to the new image using the given transform.

    def __len__(self):
        return len(self.imgs)
    #11111  return length, for the dataloader in the later stage

dataset = image_data()

"""
load the label
"""
image_label = pd.read_csv("./dataset/label.csv")
binary_label = [1 if label == "no_tumor"
               else 0 for label in image_label["label"].values]

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
index_notumor = [index for index, i in enumerate(binary_label) if i == 1] #generate index for an image lablled no tumor
show_image(np.random.choice(index_notumor), "No Tumor")
index_tumor = [index for index, i in enumerate(binary_label) if i == 0] # generate index for an image lablled tumor
show_image(np.random.choice(index_tumor), "Tumor")

"""
reshape the image
"""
binary_data = np.zeros((3000, dataset[0].reshape(-1).shape[0]))
for index, value in enumerate(dataset):
    binary_data[index, :] = value.reshape(-1)
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
def threshold(threshold):
    pca = PCA()
    pca.fit_transform(binary_data)
    # Fit the model with X and apply the dimensionality reduction on X.
    # https://www.kite.com/python/docs/sklearn.decomposition.PCA.fit_transform
    i = [i for i, value in enumerate(list(pca.explained_variance_ratio_.cumsum())) if value > threshold][0]
    plt.plot(pca.explained_variance_ratio_.cumsum())
    plt.title("PCA Dimensionality Reduction")
    plt.xlabel("Number of principle components")
    plt.ylabel("Percentage explained variance")
    plt.show()
    return i
best_i = threshold(0.9)
# Dimensionality reduction is achieved by selecting the top k features or features with importance above a
# certain threshold
# lecture note set 5.6

"""
get the best dimension
"""
pca = PCA(n_components=best_i) # Number of components to keep
binary_data_pca = pca.fit_transform(binary_data)
X_train, X_test, Y_train, Y_test = train_test_split(binary_data_pca, binary_label, test_size=0.3, random_state=0)
# test_size specifies the size of the testing dataset
# random_state performs a random split

"""
Training using decision tree use grid parameter
"""
decision_tree = DecisionTreeClassifier(random_state=0)
tree_para = dict(criterion=["gini", "entropy"], max_depth=[5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 20, 25, 30, 50], max_features=["auto", "sqrt", "log2"])
# https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
# https://towardsdatascience.com/scikit-learn-decision-trees-explained-803f3812290d
clf = GridSearchCV(decision_tree, tree_para, cv=5) # number of cross-validation 5 times
clf.fit(X_train, Y_train)


print(clf.best_score_)
print(clf.best_params_)
best_tree = DecisionTreeClassifier(**clf.best_params_)
best_tree.fit(X_train, Y_train)
accuracy = best_tree.score(X_test, Y_test)

print("The decision tree model has an accuracy of: {:.4}%".format(accuracy * 100))

'''
def plot_grid_search(cv_results, grid_param_1, grid_param_2, name_param_1, name_param_2):
    # Get Test Scores Mean and std for each grid search
    scores_mean = cv_results['mean_test_score']
    scores_mean = np.array(scores_mean).reshape(len(grid_param_2),len(grid_param_1))

    scores_sd = cv_results['std_test_score']
    scores_sd = np.array(scores_sd).reshape(len(grid_param_2),len(grid_param_1))

    # Plot Grid search scores
    _, ax = plt.subplots(1,1)

    # Param1 is the X-axis, Param 2 is represented as a different curve (color line)
    for idx, val in enumerate(grid_param_2):
        ax.plot(grid_param_1, scores_mean[idx,:], '-o', label= name_param_2 + ': ' + str(val))

    ax.set_title("Grid Search Scores", fontsize=16)
    ax.set_xlabel(name_param_1, fontsize=16)
    ax.set_ylabel('CV Average Score', fontsize=16)
    ax.legend(loc="best", fontsize=12)
    ax.grid('on')
    plt.show()

# Calling Method
tree_para_1 = dict(criterion=["gini"], max_depth=[5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 20, 25, 30, 50], max_features=["auto", "sqrt", "log2"])
clf_1 = GridSearchCV(decision_tree, tree_para_1, cv=5) # number of cross-validation 5 times
clf_1.fit(X_train, Y_train)
max_depth=[5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 20, 25, 30, 50]
max_features=["auto", "sqrt", "log2"]
plot_grid_search(clf_1.cv_results_, max_depth, max_features, 'Number of Depth (gini)', 'Max Features')

print(clf_1.best_score_)
print(clf_1.best_params_)
best_tree_1 = DecisionTreeClassifier(**clf_1.best_params_)
best_tree_1.fit(X_train, Y_train)
accuracy_1 = best_tree_1.score(X_test, Y_test)
print("The new decision tree model has an accuracy of: {:.4}%".format(accuracy_1 * 100))

tree_para_2 = dict(criterion=["entropy"], max_depth=[5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 20, 25, 30, 50], max_features=["auto", "sqrt", "log2"])
clf = GridSearchCV(decision_tree, tree_para_2, cv=5) # number of cross-validation 5 times
clf.fit(X_train, Y_train)
max_depth=[5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 20, 25, 30, 50]
max_features=["auto", "sqrt", "log2"]
plot_grid_search(clf.cv_results_, max_depth, max_features, 'Number of Depth (entropy)', 'Max Features')

'''

"""


NewPILimage = Image.open("./new_dataset/test/image/IMAGE_0010.jpg")
NewPILimage = NewPILimage.convert('L')  # translate the image to black and white
Newheight = NewPILimage.size[1]
Newwidth = NewPILimage.size[0]
NewPILimage = NewPILimage.resize((int(Newwidth * 0.1), int(Newheight * 0.1)), Image.ANTIALIAS)
Newimage = np.array(NewPILimage).reshape(1,-1)
NewPCAimage = pca.transform(Newimage)
print(BestTree.predict(NewPCAimage)[0]) # Output 1 stands for no tumor, while 0 stands for tumor.
"""


class test_Imagedata(data.Dataset):
    # An abstract class representing a dataset.

    def __init__(self, ):
        # This method called when an object is created from the class and it allow the class to initialize the attributes of a class.
        # https://micropyramid.com/blog/understand-self-and-__init__-method-in-python-class/
        path = "./new_dataset/test/image"
        ImageList = os.listdir(
            "./new_dataset/test/image")  # Return a list containing the names of the files in the directory.
        self.imgs = [os.path.join(path, i) for i in ImageList]  # 11111 get the path for each image
        # concatenates various path components with exactly one directory separator ('/') following each non-empty
        # part except the last path component.
        self.transforms = None

    def __getitem__(self, i):
        # used for list indexing, dictionary lookup, or accessing ranges of values
        # https://www.geeksforgeeks.org/__getitem__-in-python/
        ImagePath = self.imgs[i]
        PILimage = Image.open(ImagePath)
        PILimage = PILimage.convert('L')  # translate the image to black and white
        height = PILimage.size[1]
        width = PILimage.size[0]
        PILimage = PILimage.resize((int(width * 0.1), int(height * 0.1)), Image.ANTIALIAS)
        # Use the resize method to make the images smaller
        if self.transforms:
            PILimage = self.transforms(PILimage)
        else:
            PILimage = np.array(PILimage)
        return PILimage

    # 11111 decide if the images need transformations
    # Transforms this image. This method creates a new image with the given size, and the same mode as the original,
    # and copies data to the new image using the given transform.

    def __len__(self):
        return len(self.imgs)
    # 11111  return length, for the dataloader in the later stage


test_dataset = test_Imagedata()

test_ImageLabel = pd.read_csv("./new_dataset/test/label.csv")
test_BinaryLabel = [1 if label == "no_tumor"
                    else 0 for label in test_ImageLabel["label"].values]

test_BinaryData = np.zeros((200, test_dataset[0].reshape(-1).shape[0]))
for index, value in enumerate(test_dataset):
    test_BinaryData[index, :] = value.reshape(-1)


def test_threshold(threshold):
    pca = PCA()
    pca.fit_transform(test_BinaryData)
    # Fit the model with X and apply the dimensionality reduction on X.
    # https://www.kite.com/python/docs/sklearn.decomposition.PCA.fit_transform
    i = [i for i, value in enumerate(list(pca.explained_variance_ratio_.cumsum())) if value > threshold][0]
    return i


test_BestI = test_threshold(0.9)

test_pca = PCA(n_components=test_BestI)  # Number of components to keep
test_BinaryDataPCA = test_pca.fit_transform(test_BinaryData)
#test_X_test, test_Y_test = test_BinaryDataPCA
test_X_train, test_X_test, test_Y_train, test_Y_test = train_test_split(test_BinaryDataPCA, test_BinaryLabel)
clf.fit(test_X_train, test_Y_train)
test_accuracy = clf.score(test_X_test, test_Y_test)
print("The decision tree model of the new test set has an accuracy of: {:.4}%".format(test_accuracy * 100))
