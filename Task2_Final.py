# -*- coding:utf-8 -*-
"""
Date: 17/12/2021
"""


# import  necessary libraries
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.nn as nn
import PIL.Image as Image
import torch.utils.data as data
from torch.utils.data import DataLoader
from torchvision import transforms

'''
# Uncommon this block to see the plot of the accuracy while training (1/3)


import math
from matplotlib.pyplot import MultipleLocator

# Plot the line
def plot_line(x, y, x_label, y_label, x_locator, y_locator, title):
    plt.plot(x, y, c='deepskyblue')
    plt.xticks(rotation=45)
    # Make the plot title
    plt.title(title)
    plt.tick_params(axis='both', which='major', labelsize=10)
    plt.xlabel(x_label, fontsize=8)
    plt.ylabel(y_label, fontsize=8)
    # Set x axis locator
    x_major_locator = MultipleLocator(x_locator)
    # Set y axis locator
    y_major_locator = MultipleLocator(y_locator)
    ax = plt.gca()
    ax.xaxis.set_major_locator(x_major_locator)
    ax.yaxis.set_major_locator(y_major_locator)
    # Set x axis range
    plt.xlim(1, len(x))
    # Set y axis range
    plt.ylim(0, math.ceil(max(y)))
    # Show the plot
    plt.show()
'''

"""
load the data
"""
transforms = transforms.Compose([
    transforms.ToTensor(),  # change the pictures to tensor
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

# All pre-trained models expect input images normalized in the same way
# and then normalized using mean = [0.485, 0.456, 0.406] and std = [0.229, 0.224, 0.225]

class ImageData(data.Dataset):

    def __init__(self, transform):
        root = "./dataset/image"
        image_list = os.listdir("./dataset/image")
        self.imgs = [os.path.join(root, k) for k in image_list]
        self.transforms = transform
        # The dataset MNISTDataset can optionnaly be initialized with a transform function. If such transform
        # function is given it be saved in self.transforms else it will keep its default values None. When calling a
        # new item with __getitem__, it first checks if the transform is a truthy value, in this case it checks if
        # self.transforms can be coerced to True which is the case for a callable object. Otherwise it means
        # self.transforms hasn't been provided in the first place and no transform function is applied on data
        # https://stackoverflow.com/questions/65455986/what-does-does-if-self-transforms-mean

        image_label = pd.read_csv("./dataset/label.csv")
        self.multiclass_vocab = {value: index for index, value in enumerate(set(image_label["label"].values.flatten()))}
        multiclass_label = image_label["label"].apply(lambda x: self.multiclass_vocab[x]).values
        self.labels = multiclass_label  # convert each label to unique number

    def __getitem__(self, index):
        img_path = self.imgs[index]
        pil_img = Image.open(img_path)
        width = pil_img.size[0]
        height = pil_img.size[1]
        pil_img = pil_img.resize((int(width * 0.1), int(height * 0.1)), Image.ANTIALIAS)
        pil_img = self.transforms(pil_img)
        label = torch.tensor(self.labels[index])
        return pil_img, label

    def __len__(self):
        return len(self.imgs)


dataset_2 = ImageData(transforms)
dataloader = DataLoader(dataset_2, batch_size=32, shuffle=True)
# batch_size,  how many samples per batch to load  why 32 -- it is from past experience that 32 is good for CNN
# shuffle,  set to True to have the data reshuffled at every epoch


"""
Build the CNN network
"""
class CNNmodel(nn.Module):
    def __init__(self):
        super(CNNmodel, self).__init__()
        # The super(Net, self).__init__() refers to the fact that this is a subclass of nn.Module and is inheriting
        # all methods.
        # https://medium.com/@ariellemesser/pytorch-nn-module-super-classes-sub-classes-inheritance-and-call-speci-3cc277407ff5
        self.conv1 = torch.nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=6, kernel_size=2),
            # input channel is 3 because of RGB，these hyper parameters are found by several tests to achieve the
            # best performance
            nn.ReLU())

        self.conv2 = torch.nn.Sequential(
            nn.Conv2d(6, 8, 2),
            nn.ReLU())

        self.mlp1 = torch.nn.Linear(1152, 100)
        self.mlp2 = torch.nn.Linear(100, 4)
        # Try with some random number and get the error
        # mat1 and mat2 shapes cannot be multiplied (32x1152 and 115x100)
        # which means 1152 and 100 feature tensors
        # https://stackoverflow.com/questions/69778174/runtime-error-mat1-and-mat2-shapes-cannot-be-multiplied-in-pytorch

    def forward(self, out):
        # The forward function defines how to get the output of the neural net.
        # https://discuss.pytorch.org/t/understand-nn-module/8416
        # Max pooling over a (2, 2) window
        out = F.max_pool2d(self.conv1(out), 2)
        out = F.max_pool2d(self.conv2(out), 2)
        # The view function takes a Tensor and reshapes it.
        out = self.mlp1(out.view(out.size(0), -1))
        out = self.mlp2(out)
        return out


"""
Set loss function and optimizer
"""
# A loss function takes the (output, target) pair of inputs, and computes a value that estimates how far away the
# output is from the target.
# https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html#sphx-glr-beginner-blitz-neural-networks-tutorial-py
model = CNNmodel()
print(model)
error = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
# Adaptive Moment Estimation
# learning rate set to be 0.001

"""
training
"""
loss_list = []
accuracy_record = []
for epoch in range(25):  # should be 25, but for faster test this is set to 5 while coding
    flag = []
    for i, (x, y) in enumerate(dataloader):
        # https://github.com/theneuron19/pytorch/blob/master/CNN_digit_recognizer.py
        out = model(x)  # Forward propagation
        loss = error(out, y)  # Calculate softmax and ross entropy loss
        optimizer.zero_grad()  # Clear gradients
        loss.backward()  # Calculating gradients
        optimizer.step()  # Update parameters

        if i % 10 == 0:
            print("Epoch:{}".format(epoch + 1), loss.item())
            #print(loss.item())
            flag.append(loss.item())
            # append, adds a single item to the existing list
            # https://towardsdatascience.com/append-in-python-41c37453400
            # The item() method extracts the loss’s value as a Python float
            # https://discuss.pytorch.org/t/what-is-loss-item/61218


            '''
            # Uncommon this block to see the plot of the accuracy while training (2/3)
            test_accuracy = 0
            with torch.no_grad():
                for i, (test_data, test_label) in enumerate(dataloader):
                    test_output = model(test_data)
                    pred_y = torch.max(test_output, 1)[1].data.numpy()
                    test_accuracy += float((pred_y == test_label.data.numpy()).astype(int).sum()) / float(
                        test_label.size(0))
                accuracy_record.append((test_accuracy / int(i)))
            '''

    loss_list.append(flag)

'''
# Uncommon this block to see the plot of the accuracy while training (3/3)
test_epoch = [j for j in range(len(accuracy_record))]
plot_line(test_epoch, accuracy_record, 'Number of Iterations', 'Accuracy', 10, 0.1, 'CNN: Accuracy VS Number of Iterations')
'''



"""
prediction
"""
def predict_cnn():
    pred = []
    true_label = []
    with torch.no_grad():
        # skips the gradient calculation over the weights. means not changing any weight in the specified layers.
        # https://stackoverflow.com/questions/63351268/torch-no-grad-affects-on-model-accuracy

        for x, y in dataloader:
            out = model(x)
            pred.extend(list(out.argmax(axis=1).numpy()))
            # Extends list by appending elements from the iterable.
            # https://stackoverflow.com/questions/252703/what-is-the-difference-between-pythons-list-methods-append-and-extend
            # argmax axis = 1, identifies the maximum value for every row. And it returns the column index of that maximum value.
            # https://www.sharpsightlabs.com/blog/numpy-argmax/
            true_label.extend(list(y.numpy()))
    accuracy = (np.array(pred) == true_label).sum() / len(pred)
    print("The accuracy is: {:.4}%".format(accuracy * 100))


predict_cnn()

"""
plot the loss
"""
plt.figure('PyTorch_CNN_Loss')
plt.plot(loss_list[0])
plt.xlim(xmin=0)
plt.ylim(ymin=0)
plt.xlabel("Number of Epochs")
plt.ylabel("Loss")
plt.title("CNN: Loss vs Number of Epochs")
plt.show()

"""
#Uncommmon this block to output a single image with a label 

NewPILimage = Image.open("./new_dataset/test/image/IMAGE_0015.jpg")
Newwidth = NewPILimage.size[0]
Newheight = NewPILimage.size[1]
NewPILimage = NewPILimage.resize((int(Newwidth * 0.1), int(Newheight * 0.1)), Image.ANTIALIAS)
transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
Newimage = transforms(NewPILimage)
with torch.no_grad():
    pred_new = model(Newimage.reshape(1, 3, 51, 51))

print("The type of the tested image is：",pred_new.argmax().numpy())
print(dataset_2.multiclass_vocab)
"""

# the following code is to see the accuracy of an additional dataset using the trained model.
class test_ImageData(data.Dataset):

    def __init__(self, transform):
        root = "./new_dataset/test/image"
        image_list = os.listdir("./new_dataset/test/image")
        self.imgs = [os.path.join(root, k) for k in image_list]
        self.transforms = transform
        # The dataset MNISTDataset can optionnaly be initialized with a transform function. If such transform
        # function is given it be saved in self.transforms else it will keep its default values None. When calling a
        # new item with __getitem__, it first checks if the transform is a truthy value, in this case it checks if
        # self.transforms can be coerced to True which is the case for a callable object. Otherwise it means
        # self.transforms hasn't been provided in the first place and no transform function is applied on data
        # https://stackoverflow.com/questions/65455986/what-does-does-if-self-transforms-mean

        image_label = pd.read_csv("./new_dataset/test/label.csv")
        self.multiclass_vocab = {value: index for index, value in enumerate(set(image_label["label"].values.flatten()))}
        multiclass_label = image_label["label"].apply(lambda x: self.multiclass_vocab[x]).values
        self.labels = multiclass_label  # convert each label to unique number

    def __getitem__(self, index):
        img_path = self.imgs[index]
        pil_img = Image.open(img_path)
        width = pil_img.size[0]
        height = pil_img.size[1]
        pil_img = pil_img.resize((int(width * 0.1), int(height * 0.1)), Image.ANTIALIAS)
        pil_img = self.transforms(pil_img)
        label = torch.tensor(self.labels[index])
        return pil_img, label

    def __len__(self):
        return len(self.imgs)


dataset_test = test_ImageData(transforms)
dataloader_test = DataLoader(dataset_test, batch_size=32, shuffle=True)


def test_predict_cnn():
    pred = []
    true_labe = []
    with torch.no_grad():
        # skips the gradient calculation over the weights. means not changing any weight in the specified layers.
        # https://stackoverflow.com/questions/63351268/torch-no-grad-affects-on-model-accuracy

        for x, y in dataloader_test:
            out = model(x)
            pred.extend(list(out.argmax(axis=1).numpy()))
            # Extends list by appending elements from the iterable.
            # https://stackoverflow.com/questions/252703/what-is-the-difference-between-pythons-list-methods-append-and-extend
            # argmax axis = 1, identifies the maximum value for every row. And it returns the column index of that maximum value.
            # https://www.sharpsightlabs.com/blog/numpy-argmax/
            true_labe.extend(list(y.numpy()))
    accuracy = (np.array(pred) == true_labe).sum() / len(pred)
    print("The accuracy of the new test set is: {:.4}%".format(accuracy * 100))
test_predict_cnn()

