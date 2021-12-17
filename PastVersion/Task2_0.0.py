# -*- coding:utf-8 -*-
"""
Date: 12/12/2021
"""
# -*- coding:utf-8 -*-
"""
Date: 05/12/2021
"""

"""
import  necessary libraries
"""

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

"""
load the data
"""
transforms = transforms.Compose([
    transforms.ToTensor(),  # change the pictures to tensor
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])


# All pre-trained models expect input images normalized in the same way
# 11111  and then normalized using mean = [0.485, 0.456, 0.406] and std = [0.229, 0.224, 0.225]

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
# batch_size,  how many samples per batch to load 11111 why 32 -- it is from past experience that 32 is good for CNN
# shuffle,  set to True to have the data reshuffled at every epoch
