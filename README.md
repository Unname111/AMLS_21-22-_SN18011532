**1. Project Title: Applied Machine Learning Systems Coursework

**2. Introduction of the tasks

This coursework contains two tasks. The first task requires a binary classification of the MRI images with or without a tumour. The second task asks for a multiclassification, meaning different types of tumours need to be distinguished for the given image data.

**3. Description of the organization of the files

The final versions of the code are listed in the main repository. Please use these two to run.

The folder ‘DataRecord’ contains any data generated during the experiments. This is merely the raw data. The processed data is presented in the report.

The folder ‘PastVersion’ stores any code generated during the writing. Some code can be run without a problem, but some are just pieces of classes and modules. This is only for the record and evidence of the progress.

**4. How to run the code

The code is programmed in Python 3.9 but should be working in most popular used versions. The code is written in PyCharm, and all the files are named using .py format. Any suitable IDE can open and run the code.

Path of dataset

Original images  ./dataset/image

Original labels   ./dataset/label.csv

Additional images  ./new_dataset/test/image

Additional labels   ./new_dataset/test/label.csv

For Google Driver: the dataset is contained.

For GitHub, the whole data set cannot be uploaded due to the limitations of maximum upload files. Please use the dataset direcyly from Moodle.

Note: for python 3.10 there might be some problems installing the libraries. If it does happen, use Python 3.9 or try another IDE. 

**5. Necessary packages

For Task 1

torch.utils.data

PIL.Image

os

pandas

matplotlib.pyplot

numpy

sklearn.decomposition

sklearn.model_selection

sklearn.tree

sklearn.model_selection

For Task 2

os

matplotlib.pyplot

numpy

pandas

torch

torch.nn.functional

torch.nn

PIL.Image

torch.utils.data

torchvision

math


