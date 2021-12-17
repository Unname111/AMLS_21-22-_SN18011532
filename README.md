#  Applied Machine Learning Systems Coursework_18011532

## 1. Introduction of the tasks

- The programmes are written in Python, developed using PyCharm.
- Task 1 requires requires a binary classification of the MRI images with or without a tumour. – Task2 asks for a multiclassification, meaning different types of tumours need to be distinguished for the given image data.
- Task 1 builds the decision tree to solve the problem, and Task 2 builds the CNN model.



## 2. Description of the organization of the files

### ``Task_1_Final.py``

This is the final version of the code for Task 1.

### ``Task_2_Final.py``

This is the final version of the code for Task 2.

###  `dataset` and `new_dataset/test` 

`dataset` contains the training data of 3000 MRI brain images and their corresponding labels. `new_dataset/test` contains the tesing data of 200 MRI brain images and their corresponding labels.

### `DataRecord`

The folder `DataRecord` contains any data generated during the experiments. This is merely the raw data. The processed data is presented in the report.

### `PastVersion`

The folder `PastVersion` stores any code generated during the writing. Some code can be run without a problem, but some are just pieces of classes and modules. This is only for the record and evidence of the progress.

## 3. How to run the code

The code is programmed in Python 3.9 but should be working in most popular used versions. The code is written in PyCharm, and all the files are named using .py format. Any suitable IDE can open and run the code.

1. Download the whole repository.
2. Installe all the necessary Python packages/modules.
3. Choose `Task1_Final` or `Task2_Final` to run the code for Task 1 or Task 2.
4. You can uncommon some blocks in the code to achieve more functions (e.g., plotting graphs and exporting a randome image with predicted label)
Path of dataset

Note: for python 3.10 there might be some problems installing the libraries. If it does happen, use Python 3.9 or try another IDE. 

## 4. Necessary packages

| Library | Version |
| --- | --- |
| torch | 1.10.0 |
| Pillow | 8.4.0 |
|pandas      |                 1.3.4|
|matplotlib      |             3.5.0|
|numpy   |                     1.21.4|
|sklearn        |              0.0|
|torch             |           1.10.0|
|torchvision      |            0.11.1|
|math||
|os |  |

