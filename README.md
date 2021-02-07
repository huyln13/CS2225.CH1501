# CS2225.CH1501

## 1. Introduction
This project aims to identify signs of diabetic retinopathy in [Diabetic Retinopathy Detection challenge](https://www.kaggle.com/c/diabetic-retinopathy-detection/overview) for school project. Using Densenet 121 and balanced weights cross entropy for classification.

## 2. Installation and Requirements
### 2.1. Requirements:
The system requires the following:
  - Python3
  - Pytorch
  - Opencv
  - Numpy
  - jupyter notebook
  - torchinfo
 
### 2.2. Installation:
The software cloned with:
```sh
$ git clone https://github.com/huyln13/CS2225.CH1501
```
After cloning it, all dependencies can be installed using *requirements.txt*.
```sh
$ pip install -r requirements.txt
```

## 3. Running
### 3.1. Training
Training script [train.py](main/train.py).
### 3.2. Testing
Test on images using notebook [test.ipynb](main/test.ipynb).
