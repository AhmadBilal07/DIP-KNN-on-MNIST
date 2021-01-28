# DIP-KNN-on-MNIST
This code applies KNN on MNIST data by applying Fourier Transform on the images.
## Dataset
The MNIST database (Modified National Institute of Standards and Technology database) is a large database of handwritten digits that is commonly used for training various image processing systems. The dataset has been taken from Kaggle, below is the link: </br>
https://www.kaggle.com/jidhumohan/mnist-png

## Original Dataset
The original dataset consists of following numbers

### Training
Character | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9
--- | --- | --- | --- |--- |--- |--- |--- |--- |--- |---
Images Count | 5923 | 6742 | 5958 | 6131 | 5842 | 5421 | 5918 | 6265 | 5851 | 5949 

### Testing
Number | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9
--- | --- | --- | --- |--- |--- |--- |--- |--- |--- |---
Images | 980 | 1135 | 1032 | 1010 | 982 | 892 | 958 | 1028 | 974 | 1009 


## Sample Data
I took a small sample from the original dataset by selecting first 500 images from each character from training folder & similarly 200 from testing folder for training and testing classifier respectively. i.e.

### Training
Character | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9
--- | --- | --- | --- |--- |--- |--- |--- |--- |--- |---
Images Count | 500 | 500 | 500 | 500 | 500 | 500 | 500 | 500 | 500 | 500 

### Testing
Number | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9
--- | --- | --- | --- |--- |--- |--- |--- |--- |--- |---
Images | 200 | 200 | 200 | 200 | 200 | 200 | 200 | 200 | 200 | 200 

---------

