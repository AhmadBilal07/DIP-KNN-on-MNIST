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
I took a small sample from the original dataset by selecting first 500 images of each character from training folder & similarly 200 from testing folder for training and testing classifier respectively. i.e.

### Training
Character | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9
--- | --- | --- | --- |--- |--- |--- |--- |--- |--- |---
Images Count | 500 | 500 | 500 | 500 | 500 | 500 | 500 | 500 | 500 | 500 

### Testing
Number | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9
--- | --- | --- | --- |--- |--- |--- |--- |--- |--- |---
Images | 200 | 200 | 200 | 200 | 200 | 200 | 200 | 200 | 200 | 200 

---
## Working
1. Created a function *featureVectorCreator(path,name)* that takes directory path and filename as parameters.

- + It reads all the images one by one,
- + Applies Fourier Transform (DFT), 
- + Extract DFT's result which is stored in first channel, flattens it into 1-D array (Feature Vector of a Single Image)
- + Appends Label to the 1-D Array 
- + Stores feature vectors of all the images in a .csv File.

2. Extracted features for Training Data & Testing Data using above function
3. Train Sklearn's KNN classifier using our trainingvector.csv
4. Test classifer by making predictions on our testing data.
5. Check accuracy using Sklearn's accuracy_score 

## Results
 Accuracy = 81.74%  when n_neighbors = 1 <br/>
 Accuracy = 82.04%  when n_neighbors = 3 <br/>
 Accuracy = 82.79%  when n_neighbors = 5 <br/>
 Accuracy = 82.94%  when n_neighbors = 7 <br/>
---

## Usage
1. Clone Repository
2. Download Dataset
3. Update Training & Testing Path Variables
4. Prepare Training Vector by sending Training Directory Path to Function
5. Repeat process for Testing Data
6. Finally execute the classifier 
