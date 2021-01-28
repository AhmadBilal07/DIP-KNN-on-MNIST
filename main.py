import numpy as np
import cv2 as cv
from skimage.io import imread_collection
import os
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn import preprocessing as pre

# Training Directory Path
trainingDirectory = 'training/'

# Testing Directory Path
testingDirectory = 'testing/'

# A function that returns csv feature vector of the directory given in the parameter

def featureVectorCreator(path,name):


    loc = os.listdir(path)
    for i in loc:

        completePath = path + i +'/*.png'
        print(completePath)

        # creating a collection with the available images
        images = imread_collection(completePath)

        # Traversing collection and extracting features
        for img in images:
            # Applying Fourier transform (DFT)
            dft = cv.dft(np.float32(img), flags=cv.DFT_COMPLEX_OUTPUT)
            # Extracting DFT's Real Result
            realDft,imaginaryDft = cv.split(dft)
            # Flattening 2-D array into 1-D
            features = realDft.flatten()
            # Appending Label in the start of Array
            features = np.insert(features, 0, i, axis=0)
            # Storing Features of image in csv file
            with open(name + '.csv', 'a') as record_append:
                np.savetxt(record_append, np.asarray([features]), delimiter=',')




if __name__ == '__main__':

    ## For Preparing Training Feature Vector
    #featureVectorCreator(trainingDirectory,"TrainingVector")

    ## For Preparing Testing Feature Vector
    #featureVectorCreator(testingDirectory, "TestingVector")

    # Reading Training file
    trainFileName = "TrainingVector.csv"
    trainData = np.genfromtxt(fname=trainFileName, delimiter=',',dtype= float)
    print("Reading  Training Data")

    # Separating Predictors/ attributes/ features and Label
    trainX = trainData[1:,1:]  # create tranining data matix
    trainY=trainData[1:,0] # create labels array

    # Training Classifier
    print("Training Classifier")
    classifier = KNeighborsClassifier(n_neighbors=3)
    classifier.fit(trainX, trainY)

    # Reading Testing data
    testFileName = "TestingVector.csv"
    print("Reading Testing Data")
    testData = np.genfromtxt(fname=testFileName, delimiter=',',dtype= float)

    # Separating Predictors/ attributes/ features and Label
    testX = testData[1:,1:]  # create tranining data matix
    testY= testData[1:,0] # create labels array
    print("Predicting")

    # Making Prediction on testing data
    testPrediction=classifier.predict(testX)

    # Checking Accuracy of Classifier
    print(accuracy_score(testY, testPrediction))
