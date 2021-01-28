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

            dft = cv.dft(np.float32(img), flags=cv.DFT_COMPLEX_OUTPUT)
            realDft,imaginaryDft = cv.split(dft)
            features = realDft.flatten()
            features = np.insert(features, 0, i, axis=0)
            with open(name + '.csv', 'a') as record_append:
                np.savetxt(record_append, np.asarray([features]), delimiter=',')




if __name__ == '__main__':
    #featureVectorCreator(trainingDirectory,"TrainingVector")
    #featureVectorCreator(testingDirectory, "TestingVector")


    trainFileName = "TrainingVector.csv"
    trainData = np.genfromtxt(fname=trainFileName, delimiter=',',dtype= float)  # read training data from file
    print("Reading  Training Data")
    trainX = trainData[1:,1:]  # create tranining data matix
    trainY=trainData[1:,0] # create labels array
    print("Training Classifier")
    classifier = KNeighborsClassifier(n_neighbors=3)
    classifier.fit(trainX, trainY)

    # read test data
    testFileName = "TestingVector.csv"
    print("Reading Testing Data")
    testData = np.genfromtxt(fname=testFileName, delimiter=',',dtype= float)

    testX = testData[1:,1:]  # create tranining data matix
    testY= testData[1:,0] # create labels array
    print("Predicting")
    testPrediction=classifier.predict(testX)
    print(accuracy_score(testY, testPrediction))
