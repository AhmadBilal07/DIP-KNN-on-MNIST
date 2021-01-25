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


    data = []
    loc = os.listdir(path)
    for i in loc:

        completePath = path + i +'/*.png'
        print(completePath)

        # creating a collection with the available images
        images = imread_collection(completePath)

        # Traversing collection and extracting features
        for img in images:

            dft = cv.dft(np.float32(img), flags=cv.DFT_COMPLEX_OUTPUT)
            dft_shift = np.fft.fftshift(dft)
            magnitude_spectrum = 20 * np.log(cv.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))
            magnitude_spectrum = np.around(magnitude_spectrum, decimals=4)
            if np.any(np.isnan(magnitude_spectrum)):
                print("error")
            features = magnitude_spectrum.flatten()
            features = np.insert(features, 0, i, axis=0)

            data.append(features)

    DF = pd.DataFrame(data)
    # save the dataframe as a csv file
    DF.to_csv(name + ".csv",index=False)



if __name__ == '__main__':
    # featureVectorCreator(trainingDirectory,"TrainingVector")
    # featureVectorCreator(testingDirectory, "TestingVector")


    trainFileName = "TrainingVector.csv"
    trainData = np.genfromtxt(fname=trainFileName, delimiter=',',dtype= float)  # read training data from file
    print("Reading  Training Data")
    trainX = trainData[1:,1:]  # create tranining data matix
    trainY=trainData[1:,0] # create labels array
    print("Training Classifier")
    classifier = KNeighborsClassifier(n_neighbors=15)
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
