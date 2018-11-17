import predictTNA
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn import svm
from sklearn.model_selection import KFold
from sklearn import cross_validation as cv
from sklearn import metrics
from sklearn import preprocessing
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
import pickle
import matplotlib.pyplot as plt


# This method is passed predicted and observed values, simply returning the AUC of the given predictions/model
def accuracy(predicted, observations):
    observationsDataframe = observations.copy()
    observationsArray = observationsDataframe.values

    # Simply initialises the necessary variables needed for calculating the AUC
    fpr, tpr, thresholds = metrics.roc_curve(observationsArray, predicted)
    AUC = metrics.auc(fpr, tpr)

    # AUC is calculated and returned
    return AUC


# This model is passed a model, the data it will be validated against and the target data column name
def testModel(model, test, tLabel):
    testCopy = test.copy()

    # testVals represents the values which will be used in the prediction step
    testVals = testCopy.drop(tLabel, axis=1)

    # predictedValues represents the values predicted by the model for the given testVals
    predictedValues = model.predict(testVals)

    # The actual observations for which we predicted are then copied into observedValues
    observedValues = testCopy[tLabel].copy()

    # We then return what the method accuracy() returns when passed the predictedValues and the actual observedValues
    return accuracy(predictedValues, observedValues)


# This method, as the name suggests, is responsible for training the models
# It is passed a dataFrame, which contains the training data
# tLabel which is the name of the target data column
# depth which defines depth of trees
# estimators defines number of trees


def trainData(dataFrame, tLabel, depth, estimators):
    df = dataFrame.copy()
    # Data is partitioned between the data used as parameters (training) and the target data (targetData)
    training = df.drop(tLabel, axis=1)
    targetData = dataFrame[tLabel].copy()

    # produces the model
    model = RandomForestClassifier(n_estimators=estimators, max_depth=depth, random_state=0)
    # the model is then fit to the training data
    model.fit(training, targetData)

    print(depth)
    print(model.feature_importances_)
    return model


# This method is passed a dataframe, a depth which defines the depth of the tree, estimators which determiens number of trees made
# splits which defines the number of 'folds' to make in the data for training and validation
# tLabel which defines the target label, i.e. the data for which we are training to predict


def crossValidation(dataFrame, depth, estimators, splits, tLabel):
    df = dataFrame.copy()

    # Initialise kf with the given number of splits
    kf = KFold(n_splits=splits)
    bestModel = None
    bestModelResult = 0

    # For the training data and test data in each "split"/fold, we produce a model using the training
    # data, tLabel and k, where we then test this model using the produced model, test data and the target label
    for training, test in kf.split(df):
        trainingData = df.ix[training]
        testData = df.ix[test]

        model = trainData(trainingData, tLabel, depth, estimators)
        result = testModel(model, testData, tLabel)

        # This simply selects the best model seen so far out of all the folds
        # For example with splits = 10, this loop will be run 10x producing 10 models
        # This code then returns the best of these 10 models, i.e. the best model for this given k
        if (result > bestModelResult):
            bestModelResult = result
            bestModel = model

    return [bestModelResult, depth, bestModel, estimators]


# This method is passed a dataFrame and a column name to be dropped (so it doesn't have preprocessing run on it)
# & returns the preprocessed dataFrame


def preprocessing(dataFrame, tLabel):
    df = dataFrame.copy()

    # 'none' is a special tLabel value passed in when we dont want to drop any column from the dataFrame
    # If it is not equal to 'none' then dorp the iven column name and continue with the preprocessing
    if (tLabel != 'none'):
        df = df.drop(tLabel, axis=1)

    # The mean is calculated
    mean = df.mean(axis=0)

    # standard variation calculated
    stdVar = df.std(axis=0)

    # The data is then scaled by subtracting the mean from each value and dividing through by the standard variation
    df = df.sub(mean, axis=1)
    df = df / stdVar

    return df


# This method is responsible for taking a dataframe, three column names and outputting a dataframe
# with the three columns ordered such that col1Val <= col2Val <= col3Val
# This was done as it was seen that the data appeared to be "out-of-order", and after re-arranging the values
# performance appeared to increase


def orderData(dataFrame, col1, col2, col3):
    df = dataFrame.copy()

    # Iterate through row by row in the dataframe
    for index, row in df.iterrows():
        # Assign the values in column 1, 2 & 3 to their corresponding variables
        col1Val = row[col1]
        col2Val = row[col2]
        col3Val = row[col3]

        # The values are then put in an array where the built in .sort() method is used to arrange the values
        values = [col1Val, col2Val, col3Val]
        values.sort()

        # Now the data in the given row is simply re-assigned such that col1Val <= col2Val <= col3Val
        row[col1] = values[0]
        row[col2] = values[1]
        row[col3] = values[2]
    return df


def main():
    # Reads the probeA csv file into a panda dataframe
    probeAData = pd.read_csv('../probeA.csv')

    # This chunk of code calls orderData, which goes through the passed in
    # headers, swapping values such that the first header has all the smallest
    # values, the middle with the middle value and the last header with the largest
    # value. This was done as after inspecting the data, where it became apparent this
    # was a common trend in the data, with them occasionally being out of order.
    # After swapping these around, an improvement was seen in the AUC score, so was kept
    firstChemOrdered = orderData(probeAData, 'c1', 'c2', 'c3')
    secondChemOrdered = orderData(firstChemOrdered, 'm1', 'm2', 'm3')
    thirdChemOrdered = orderData(secondChemOrdered, 'n1', 'n2', 'n3')
    fourthChemOrdered = orderData(thirdChemOrdered, 'p1', 'p2', 'p3')
    probeAData = fourthChemOrdered.copy()

    # After swapping the data around into a fixed order, the data is then passed to preprocessing
    # As the name suggests, this method performs preprocessing on the data, returning a
    # dataframe containing the preprocessed data
    # 'tna' is passed into the method to ensure that it isn't included in the preprocessing
    # step. As, testing showed that performing preprocessing steps on tna lead to a decrease
    # in performance in terms of accuracy and AUC
    probeAPreprocessed = preprocessing(probeAData, 'tna')

    # Read in the classA csv file into a panda dataframe
    probeAResults = pd.read_csv('../classA.csv')

    # Here the tna column, (which is dropped in the preprocessing call as it isn't to have
    # any preprocessing run on it), is concatenated with the preprocessed data and the contents
    # of the classA csv file (which contains the classification assignments)
    probeAConcatenated = pd.concat([probeAData['tna'], probeAPreprocessed, probeAResults], axis=1)
    # probeAConcatenated = pd.concat([probeAScaled, probeAResults], axis=1)

    bestModel = [0, 0, 0]
    tLabel = 'class'
    # Defines the number of splits we're going to use on the training data. In this case we use 10 meaning we will
    # Produce 10 folds, 900 training 100 validation
    splits = 10
    # Run cross validation for depth in range 1 to 20
    for depth in range(1, 20):
        #calls crossvalidation for given depth level with a fixed number of 100 estimators
        result = crossValidation(probeAConcatenated, depth, 200, splits, tLabel)
        if (result[0] > bestModel[0]):
                bestModel = result
    model = bestModel[2]
    # Simple print statement which prints the overall best model's AUC and the depth it used.
    print("BEST MODEL PREDICTED: ")
    print(str(bestModel[0]) + "______" + str(bestModel[1]) + "______" + str(bestModel[3]))
    print(model.feature_importances_)
main()