import predictTNA
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn import svm
from sklearn.model_selection import KFold
from sklearn import cross_validation as cv
from sklearn import metrics
from sklearn import preprocessing
from sklearn.preprocessing import PolynomialFeatures
import pickle


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
# k which defines the number of neighbours to use in the k-NN algorithm


def trainData(dataFrame, tLabel, k):
    df = dataFrame.copy()
    # Data is partitioned between the data used as parameters (training) and the target data (targetData)
    training = df.drop(tLabel, axis=1)
    targetData = dataFrame[tLabel].copy()

    # A model named neighbors is produced using the given k used as the paramter for number of neighbours
    neighbors = KNeighborsClassifier(n_neighbors=k)
    # the model is then fit to the training data
    neighbors.fit(training, targetData)

    return neighbors


# This method is passed a dataframe, a k which defines the number of neighbours to use in k-NN,
# splits which defines the number of 'folds' to make in the data for training and validation
# tLabel which defines the target label, i.e. the data for which we are training to predict


def crossValidation(dataFrame, k, splits, tLabel):
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

        model = trainData(trainingData, tLabel, k)
        result = testModel(model, testData, tLabel)

        # This simply selects the best model seen so far out of all the folds
        # For example with splits = 10, this loop will be run 10x producing 10 models
        # This code then returns the best of these 10 models, i.e. the best model for this given k
        if (result > bestModelResult):
            bestModelResult = result
            bestModel = model

    return [bestModelResult, k, bestModel]


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
    # Defines the filename in which the models weights/paramaters have been pre-calculated and stored
    filename = "predictClassModel.sav"
    model = None
    try:
        # If the model has been precomputed this will not throw an exception
        model = pickle.load(open(filename, 'rb'))
    except Exception as e:
        # If the model has not been precomputed, the filename will not exist, cannot be opened and thus throw an error
        # This will be caught and come into this section, where the models will then be calculated and stored in the given
        # Filename, meaning in the future they need not be calculated again

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
        # Run cross validation for all k in the range of 1 to 50
        # No need to exceed 50 as research suggests that k > the root of the training set yield no further improvement
        # So k=50 is even excessive
        for k in range(1, 50):
            # This call to crossValidation returns the best model, it's k and its corresponding AUC result produced using the
            # Given k passed into it
            result = crossValidation(probeAConcatenated, k, splits, tLabel)

            # If the best models AUC for the given k is better than the current best seen then we set the new best model to
            # Equal the result, which is the returned model for this given k
            if (result[0] > bestModel[0]):
                bestModel = result

        # Simple print statement which prints the overall best model's AUC and the k-neighbours it used.
        print("BEST MODEL PREDICTED: ")
        print(str(bestModel[0]) + "______" + str(bestModel[1]))

        # This asssigns model to be the best model found from all the given k's tested
        # Remember bestModel returns an array of [AUCresult, k, model]
        # So bestModel[2] is simply returning the actual model itself
        model = bestModel[2]

        # We are still in the caught exception scope, meaning the model was not already precalculated and thus needs to be
        # saved for future use
        pickle.dump(model, open(filename, 'wb'))

    # Now the model has either been loaded or computed we can start loading in the data needed for predictions
    # ProbeB csv file is opened and read into a panda dataFrame
    probeBData = pd.read_csv('../probeB.csv')

    # Here the tnaB csv file is read, this contains the predictions from the 'predictTNA.py' script for the TNA of probe B
    # This file is guaranteed to exist as 'predictTNA' is imported at the top of the file, so predictTNA will be run beforehand
    TNAPredictions = pd.read_csv('tnaB.csv')

    # This section shuffles the data in the columns to assure that the values appear in the order of:
    # col1val <= col2val <= col3val
    BfirstChemOrdered = orderData(probeBData, 'c1', 'c2', 'c3')
    BsecondChemOrdered = orderData(BfirstChemOrdered, 'm1', 'm2', 'm3')
    BthirdChemOrdered = orderData(BsecondChemOrdered, 'n1', 'n2', 'n3')
    BfourthChemOrdered = orderData(BthirdChemOrdered, 'p1', 'p2', 'p3')

    # The final BFourthChemOrdered will now contain all the ordered columnds, so is simply copied into a new, easier to
    # handle dataframe
    probeBData = BfourthChemOrdered.copy()

    # Performs preprocessing on the probeBData
    # 'none' is passed as no column needs to be dropped
    probeBScaled = preprocessing(probeBData, 'none')

    # The predicted TNA values is concatenated with the now scaled probe B data to produce a single dataframe for predicting
    # the classes
    probeBConcatenated = pd.concat([TNAPredictions, probeBScaled], axis=1)

    # The model predicts the probabilities for the given data
    predictions = model.predict_proba(probeBConcatenated)

    class1Predictions = []

    # The predictions are then printed to a classB csv file with a header of "class1"
    for row in predictions:
        class1Predictions.append(row[1])
    dataFramePredictions = pd.DataFrame(class1Predictions, columns=["class1"])
    dataFramePredictions.to_csv('classB.csv', index=False)


main()