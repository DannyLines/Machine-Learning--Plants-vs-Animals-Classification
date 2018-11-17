#!/usr/bin/python
import pandas as pd
import numpy as np
import pickle
from sklearn import linear_model
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn import svm
from sklearn.model_selection import KFold
from sklearn import cross_validation as cv
from sklearn import linear_model
from sklearn.linear_model import LassoCV
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn import preprocessing
from sklearn.preprocessing import PolynomialFeatures


# This method is passed predicted and observed values, simply returning the R2 of the given predictions/model
def accuracy(predicted, observations):
    observationsDataframe = observations.copy()
    observationsArray = observationsDataframe.values

    # Simply calculates the R2 score given the observed data and predicted values
    r2Score = r2_score(observationsArray, predicted)
    return r2Score


# This model is passed a model, the data it will be validated against and the target data column name
def testModel(model, test, tLabel):
    testCopy = test.copy()
    
    #testVals represents the values which will be used in the prediction step
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
# a which represents the alpha to be used in the production of the lasso model
def trainData(dataFrame, tLabel, a):
    df = dataFrame.copy()
    
    # Data is partitioned between the data used as parameters (training) and the target data (targetData)
    training = df.drop(tLabel, axis=1)
    targetData = dataFrame[tLabel].copy()

    # We initialise a lasso model with alpha = a and normalize set to true
    reg = linear_model.Lasso(alpha=a, normalize=True)
    
    # We then fit this model to the training and target data
    reg.fit(training, targetData)
    return reg

# This method is passed a dataframe, splits which defines the number of 'folds' to make in the data for training and validation
# tLabel which defines the target label, i.e. the data for which we are training to predict
# & alphas which is an array of alphas to be tested, where it will return the best model from all those produced using the
# given alphas
def crossValidation(dataFrame, splits, tLabel, alphas):
    df = dataFrame.copy()
    
    # Initialise kf with the given number of splits
    kf = KFold(n_splits=splits)
    bestModel = None
    
    # Initialises bestModelResult and bestAlpha to a value which will be surpassed 1st time no matter what
    bestModelResult = -100
    bestAlpha = -1
    
    # For the training data and test data in each "split"/fold, we iterate through every alpha in the alphas
    # array passed into the method. So for eevery alpha in alphas for every fold/split we produce a model
    # using the traininf data, tLabel and given alpha
    for training, test in kf.split(df):
        trainingData = df.ix[training]
        testData = df.ix[test]
        
        # We then produce a model using every alpha, keeping track of the vest model so far
        for alpha in alphas:
	    # Trains the model
	    model = trainData(trainingData, tLabel, alpha)
	    
	    # Assigns result the R2 value the given model achieves on the validation set
	    result = testModel(model, testData, tLabel)
	    
	    # Simply keeps track of the best model so far
	    if (result > bestModelResult):
		bestModel = model
		bestModelResult = result
		bestAlpha = alpha

    return [bestModel, bestModelResult, bestAlpha]


# This method is passed a dataFrame and a column name to be dropped (so it doesn't have preprocessing run on it)
# & returns the preprocessed dataFrame

def preprocessing(dataFrame, tLabel):
    df = dataFrame.copy()
    
    #'none' is a special tLabel value passed in when we dont want to drop any column from the dataFrame
    # If it is not equal to 'none' then dorp the iven column name and continue with the preprocessing
    if(tLabel != 'none'):
        df = df.drop(tLabel, axis=1)
    
    # The mean is calculated
    mean = df.mean(axis=0)
    
    # standard variation calculated
    stdVar = df.std(axis=0)
    
    # The data is then scaled by subtracting the mean from each value and dividing through by the standard variation
    df = df.sub(mean, axis=1)
    df = df / stdVar

    # Further preprocessing takes place here whereby the attributes (columns) are combined together with the degree
    # defined in the polynomialFeatures(), which allows for a wider and more useful set of data for training
    poly = PolynomialFeatures(2)
    df_scaled = poly.fit_transform(df)

    return pd.DataFrame(df_scaled)

# This method is responsible for taking a dataframe, three column names and outputting a dataframe
# with the three columns ordered such that col1Val <= col2Val <= col3Val
# This was done as it was seen that the data appeared to be "out-of-order", and after re-arranging the values
# performance appeared to increase

def orderData(dataFrame, col1, col2, col3):
    df = dataFrame.copy()
    for index, row in df.iterrows():
	# Assign the values in column 1, 2 & 3 to their corresponding variables
        col1Val = row[col1]
        col2Val = row[col2]
        col3Val = row[col3]
  
	# The values are then put in an array where the built in .sort() method is used to arrange the values
        values = [col1Val, col2Val, col3Val]
        values.sort()

	#Now the data in the given row is simply re-assigned such that col1Val <= col2Val <= col3Val
        row[col1] = values[0]
        row[col2] = values[1]
        row[col3] = values[2]
    return df


def main():
    # Defines the filename in which the models weights/paramaters have been pre-calculated and stored
    filename = "predictTNAModel.sav"
    model = None
    try:
      # If the model has been precomputed this will not throw an exception
      model = pickle.load(open(filename, 'rb'))
    except Exception as e:
      # If the model has not been precomputed, the filename will not exist, cannot be opened and thus throw an error
      # This will be caught and come into this section, where the models will then be calculated and stored in the given
      # Filename, meaning in the future they need not be calculated again
      
      tLabel = 'tna'
      
      # Reads the probeA csv file into a panda dataframe
      probeAData = pd.read_csv('../probeA.csv')
      
      # This chunk of code calls orderData, which goes through the passed in
      # headers, swapping values such that the first header has all the smallest
      # values, the middle with the middle value and the last header with the largest
      # value. This was done as after inspecting the data, where it became apparent this
      # was a common trend in the data, with them occasionally being out of order.
      # After swapping these around, an improvement was seen in the R2 score, so was kept
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
      # in performance in terms of R2
      probeAScaled = preprocessing(probeAData, tLabel)
      
      # A dataFrame 'probeAConcatenated' is produced containing the tna and preprocessed data 
      # Which will be used for training
      probeAConcatenated = pd.concat([probeAData['tna'], probeAScaled], axis=1)
      probeATraining = probeAScaled.copy()

      # Defines number of splits/folds to use
      splits = 10
      
      # This creates an array of 800 alpha values to test in a range between 0.00000001 and 2 which are produced non-linearly
      alphas = np.geomspace(0.00000001, 2, 800)
      
      # The best model is retrieved by running crossValidation on the data, with a defined
      # number of splits, with a given target label with the defined set of alphas
      bestModelResult = crossValidation(probeAConcatenated, splits, tLabel, alphas)
      
      model = bestModelResult[0]
      result = bestModelResult[1]
      bestAlpha = bestModelResult[2]
      print("BEST MODEL - " + str(result) + "__" + str(bestAlpha))
      
      #Can use TNA in prediction of t1, but need to check effect of this
      #On average the prediction is 0.14 out, so make gaussian distribution ot mdoel noise of 0.14 and apply to
      #task 1 tna column and check results
      pickle.dump(model, open(filename, 'wb'))
    
    
    # Now the model has either been loaded or computed we can start loading in the data needed for predictions
    # ProbeB csv file is opened and read into a panda dataFrame
    probeBData = pd.read_csv('../probeB.csv')
    
    #This section shuffles the data in the columns to assure that the values appear in the order of:
    # col1val <= col2val <= col3val
    BfirstChemOrdered = orderData(probeBData, 'c1', 'c2', 'c3')
    BsecondChemOrdered = orderData(BfirstChemOrdered, 'm1', 'm2', 'm3')
    BthirdChemOrdered = orderData(BsecondChemOrdered, 'n1', 'n2', 'n3')
    BfourthChemOrdered = orderData(BthirdChemOrdered, 'p1', 'p2', 'p3')
    probeBData = BfourthChemOrdered.copy()
    
    # Performs preprocessing on the probeBData
    # 'none' is passed as no column needs to be dropped (as it doesn't contain a tna column)
    probeBScaled = preprocessing(probeBData, 'none')

    # The model predicts the probabilities for the given data
    predictions = model.predict(probeBScaled)
    
    # The predictions are then printed to a tnaB csv file with a header of "tna"
    dataFramePredictions = pd.DataFrame(predictions, columns=["tna"])
    dataFramePredictions.to_csv('tnaB.csv', index=False)
main()