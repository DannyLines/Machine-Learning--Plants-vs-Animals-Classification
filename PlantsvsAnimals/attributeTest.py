import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def main():
    #Read the probeA csv file into a panda dataframe
    probeAData = pd.read_csv('../probeA.csv')

    plt.figure(figsize=(14, 5))
    plt.scatter(probeAData['c1'], probeAData['tna'])
    plt.ylabel('Feature Y')
    plt.xlabel('Feature X1')
main()