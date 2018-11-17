import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

probeAData = pd.read_csv('../../probeA.csv')
probeAClass = pd.read_csv('../../classA.csv')
probeAConcatenated = pd.concat([probeAData, probeAClass], axis=1)
colors = []

def produceGraph(input, target):
    plt.scatter(probeAConcatenated[input], probeAConcatenated[input], color=colors, marker='|', alpha=0.5)

def main():
    for row in probeAClass["class"]:
        if row == 0:
            colors.append("red")
        else:
            colors.append("blue")

    loop = True
    loopCount = 1
    count = 1
    iteration = 0
    chemicals = ['c', 'm', 'n', 'p']

    while(loop):
        if(count % 4 == 0):
            iteration += 1
            count = 1
            if (iteration == 4):
                loop = False
                break
        loopCount = (iteration * 3) + count

        currentChem = chemicals[iteration]
        chemString = str(currentChem) + str(count)
        plt.subplot(4, 4, loopCount)
        plt.xlabel(chemString)
        plt.ylabel(chemString)
        produceGraph(chemString, "class")

        count += 1
    plt.subplot(4, 4, loopCount+1)
    plt.xlabel("tna")
    plt.ylabel("tna")
    produceGraph("tna", "class")
    plt.show()
main()