import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import normalize
from sklearn.preprocessing import MinMaxScaler
def main():
    data = [[0.712643678161],  [0.675],   [0.633333333] , [0.65833333],  [0.681006493506] , [0.703183133526], [.62916666666] , [0.64166666666] , [0.615353037767] , [0.668205539944],  [0.632699462588] , [0.66767676767], [.631665977677]]
    scaler = MinMaxScaler()

    scaler.fit(data)
    scaledData = scaler.transform(data)
    result = []
    for entry in scaledData:
        result.append(entry[0])
    print(result)
    plt.bar(['tna', 'c1', 'c2', 'c3', 'n1', 'n2', 'n3', 'm1', 'm2', 'm3', 'p1', 'p2', 'p3'], result)
    plt.show()
main()