import numpy as np
import nnfs
from nnfs.datasets import spiral_data

nnfs.init()

weights = [[1,2,3,4,5],
           [2,3,4,5,6],
           [3,4,5,6,7]]
bias = [1,2,3]
weights2 = [[1,2,3],
           [2,3,4],
           [3,4,5]]
bias2 = [1,2,3]
input = [[1,2,3,4,5]] # typically the inputs would be a matrix as well already

output1 = np.dot(np.array(input), np.array(weights).T) + bias

print(output1)

output2 = np.dot(output1, np.array(weights2).T)
print(output2)



