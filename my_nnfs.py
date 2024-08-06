import numpy as np
import nnfs
from nnfs.datasets import spiral_data
import matplotlib.pyplot as plt



plt.close('all')
nnfs.init()
import matplotlib.pyplot as plt
X,y = spiral_data(samples=100, classes = 3)
plt.scatter(X[:,0],X[:,1], c=y, cmap='brg')
# plt.show(block = False)
# plt.pause(0.001)
# input('hit any key to end: ')
# plt.close('all')

class Dense_layer:
    def __init__(self, n_inputs, n_neurons):
        pass
    def forward(self, n_inputs):
        pass
