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

class Activation_ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)

class Dense_layer:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.01*np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons)) # we want all of them to be matrices instead of just arrays
    
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

if __name__ == '__main__':
    X, y = spiral_data(samples=100, classes=3)

    dense1 = Dense_layer(2,3)
    activation = Activation_ReLU()

    dense1.forward(X)
    activation.forward(dense1.output)
    print(activation.output[:5])