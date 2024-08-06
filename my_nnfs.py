import numpy as np
import nnfs
from nnfs.datasets import spiral_data

nnfs.init()

# if you want to print you can just call the print function of the variables of the class
class Dense_layer:
    def __init__(self, num_inputs, num_neurons):
        self.weights = 0.001 * np.random.randn(num_inputs, num_neurons)
        self.biases = np.zeros(num_neurons)

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

class ReLU():
    def forward(self, input):
        self.output = np.maximum(0,input)

class SoftMax():
    def forward(self, input):
        exponentials = np.exp(input - np.max(input, axis=1, keepdims = True))
        probabilities = exponentials/(np.sum(exponentials, axis=1, keepdims=True))
        self.output = probabilities


if __name__ == "__main__":
    X,y = spiral_data(samples=100, classes=3)
    dense = Dense_layer(2,3)
    dense2 = Dense_layer(3,3)
    relu = ReLU()
    relu2 = ReLU()
    dense.forward(X)
    relu.forward(dense.output)
    dense2.forward(relu.output)
    softmax = SoftMax()
    softmax.forward(dense2.output)
    print(softmax.output[:5])




        



