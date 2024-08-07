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

class Activation_ReLU():
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)

class Dense_layer():
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
        exponentials = np.exp(input - np.max(input, axis=1, keepdims = True)) # we want this to return an inside value of -inf to 0, so that the exponential will always be from 0 to 1
        probabilities = exponentials/(np.sum(exponentials, axis=1, keepdims=True))
        self.output = probabilities

class Loss():
    def calculate(self, output, y):
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)
        return data_loss
    
class Loss_CategoricalCrossentropy(Loss):
    def forward(self, y_pred, y_true):
        samples = len(y_pred)

        y_pred_clipped = np.clip(y_pred, 1e-7, 1e-7)



        if len(y_true.shape) == 1: # this is for if the shape is a list
            correct_confidences = y_pred_clipped[range(samples), y_true]
        
        if len(y_true.shape) == 2:
            corrent_confidences = np.sum(y_pred_clipped*y_true, axis=1)

        negative_log_likelihoods = -np.log(correct_confidences)
        return regative_log_likelihoods


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

    softmax_outputs = np.array([[0.7, 0.1, 0.2],
                                [0.1, 0.5, 0.4],
                                [0.02, 0.9,0.08]])
    class_targets = np.array([[1, 0, 0],
                            [0, 1, 0],
                            [0, 1, 0]])
    if len(class_targets.shape) == 1:
        correct_confidences = softmax_outputs[range(len(softmax_outputs)), class_targets]

    class_targets = [0,1,1]
    print(softmax_outputs[range(len(softmax_outputs)), class_targets])
    print(range(len(softmax_outputs)))
    print(softmax_outputs[1,class_targets])


        



