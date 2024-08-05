import numpy as np

input = [0.1,0.2,0.3,0.4]
weights = [[0.2,0.1,0.3,0.4],[0.4,0.6,0.1,0.9],[0.7,0.6,0.1,0.1]]
bias = [0.2,0.3,0.4]

layer_output = []
for n_weight, n_bias in zip(weights, bias):
    n_output = 0
    for n_input, weight in zip(input, n_weight):
        n_output += n_input * weight
    n_output += n_bias
    layer_output.append(n_output)

print(layer_output) 

