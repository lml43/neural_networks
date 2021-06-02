import pandas
import numpy as np
import random


class Layer:
    def __init__(self, size, activation, derivation, bias):
        self.activation = activation
        self.derivation = derivation
        self.size = size
        self.bias = bias


# Activation functions
def logistic(x):
    return 1.0 / (1 + np.exp(-x))


def logistic_deriv(x):
    return logistic(x) * (1 - logistic(x))


def reLU(x):
    return np.maximum(0, x)


def reLU_deriv(x):
    return np.maximum(0, x)


def softmax(x):
    return np.exp(x) / sum(np.exp(x))


# Variables
learning_rate = 0.8

I_dim = 2
O_dim = 2

epoch_count = 2
batch_size = 16

# Initialise the hidden layers
hidden_layers = [
    Layer(4, logistic, logistic_deriv, 0),
    Layer(3, reLU, reLU_deriv, 0),
]

# Generate weight matrices with random values in [-0.5, 0.5]
# The matrices' sizes are corresponding to the layers' sizes
weight_matrices = []

# TODO check order of dimensions
for i in range(len(hidden_layers)):
    if i == 0:
        weight_matrices.append(np.random.rand(hidden_layers[0].size, I_dim) - 0.5)
    else:
        weight_matrices.append(np.random.rand(hidden_layers[i].size, hidden_layers[i-1].size) - 0.5)

weight_matrices.append(np.random.rand(O_dim, hidden_layers[-1].size) - 0.5)

input_file = 'data/classification/data.simple.train.100.csv'
print("Import file {} ...".format(input_file))
training_data = pandas.read_csv(input_file)
training_class = training_data.cls
training_data = training_data.drop(['cls'], axis=1)
training_data = np.asarray(training_data)
training_count = len(training_data[:, 0])


preActivations = []
postActivations = []

for i in range(len(hidden_layers)):
    preActivations.append(np.zeros(hidden_layers[i].size))
    postActivations.append(np.zeros(hidden_layers[i].size))


def forward_propagate(network, weights, training_point):
    layer_input = training_point

    for i in range(len(network)):
        layer = network[i]
        pre = np.zeros(layer.size)
        post = np.zeros(layer.size)

        for neuron in range(layer.size):
            pre[neuron] = np.dot(weights[i][neuron, :], layer_input)
            post[neuron] = layer.activation(pre[neuron])

        layer_input = post

    f_output = np.zeros(O_dim)
    for neuron in range(O_dim):
        f_output[neuron] = np.dot(weights[-1][neuron, :], layer_input)

    print(f_output)
    return f_output.max()


# training
for epoch in range(epoch_count):
    print("\n\nEpoch {}".format(epoch))

    # Shuffle the array
    random.shuffle(training_data)

    # Iteratively picking up points in a batch
    for index in range(0, training_count, batch_size):
        for point in range(index, min(index + batch_size, training_count)):
            output = forward_propagate(hidden_layers, weight_matrices, training_data[point, :])
            error = output - training_class[point]

            #
            # Backpropagation
            # TODO for each layer
            # for H_node in range(H_dim):
            #     S_error = error * logistic_deriv(output)
            #     # TODO try some methods of optimizing
            #     for I_node in range(I_dim):
            #         input_value = training_data[point, I_node]
            #         gradient_ItoH = S_error * weights_HtoO[H_node] * logistic_deriv(preActivations[H_node]) * input_value
            #         weights_ItoH[I_node, H_node] -= learning_rate * gradient_ItoH
            #
            #     gradient_HtoO = S_error * postActivations[H_node]
            #     weights_HtoO[H_node] -= learning_rate * gradient_HtoO

    print('I to H:')
    print(weights_ItoH)

    print()
    print('H to O:')
    print(weights_HtoO)
