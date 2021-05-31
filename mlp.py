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
H_dim = 4
O_dim = 2

epoch_count = 100
batch_size = 16

# Initialise the hidden layers
hidden_layers = [
    Layer(10, logistic, logistic_deriv, 0),
    Layer(12, reLU, reLU_deriv, 0),
]

# Generate weight matrices with random values in [-0.5, 0.5]
# The matrices' sizes are corresponding to the layers' sizes
weight_matrices = []

for i in range(len(hidden_layers)):
    if i == 0:
        weight_matrices.append(np.random.rand(hidden_layers[0].size, I_dim) - 0.5)
    else:
        weight_matrices.append(np.random.rand(hidden_layers[i].size, hidden_layers[i-1].size) - 0.5)

weight_matrices.append(np.random.rand(O_dim, hidden_layers[-1].size) - 0.5)

# TODO delete in future
weights_ItoH = np.random.uniform(-1, 1, (I_dim, H_dim))
weights_HtoO = np.random.uniform(-1, 1, H_dim)

# TODO create arrays
preActivation_H = np.zeros(H_dim)
postActivation_H = np.zeros(H_dim)

input_file = 'data/classification/data.simple.train.10000.csv'
print("Import file {} ...".format(input_file))
training_data = pandas.read_csv(input_file)
training_class = training_data.cls
training_data = training_data.drop(['cls'], axis=1)
training_data = np.asarray(training_data)
training_count = len(training_data[:, 0])


# training
for epoch in range(epoch_count):
    print("\n\nEpoch {}".format(epoch))

    # Shuffle the array
    random.shuffle(training_data)

    # Iteratively picking up points in a batch
    for index in range(0, training_count, batch_size):
        for point in range(index, min(index + batch_size, training_count)):
            # TODO for each layer
            for layer_node in range(H_dim):
                preActivation_H[layer_node] = np.dot(training_data[point, :], weights_ItoH[:, layer_node])
                postActivation_H[layer_node] = logistic(preActivation_H[layer_node])

            # TODO check classification or regression
            preActivation_O = np.dot(postActivation_H, weights_HtoO)
            postActivation_O = logistic(preActivation_O)

            error = postActivation_O - training_class[point]

            # Backpropagation
            # TODO for each layer
            for H_node in range(H_dim):
                S_error = error * logistic_deriv(preActivation_O)
                gradient_HtoO = S_error * postActivation_H[H_node]
                for I_node in range(I_dim):
                    input_value = training_data[point, I_node]
                    gradient_ItoH = S_error * weights_HtoO[H_node] * logistic_deriv(preActivation_H[H_node]) * input_value
                    weights_ItoH[I_node, H_node] -= learning_rate * gradient_ItoH

                weights_HtoO[H_node] -= learning_rate * gradient_HtoO

    print('I to H:')
    print(weights_ItoH)

    print()
    print('H to O:')
    print(weights_HtoO)
