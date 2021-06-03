import pandas
import numpy as np
import random


class Layer:
    def __init__(self, size, activation, derivation):
        self.activation = activation
        self.derivation = derivation
        self.size = size


# Activation functions
def logistic(x):
    return 1.0 / (1 + np.exp(-x))


def logistic_deriv(x):
    return logistic(x) * (1 - logistic(x))


def reLU(x):
    return np.maximum(0, x)


def reLU_deriv(x):
    return x > 0


def softmax(x):
    return np.exp(x) / sum(np.exp(x))


def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, Y.max()))
    one_hot_Y[np.arange(Y.size), Y-1] = 1
    return one_hot_Y


def get_predictions(Z):
    return np.argmax(Z, 0)


def get_accuracy(predictions, Y):
    return np.sum(predictions == Y) / Y.size


def forward_propagate(network, weights, training_point):
    layer_input = training_point
    pre_activations = []
    post_activations = [layer_input]

    for i in range(len(network)):
        layer = network[i]

        pre = np.asarray(np.dot(weights[i], layer_input.T) + bias[i]).reshape(-1)
        post = layer.activation(pre)

        layer_input = post
        pre_activations.append(pre)
        post_activations.append(post)

    f_output = np.zeros(O_dim)
    for neuron in range(O_dim):
        f_output[neuron] = np.dot(weights[-1][neuron, :], layer_input)

    return softmax(f_output), pre_activations, post_activations


def predict(x):
    out, _, _ = forward_propagate(hidden_layers, weight_matrices, x)
    predictions = get_predictions(out)
    return predictions


# Variables
learning_rate = 0.1

I_dim = 2
O_dim = 2

epoch_count = 10
batch_size = 32

# Initialise the hidden layers
hidden_layers = [
    Layer(3, reLU, reLU_deriv),
    # Layer(4, logistic, logistic_deriv),

]

# Generate weight matrices with random values in [-0.5, 0.5]
# The matrices' sizes are corresponding to the layers' sizes
weight_matrices = []
bias = np.random.rand(len(hidden_layers) + 1) - 0.5

# TODO check order of dimensions
for i in range(len(hidden_layers)):
    if i == 0:
        weight_matrices.append(np.random.rand(hidden_layers[0].size, I_dim) - 0.5)
    else:
        weight_matrices.append(np.random.rand(hidden_layers[i].size, hidden_layers[i-1].size) - 0.5)

weight_matrices.append(np.random.rand(O_dim, hidden_layers[-1].size) - 0.5)

input_file = 'data/classification/data.simple.train.1000.csv'
print("Import file {} ...".format(input_file))
x_train = pandas.read_csv(input_file)
y_train = x_train.cls
x_train = x_train.drop(['cls'], axis=1)
x_train = np.asarray(x_train)
training_count = len(x_train[:, 0])


one_hot_y = one_hot(y_train)


# Test
input_file = 'data/classification/data.simple.test.500.csv'
x_test = pandas.read_csv(input_file)
y_test = x_test.cls
x_test = x_test.drop(['cls'], axis=1)
x_test = np.asarray(x_test)
test_count = len(x_test[:, 0])

expected = y_test.to_numpy() - 1

# training
for epoch in range(epoch_count):

    if epoch % 1 == 0:
        predictions = []
        for point in x_test:
            predictions.append(predict(point))

        acc = get_accuracy(predictions, expected)
        print("Epoch {}, accuracy: {}".format(epoch, acc))

    # Shuffle the array
    # random.shuffle(x_train)

    # Iteratively picking up points in a batch
    for index in range(0, training_count, batch_size):
        for point in range(index, min(index + batch_size, training_count)):

            output, pre_activations, post_activations = forward_propagate(hidden_layers, weight_matrices, x_train[point, :])

            # Backpropagation
            diffW = []
            diffB = []

            dZ = np.matrix(output - one_hot_y[point]).T
            dW = np.dot(dZ, np.matrix(post_activations[len(hidden_layers)]))
            dB = np.sum(dZ)

            diffW.append(dW)
            diffB.append(dB)

            for i in reversed(range(len(hidden_layers))):
                layer = hidden_layers[i]
                dZ = np.asarray(weight_matrices[i+1].T.dot(dZ)).reshape(-1) * layer.derivation(pre_activations[i])
                dW = np.dot(np.matrix(dZ).T, np.matrix(post_activations[i]))
                dB = np.sum(dZ)

                diffW.append(dW)
                diffB.append(dB)

            diffW.reverse()
            diffB.reverse()

            for i in range(len(weight_matrices)):
                weight_matrices[i] = weight_matrices[i] - learning_rate * diffW[i]
                bias[i] = bias[1] - learning_rate * diffB[i]

    # print(weight_matrices[0])






