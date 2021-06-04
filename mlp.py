import pandas
import numpy as np


class Layer:
    def __init__(self, size, activation, derivation):
        self.activation = activation
        self.derivation = derivation
        self.size = size

    # Activation functions
    @staticmethod
    def logistic(x):
        return 1.0 / (1 + np.exp(-x))

    @staticmethod
    def logistic_deriv(x):
        return Layer.logistic(x) * (1 - Layer.logistic(x))

    @staticmethod
    def reLU(x):
        return np.maximum(0, x)

    @staticmethod
    def reLU_deriv(x):
        return x > 0

    @staticmethod
    def softmax(x):
        return np.exp(x) / sum(np.exp(x))

    @staticmethod
    def softmax_deriv(x, target):
        return x - target

    @staticmethod
    def identity(x, target):
        return x


class NeuralNetwork:
    def __init__(self, train_file, test_file, hidden_layers, network_type, epoch_count, batch_size, learning_rate):
        self.epoch_count = epoch_count
        self.batch_size = batch_size
        self.learning_rate = learning_rate

        self.x_train, self.y_train, self.training_count = self.train_test_from_file(train_file)
        self.x_test, self.y_test, self.test_count = self.train_test_from_file(test_file)
        self.one_hot_y = self.one_hot(self.y_train)

        self.I_dim = self.x_train.shape[0]
        self.O_dim = self.one_hot_y.shape[0]

        self.expected = self.y_test.to_numpy() - 1
        self.network_type = network_type
        if network_type == 'regression':
            hidden_layers.append(Layer(1, Layer.identity, Layer.identity))
        else:
            hidden_layers.append(Layer(self.O_dim, Layer.softmax, Layer.softmax_deriv))

        self.layers = hidden_layers

        # Generate weight matrices with random values in [-0.5, 0.5]
        # The matrices' sizes are corresponding to the layers' sizes
        self.weights = []
        self.bias = []

        for i in range(len(self.layers)):
            self.bias.append(np.zeros(self.layers[i].size))
            if i == 0:
                self.weights.append(np.random.rand(self.layers[0].size, self.I_dim) - 0.5)
            else:
                self.weights.append(np.random.rand(self.layers[i].size, hidden_layers[i - 1].size) - 0.5)

    def train_test_from_file(self, filename):
        x = pandas.read_csv(filename)
        y = x.cls
        x = x.drop(['cls'], axis=1)
        x = np.transpose(np.asarray(x))
        count = len(x[0, :])
        return x, y, count

    def one_hot(self, Y):
        one_hot_Y = np.zeros((Y.size, Y.max()))
        one_hot_Y[np.arange(Y.size), Y - 1] = 1
        return np.transpose(one_hot_Y)

    def get_predictions(self, Z):
        return np.argmax(Z, 0)

    def get_accuracy(self, predictions, Y):
        return np.sum(predictions == Y) / Y.size

    def forward_propagate(self, layers, weights, bias, training_point):
        layer_input = training_point
        pre_activations = []
        post_activations = [layer_input]

        for i in range(len(layers)):
            layer = layers[i]

            pre = np.dot(weights[i], layer_input) + bias[i]
            post = layer.activation(pre)

            layer_input = post
            pre_activations.append(pre)
            post_activations.append(post)

        return layer_input, pre_activations, post_activations

    def predict(self, x):
        out, _, _ = self.forward_propagate(self.layers, self.weights, self.bias, x)
        predictions = self.get_predictions(out)
        return predictions

    def test(self):
        predictions = []
        for point in range(self.x_test.shape[1]):
            predictions.append(self.predict(self.x_test[:, point]))

        acc = self.get_accuracy(predictions, self.expected)
        return acc

    def dot_vec(self, a, b):
        res = np.zeros((len(a), len(b)))
        for i in range(len(a)):
            for j in range(len(b)):
                res[i, j] += a[i] * b[j]

        return res

    def train(self):
        for epoch in range(epoch_count):
            if epoch % 1 == 0:
                accuracy = self.test()
                print("Epoch {}, accuracy: {}".format(epoch, accuracy))

            # Iteratively picking up points in a batch
            for index in range(0, self.training_count, self.batch_size):
                for point in range(index, min(index + self.batch_size, self.training_count)):
                    output, pre_activations, post_activations = self.forward_propagate(self.layers, self.weights, self.bias, self.x_train[:, point])

                    # Backpropagation
                    diff_w = []
                    diff_b = []

                    d_z = output - self.one_hot_y[:, point]
                    d_w = self.dot_vec(d_z, post_activations[len(self.layers)-1])
                    d_b = np.sum(d_z)
                    diff_w.append(d_w)
                    diff_b.append(d_b)

                    for i in reversed(range(0, len(self.layers)-1)):
                        layer = self.layers[i]

                        d_z = self.weights[i + 1].T.dot(d_z) * layer.derivation(pre_activations[i])
                        d_w = self.dot_vec(d_z, post_activations[i])
                        d_b = np.sum(d_z)

                        diff_w.append(d_w)
                        diff_b.append(d_b)

                    diff_w.reverse()
                    diff_b.reverse()

                    for i in range(len(self.weights)):
                        self.weights[i] = self.weights[i] - learning_rate * diff_w[i]
                        self.bias[i] = self.bias[i] - learning_rate * diff_b[i]


# Variables
learning_rate = 0.1

epoch_count = 100
batch_size = 32

# Initialise the hidden layers
hidden_layers = [
    # Layer(4, Layer.logistic, Layer.logistic_deriv),
    Layer(3, Layer.reLU, Layer.reLU_deriv),
]

output_type = 'classification'
# output_type = 'regression'
train_file = 'data/classification/data.simple.train.100.csv'
test_file = 'data/classification/data.simple.test.100.csv'

network = NeuralNetwork(train_file, test_file, hidden_layers, output_type, epoch_count, batch_size, learning_rate)
network.train()