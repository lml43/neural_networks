import pandas
import numpy as np
import matplotlib.pyplot as plt
import csv


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
    def gauss(x):
        return np.exp(-x * x)

    @staticmethod
    def gauss_deriv(x):
        return - 2 * x * Layer.gauss(x)

    @staticmethod
    def identity(x):
        return x

    @staticmethod
    def identity_loss(x, target):
        return -2 * (target - x)


class NeuralNetwork:
    def __init__(self, x_train, y_train, x_test, y_test, hidden_layers):

        self.x_train = x_train
        self.y_train = y_train
        self.train_count = x_train.shape[1]

        self.x_test = x_test
        self.y_test = y_test
        self.test_count = x_test.shape[1]

        self.I_dim = 784
        self.O_dim = 10


        self.one_hot_y = self.one_hot(self.y_train)
        hidden_layers.append(Layer(self.O_dim, Layer.softmax, Layer.softmax_deriv))

        self.layers = hidden_layers

        # Generate weight matrices with random values in [-0.5, 0.5]
        # The matrices' sizes are corresponding to the layers' sizes
        self.weights = []
        self.bias = []

        # np.random.seed(1)
        for i in range(len(self.layers)):
            self.bias.append(np.zeros((self.layers[i].size, 1)))
            if i == 0:
                self.weights.append(np.random.rand(self.layers[0].size, self.I_dim) - 0.5)
            else:
                self.weights.append(np.random.rand(self.layers[i].size, hidden_layers[i - 1].size) - 0.5)

    def one_hot(self, Y):
        one_hot_Y = np.zeros((Y.size, Y.max() + 1))
        one_hot_Y[np.arange(Y.size), Y] = 1
        one_hot_Y = one_hot_Y.T
        return one_hot_Y

    def get_predictions(self, Z):
        return np.argmax(Z, 0)

    def get_accuracy(self, predictions, Y):
        return np.sum(predictions == Y) / Y.size

    def forward_propagate(self, layers, weights, bias, training_point):
        layer_input = training_point
        pre_activations = []
        post_activations = []

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

        return predictions, out

    def test(self):
        predictions = []
        predict, out = self.predict(self.x_test)
        predictions.append(predict)

        loss = self.loss_function(out, self.one_hot(self.y_test))
        acc = self.get_accuracy(predictions, self.y_test)
        return acc, loss

    def dot_vec(self, a, b):
        res = np.zeros((len(a), len(b)))
        for i in range(len(a)):
            for j in range(len(b)):
                res[i, j] += a[i] * b[j]

        return res

    def loss_function(self, output, target):
        loss = np.sum( -np.multiply(target,np.log(output)), axis=0 )
        loss = np.sum( loss )/output.shape[1]
        return loss

    def train(self, epoch_count, batch_size, learning_rate):
        loss_train = []
        loss_test = []
        acc_test = []

        for epoch in range(epoch_count):
            loss_epoch = 0
            batch_count = 0

            # Iteratively picking up points in a batch
            for index in range(0, self.train_count, batch_size):
                batch_count += 1

                input_batch = self.x_train[:, index:min(index + batch_size, self.train_count)]
                target_batch = self.one_hot_y[:, index:min(index + batch_size, self.train_count)]
                sample_count = input_batch.shape[1]

                output, pre_activations, post_activations = \
                    self.forward_propagate(self.layers, self.weights, self.bias, input_batch)
                loss_epoch += self.loss_function(output, target_batch)

                # Backpropagation
                diff_b, diff_w = self.backpropagation(input_batch, output, post_activations, pre_activations, sample_count,
                                                      target_batch)

                self.update_weights(diff_b, diff_w, learning_rate)

            loss_train.append(loss_epoch / batch_count)

            acc_test_epoch, loss_test_epoch = self.test()
            loss_test.append(loss_test_epoch)
            acc_test.append(acc_test_epoch)

            if epoch % 10 == 9:
                print("Epoch {}, accuracy = {}".format(epoch, acc_test_epoch))

        return acc_test, loss_test, loss_train

    def update_weights(self, diff_b, diff_w, learning_rate):
        for i in range(len(self.weights)):
            self.weights[i] = self.weights[i] - learning_rate * diff_w[i]
            self.bias[i] = self.bias[i] - learning_rate * diff_b[i]

    def backpropagation(self, input_batch, output, post_activations, pre_activations, sample_count, target_batch):
        diff_w = []
        diff_b = []
        prevA = post_activations[len(self.layers) - 2].T
        d_z = self.layers[-1].derivation(output, target_batch)
        d_w = np.dot(d_z, prevA) / sample_count
        d_b = np.sum(d_z) / sample_count
        diff_w.append(d_w)
        diff_b.append(d_b)
        for i in reversed(range(0, len(self.layers) - 1)):
            layer = self.layers[i]

            if i == 0:
                prevA = input_batch.T
            else:
                prevA = post_activations[i - 1].T

            d_z = self.weights[i + 1].T.dot(d_z) * layer.derivation(pre_activations[i])
            d_w = np.dot(d_z, prevA) / sample_count
            d_b = np.sum(d_z) / sample_count

            diff_w.append(d_w)
            diff_b.append(d_b)
        diff_w.reverse()
        diff_b.reverse()
        return diff_b, diff_w


def train_test_split(data, m_test):
    data = np.array(data)
    m, n = data.shape
    np.random.shuffle(data)

    data_test = data[0:m_test].T
    y_test = data_test[0]
    x_test = data_test[1:n]
    x_test = x_test / 255.

    data_train = data[m_test:m].T
    y_train = data_train[0]
    x_train = data_train[1:n]
    x_train = x_train / 255.
    _, m_train = x_train.shape
    return x_train, y_train, x_test, y_test


# Variables
learning_rate = 0.001
epoch_count = 500
batch_size = 64

# Initialise the hidden layers
hidden_layers = [
    # Layer(10, Layer.gauss, Layer.gauss_deriv),
    # Layer(10, Layer.logistic, Layer.logistic_deriv),
    # Layer(10, Layer.logistic, Layer.logistic_deriv),
    Layer(15, Layer.reLU, Layer.reLU_deriv),
    Layer(15, Layer.reLU, Layer.reLU_deriv),
]


train_file = 'data/mnist/train.csv'
data_train = pandas.read_csv(train_file)
x_train, y_train, x_test, y_test = train_test_split(data_train, 2000)

network = NeuralNetwork(x_train, y_train, x_test, y_test, hidden_layers)
acc_test, loss_test, loss_train = network.train(epoch_count, batch_size, learning_rate)

plt.plot(loss_train, label="Loss Training set")
plt.plot(loss_test, label="Loss Test set")
plt.plot(acc_test, label="Accuracy Test set")
plt.legend(loc="upper right")
plt.show()

# Write to file
test_file = 'data/mnist/test.csv'
data_test = pandas.read_csv(test_file)
data_test = np.array(data_test)
data_test = data_test.T
data_test = data_test / 255.

predictions, _ = network.predict(data_test)
print(predictions)

header = ['ImageId', 'Label']
data = []

for i in range(28000):
    data.append([i+1, predictions[i]])

with open('out.csv', 'w', encoding='UTF8', newline='') as f:
    writer = csv.writer(f)

    # write the header
    writer.writerow(header)

    # write multiple rows
    writer.writerows(data)
