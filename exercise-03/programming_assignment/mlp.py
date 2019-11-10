import random
from abc import abstractmethod

import numpy as np
import matplotlib.pyplot as plt

T_TANH = 'tanh'
T_LOGISTIC = 'logistic'
T_IDENTITY = 'identity'


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def derivative(x, transfer=T_TANH):
    if transfer == T_TANH:
        return 1 - np.tanh(x) ** 2
    elif transfer == T_LOGISTIC:
        sig = sigmoid(x)
        return sig * (1 - sig)
    elif transfer == T_IDENTITY:
        return 1
    else:
        raise Exception('Activation function not correct.')


class Layer(object):

    def __init__(self, n_inputs, n_outputs, learning_rate=0.01, transfer=T_TANH, final=False):
        """
        Initialize the network with the hyperparameters.
        :param n_inputs:
        :param n_outputs:
        :param learning_rate:
        :param transfer:
        """
        self.net = None
        self.output = None
        self.sigma = None
        self.delta = None
        self.layer_above = None
        self.layer_below = None
        self.biases = np.random.uniform(-2, 2, size=(n_outputs,))
        self.weights = np.random.uniform(-2, 2, size=(n_inputs, n_outputs))
        self.learning_rate = learning_rate
        self.transfer = transfer
        self.final = final

    def forward(self, X):
        """
        Forward propagation.
        :param X:
        :return:
        """
        if not self.layer_above:
            return self.activation(X)
        else:
            return self.activation(self.layer_above.output)

    @abstractmethod
    def backward(self, teacher=None, pattern=None):
        raise NotImplementedError('Implement the function')

    def activation(self, X):
        """
        Activation function.
        :param X:
        :return:
        """
        self.net = np.dot(X, self.weights) + self.biases
        if self.transfer == T_TANH:
            self.output = np.tanh(self.net)
        elif self.transfer == T_LOGISTIC:
            self.output = sigmoid(self.net)
        elif self.transfer == T_IDENTITY:
            self.output = self.net
        else:
            raise Exception('Activation function not correct.')
        return self.output

    def apply_delta(self):
        self.biases = self.learning_rate * self.sigma
        self.weights += self.delta


class Hidden(Layer):
    """
    """

    def __init__(self,
                 n_inputs,
                 n_outputs,
                 learning_rate=0.01,
                 transfer=T_TANH):
        super().__init__(n_inputs, n_outputs, learning_rate, transfer)

    def backward(self, teacher=None, pattern=None):
        # sigma_h = (sum(sigma_k * w_hk) * f'(net_h)
        self.sigma = (np.dot(
            self.layer_below.sigma,
            self.layer_below.weights.T
        ) * derivative(self.net, self.transfer))

        # retrieve out_g
        if not self.layer_above:
            # first layer = patterns
            out_g = pattern.T
        else:
            # out_g from layer above
            out_g = self.layer_above.output
            out_g = np.reshape(out_g, (out_g.shape[1], 1))

        # delta_W = n * sigma * out_g
        self.delta = (self.learning_rate * np.dot(out_g, self.sigma))


class Output(Layer):
    """
    """

    def __init__(self,
                 n_inputs,
                 n_outputs,
                 learning_rate=0.01,
                 transfer=T_TANH):
        super().__init__(n_inputs, n_outputs, learning_rate, transfer)

    def backward(self, teacher=None, pattern=None):
        # sigma_m = ((y'm - ym) * f'(net_m)
        self.sigma = ((teacher - self.output) * derivative(self.net, self.transfer))

        # retrieve out_h
        out_h = self.layer_above.output
        out_h = np.reshape(out_h, (out_h.shape[1], 1))  # shape[0]

        # delta_W = n * sigma_m * out_h
        self.delta = self.learning_rate * np.dot(out_h, self.sigma)


class Network(object):
    """
    Multi Layer Perzeptron.
    """

    def __init__(self, epochs=50, seed=999):
        """
        Initialize the MLP.
        """
        np.random.seed(seed)

        self.X = None
        self.y = None
        self.errors = []
        self.layers = []
        self.epochs = epochs

    def add_layer(self, layer):
        self.layers.append(layer)

    def end_layers(self):
        for idx, layer in enumerate(self.layers):
            if idx == 0:
                layer.layer_below = self.layers[idx + 1]
                continue
            elif idx + 1 == len(self.layers):
                layer.layer_above = self.layers[idx - 1]
                continue

            layer.layer_above = self.layers[idx - 1]
            layer.layer_below = self.layers[idx + 1]

    def fit(self, X, y):
        self.X = X
        self.y = y

        for ep in range(1, self.epochs + 1):
            error_sum = 0
            for pattern, teacher in zip(X, y):
                pattern = np.reshape(pattern, (1, pattern.shape[0]))
                # forward computation
                output = None
                for layer in self.layers:
                    output = layer.forward(pattern)

                # back-propagation
                for layer in reversed(self.layers):
                    layer.backward(teacher, pattern)

                # apply deltas
                for layer in self.layers:
                    layer.apply_delta()

                error_sum += self.compute_error(teacher, output)
            self.errors.append(error_sum)
            print('Epoch ({})\t||\tloss: {:.4f}'.format(ep, self.errors[-1]))
        self.save_errors()

    @staticmethod
    def compute_error(teacher, output):
        error = np.sum((teacher - output) ** 2)
        return error

    def predict(self, X):
        outputs = []
        for pattern in X:
            # forward computation
            pattern = np.reshape(pattern, (1, pattern.shape[0]))
            output = None
            for layer in self.layers:
                output = layer.forward(pattern)
            outputs.append(output[0, :])
        return np.array(outputs)

    def save_errors(self):
        f = open('learning.curve', 'w')
        print('#\tX\tY', file=f)
        for x, y in enumerate(self.errors):
            print('\t{}\t{}'.format(x, y), file=f)
        f.close()


def read_dat(name):
    """Read data from file.
    """
    X = []
    y = []
    with open(name) as f:
        for line in f:
            if line.startswith('# P'):
                # second line
                # P=350    N=2    M=1
                splits = line.split('    ')
                N = int(splits[1][2:])
                M = int(splits[2][2:])
                continue
            elif line[0] == '#':
                continue
            line = line.strip()
            elements = line.split(' ')
            if '' in elements:
                elements = list(filter(''.__ne__, elements))
            X.append(elements[:N])
            y.append(elements[N:N + M])
        X = np.array(X).astype(np.float)
        y = np.array(y).astype(np.float)

    return X, y


def cross_validation(X, y, split=0.75):
    assert X.shape[0] == y.shape[0]
    size = X.shape[0]
    sep = int(split * size)

    for i in range(size):
        j = random.randint(0, size - 1)
        x_tmp = X[i, :]
        X[i, :] = X[j, :]
        X[j, :] = x_tmp
        y_tmp = y[i, :]
        y[i, :] = y[j, :]
        y[j, :] = y_tmp
        
    return X[:sep, :], y[:sep, :], X[sep:, :], y[sep:, :]


if __name__ == "__main__":
    # read dataset
    X, y = read_dat('PA-B-train-04.dat')
    X_train, y_train, X_val, y_val = cross_validation(X, y, split=0.80)

    # initialization of the network
    # and creating the layers
    net = Network(epochs=200, seed=20)
    net.add_layer(Hidden(X_train.shape[1], 15, learning_rate=0.01, transfer=T_TANH))
    net.add_layer(Hidden(15, 10, learning_rate=0.08, transfer=T_TANH))
    net.add_layer(Hidden(10, 5, learning_rate=0.04, transfer=T_TANH))
    net.add_layer(Output(5, y_train.shape[1], learning_rate=0.005, transfer=T_TANH))
    net.end_layers()

    # training the network
    net.fit(X_train[:, :], y_train[:, :])

    # print random prediction on validation data
    print(net.predict(X_val[5:20, :]).T)
    print(y_val[5:20, :].T)

    # plot errors
    plt.plot(net.errors)
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.show()
