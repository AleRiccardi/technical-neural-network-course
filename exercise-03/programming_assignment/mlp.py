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
        self.weights = np.random.uniform(-2, 2, size=(n_inputs +1 , n_outputs))
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
        weights_T = self.layer_below.weights.T
        self.sigma = np.dot(self.layer_below.sigma, weights_T) * derivative(self.net, self.transfer)
        # delta W = n * sigma * out_above
        if self.layer_above:
            shape = self.layer_above.output.shape
            output_T = np.reshape(self.layer_above.output, (shape[0], 1))
        else:
            shape = pattern.shape
            output_T = np.reshape(pattern, (shape[0], 1))
        self.delta = self.learning_rate * np.dot(output_T, self.sigma)


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
        self.sigma = ((teacher - self.output) * derivative(self.net, self.transfer))
        self.sigma = np.reshape(self.sigma, (1, self.sigma.shape[0]))
        # delta W = n * sigma * out_above
        shape = self.layer_above.output.shape
        output_T = np.reshape(self.layer_above.output, (shape[0], 1))
        self.delta = self.learning_rate * np.dot(output_T, self.sigma)


class Network(object):
    """
    Multi Layer Perzeptron.
    """

    def __init__(self, epochs=50):
        """
        Initialize the MLP.
        """
        np.random.seed(999)

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
            print('Epoch n: {}'.format(ep))
            for pattern, teacher in zip(X, y):
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
                self.compute_error(teacher, output)

    def compute_error(self, teacher, output):
        self.errors.append(np.sum((teacher - output) ** 2) * (1 / 2))

    def predict(self, X):
        pass


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


if __name__ == "__main__":
    # read dataset
    X_train, y_train = read_dat('PA-B-train-04.dat')
    # Initialize the MLP
    print(X_train.shape)

    net = Network()
    net.add_layer(Hidden(X_train.shape[1], 30))
    net.add_layer(Output(30, y_train.shape[1]))
    net.end_layers()

    net.fit(X_train, y_train)
