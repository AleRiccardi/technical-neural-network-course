import numpy as np


class Layer(object):
    """
    """

    def __init__(self, n_inputs, n_outputs, learingn_rate=0.01, transfer='tanh'):
        """Initialize the perzeptron with the given
        learning rate and number of epochs.
        """
        self.biases = np.zeros(n_outputs)
        self.weights = np.random.uniform(-0.5, 0.5, size=(n_inputs, n_outputs))
        self.learning_rate = learingn_rate
        self.transfer = transfer

    def forward(self, input):
        """
        Forward propagation.
        """
        return np.dot(input, self.weights) + self.biases

    def backward(self, output, delta):
        """
        Backward propagation.
        """
        return None


class Network(object):
    """
    Multi Layer Perzeptron.
    """

    def __init__(self, epochs=50):
        """
        Initialize the MLP.
        """
        np.random.seed(999)
        self.epochs = epochs
        self.layers = []

    def add_layer(self, layer):
        self.layers.append(layer)

    def fit(self, X, y):
        for ep in range(self.epochs):
            print('Epoch n: {}'.format(ep))
            for pattern, output in zip(X, y):
                outputs = []
                for layer in self.layers:
                    outputs.append(layer.forward(pattern))

                # back-propagation
                pass

    def predict(self, X):
        pass


def read_dat(name):
    """Read data from file.
    """
    X = []
    y = []
    with open(name) as f:
        for line in f:
            if line[0] == '#':
                continue
            line = line.strip()
            X_string = line.split('\t')[0].strip()
            y_string = line.split('\t')[1].strip()
            X.append(X_string.split(' '))
            y.append(y_string.split(' '))
    X = np.matrix(X).astype(np.int8)
    y = np.matrix(y).astype(np.int8)
    return X, y


if __name__ == "__main__":
    # read dataset
    X_train, y_train = read_dat('PA-A-train.dat')
    # Initialize the MLP
    net = Network()
    net.add_layer(Layer(X_train.shape[0], 30))
    net.add_layer(Layer(30, 4))

    net.fit(X_train, y_train)
