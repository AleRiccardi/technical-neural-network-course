import numpy as np


class Perzeptron(object):
    """Implementation of a single perzeptron.
    """

    def __init__(self, n=0.01, epochs=50):
        """Initialize the perzeptron with the given
        learining rate and number of epoches.
        """
        self.n = n
        self.epochs = epochs

    def train(self, X_train, y_train):
        """Train the perzeptron given as input
        a matrix of values (X) and an array of
        desired output (y).
        """
        self.w = np.random.uniform(-0.5, 0.5, X_train.shape[1]+1)
        # Iterate for the number of epoches
        # required.
        for _ in range(self.epochs):
            # Iterate for the traing date with
            # the desired ouput.
            for values, target in zip(X_train, y_train):
                netm = self.predict(values)
                # Learning rule
                error = target - netm
                self.w[0] += self.n * error
                self.w[1:] = self.w[1:] + (self.n * error * values)
        return self

    def predict(self, X):
        """Prediction of the output, the value is then
        casted to 1 if it is major of 0, 0 otherwise.
        """
        net_m = np.dot(X, self.w[1:]) + self.w[0]
        y_pred = net_m
        return np.where(y_pred >= 0, 1, 0)


class Network(object):
    """Multi Layet Perzeptron of 2-layers.
    """

    def __init__(self, n=0.01, epochs=50, load_w=False):
        """Initialize the MLP with the given
        learining rate and number of epoches.
        """
        self.n = n
        self.epochs = epochs
        self.perzs = []
        if load_w:
            self.load_weights()

    def load_weights(self):
        """Load weights from file.
        """
        with open('weights.txt') as f:
            lines = f.readlines()
            for line in lines:
                perz = Perzeptron(self.n, self.epochs)
                myarray = np.fromstring(
                    line, dtype=float, sep=',')
                perz.w = myarray
                self.perzs.append(perz)

    def train(self, X_train, y_train):
        """Train the perzeptrons given as input
        a matrix of values (X) and an array of
        desired output (y).
        Return Self if the training succeed, False
        otherwise.
        """
        self.X_train = X_train
        self.y_train = y_train
        # control values
        self.control_data(X_train, y_train)
        for i in range(self.y_train.shape[1]):
            perz = Perzeptron(self.n, self.epochs)
            perz.train(self.X_train, self.y_train[:, i])
            self.perzs.append(perz)

    def predict(self, X):
        """Prediction of the output.
        """
        out_shape = (X.shape[0], len(self.perzs))
        output = np.zeros(out_shape)
        for i in range(out_shape[1]):
            output[:, i] = self.perzs[i].predict(X)
        return output

    def control_data(self, X, y):
        correct = True
        msg = ''
        if X.shape[0] >= 200:
            msg += 'P shall be less than 200.\n'
            correct = False
        if X.shape[1] >= 101:
            msg += 'N shall be less than 101.\n'
            correct = False
        if y.shape[1] >= 30:
            msg += 'M shall be less than 30.\n'
            correct = False

        if not correct:
            raise ValueError(msg)


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
    # Initialize the MLP with the right
    # hyperparameters.
    net = Network(n=0.01, epochs=15, load_w=False)
    # Train the MLP for 8 epochs
    net.train(X_train, y_train)
    # Print the predictions
    print(net.predict(X_train[:5, :]))
