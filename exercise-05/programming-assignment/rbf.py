import random
from abc import abstractmethod

import numpy as np
import matplotlib.pyplot as plt


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
        self.layer = []
        self.epochs = epochs

    def fit(self, X, y):
        pass

    @staticmethod
    def compute_error(teacher, output):
        error = np.sum((teacher - output) ** 2)
        return error

    def predict(self, X):
        pass

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
