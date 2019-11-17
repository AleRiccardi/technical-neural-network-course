import random
from abc import abstractmethod
import math

from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt


def euclidian_distance(a, b):
    assert len(a) == len(b)
    d = 0
    for i in range(len(a)):
        d += (a[i] - b[i]) ** 2

    return d ** 0.5


class Network(object):
    """
    Radial Basis Function network
    """

    def __init__(self, epochs=50, seed=999):
        """
        Initialize the network
        """
        np.random.seed(seed)

        self.X = None
        self.y = None
        self.errors = []
        self.rbf_layer = None
        self.output_layer = None
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


class RBFlayer:
    def __init__(self, num_neurons, len_input, closest_percent=0.1):
        self.centers = np.array((num_neurons, len_input))
        self.sizes = np.array((num_neurons,))
        self.num_neurons = num_neurons
        self.distances_matrix = np.array((num_neurons, num_neurons))
        # how many closest centers to consider for eaech center
        # when computing its radius
        self.closest_percent = closest_percent

    def find_centers(self, all_inputs):
        # find center vectors
        kmeans = KMeans(n_clusters=self.num_neurons)
        kmeans.fit(all_inputs)
        self.centers = kmeans.cluster_centers_

    def find_sizes(self):
        # fill in distance matrix
        for i in range(self.num_neurons):
            for j in range(i+1, self.num_neurons):
                if i == j:
                    self.distances_matrix[i, j] = 0
                else:
                    self.distances_matrix[i, j] = self.distances_matrix[j, i] = euclidian_distance(self.centers[i], self.centers[j])

        # set sizes
        num_closest = math.ceil(self.num_neurons * self.closest_percent)
        sorted_distances = np.sort(self.distances_matrix)
        for i, c in enumerate(self.centers):
            self.sizes[i] = np.mean(sorted_distances[i, 1:num_closest+1])

    def forward(self, X):
        distances = []
        for i in range(self.num_neurons):
            distances.append(euclidian_distance(self.centers[i], X))
        distances = np.array(distances, dtype=float)
        distances /= 2 * np.square(self.sizes)
        distances = np.exp(-distances)
        return distances


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
