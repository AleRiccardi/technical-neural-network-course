import random
from abc import abstractmethod
import math

from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt


def euclidian_distance(a, b):
    """
    Calculating Euclidian distance between 2 vectors
    :param a: 1st vector
    :param b: 2nd vector
    :return:
    """
    assert len(a) == len(b)
    d = 0
    for feature in range(len(a)):
        d += (a[feature] - b[feature]) ** 2

    return d ** 0.5


class Network(object):
    """
    Radial Basis Function network
    """

    def __init__(self, num_rbf, epochs=50, seed=999):
        """
        Initialize the network
        """
        np.random.seed(seed)

        self.X = None
        self.y = None
        self.num_rbf = num_rbf
        self.errors_train = []
        self.errors_val = []
        self.rbf_layer = None
        self.output_layer = None
        self.epochs = epochs

    def fit(self, X_train, y_train, X_val, y_val):
        self.rbf_layer = RBFlayer(self.num_rbf, X.shape[1])
        self.output_layer = OutLayer(self.num_rbf, y.shape[1])

        self.rbf_layer.find_centers(X)
        self.rbf_layer.find_sizes()
        # loop per every epochs and pattern
        for ep in range(1, self.epochs + 1):
            errors_train = []
            for pattern, teacher in zip(X_train, y_train):
                # rbf layer
                R_vect = self.rbf_layer.forward(pattern)
                output = self.output_layer.forward(R_vect)
                self.output_layer.adjust_weights(teacher)
                loss = self.mean_squared_error(teacher, output)
                errors_train.append(loss)
            self.errors_train.append(sum(errors_train) / len(errors_train))

            # validation
            errors_val = []
            for pattern, teacher in zip(X_val, y_val):
                R_vect = self.rbf_layer.forward(pattern)
                output = self.output_layer.forward(R_vect)
                loss = self.mean_squared_error(teacher, output)
                errors_val.append(loss)
            self.errors_val.append(sum(errors_val) / len(errors_val))

            print('Epoch ({})\t||\ttrain loss: {:.4f}\t||\tval loss: {:.4f}'.format(ep, self.errors_train[-1],
                                                                                    self.errors_val[-1]))
        self.save_errors()

    @staticmethod
    def mean_squared_error(teacher, output):
        error = np.sum((teacher - output) ** 2)
        return error

    def predict(self, X):
        outputs = []
        for pattern in X:
            # forward computation
            R_vect = self.rbf_layer.forward(pattern)
            output = self.output_layer.forward(R_vect)

            outputs.append(output)
        return np.array(outputs)

    def save_errors(self):
        f = open('learning.curve', 'w')
        print('#\tX\tY', file=f)
        for x, y in enumerate(self.errors_val):
            print('\t{}\t{}'.format(x, y), file=f)
        f.close()


class RBFlayer:
    def __init__(self, num_neurons, len_input, closest_percent=0.1):
        self.centers = np.zeros((num_neurons, len_input))
        self.sizes = np.zeros((num_neurons,))
        self.num_neurons = num_neurons
        self.distances_matrix = np.zeros((num_neurons, num_neurons))
        # how many closest centers to consider for each center
        # when computing its radius
        self.closest_percent = closest_percent

    def find_centers(self, all_inputs):
        """
        Sets self.centers with centers found with K-means clustering algorithm
        the number of centers is the number of neurons.
        :param all_inputs:
        :return:
        """
        # find center vectors with Kmeans clustering method
        kmeans = KMeans(n_clusters=self.num_neurons)
        kmeans.fit(all_inputs)
        self.centers = kmeans.cluster_centers_

    def find_sizes(self):
        # fill in distance matrix
        for i in range(self.num_neurons):
            for j in range(i + 1, self.num_neurons):
                if i == j:
                    self.distances_matrix[i, j] = 0
                else:
                    a = self.centers[i, :]
                    b = self.centers[j, :]
                    dist = euclidian_distance(a, b)
                    self.distances_matrix[i, j] = dist
                    self.distances_matrix[j, i] = dist

        # set size for each center to the mean of the distances
        # to 'closest_percent' of the closest centers
        num_closest = math.ceil(self.num_neurons * self.closest_percent)
        # sorting each row of the distance matrix
        sorted_distances = np.sort(self.distances_matrix)
        for i, c in enumerate(self.centers):
            # and taking 'num_closest' distances starting from the second one
            # because first is 0 distance between a center and itself
            self.sizes[i] = np.mean(sorted_distances[i, 1:num_closest + 1])

    def forward(self, X):
        """
        calculate the output of the RBF layer
        :param X: input pattern
        :return:
        """
        distances = []
        # get distances from centers
        for i in range(self.num_neurons):
            distances.append(euclidian_distance(self.centers[i], X))

        # apply the rest of the formula
        distances = np.array([distances], dtype=float)
        distances /= 2 * np.square(self.sizes)
        distances = np.exp(-distances)
        return distances


class OutLayer:

    def __init__(self,
                 n_rbf,
                 n_outputs,
                 learning_rate=0.01):
        self.net = None
        self.out_rbf = None
        self.output = None
        self.sigma = None
        # initialize weights randomly
        self.weights = np.random.uniform(-.5, .5, size=(n_rbf, n_outputs))
        self.learning_rate = learning_rate

    def forward(self, R):
        """
        Forward propagation.
        :param X:
        :return:
        """
        self.out_rbf = R
        self.output = np.dot(R, self.weights)
        return self.output

    def adjust_weights(self, teacher):
        out_sub = (teacher - self.output)
        # calculate weight changes with delta rule
        delta = self.learning_rate * np.dot(self.out_rbf.T, out_sub)
        # apply weight changes
        self.weights += delta


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


def train_test_split(X, y, split=0.75):
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
    # take 'split' percent of the data
    # and further split it to train and validation samples
    X_train, y_train, X_val, y_val = train_test_split(X, y, split=.8)

    # initialize the network
    net = Network(num_rbf=50, epochs=100, seed=10)
    # train and validate each epoch
    net.fit(X_train, y_train, X_val, y_val)

    # test the network by predicting output from the unseen data
    print('Test prediction:')
    prediction = net.predict(X_val)
    # print predicted and true values side bt side
    for i, (y_pred, y_val) in enumerate(zip(prediction, y_val)):
        print('Pattern ({}) || prediction: {:.5f}, actual value: {:.5f}'.format(i, y_pred[0, 0], y_val[0]))

    # plot errors for training and validation
    plt.plot(net.errors_train)
    plt.plot(net.errors_val)
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.show()
