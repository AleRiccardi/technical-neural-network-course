import numpy as np
from mlxtend.plotting import plot_decision_regions
import matplotlib.pyplot as plt


class Perzeptron(object):

    def __init__(self, n=0.01, epochs=50):
        self.n = n
        self.epochs = epochs

    def train(self, X, y):
        self.w = np.zeros(X.shape[1]+1)
        self.errors = []
        for _ in range(self.epochs):
            for values, target in zip(X, y):
                netm = self.predict(values)
                error = target - netm
                self.w[0] += self.n * error
                self.w[1:] += self.n * error * values
        return self

    def calculate(self, X):
        return np.dot(X, self.w[1:]) + self.w[0]

    def predict(self, X):
        return np.where(self.calculate(X) >= 0, 1, -1)


if __name__ == "__main__":
    e = 1
    net = Perzeptron(0.01, 10)
    X = np.array([[0, 0], [2, 3], [1-e, 3-e],
                  [-1+e, 3+e], [-1, 2], [3, 1], [5, -2+e], [5-e, -2]])
    y = np.array([1, -1, 1, -1, 1, -1, -1, 1])
    net.train(X, y)
    print('w1: {:.2f}, w2: {:.2f}, wb: {:.2f}'.format(
        net.w[1], net.w[2], net.w[0]))
    plot_decision_regions(X, y, clf=net)
    plt.title('Perceptron')
    plt.xlabel('sepal length [cm]')
    plt.ylabel('petal length [cm]')
    plt.show()
