import numpy as np
import random

letter_C = np.array([
    [1, 1, 1, 1, 1],
    [1, 0, 0, 0, 0],
    [1, 0, 0, 0, 0],
    [1, 0, 0, 0, 0],
    [1, 1, 1, 1, 1],
])
noisy_C = np.array([
    [1, 1, 1, 1, 1],
    [0, 1, 0, 0, 1],
    [1, 0, 0, 0, 0],
    [1, 0, 0, 1, 0],
    [1, 0, 1, 1, 1],
])

letter_I = np.array([
    [0, 1, 1, 1, 1],
    [0, 0, 1, 0, 0],
    [0, 0, 1, 0, 0],
    [0, 0, 1, 0, 0],
    [1, 1, 1, 1, 1],
])
noisy_I = np.array([
    [1, 1, 1, 1, 1],
    [0, 0, 1, 0, 0],
    [0, 1, 1, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 1, 0, 1, 1],
])

letter_T = np.array([
    [1, 1, 1, 1, 1],
    [0, 0, 1, 0, 0],
    [0, 0, 1, 0, 0],
    [0, 0, 1, 0, 0],
    [0, 0, 1, 0, 0],
])
noisy_T = np.array([
    [1, 1, 0, 1, 0],
    [0, 0, 1, 0, 0],
    [0, 1, 1, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0],
])


class HopfieldNet:
    def __init__(self, num_neurons, threshold=None):
        assert num_neurons <= 1000
        self.weights = np.zeros((num_neurons, num_neurons)).astype(np.int)
        self.state = np.array((1, num_neurons))
        if threshold:
            self.thresholds = np.array([threshold for _ in num_neurons])
        else:
            self.thresholds = np.zeros((num_neurons,))

    def fit(self, X):
        num_p = X.shape[0]
        num_k = X.shape[1]
        # check right number of pattern
        assert num_p < num_k * 0.138

        num_k = X.shape[1]
        for p in range(X.shape[0]):
            X_p = X[p, :].reshape((1, num_k))
            matrix_lr = np.dot(X_p.T, X_p).astype(np.int)
            np.fill_diagonal(matrix_lr, 0)
            self.weights += matrix_lr

    def predict(self, X, show_energy=False):
        num_k = X.shape[1]
        X_pred = X.copy()

        # loop per every pattern
        for p in range(X_pred.shape[0]):
            differ = True
            time_s = 0

            # loop until the state
            # stay the same
            while differ:
                X_prev = X_pred[p].copy()
                # print energy
                if show_energy:
                    self.print_energy(X_pred[p], p, time_s)

                # loop per every neuron
                for k in range(num_k):
                    val = np.dot(X_pred[p], self.weights[:, k])
                    val_thres = 1 if val > self.thresholds[k] else -1
                    X_pred[p, k] = val_thres

                # check if the new state differs from the previous one
                differ = False if np.array_equal(X_pred[p], X_prev) else True

                time_s += 1

        return X_pred

    def print_energy(self, state, num_p, time_s):
        first_term = 0
        second_term = 0

        for i in range(state.shape[0]):
            for j in range(state.shape[0]):
                first_term += self.weights[i, j] * state[i] * state[j]

        for k in range(state.shape[0]):
            second_term += self.thresholds[k] * state[k]

        energy = -0.5 * first_term + second_term
        print('Pattern: {}\t||\tTime stamp: {}\t||\tEnergy: {:7.0f}'.format(num_p, time_s, energy))
        return energy


def print_char(sequence):
    l = np.sqrt(sequence.shape[0])
    # check if correct sequence
    assert l % 1 == 0

    for y in range(int(l)):
        for x in range(int(l)):
            idx = int(y * l + x)
            val = '*' if sequence[idx] > 0 else ' '
            print(val, end=' ')
        print('', sep='', end='\n')
    print('', sep='', end='\n')


def test_w_less_101():
    print('\n================')
    print('K < 101')
    print('================\n')
    X = np.array([
        letter_C.flatten(),
        letter_I.flatten(),
        letter_T.flatten(),
    ])
    X = np.where(X > 0, 1, -1)
    net = HopfieldNet(X.shape[1])
    net.fit(X)

    X_test = np.array([
        noisy_C.flatten(),
        noisy_I.flatten(),
        noisy_T.flatten(),
    ])
    X_test = np.where(X_test > 0, 1, -1)

    X_out = net.predict(X_test)

    if X.shape[1] <= 100:
        for p in range(X_out.shape[0]):
            print('Test: {}'.format(p + 1))
            print('(Noisy)')
            print_char(X_test[p, :])
            print('(Corrected)')
            print_char(X_out[p, :])


def test_w_more_100():
    print('\n================')
    print('K > 100')
    print('================\n')
    num_k = random.randint(101, 1000)
    binary = 2
    X = np.array([
        np.random.randint(binary, size=num_k),
        np.random.randint(binary, size=num_k),
        np.random.randint(binary, size=num_k),
        np.random.randint(binary, size=num_k),
        np.random.randint(binary, size=num_k),
        np.random.randint(binary, size=num_k),
        np.random.randint(binary, size=num_k),
        np.random.randint(binary, size=num_k),
        np.random.randint(binary, size=num_k),
    ])
    X = np.where(X > 0, 1, -1)
    net = HopfieldNet(X.shape[1])
    net.fit(X)

    X_test = np.array([
        np.random.randint(binary, size=num_k),
        np.random.randint(binary, size=num_k),
        np.random.randint(binary, size=num_k),
    ])
    X_test = np.where(X_test > 0, 1, -1)

    X_out = net.predict(X_test, True)


if __name__ == '__main__':
    test_w_less_101()
    test_w_more_100()
