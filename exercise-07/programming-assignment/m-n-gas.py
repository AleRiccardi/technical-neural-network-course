import numpy as np
import random
from matplotlib import pyplot as plt


def euclidian_distance(a, b):
    return np.linalg.norm(a - b)


class NeuralGas:

    def __init__(self, num_patterns, centers, min_l_rate=0.01, max_l_rate=0.1, epochs=50):
        self.rank_dist = []
        self.centers = centers
        self.min_l_rate = min_l_rate
        self.max_l_rate = max_l_rate
        self.num_patterns = num_patterns
        self.epochs = epochs

    def fit(self, pattern, epoch, t):
        """
        Fitting of the neural gas.
        :param pattern:
        :param epoch:
        :param t:
        :return:
        """
        for pos, rank_c in enumerate(self.rank_dist):
            center = self.centers[rank_c[0], :]
            sub = (pattern - center)
            neigh_funct = self.neighbourhood(pos, epoch)
            learn_rate = self.learning_rate(t)
            delta_vector = learn_rate * neigh_funct * sub
            self.centers[rank_c[0], :] += delta_vector

    def get_closest_distance(self):
        """
        Get the closes distance after having
        executed compute_distances func.
        :return:
        """
        return self.rank_dist[1]

    def compute_distances(self, pattern):
        """
        Compute distance between all
        the center given a pattern
        :param pattern: numpy array.
        """
        self.rank_dist = []
        for c in range(self.centers.shape[0]):
            # append a tuple of (number of neuron, distance)
            # to the ranked list
            dist = euclidian_distance(self.centers[c, :], pattern)
            self.rank_dist.append((c, dist))
        # actually rank the distances
        self.rank_dist.sort(key=lambda tup: tup[1])

    def neighbourhood(self, pos, epoch, sigma=0.5):
        """
        Neighbourhood function.
        :param pos: position of the center in the
        ranking array.
        :param epoch: number of epoch.
        :param sigma: sigma value of the gaussian
        neighbourhood function
        :return: the value of the function
        """
        norm_pos = pos / self.centers.shape[0]
        num = 1 / (np.sqrt(2 * np.pi) * sigma)
        den = np.exp(norm_pos ** 2 / (2 * sigma ** 2))
        t = ((self.epochs - epoch) - 0.25) / (self.epochs - 0.25)
        return num * den * t

    def learning_rate(self, t):
        """
        Learning rate.
        :param t:
        :return:
        """
        # get num_patterns from looping code every time
        # universal for training and validation
        t_norm = t / self.num_patterns
        l_rate = np.exp(-10 * t_norm)
        return self.max_l_rate * (self.min_l_rate / self.max_l_rate) ** l_rate


class MNGas:

    def __init__(self, num_nets, k_range, X_train, epochs=50):
        """
        Initialization.
        :param num_nets: number of networks.
        :param k_range: tuple (pair) of values that specify the
        range of neurons that can be chosen randomly for each network.
        :param X_train: 2D numpy training data, the axis=0 specify the patterns
        and the axis=1 specify the features of the input pattern.
        """
        self.m_nets = num_nets
        self.k_range = k_range
        self.num_patterns = X_train.shape[0]
        self.num_features = X_train.shape[1]
        self.X_train = X_train
        self.epochs = epochs
        self.gasses = self.init_gasses()
        self.plot_centers(X_train)

    def init_gasses(self):
        gasses = []
        m_centers = self.generate_nets_centers()
        for m in range(self.m_nets):
            gasses.append(NeuralGas(self.num_patterns, m_centers[m], 0.01, 0.1, self.epochs))
        return gasses

    def generate_nets_centers(self):
        # list of number of centers per every net
        m_k_centers = [random.randint(self.k_range[0], self.k_range[1]) for _ in range(self.m_nets)]

        # list of indexes of patterns
        center_patterns = random.sample(range(self.num_patterns), sum(m_k_centers))

        last = 0
        m_centers = []
        # loop per every net
        for i, num_centers in enumerate(m_k_centers):
            # retrieve the right number of indexes of centers
            # for the specific network
            index_patterns = center_patterns[last: (last + num_centers)]
            centers = self.X_train[index_patterns, :]
            m_centers.append(centers)
            last += num_centers

        return m_centers

    def fit(self):
        for epoch in range(self.epochs):
            print('Epoch: {}/{}'.format(epoch + 1, self.epochs))
            interval = self.num_patterns / 6

            for x in range(self.X_train.shape[0]):
                idx_net = self.get_closest_net(x)
                self.gasses[idx_net].fit(self.X_train[x], epoch, x)

                if x >= interval:
                    # if True:
                    interval += x
                    self.plot_centers(self.X_train)

    def get_closest_net(self, x):
        pattern = self.X_train[x, :]
        # (dist, num_net)
        closest_net = (-1, None)
        for m in range(self.m_nets):
            self.gasses[m].compute_distances(pattern)
            dist = self.gasses[m].get_closest_distance()
            if closest_net[0] == -1:
                closest_net = (dist, m)
            if dist < closest_net[0]:
                closest_net = (dist, m)
            if dist == 0:
                print('hopppla')

        # get idx of the closest net
        return closest_net[1]

    def plot_centers(self, X):
        points = ['r.', 'g.', 'b.', 'm.']
        plt.plot(X[:, 0], X[:, 1], 'k.', linewidth=1)
        for m in range(self.m_nets):
            centers = self.gasses[m].centers
            plt.plot(centers[:, 0], centers[:, 1], points[m], label='line 1', linewidth=2)
        plt.show()


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
                line = line.replace('  ', ' ')
                line = line.replace('   ', ' ')
                splits = line.split(' ')
                N = int(splits[2][2:])
                M = int(splits[3][2:])
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


def generate_2_areas():
    """
    Generate 2 circular non overlapping areas.
    :return:
    """
    # return X patterns
    return None


def generate_3_areas():
    """
    Generate 3 circular non overlapping areas.
    :return:
    """
    # return X patterns
    return None


if __name__ == '__main__':
    X, _ = read_dat('PA-D-train.dat.txt')
    m_gas = MNGas(4, (100, 200), X, epochs=50)
    m_gas.fit()
