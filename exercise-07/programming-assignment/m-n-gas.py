import numpy as np
import random


class NeuralGas:

    def __init__(self):
        self.rank_dist = []

    def fit(self, pattern):
        pass

    def confirm_fit(self):
        pass

    def get_shortest_dist(self):
        pass

    def neighbour_func(self):
        pass

    def learning_rate(self):
        pass


class MNGas:

    def __init__(self, num_nets, k_range, X_train):
        """
        Initialization.
        :param m_nets: number of networks.
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
        self.gasses = self.init_gasses()

    def init_gasses(self):
        gasses = []
        m_centers = self.generate_nets_centers()
        for m in range(self.m_nets):
            gasses.append(NeuralGas(self.num_patterns, m_centers[m]))
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

    def fit(self, X_train):
        for x in range(X_train.shape[0]):
            pattern = X_train[x, :]
            # (dist, num_net)
            closest_net = (-1, None)
            for m in range(self.m_nets):
                self.gasses[m].compute_distances(pattern)
                dist = self.gasses[m].closest_distance()
                if closest_net[0] == -1:
                    closest_net = (dist, m)
                if dist < closest_net[0]:
                    closest_net = (dist, m)

            # get num of the closest net
            num_net = closest_net[1]
            # fit the closest net
            self.m_nets[num_net].fit(pattern, x)


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
    m_gas = MNGas(4, (50, 100), X)
