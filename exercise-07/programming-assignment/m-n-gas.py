import numpy as np
import random

def euclidian_distance(a, b):
    assert len(a) == len(b)
    sum_sq_dif = 0
    for i in range(a.shape[0]):
        sum_sq_dif += (a[i] - b[i]) ** 2
    return sum_sq_dif ** 0.5


class NeuralGas:

    def __init__(self, num_patterns, centers, min_learning_rate=0.01, max_learning_rate=0.1, max_t=30):
        self.rank_dist = []
        self.centers = centers
        self.min_learning_rate = min_learning_rate
        self.max_learning_rate = max_learning_rate
        self.num_patterns = num_patterns
        self.max_t = max_t

    def fit(self, x, t):
        for k in self.rank_dist:
            delta_vector = self.learning_rate(t) * self.neighbourhood(k, t) * (x - self.centers[k])
            self.centers[k] += delta_vector

    def closest_distance(self):
        return self.rank_dist[0]

    def compute_distances(self, x):
        self.rank_dist = []
        for c in range(self.centers.shape[0]):
            # append a tuple of (number of neuron, distance) to the ranked list
            self.rank_dist.append((c, euclidian_distance(self.centers[c,:], x)))
        # actually rank the distances
        self.rank_dist.sort(key=lambda tup: tup[1])
        self.rank_dist = [x[0] for x in self.rank_dist]



    def neighbourhood(self, dist, t):
        # get num_patterns from looping code every time
        # universal for training and validation
        t = (t / self.num_patterns) * self.max_t
        n = 1 - (0.99 + t) * dist
        return n if n >=0 else 0


    def learning_rate(self, t):
        # get num_patterns from looping code every time
        # universal for training and validation
        return self.max_learning_rate * (self.min_learning_rate / self.max_learning_rate)\
               ** (t/self.num_patterns)


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


def generate_2_areas(num_patterns, num_dimensions, radius=0.1):
    """
    Generate 2 circular non overlapping areas.
    :return:
    """
    i = 0
    patterns = []
    while i < num_patterns:
        pattern = []
        for j in range(num_dimensions):
            pattern.append(0.33 + random.uniform(-radius, radius))
        patterns.append(pattern)
        i += 1

        if i == num_patterns:
            break

        for j in range(num_dimensions):
            pattern.append(0.66 + random.uniform(-radius, radius))
        patterns.append(pattern)
        i += 1

    return patterns


def generate_3_areas(num_patterns, num_dimensions, radius=0.1):
    """
    Generate 3 circular non overlapping areas.
    :return:
    """
    i = 0
    patterns = []
    while i < num_patterns:
        pattern = []
        for j in range(num_dimensions):
            pattern.append(0.25 + random.uniform(-radius, radius))
        patterns.append(pattern)
        i += 1

        if i == num_patterns:
            break

        for j in range(num_dimensions):
            pattern.append(0.5 + random.uniform(-radius, radius))
        patterns.append(pattern)
        i += 1

        if i == num_patterns:
            break

        for j in range(num_dimensions):
            pattern.append(0.75 + random.uniform(-radius, radius))
        patterns.append(pattern)
        i += 1

    return patterns


if __name__ == '__main__':
    X, _ = read_dat('PA-D-train.dat.txt')
    m_gas = MNGas(4, (50, 100), X)
