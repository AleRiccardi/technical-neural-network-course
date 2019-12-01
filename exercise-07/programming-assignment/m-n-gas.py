import numpy as np

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

    def __init__(self):
        self.gases = []
        pass

    def init_gases(self):


    def fit(self, X_train):
        # + loop for each pattern
        # present each pattern to each network
        # ++ Loop net in networks:
        # close_net = -1
        # if net.get_shortest_dist() < close_dist:
        # close_net = pair(net.get_shortest_dist(), net)
        # -- close loop
        # close_net[1].confirm_fit()
        # - end loop
        pass

    def predict(self, X):
        # TODO: chance below
        # + loop for each pattern
        # present each pattern to each network
        # ++ Loop net in networks:
        # close_net = -1
        # if net.get_shortest_dist() < close_dist:
        # close_net = pair(net.get_shortest_dist(), net)
        # -- close loop
        # close_net[1].confirm_fit()
        # - end loop

        # return list of pattern
        pass


if __name__ == '__main__':
    print('ciao')
