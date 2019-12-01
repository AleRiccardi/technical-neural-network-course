import numpy as np


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

    def __init__(self):
        pass

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
