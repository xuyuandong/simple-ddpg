import sys

class Env(object):
    def __init__(self):
        self.state_space = 1000000
        self.action_dim = 1
        self.timestep_limit = 300
        pass

    def read_data(self, f):
        '''parse training data, generate recommendataion sequence'''
        pass

    def reset(self):
        '''select the next recommendation sequence'''
        pass

    def step(self):
        '''return one sample'''
        pass

    def rand(self):
        '''random select one recommendation sequence sample'''
        return 0

    def search(self, state, action):
        '''knn find the possible reward'''
        return 0.0
