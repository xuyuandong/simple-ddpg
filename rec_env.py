import sys
import random

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
        dim = 4
        state = [random.randint(0, self.state_space) for i in range(dim)] 
        action = random.random() - 0.5
        reward = random.random() - 0.5
        next_state = [random.randint(0, self.state_space) for i in range(dim)]
        done = random.randint(0,1)
        return state, action, reward, next_state, done

    def rand(self):
        '''random select one recommendation sequence sample'''
        return 0

    def search(self, state, action):
        '''knn find the possible reward'''
        return 0.0


if __name__ == '__main__':
    e = Env()
    for i in range(4):
        print e.step()
