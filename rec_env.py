import sys
import random

class Env(object):
    def __init__(self):
        self.state_space = 1000000
        self.action_dim = 1
        self.timestep_limit = 10
        pass

    def read_data(self, f):
        '''parse training data, generate recommendataion sequence'''
        pass

    def reset(self):
        '''select the next recommendation sequence'''
        pass

    def step(self):
        '''return one sample'''
        dim = random.randint(4,20)
        state = [random.randint(0, self.state_space) for i in range(dim)] 
        dim = random.randint(4,20)
        next_state = [random.randint(0, self.state_space) for i in range(dim)]
        action = random.random() - 0.5
        reward = random.random() - action
        done = random.randint(0,1)
        return state, action, reward, next_state, done
        
    def pretrained_step(self):
        state = [random.uniform(-1.0, 1.0) for i in range(self.state_space)] 
        next_state = [random.uniform(-1.0, 1.0) for i in range(self.state_space)] 
        action = random.random() - 0.5
        reward = random.random() - 0.5
        next_state[0] += action
        reward += random.uniform(1.0, 0.2) * action
        done = random.randint(0,1)
        return state, action, reward, next_state, done

    def rand(self):
        '''random select one recommendation sequence sample'''
        state = [random.uniform(-1.0, 1.0) for i in range(self.state_space)] 
        return state

    def search(self, state, action):
        '''knn find the possible reward'''
        reward = random.random() - 0.5
        reward += random.uniform(1.0, 0.2) * action
        next_state = [random.uniform(-1.0, 1.0) for i in range(self.state_space)] 
        next_state[0] += action
        return next_state, reward


if __name__ == '__main__':
    e = Env()
    for i in range(4):
        print (e.pretrained_step())
