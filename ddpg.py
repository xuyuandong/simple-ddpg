# -----------------------------------
# Deep Deterministic Policy Gradient
# -----------------------------------
import tensorflow as tf
import numpy as np
from ou_noise import OUNoise
from actor_critic_network import ActorCriticNetwork
from replay_buffer import ReplayBuffer

# Hyper Parameters:

REPLAY_BUFFER_SIZE = 1000000
REPLAY_START_SIZE = 10000
BATCH_SIZE = 64
GAMMA = 0.9


class DDPG:
    """docstring for DDPG"""
    def __init__(self, state_space, action_dim):
        self.name = 'DDPG' # name for uploading results
        self.sess = tf.Session()  

        # Randomly initialize actor network and critic network
        # with both their target networks
        self.state_space = state_space
        self.action_dim = action_dim # 1


        self.ac_network = ActorCriticNetwork(self.sess,self.state_space,self.action_dim)
        
        # initialize replay buffer
        self.replay_buffer = ReplayBuffer(REPLAY_BUFFER_SIZE)

        # Initialize a random process the Ornstein-Uhlenbeck process for action exploration
        self.exploration_noise = OUNoise(self.action_dim)

    def train(self):
        # Sample a random minibatch of N transitions from replay buffer
        minibatch = self.replay_buffer.get_batch(BATCH_SIZE)
        state_batch = np.asarray([data[0] for data in minibatch])
        state_batch = self.sparse_tensor(state_batch, self.state_space)
        action_batch = np.asarray([data[1] for data in minibatch])
        reward_batch = np.asarray([data[2] for data in minibatch])
        next_state_batch = np.asarray([data[3] for data in minibatch])
        next_state_batch = self.sparse_tensor(next_state_batch, self.state_space)
        done_batch = np.asarray([data[4] for data in minibatch])

        # for action_dim = 1
        action_batch = np.resize(action_batch,[BATCH_SIZE,self.action_dim])

        # Run policy by target actor network
        # a' = pi(s')
        next_action_batch = self.ac_network.target_actions(next_state_batch)
        # maxQ(s',a')
        q_value_batch = self.ac_network.target_q(next_state_batch,next_action_batch)

        # Calculate target maxQ(s,a): y = reward + GAMMA * maxQ(s',a')
        y_batch = []  
        for i in range(len(minibatch)): 
            if done_batch[i]:
                y_batch.append(reward_batch[i])
            else :
                y_batch.append(reward_batch[i] + GAMMA * q_value_batch[i])
        y_batch = np.resize(y_batch,[BATCH_SIZE,1])
        
        # Update eval critic network by minimizing the loss L
        cost = self.ac_network.train_critic(y_batch,state_batch,action_batch)
        print 'step_%d critic cost:'%self.ac_network.time_step, cost

        # a = pi(s)
        action_batch_for_gradients = self.ac_network.actions(state_batch)
        # ga from maxQ(s,a), -ga for maxmizing maxQ(s,a)
        q_gradient_batch = self.ac_network.gradients(state_batch,action_batch_for_gradients)
        
        # Update eval actor policy using the sampled gradient:
        self.ac_network.train_actor(q_gradient_batch,state_batch)
        #TODO: no sampling?

        # Update the target networks
        self.ac_network.update_target()
        self.ac_network.update_target()

    def noise_action(self,state):
        # Select action a_t according to the current policy and exploration noise
        action = self.ac_network.action(state)
        return action+self.exploration_noise.noise()

    def action(self,state):
        action = self.ac_network.action(state)
        return action

    def perceive(self,state,action,reward,next_state,done):
        # Store transition (s_t,a_t,r_t,s_{t+1}) in replay buffer
        self.replay_buffer.add(state,action,reward,next_state,done)

        # Store transitions to replay start size then start training
        if self.replay_buffer.count() >  REPLAY_START_SIZE:
            self.train()

        #if self.time_step % 10000 == 0:
            #self.actor_network.save_network(self.time_step)
            #self.critic_network.save_network(self.time_step)

        # Re-iniitialize the random process when an episode ends
        if done:
            self.exploration_noise.reset()

    def sparse_tensor(self, state_batch, state_space):
        row = len(state_batch)
        indices = []
        for r in xrange(row):
            indices += [(r, c) for c in state_batch[r]]
        values = [1.0 for i in xrange(len(indices))]
        return tf.SparseTensorValue(indices=indices, values=values, dense_shape=[row, state_space])









