import tensorflow as tf 
import numpy as np
import math


# Hyper Parameters
LAYER1_SIZE = 128
LAYER2_SIZE = 64
LEARNING_RATE = 1e-4
TAU = 0.001
L2 = 0.01
BATCH_SIZE = 64

class ActorCriticNetwork:
        """docstring for ActorNetwork"""
        def __init__(self,sess,state_space,action_dim):
                self.time_step = 0
                self.sess = sess
                self.state_space = state_space
                self.action_dim = action_dim
                
                state_dim = 32

                # create eval network
                self.state_input = tf.sparse_placeholder(tf.float32, shape=[None, state_space])  # None * state_space
                self.state_embedding = tf.Variable(tf.random_normal([state_space, state_dim], 0.0, 0.01), name='state_embedding')  
                state_embed_input = tf.sparse_tensor_dense_matmul(self.state_input, self.state_embedding)

                self.action, self.actor_net \
                        = self.create_eval_actor_network(state_embed_input, state_dim, action_dim)
                self.q_value, self.critic_net \
                        = self.create_eval_critic_network(state_embed_input, self.action, state_dim, action_dim)

                # create target network
                self.ema = tf.train.ExponentialMovingAverage(decay=1-TAU)
                self.target_state_input = tf.sparse_placeholder(tf.float32, shape=[None, state_space])  # None * state_space
                self.target_state_update = self.ema.apply([self.state_embedding])
                self.target_state_embedding = self.ema.average(self.state_embedding)
                target_state_embed_input = tf.sparse_tensor_dense_matmul(self.target_state_input, self.target_state_embedding)
                
                self.target_action, self.target_actor_net, self.target_actor_update,\
                        = self.create_target_actor_network(self.ema, target_state_embed_input, self.actor_net)
                self.target_q_value,self.target_critic_net,self.target_critic_update \
                        = self.create_target_critic_network(self.ema, target_state_embed_input, self.target_action, self.critic_net)

                # define training rules
                self.create_training_actor_method([self.state_embedding] + self.actor_net)
                self.create_training_critic_method([self.state_embedding] + self.critic_net)

                self.sess.run(tf.initialize_all_variables())
                self.update_target()

        def create_training_critic_method(self, net):
                self.y_input = tf.placeholder("float",[None,1])
                weight_decay = tf.add_n([L2 * tf.nn.l2_loss(var) for var in net])
                self.cost = tf.reduce_mean(tf.square(self.y_input - self.q_value)) + weight_decay
                self.critic_optimizer = tf.train.AdamOptimizer(LEARNING_RATE).\
                        minimize(self.cost, var_list=[self.state_embedding]+self.critic_net)

        def create_training_actor_method(self, net):
                self.actor_loss = -tf.reduce_mean(self.q_value)
                self.actor_optimizer = tf.train.AdamOptimizer(LEARNING_RATE).\
                        minimize(self.actor_loss, var_list=[self.state_embedding] + self.actor_net)

        def create_eval_actor_network(self, state_embed_input, state_dim, action_dim):
                layer1_size = LAYER1_SIZE
                layer2_size = LAYER2_SIZE
                W1 = self.variable([state_dim,layer1_size],state_dim)
                b1 = self.variable([layer1_size],state_dim)
                W2 = self.variable([layer1_size,layer2_size],layer1_size)
                b2 = self.variable([layer2_size],layer1_size)
                W3 = tf.Variable(tf.random_uniform([layer2_size,action_dim],-3e-3,3e-3))
                b3 = tf.Variable(tf.random_uniform([action_dim],-3e-3,3e-3))

                layer1 = tf.nn.relu(tf.matmul(state_embed_input,W1) + b1)
                layer2 = tf.nn.relu(tf.matmul(layer1,W2) + b2)
                action_output = tf.tanh(tf.matmul(layer2,W3) + b3)

                return action_output, [W1,b1,W2,b2,W3,b3]
        
        def create_eval_critic_network(self,state_embed_input, action_input, state_dim, action_dim):
                layer1_size = LAYER1_SIZE
                layer2_size = LAYER2_SIZE
                W1 = self.variable([state_dim,layer1_size],state_dim)
                b1 = self.variable([layer1_size],state_dim)
                W2 = self.variable([layer1_size,layer2_size],layer1_size+action_dim)
                W2_action = self.variable([action_dim,layer2_size],layer1_size+action_dim)
                b2 = self.variable([layer2_size],layer1_size+action_dim)
                W3 = tf.Variable(tf.random_uniform([layer2_size,1],-3e-3,3e-3))
                b3 = tf.Variable(tf.random_uniform([1],-3e-3,3e-3))

                layer1 = tf.nn.relu(tf.matmul(state_embed_input,W1) + b1)
                layer2 = tf.nn.relu(tf.matmul(layer1,W2) + tf.matmul(action_input,W2_action) + b2)
                q_value_output = tf.identity(tf.matmul(layer2,W3) + b3)

                return q_value_output,[W1,b1,W2,W2_action,b2,W3,b3]

        def create_target_actor_network(self, ema, state_embed_input, actor_net):
                target_update = ema.apply(actor_net)
                target_net = [ema.average(x) for x in actor_net]
                
                layer1 = tf.nn.relu(tf.matmul(state_embed_input,target_net[0]) + target_net[1])
                layer2 = tf.nn.relu(tf.matmul(layer1,target_net[2]) + target_net[3])
                action_output = tf.tanh(tf.matmul(layer2,target_net[4]) + target_net[5])
                
                return action_output, target_net, target_update

        def create_target_critic_network(self, ema, state_embed_input, action_input, critic_net):
                target_update = ema.apply(critic_net)
                target_net = [ema.average(x) for x in critic_net]
                
                layer1 = tf.nn.relu(tf.matmul(state_embed_input,target_net[0]) + target_net[1])
                layer2 = tf.nn.relu(tf.matmul(layer1,target_net[2]) + tf.matmul(action_input,target_net[3]) + target_net[4])
                q_value_output = tf.identity(tf.matmul(layer2,target_net[5]) + target_net[6])
                
                return q_value_output, target_net, target_update

        def update_target(self):
                self.sess.run([self.target_state_update, self.target_actor_update, self.target_critic_update])

        def train_critic(self,y_batch,state_batch,action_batch):
                self.time_step += 1
                cost,_ = self.sess.run([self.cost,self.critic_optimizer],feed_dict={
                        self.y_input:y_batch,
                        self.state_input:state_batch,
                        self.action:action_batch
                        })
                return cost

        def train_actor(self,state_batch):
                self.sess.run(self.actor_optimizer,feed_dict={
                        self.state_input:state_batch
                        })

        ''' critic net '''
        def target_q(self,state_batch):
                return self.sess.run(self.target_q_value,feed_dict={
                        self.target_state_input:state_batch,
                        })

        def q_value(self,state_batch,action_batch):
                return self.sess.run(self.q_value,feed_dict={
                        self.state_input:state_batch,
                        self.action:action_batch})

        ''' actor net '''
        def actions(self,state_batch):
                return self.sess.run(self.action,feed_dict={
                        self.state_input:state_batch
                        })

        def action(self,state):
                return self.sess.run(self.action,feed_dict={
                        self.state_input:[state]
                        })[0]

        # f fan-in size
        def variable(self,shape,f):
                return tf.Variable(tf.random_uniform(shape,-1/math.sqrt(f),1/math.sqrt(f)))
'''
        def load_network(self):
                self.saver = tf.train.Saver()
                checkpoint = tf.train.get_checkpoint_state("saved_actor_networks")
                if checkpoint and checkpoint.model_checkpoint_path:
                        self.saver.restore(self.sess, checkpoint.model_checkpoint_path)
                        print "Successfully loaded:", checkpoint.model_checkpoint_path
                else:
                        print "Could not find old network weights"
        def save_network(self,time_step):
                print 'save actor-network...',time_step
                self.saver.save(self.sess, 'saved_actor_networks/' + 'actor-network', global_step = time_step)

'''

                
