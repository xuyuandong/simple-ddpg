import tensorflow as tf 
import numpy as np
import math


# Hyper Parameters
LAYER1_SIZE = 128
LAYER2_SIZE = 64
LEARNING_RATE = 1e-4
TAU = 0.001
BATCH_SIZE = 64

class ActorCriticNetwork:
	"""docstring for ActorNetwork"""
	def __init__(self,sess,state_space,action_dim):

		self.sess = sess
		self.state_space = state_space
		self.action_dim = action_dim

		# create eval network
		self.state_input, self.state_embedding,\
                        self.action, self.actor_net,\
                        self.action_input, self.q_value, self.critic_net \
                        = self.create_eval_network(state_space, action_dim)

		# create target network
		self.target_state_input,self.target_state_embedding,self.target_state_update,\
                        self.target_action,self.target_actor_net,self.target_actor_update,\
                        self.target_action_input, self.target_q_value,self.target_critic_net,self.target_critic_update \
                        = self.create_target_network(state_space, self.state_embedding, self.actor_net, self.critic_net)

		# define training rules
                self.create_training_actor_method([self.state_embedding] + self.actor_net)
		self.create_training_critic_method([self.state_embedding] + self.critic_net)

		self.sess.run(tf.initialize_all_variables())

		self.update_target()

	def create_training_critic_method(self, net):
                ''' for eval network '''
		self.q_input = tf.placeholder("float",[None,1])
		weight_decay = tf.add_n([L2 * tf.nn.l2_loss(var) for var in net])
		self.cost = tf.reduce_mean(tf.square(self.q_input - self.q_value)) + weight_decay
		self.optimizer = tf.train.AdamOptimizer(LEARNING_RATE).minimize(self.cost)
		self.action_gradients = tf.gradients(ys=self.q_value, xs=self.action_input)

	def create_training_actor_method(self, net):
                ''' for eval network '''
		self.q_gradient_input = tf.placeholder("float",[None,self.action_dim])
		self.parameters_gradients = tf.gradients(ys=self.action, xs=net, grad_ys=-self.q_gradient_input)
		self.optimizer = tf.train.AdamOptimizer(LEARNING_RATE).apply_gradients(zip(self.parameters_gradients, net))

	def create_eval_network(self, state_space, action_dim):
                state_dim = 32
                state_input = tf.sparse_placeholder(tf.float32, shape=[None, state_space])  # None * state_space
		action_input = tf.placeholder("float",[None,action_dim])
                
                state_embedding = tf.Variable(tf.random_normal([state_space, state_dim], 0.0, 0.01), name='state_embedding')  
                state_embed_input = tf.sparse_tensor_dense_matmul(state_input, state_embedding)
                state_embed_input = tf.reduce_mean(state_embed_input, axis=1)

                action, actor_net = self.create_eval_actor_network(state_embed_input, state_dim, action_dim)
                q_value, critic_net = self.create_eval_critic_network(state_embed_input, action_input, state_dim, action_dim)
                return state_input, state_embedding, action, actor_net, action_input, q_value, critic_net

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
	
        def create_eval_critic_network(self,state_embed_input, action_input, state_dim,action_dim):
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

	def create_target_network(self, state_space, state_embedding, actor_net, critic_net):
                state_input = tf.sparse_placeholder(tf.float32, shape=[None, state_space])  # None * state_space
		action_input = tf.placeholder("float",[None,action_dim])
                ema = tf.train.ExponentialMovingAverage(decay=1-TAU)
		
                # state embedding
                state_embedding_update = ema.apply([state_embedding])
                target_state_embedding = ema.average(state_embedding)

                state_embed_input = tf.sparse_tensor_dense_matmul(state_input, target_state_embedding)
                state_embed_input = tf.reduce_mean(state_embed_input, axis=1)

                # actor net
		target_actor_update = ema.apply(actor_net)
		target_actor_net = [ema.average(x) for x in actor_net]
                action = self.create_target_actor_network(state_embed_input, target_actor_net)	

                # critic net
                target_critic_update = ema.apply(critic_net)
                target_critic_net = [ema.average(x) for x in critic_net]
                q_value = self.create_target_critic_network(state_embed_input, action_input, target_critic_net)

		return state_input,target_state_embedding,state_embedding_update,\
                        action,target_actor_net, target_actor_update,\
                        action_input, q_value, target_critic_net, target_critic_update

        def create_target_actor_network(self, state_embed_input, target_net):
                layer1 = tf.nn.relu(tf.matmul(state_embed_input,target_net[0]) + target_net[1])
		layer2 = tf.nn.relu(tf.matmul(layer1,target_net[2]) + target_net[3])
		action_output = tf.tanh(tf.matmul(layer2,target_net[4]) + target_net[5])
                return action_output

        def create_target_critic_network(self, state_embed_input, action_input, target_net):
                layer1 = tf.nn.relu(tf.matmul(state_embed_input,target_net[0]) + target_net[1])
                layer2 = tf.nn.relu(tf.matmul(layer1,target_net[2]) + tf.matmul(action_input,target_net[3]) + target_net[4])
                q_value_output = tf.identity(tf.matmul(layer2,target_net[5]) + target_net[6])
                return q_value_output


	def update_target(self):
		self.sess.run([self.target_state_update, self.target_actor_update, self.target_critic_update])

	def train_critic(self,y_batch,state_batch,action_batch):
		self.time_step += 1
		self.sess.run(self.optimizer,feed_dict={
			self.y_input:y_batch,
			self.state_input:state_batch,
			self.action_input:action_batch
			})

	def train_actor(self,q_gradient_batch,state_batch):
		self.sess.run(self.optimizer,feed_dict={
			self.q_gradient_input:q_gradient_batch,
			self.state_input:state_batch
			})

        ''' critic net '''
	def gradients(self,state_batch,action_batch):
		return self.sess.run(self.action_gradients,feed_dict={
			self.state_input:state_batch,
			self.action_input:action_batch
			})[0]
	
        def target_q(self,state_batch,action_batch):
		return self.sess.run(self.target_q_value,feed_dict={
			self.target_state_input:state_batch,
			self.target_action_input:action_batch
			})

	def q_value(self,state_batch,action_batch):
		return self.sess.run(self.q_value,feed_dict={
			self.state_input:state_batch,
			self.action_input:action_batch})

        ''' actor net '''
	def actions(self,state_batch):
		return self.sess.run(self.action,feed_dict={
			self.state_input:state_batch
			})

	def action(self,state):
		return self.sess.run(self.action,feed_dict={
			self.state_input:[state]
			})[0]


	def target_actions(self,state_batch):
		return self.sess.run(self.target_action,feed_dict={
			self.target_state_input:state_batch
			})

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

		
