from ddpg import *
#from new_ddpg import *
#from pretrained_ddpg import *
import rec_env
import sys
import gc
gc.enable()

EPISODES = 100000
TEST_NUM = 10
flag_test = False

def main():
    env = rec_env.Env()
    agent = DDPG(env.state_space, env.action_dim)

    for episode in range(EPISODES):
        env.reset()
        # Train
        for step in range(env.timestep_limit):
            state,action,reward,next_state,done = env.step()
            agent.perceive(state,action,reward,next_state,done)
            if done:
                break
        # Testing:
        if flag_test and episode > 0:
            total_reward = 0
            for i in xrange(TEST_NUM):
                state = env.rand()
                for step in range(env.timestep_limit):
                    action = agent.action(state) # direct action for test
                    state, reward = env.search(state, action)
                    total_reward += reward
            ave_reward = total_reward/TEST_NUM
            print ('episode: ',episode,'Evaluation Average Reward:',ave_reward)

if __name__ == '__main__':
    main()
