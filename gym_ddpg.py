#-*-coding:utf-8 -*-
import warnings
warnings.filterwarnings("ignore")
import filter_env
from ddpg import *
import gc
gc.enable()
from cobalt_simulation_2 import cobalt_removal
import pdb

#ENV_NAME = 'InvertedPendulum-v1'#倒立摆
#ENV_NAME = 'InvertedPendulum-v2'
ENV_NAME='Pendulum-v0'#'Acrobot-v1'#'MountainCar-v0'#‘CartPole_v0'#离散的
EPISODES = 10000
TEST = 5

def main():
    #env = filter_env.makeFilteredEnv(gym.make(ENV_NAME))
    env = filter_env.makeFilteredEnv(cobalt_removal())
    agent = DDPG(env)
    #env.monitor.start('experiments/' + ENV_NAME,force=True)

    for episode in xrange(EPISODES):
        state = env.reset()
        #print "episode:",episode
        # Train
        for step in xrange(env.spec.timestep_limit):
            for t in state:
                if np.isinf(t):
                    pdb.set_trace()
            action = agent.noise_action(state)
            next_state,reward,done,_ = env.step(action)
            agent.perceive(state,action,reward,next_state,done)
            state = next_state
            if done:
                break
        # Testing:
        if episode % 10 == 0 and episode > 5:
			total_reward = 0
			for i in xrange(TEST):
				state = env.reset()
				for j in xrange(env.spec.timestep_limit):
					#env.render()
					action = agent.action(state) # direct action for test
					state,reward,done,_ = env.step(action)
					total_reward += reward
					if done:
						break
			ave_reward = total_reward/TEST            
			print 'episode: ',episode,'Evaluation Average Reward:',ave_reward
    #env.monitor.close()

if __name__ == '__main__':
    main()
