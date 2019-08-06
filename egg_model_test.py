import gym
import ecl_gym
from random import randint
from pyprind import ProgBar

env = gym.make('ecl-v0')
n_steps = 100
bar = ProgBar(n_steps, bar_char='█')
for i_episode in range(1):
    ## reinitialize the environment 
    observation = env.reset()
    ## the simulation for n_steps timesteps
    for t in range(n_steps):
        ##  value, is_rate, is_producer, is_open         
        actions_inje = [[randint(410,430), False, False, True] for _ in range(8)]
        actions_prod = [[randint(220,250), False, True, True] for _ in range(4)]
        ## Advance the simulation forward 
        observation, reward, done, observation_full = \
            env.step(actions_inje + actions_prod)  
        # print (reward) 
        bar.update()
        if done.any():
            print("Episode finished after {} timesteps".format(t+1))
            break
env.close()

