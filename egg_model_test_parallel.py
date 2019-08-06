## TODO
# SubprocVecEnv close method 
# makes no effect

import gym
import ecl_gym
from random import randint
from pyprind import ProgBar
from multiprocessing_env import SubprocVecEnv, VecNormalize
env_name = 'ecl-v0'
N_ENVS = 1
def make_env():
        return gym.make(env_name)
if __name__ == '__main__':
    envs = [make_env for i in range(N_ENVS)]
    envs = SubprocVecEnv(envs)
    obs = envs.reset()
    print ("OBSERVATION ", obs[0])
    obs = obs.reshape(-1)
    obs_shape = obs.shape
    envs = VecNormalize(envs, obs_shape, ob=False, gamma=0.99)

    n_steps = 100
    bar = ProgBar(n_steps, bar_char='█')
    for i_episode in range(2):
        ## reinitialize the environment 
        observation = envs.reset()
        ## the simulation for n_steps timesteps
        for t in range(n_steps):
            ##  value, is_rate, is_producer, is_open         
            actions_inje = [[randint(410,430), False, False, True] for _ in range(8)]
            actions_prod = [[randint(220,250), False, True, True] for _ in range(4)]
            ## Advance the simulation forward 
            observation, reward, done, observation_full = \
                envs.step([(actions_inje + actions_prod) for _ in range(N_ENVS)])  
            # print (reward) 
            bar.update()
            if done.any():
                print("Episode finished after {} timesteps".format(t+1))
                break
    envs.close()