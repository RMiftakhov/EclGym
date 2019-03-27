# EclGym - Reservoir Simulation environment for Reinforcement Learning

Eclipse Integration for [Gym toolkit](https://gym.openai.com/docs/). Read my [LinedIn article](https://www.linkedin.com/pulse/eclipse-gym-reservoir-simulation-environment-ruslan-miftakhov/) for more information. 

![EclGym](/figures/EclGymPoster.png)

## Installation

The EclGym depends on [gym toolkit](https://gym.openai.com/docs/) for portability and [libecl](https://github.com/equinor/libecl) for reading eclipse binary files. If there is a need, then libecl can later be easily swapped with other python based eclipse file readers. I developed the code and did testing only on Windows operating system with Python 3.5+ installed. Later I will probably docker everything.

Install dependencies using pip:
```python
git clone https://github.com/RMiftakhov/EclGym.git
cd EclGym
pip install -r requirements.txt
```

Installation of libecl is a bit longer, but it's quite manageable. Since the libecl library is written in C/C++, you need to have C++ compiler. We need to build the library with python wrappers by adding the option -DENABLE_PYTHON=ON when running cmake.

```python
git clone https://github.com/Equinor/libecl
cd libecl
pip install -r requirements.txt
```

![cmake](/figures/Cmake.png)

After building the libecl, we need to show your Python interpreter where to find the library by modifying your system's environment variables:

```python
PATH = /path/to/install/lib/python3.7/site-packages
LD_LIBRARY_PATH = /path/to/install/lib/Release
```

Test the installation by writing a short python script that examines FOPT data:

```python
from ecl.summary import EclSum
import sys

summary = EclSum(/path/to/eclipse/generated/data/file)
print(summary.numpy_vector("FOPT"))
```

## Environments

The EclGym repository contains python files, the ones that require some attention are ecl_env.py and egg_model_test.py.

To test the installation and get yourself familiar with the package, I provide eclipse datafile for EggModel benchmark in the example directory. Please, take a look at EGG_MODEL_ECL.DATA because it uses all the necessary keywords that the package expects from a simulation datafile to function correctly. In essence, the library requires SAVE keywords in RUNSPEC section and RPTSOL with RESTART=1 in SOLUTION section. To run the example, you need to specify the path to a simulator executable in ecl_env.py:

```python
# User sys variables
# Path to a simulator exe file
ECL_EXECUTION_PATH = "path/to/eclipse.exe"
```

Then you can run egg_model_test.py. The simulation process gets started by calling reset(), which returns an initial state (observation) and evolves by calling step() function. The process generates new timesteps for a defined number of steps or while the done flag is false:

```python
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
        ## Advances the simulation with random values 
        observation, reward, done, observation_full = \
            env.step(actions_inje + actions_prod)  
        bar.update()
        if done.any():
            print("Episode finished after {} timesteps".format(t+1))
            break
env.close()
```

![Observation](figures/ObsFull.gif)

# Observations

The environment's step() function returns four value for each agent (well):

* Observation (object): an environment-specific object representing an agent's observation of the environment. For example, volumetric variables like pressure or water saturation. Agent's region of interest (partial observation), as well as a selection of volumetric variables to track, should be specified in ecl_env.py file.
  
```python
# State information related variables
## Half window from the well's position in x direction
OBS_HALF_X = 3
## Half window from the well's position in y direction
OBS_HALF_Y = 3
## Number of cell from first gridblock
OBS_NZ = 3
## State information variables
OBS_RETURN = ["PRESSURE", "SWAT"]
```

* Reward (float): an amount of reward an agent gained by the previous action. The reward scale can vary between different reward functions, but an agent's goal is always to increase the total reward. I implemented a trivial reward system that is based on field oil recovery; the reward function can be modified in ecl_env.py to fit your purpose.

```python
def __reward_function(self, summary_file, well_name): 
    return summary_file.numpy_vector("FOPR")[-1]
```

* Done (boolean): determines if it is time to reset the environment. A true value indicates that some conditions of termination are met. As a condition could be the end of the simulation or any other physical restriction. You can also define your custom done function. The implemented condition only checks if all agents can produce/inject.

```python
def __done_function(self, summary_file, well_name, is_producer):if is_producer is True:
       value = summary_file.numpy_vector("WOPR:"+ well_name.replace("'", ""))
    else:
       value = summary_file.numpy_vector("WWIR:"+ well_name.replace("'", ""))
    return bool(value[-1] == 0.0)
```

* Observation_full (object): an environment-specific object representing full field observation of the environment without leaving out any data. The only difference with the previously discussed "observation" value that "observation_full" is not agent specific.

## Spaces

The environment comes with an action_space and an observation_space, and they provide shape information for actions and observations:

```python
import gym
import ecl_gym
env = gym.make('ecl-v0')
print(env.action_space)
#> (12,4)
print(env.observation_space)
#> (12,2,147)
```