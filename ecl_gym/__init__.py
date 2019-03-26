## @package ecl_gym
#  This is a reservoir simulation
#  integration into gym toolkit

from gym.envs.registration import register
register(
    id='ecl-v0',
    entry_point='ecl_gym.envs:EclEnv',
)
