from gymnasium.envs.registration import register
#from ofc_gym import OfcEnv

register(
    id='ofc-v0',
    entry_point='ofcenv.gym.ofc_gym:OfcEnv'
)