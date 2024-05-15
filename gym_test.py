import gymnasium as gym
from gymnasium.envs.registration import register

register(
    id='ofc-v0',
    entry_point='gym.ofc_gym:OfcEnv',
)

#print(gym.envs.registry)

env = gym.make('ofc-v0')
# observation, info = env.reset()
# print(observation)
# env.render()


from stable_baselines3.common.env_checker import check_env
# If the environment don't follow the interface, an error will be thrown
# check_env(env, warn=True)

from stable_baselines3 import PPO
model = PPO("MlpPolicy", env, verbose=2)
model.learn(total_timesteps=50_000, progress_bar=False)

vec_env = model.get_env()
obs = vec_env.reset()
for i in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = vec_env.step(action)
    vec_env.render()
    # VecEnv resets automatically
    # if done:
    #   obs = env.reset()

env.close()



# sp = gym.spaces.MultiDiscrete([[[14, 14, 14, 1, 1], [14, 14, 14, 14, 14], [14, 14, 14, 14, 14], [14, 14, 14, 14, 14], [14, 14, 14, 14, 14]], [[5, 5, 5, 1, 1], [5, 5, 5, 5, 5], [5, 5, 5, 5, 5], [5, 5, 5, 5, 5], [5, 5, 5, 5, 5]]])
# for _ in range(1):
#     print(sp.sample())