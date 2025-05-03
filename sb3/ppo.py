import gymnasium as gym
from gymnasium.envs.registration import register

register(
    id='ofc-v0',
    entry_point='gym_env.ofc_gym:OfcEnv',
)


env = gym.make('ofc-v0')


from stable_baselines3 import PPO
policy_kwargs = dict(net_arch=[256, 256, 128, 256, 256])
model = PPO("MlpPolicy", env, policy_kwargs=policy_kwargs, verbose=1)
model.learn(total_timesteps=10_000, progress_bar=False)

vec_env = model.get_env()
obs = vec_env.reset()
vec_env.render()
for i in range(10):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = vec_env.step(action)
    if (done):
        # print(info[0]['terminal_observation'])
        print('Reward: ', reward)
    vec_env.render()
    # VecEnv resets automatically
    # if done:
    #   obs = env.reset()

env.close()

