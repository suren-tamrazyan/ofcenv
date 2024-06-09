import gymnasium as gym
from gymnasium.envs.registration import register

register(
    id='ofc-v0',
    entry_point='gym.ofc_gym:OfcEnv',
)


env = gym.make('ofc-v0')


import torch as th
import torch.nn as nn
from gymnasium import spaces

from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class CustomCNN(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: spaces.Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 256, kernel_size=5, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 256, kernel_size=5, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 256, kernel_size=5, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 256, kernel_size=5, stride=2, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with th.no_grad():
            n_flatten = self.cnn(
                th.as_tensor(observation_space.sample()[None]).float()
            ).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(self.cnn(observations))

policy_kwargs = dict(
    features_extractor_class=CustomCNN,
    features_extractor_kwargs=dict(features_dim=128),
)
model = PPO("CnnPolicy", env, policy_kwargs=policy_kwargs, verbose=1)
model.learn(1000)

vec_env = model.get_env()
obs = vec_env.reset()
vec_env.render()
for i in range(10):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = vec_env.step(action)
    vec_env.render()
    # VecEnv resets automatically
    # if done:
    #   obs = env.reset()

env.close()

