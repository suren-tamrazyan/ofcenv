import gymnasium as gym
from gymnasium.envs.registration import register
import numpy as np

register(
    id='ofc-v0',
    entry_point='gym.ofc_gym:OfcEnv',
)


env = gym.make('ofc-v0', observe_summary=True)


import torch as th
import torch.nn as nn
from gymnasium import spaces

from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class CustomCombinedExtractor(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: spaces.Box, features_dim: int = 128):
        super().__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        # n_input_channels = observation_space.shape[0]
        board_shape = observation_space.spaces['board'].shape
        n_input_channels = board_shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with th.no_grad():
            n_flatten = self.cnn(
                # th.as_tensor(observation_space.sample()[None]).float()
                th.as_tensor(np.zeros(board_shape)).float().unsqueeze(0)
            ).shape[1]

        board_features_dim = 128
        self.linear_cnn = nn.Sequential(
            nn.Linear(n_flatten, board_features_dim),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

        summary_shape = observation_space.spaces['summary'].shape[0]
        summary_features_dim = 64
        self.linear_summary = nn.Sequential(
            nn.Linear(summary_shape, summary_features_dim),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

        self.final_layers = nn.Sequential(
            nn.Linear(board_features_dim + summary_features_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, features_dim),
            nn.ReLU()
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        board = observations['board']
        summary = observations['summary']
        board_features = self.linear_cnn(self.cnn(board))
        summary_features = self.linear_summary(summary)
        combined_features = th.cat((board_features, summary_features), dim=1)
        return self.final_layers(combined_features)

policy_kwargs = dict(
    features_extractor_class=CustomCombinedExtractor,
    features_extractor_kwargs=dict(features_dim=128),
)
model = PPO("MultiInputPolicy", env, policy_kwargs=policy_kwargs, verbose=1, tensorboard_log="./tb_logs/")
model.learn(100_000)

vec_env = model.get_env()
obs = vec_env.reset()
vec_env.render()
for i in range(100):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = vec_env.step(action)
    if done:
        # print(info[0]['terminal_observation'])
        print('Reward: ', reward)
    vec_env.render()
    # VecEnv resets automatically
    # if done:
    #   obs = env.reset()

env.close()

