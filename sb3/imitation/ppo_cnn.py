import json

import numpy as np
import gymnasium as gym
from gymnasium.envs.registration import register
from imitation.algorithms.bc import BC
from imitation.data import types, rollout
from imitation.algorithms.adversarial.gail import GAIL
from imitation.rewards.reward_nets import BasicShapedRewardNet
from stable_baselines3.common.vec_env import DummyVecEnv

from gym_env.ofc_gym import OfcEnv
from sb3.imitation.expert_data_loader import load_expert_trajectories

# register(
#     id='ofc-v0',
#     entry_point='gym_env.ofc_gym:OfcEnv',
# )
#
#
# env = gym_env.make('ofc-v0', observe_summary=False)

env = DummyVecEnv([lambda: OfcEnv()])

import torch as th
import torch.nn as nn
from gymnasium import spaces

from stable_baselines3 import PPO, DQN, A2C
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class CustomCNN(BaseFeaturesExtractor):
    """
    :param observation_space: (gym_env.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: spaces.Box, features_dim: int = 128):
        super().__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        n_input_channels = observation_space.shape[0]
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
                th.as_tensor(observation_space.sample()[None]).float()
            ).shape[1]

        # self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())
        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(features_dim, features_dim),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(self.cnn(observations))




def load_game_history(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

# Шаг 2: Преобразование данных в формат, подходящий для imitation
def process_data(data):
    observations = []
    actions = []
    rewards = []
    next_observations = []
    dones = []
    infos = []
    for episode in data['episodes']:
        episode_obs = []
        episode_actions = []
        episode_rewards = []
        episode_infos = []
        for obs in episode['observations']:
            episode_obs.append(np.array(obs['state'], dtype=np.int8))
            episode_actions.append(obs['action'])
            episode_rewards.append(obs['reward'])
            episode_infos.append({})  # Пустой словарь, если нет дополнительной информации
        observations.extend(episode_obs)
        actions.extend(episode_actions)
        rewards.extend(episode_rewards)
        next_observations.extend(episode_obs[1:] + [episode_obs[-1]])  # Последнее next_obs такое же, как последнее obs
        dones.extend([False] * (len(episode_obs) - 1) + [True])
        infos.extend(episode_infos)
    return np.array(observations), np.array(actions), np.array(rewards), np.array(next_observations), np.array(dones), infos

file_path = './expert_hh/bit.json'
# data = load_game_history(file_path)
# observations, actions, rewards, next_observations, dones, infos = process_data(data)
#
# transitions = types.Transitions(
#     obs=observations,
#     acts=actions,
#     next_obs=next_observations,
#     dones=dones,
#     infos=infos
# )
# Загружаем экспертные данные
expert_trajectories = load_expert_trajectories(file_path)
# Преобразуем траектории в transitions
transitions = rollout.flatten_trajectories(expert_trajectories)

# Создание сети вознаграждений
reward_net = BasicShapedRewardNet(env.observation_space, env.action_space)

policy_kwargs = dict(
    features_extractor_class=CustomCNN,
    features_extractor_kwargs=dict(features_dim=128),
)

# model = DQN("CnnPolicy", env, policy_kwargs=policy_kwargs, verbose=1, tensorboard_log="./tb_logs/")
# model = A2C("CnnPolicy", env, policy_kwargs=policy_kwargs, verbose=1, tensorboard_log="./tb_logs/")
# model.learn(1_000, tb_log_name="PPO CNN 1")

# model = PPO("CnnPolicy", env, policy_kwargs=policy_kwargs, verbose=1)
# gail_trainer = GAIL(
#     venv=env,
#     demonstrations=transitions,
#     gen_algo=model,
#     demo_batch_size=32,
#     reward_net=reward_net,
#     allow_variable_horizon=True
# )
# gail_trainer.train(total_timesteps=10000)
# gail_trainer.gen_algo.save("./expert_hh/ppo_ofc_pineapple")


model = A2C("CnnPolicy", env, policy_kwargs=policy_kwargs, verbose=1)
rng = np.random.default_rng(0)
bc_trainer = BC(
    observation_space=env.observation_space,
    action_space=env.action_space,
    demonstrations=transitions,
    policy=model,
    batch_size=32,
    rng=rng
 )
bc_trainer.train(n_epochs=50)
bc_trainer.policy.save("./expert_hh/model_saves/a2c_ofc_pineapple_bc")


vec_env = model.get_env()
obs = vec_env.reset()
vec_env.render()
for i in range(100):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = vec_env.step(action)
    if done:
        print('Reward: ', reward)
    vec_env.render()

env.close()

