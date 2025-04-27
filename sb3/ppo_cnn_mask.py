import gymnasium as gym
import numpy as np
from gymnasium.envs.registration import register
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy, MaskableActorCriticCnnPolicy
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib import MaskablePPO

from ofc_encoder import is_legal_action

register(
    id='ofc-v0',
    entry_point='gym.ofc_gym:OfcEnv',
)


# env = gym.make('ofc-v0', observe_summary=False)
def mask_fn(env) -> np.ndarray:
    # Получаем маску из текущего окружения (уже обернутого в ActionMasker)
    return env.unwrapped.get_action_mask()

env = gym.make("ofc-v0", observe_summary=False)
# Сначала оборачиваем в ActionMasker, потом в DummyVecEnv
env = ActionMasker(env, mask_fn)
env = DummyVecEnv([lambda: env])


# env = gym.make("ofc-v0", observe_summary=False)
# env = DummyVecEnv([lambda: env])  # Сначала векторизуем
# env = ActionMasker(env, mask_fn)  # Затем применяем маскировку


# тестирование перед обучением
obs = env.unwrapped.reset()
# Получаем маску через метод env_method для векторизованного окружения
current_mask = env.env_method("get_action_mask")[0]  # [0] - первый (и единственный) env instance
print("Valid actions:", np.where(current_mask)[0])
# exit(0)

import torch as th
import torch.nn as nn
from gymnasium import spaces

from stable_baselines3 import PPO, DQN, A2C
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


# class CustomCNN(BaseFeaturesExtractor):
#     """
#     :param observation_space: (gym.Space)
#     :param features_dim: (int) Number of features extracted.
#         This corresponds to the number of unit for the last layer.
#     """
#     def __init__(self, observation_space: spaces.Dict, features_dim: int = 128):
#         # 1. Используем spaces.Dict, если наблюдение содержит несколько компонентов
#         super().__init__(observation_space, features_dim)
#         # 2. CNN для обработки карточной матрицы (board)
#         n_input_channels = observation_space['observation'].shape[0]
#         self.cnn = nn.Sequential(
#             nn.Conv2d(n_input_channels, 32, kernel_size=3, stride=1, padding=1),
#             nn.ReLU(),
#             nn.BatchNorm2d(32),
#             nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
#             nn.ReLU(),
#             nn.BatchNorm2d(64),
#             nn.MaxPool2d(kernel_size=2, stride=2),
#             nn.Flatten(),
#         )
#         mask_out_dim = 32;
#         # 3. Ветка для обработки маски действий
#         self.mask_processor = nn.Sequential(
#             nn.Linear(259, 64),
#             nn.ReLU(),
#             nn.Linear(64, mask_out_dim)
#         )
#         # 4. Вычисление размерности после CNN
#         with th.no_grad():
#             sample_input = th.as_tensor(observation_space['observation'].sample()[None]).float()
#             cnn_output_dim = self.cnn(sample_input).shape[1]
#
#         # 5. Финальные полносвязные слои
#         self.linear = nn.Sequential(
#             nn.Linear(cnn_output_dim + mask_out_dim, features_dim),  # Объединяем CNN и маску
#             # nn.Linear(n_flatten, features_dim),
#             nn.ReLU(),
#             nn.Dropout(0.5),
#             nn.Linear(features_dim, features_dim),
#             nn.ReLU(),
#             nn.Dropout(0.5)
#         )
#
#     def forward(self, observations: dict) -> th.Tensor:
#         # 6. Разделяем наблюдение на компоненты
#         board = observations['observation']
#         action_mask = observations['action_mask']
#         # 7. Обрабатываем карточную матрицу через CNN
#         board_features = self.cnn(board)
#         # 8. Обрабатываем маску действий
#         mask_features = self.mask_processor(action_mask.float())
#         # 9. Объединяем признаки
#         combined = th.cat([board_features, mask_features], dim=1)
#         # 10. Финальное преобразование
#         return self.linear(combined)

class CustomCNN(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: spaces.Dict, features_dim: int = 128):
        super().__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        n_input_channels = observation_space['observation'].shape[0]
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
                th.as_tensor(observation_space['observation'].sample()[None]).float()
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

    def forward(self, observations: dict) -> th.Tensor:
        return self.linear(self.cnn(observations['observation']))


policy_kwargs = dict(
    features_extractor_class=CustomCNN,
    features_extractor_kwargs=dict(features_dim=128),
)
model = MaskablePPO(
    MaskableActorCriticCnnPolicy,
    env,
    policy_kwargs=policy_kwargs,
    verbose=1,
    tensorboard_log="./tb_logs/"
)
# model = PPO("CnnPolicy", env, policy_kwargs=policy_kwargs, verbose=1, tensorboard_log="./tb_logs/")
# model = PPO.load("./expert_hh/ppo_ofc_pineapple", env=env)

# model = DQN("CnnPolicy", env, policy_kwargs=policy_kwargs, verbose=1, tensorboard_log="./tb_logs/")
# model = A2C("CnnPolicy", env, policy_kwargs=policy_kwargs, verbose=1, tensorboard_log="./tb_logs/")

# class ActionValidatorCallback(BaseCallback):
#     def __init__(self, verbose=0):
#         super().__init__(verbose)
#     def _on_step(self) -> bool:
#         for env in self.model.env.envs:
#             if env.unwrapped.game.current_player().to_play:
#                 action = self.locals["actions"][0]
#                 if not is_legal_action(env.unwrapped.game.hero_player(), action):
#                     print(f"Illegal action {action} detected!")
#         return True
class ActionValidatorCallback(BaseCallback):
    def _on_step(self):
        for env_idx in range(self.model.env.num_envs):
            current_mask = self.model.env.env_method("get_action_mask", indices=env_idx)[0]
            action = self.locals["actions"][env_idx]
            if not current_mask[action]:
                print(f"Illegal action {action}. Valid: {np.where(current_mask)[0]}")
        return True

model.learn(
    100_000,
    callback=ActionValidatorCallback(),
    tb_log_name="PPO_Masked",
)

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

