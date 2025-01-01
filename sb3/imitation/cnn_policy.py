import gymnasium as gym
import torch as th
from torch import nn
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class CNNPolicy(ActorCriticPolicy):
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule,
        *args,
        **kwargs,
    ):
        # Переопределяем features_extractor_class
        if "features_extractor_class" not in kwargs:
            kwargs["features_extractor_class"] = CNNExtractor
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            *args,
            **kwargs,
        )

class CNNExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Space, features_dim: int = 512):
        super().__init__(observation_space, features_dim)
        
        # Определяем размерности выходов для policy и value networks
        self.latent_dim_pi = 256  # Размерность выхода для policy head
        self.latent_dim_vf = 256  # Размерность выхода для value head
        
        n_input_channels = observation_space.shape[0]  # Должно быть 12
        
        # CNN для обработки карточного состояния (12, 4, 13)
        self.cnn = nn.Sequential(
            # nn.Conv2d(n_input_channels, 128, kernel_size=3, stride=1, padding=1),
            # nn.ReLU(),
            # nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            # nn.ReLU(),
            # nn.Flatten(),

            nn.Conv2d(n_input_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),

        )
        
        # Вычисляем размер выхода CNN
        with th.no_grad():
            n_flatten = self.cnn(th.zeros(1, n_input_channels, 4, 13)).shape[1]
        
        self.shared_net = nn.Sequential(
            nn.Linear(n_flatten, 512),
            nn.ReLU(),
            nn.Linear(512, features_dim),
            nn.ReLU(),
        )
        
        # Отдельные головы для policy и value
        self.policy_net = nn.Sequential(
            nn.Linear(features_dim, self.latent_dim_pi),
            nn.ReLU()
        )
        
        self.value_net = nn.Sequential(
            nn.Linear(features_dim, self.latent_dim_vf),
            nn.ReLU()
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        """
        Обрабатывает входные данные через CNN и возвращает features
        """
        # Убеждаемся, что входные данные имеют правильную форму
        if len(observations.shape) == 3:
            observations = observations.unsqueeze(0)
            
        features = self.cnn(observations)
        return self.shared_net(features)

    def forward_actor(self, features: th.Tensor) -> th.Tensor:
        return self.policy_net(features)

    def forward_critic(self, features: th.Tensor) -> th.Tensor:
        return self.value_net(features) 