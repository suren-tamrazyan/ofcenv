import gymnasium as gym
import numpy as np
from gymnasium.envs.registration import register
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from sb3_contrib.common.maskable.policies import MaskableActorCriticCnnPolicy
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib import MaskablePPO

import torch as th
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from ofc_encoder import is_legal_action

register(
    id='ofc-v0',
    entry_point='gym.ofc_gym:OfcEnv',
)

# Define the mask function
def mask_fn(env) -> np.ndarray:
    return env.unwrapped.get_action_mask()

# Create environment with proper wrapping order
env = gym.make("ofc-v0", observe_summary=False)
env = ActionMasker(env, mask_fn)  # First apply masking
env = DummyVecEnv([lambda: env])   # Then vectorize

# Test masking before training
obs = env.reset()
current_mask = env.env_method("get_action_mask")[0]
print(f"Number of valid actions: {np.sum(current_mask)}")
print(f"Valid actions: {np.where(current_mask)[0]}")

# Define custom CNN feature extractor
class CustomCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Dict, features_dim: int = 128):
        super().__init__(observation_space, features_dim)
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

# Define policy kwargs with feature extractor
policy_kwargs = dict(
    features_extractor_class=CustomCNN,
    features_extractor_kwargs=dict(features_dim=128),
)

# Create MaskablePPO model
model = MaskablePPO(
    MaskableActorCriticCnnPolicy,
    env,
    policy_kwargs=policy_kwargs,
    verbose=1,
    tensorboard_log="./tb_logs/"
)

# Define action validator callback
class ActionValidatorCallback(BaseCallback):
    def _on_step(self):
        for env_idx in range(self.model.env.num_envs):
            current_mask = self.model.env.env_method("get_action_mask", indices=env_idx)[0]
            action = self.locals["actions"][env_idx]
            print(f"Action: {action}, Is valid according to mask: {current_mask[action]}")
            # Дополнительная проверка через is_legal_action
            if not is_legal_action(self.model.env.envs[env_idx].unwrapped.game.hero_player(), action):
                print(f"Illegal action {action} detected by is_legal_action!")
                valid_actions = np.where(current_mask)[0]
                print(f"Valid actions according to mask: {valid_actions}")
        return True

# Train with masking enabled
model.learn(
    total_timesteps=100_000,
    # callback=ActionValidatorCallback(),
    tb_log_name="PPO_Masked",
    use_masking=True  # This is crucial!
)

# Test trained model
vec_env = model.get_env()
obs = vec_env.reset()
for i in range(100):
    # Get the action mask for prediction
    action_masks = vec_env.env_method("get_action_mask")
    # Use the mask for prediction
    action, _states = model.predict(obs, action_masks=action_masks, deterministic=True)
    obs, reward, done, info = vec_env.step(action)
    vec_env.render()
    if done:
        print('Reward: ', reward)
        obs = vec_env.reset()

env.close()
