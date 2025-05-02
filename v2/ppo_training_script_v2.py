from typing import Dict

import gymnasium as gym
import numpy as np
import torch
from gymnasium.envs.registration import register
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.callbacks import BaseCallback
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib import MaskablePPO

# Импортируем новую среду и архитектуру
from ofc_gym_v2 import OfcEnvV2
from ofc_neural_network_architecture import OFCFeatureExtractor, OFCPolicyNetwork

# Регистрируем новую среду
register(
    id='ofc-v2',
    entry_point='ofc_gym_v2:OfcEnvV2', # Указываем новый класс среды
)

# --- Кастомный Feature Extractor для SB3 ---
class SB3OFCFeaturesExtractor(BaseFeaturesExtractor):
    """
    Обертка над OFCFeatureExtractor для использования с SB3.
    Принимает словарь наблюдений Dict и передает его в OFCFeatureExtractor.
    """
    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int = 512): # features_dim - это выход экстрактора
        # Определяем game_state_dim из observation_space
        game_state_dim = observation_space['game_state'].shape[0]
        # Создаем наш основной экстрактор
        self.ofc_feature_extractor = OFCFeatureExtractor(game_state_dim=game_state_dim)
        # Сообщаем SB3 итоговую размерность признаков
        calculated_features_dim = self.ofc_feature_extractor.feature_dim
        super().__init__(observation_space, features_dim=calculated_features_dim)
        print(f"SB3 Feature Extractor Wrapper Initialized. Output dim: {calculated_features_dim}")

    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        # Просто передаем словарь наблюдений в наш экстрактор
        return self.ofc_feature_extractor(observations)

# --- Кастомная политика для MaskablePPO ---
# Используем стандартную MaskableActorCriticPolicy, но передаем ей наш экстрактор
# Actor/Critic сети будут созданы внутри стандартной политики поверх признаков экстрактора

# --- Настройка среды ---
def mask_fn(env: gym.Env) -> np.ndarray:
    """Получает маску из словаря наблюдений среды."""
    # В новой среде маска уже является частью наблюдения
    # Но ActionMasker ожидает функцию, которая вызывает метод среды
    # Сделаем так, чтобы среда возвращала маску отдельным методом, если нужно,
    # или просто извлечем ее из наблюдения здесь (менее чисто)
    # Вариант 1: Добавить метод в OfcEnvV2: get_current_action_mask()
    # return env.get_current_action_mask()
    # Вариант 2: Извлечь из последнего наблюдения (менее надежно)
    # current_obs = env.unwrapped._get_obs() # Доступ к приватному методу - плохая практика
    # return current_obs['action_mask']
    # Вариант 3 (предпочтительный с ActionMasker):
    # ActionMasker сам извлечет маску, если она есть в observation space под ключом 'action_mask'
    # Поэтому функция может быть фиктивной или не использоваться напрямую, если SB3 это поддерживает
    # Давайте попробуем без явной функции, полагаясь на ключ 'action_mask'
    pass # Оставляем пустым, ActionMasker должен найти ключ 'action_mask'


# Создаем окружение
env = gym.make("ofc-v2")

# Оборачиваем в ActionMasker. Он должен автоматически найти 'action_mask' в observations space
env = ActionMasker(env) # Убрали mask_fn
env.reset()

# Векторизуем (для PPO нужна векторная среда)
env = DummyVecEnv([lambda: env])

print("Observation Space:", env.observation_space)
print("Action Space:", env.action_space)

# --- Настройка модели ---
policy_kwargs = dict(
    features_extractor_class=SB3OFCFeaturesExtractor,
    # features_extractor_kwargs=dict(features_dim=?) # features_dim определяется внутри экстрактора
    # Укажем архитектуру сетей Actor/Critic после экстрактора
    # Размеры скрытых слоев для Actor (pi) и Critic (vf)
    net_arch=dict(pi=[512, 256], vf=[512, 256]) # Эти размеры из OFCPolicyNetwork
)

model = MaskablePPO(
    "MultiInputPolicy", # Используем эту политику для Dict observation space
    env,
    policy_kwargs=policy_kwargs,
    verbose=1,
    tensorboard_log="./tb_logs_v2/",
    learning_rate=3e-4, # Можно подбирать
    n_steps=2048,       # Стандартное значение для PPO
    batch_size=64,      # Стандартное значение
    n_epochs=10,        # Стандартное значение
    gamma=0.99,         # Фактор дисконтирования
    gae_lambda=0.95,    # Параметр GAE
    clip_range=0.2,     # Параметр клиппинга PPO
    ent_coef=0.0,       # Коэффициент энтропии (можно увеличить для exploration)
    vf_coef=0.5,        # Коэффициент Value Function loss
    max_grad_norm=0.5,  # Ограничение нормы градиента
    seed=42             # Для воспроизводимости
)

# --- Коллбэк для проверки легальности действий ---
class ActionValidatorCallback(BaseCallback):
    def _on_step(self):
        # Получаем маски для всех окружений в векторе
        # Используем env_method для вызова метода базовой среды
        try:
            # Попытка получить маску из наблюдения (предпочтительно)
            masks = self.training_env.get_attr("last_obs")[0]['action_mask']
            # Если get_attr не работает или last_obs не содержит маску, нужен другой способ
        except Exception:
             # Запасной вариант: вызвать метод среды, если он есть
             # masks = self.training_env.env_method("get_action_mask") # Раскомментировать, если добавили метод
             print("Warning: Could not reliably get action mask for validation.")
             return True # Пропускаем проверку

        actions = self.locals["actions"]
        for env_idx in range(self.model.env.num_envs):
            action = actions[env_idx]
            mask = masks # [env_idx] # Если masks - список масок
            if isinstance(mask, list): mask = mask[env_idx] # Если env_method вернул список

            if not mask[action]:
                print(f"Illegal action {action} detected in env {env_idx}! Valid: {np.where(mask)[0]}")
                # Можно добавить логирование или остановку обучения
        return True

# --- Обучение ---
print("Starting training...")
model.learn(
    total_timesteps=500_000, # Увеличьте для реального обучения
    callback=ActionValidatorCallback(),
    tb_log_name="MaskablePPO_OFC_v2",
    progress_bar=True
)

# --- Сохранение модели ---
model.save("ppo_ofc_v2_model")
print("Model saved.")

# --- Оценка (пример) ---
print("Evaluating model...")
vec_env = model.get_env()
obs = vec_env.reset()
total_reward = 0
num_episodes = 10

for episode in range(num_episodes):
    episode_reward = 0
    terminated = False
    truncated = False
    while not terminated and not truncated:
        # Получаем маску для предсказания (MaskablePPO делает это внутри)
        # action_masks = vec_env.env_method("get_action_mask") # Не нужно для predict
        action, _states = model.predict(obs, deterministic=True) # deterministic=True для оценки
        obs, reward, terminated, info = vec_env.step(action)
        episode_reward += reward[0] # Суммируем награду для первого (и единственного) env
        if terminated[0] or truncated[0]:
             print(f"Episode {episode + 1} finished. Reward: {episode_reward}")
             total_reward += episode_reward
             # obs = vec_env.reset() # VecEnv сбрасывается автоматически
             break # Выход из while

print(f"Average reward over {num_episodes} episodes: {total_reward / num_episodes}")

env.close()