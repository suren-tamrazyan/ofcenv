import gymnasium as gym
import numpy as np
import torch
from gymnasium.envs.registration import register
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import get_linear_fn
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv # Импортируем VecEnv
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, StopTrainingOnRewardThreshold
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib import MaskablePPO
import os
from typing import Optional, Dict # Добавим типизацию

# Импортируем новую среду и компоненты архитектуры
from ofc_gym_v2 import OfcEnvV2
from ofc_neural_network_architecture import OFCFeatureExtractor

# --- Регистрация среды (если еще не сделано) ---
ENV_ID = 'ofc-v2'
try:
    gym.spec(ENV_ID)
except gym.error.NameNotFound:
     print(f"Registering {ENV_ID} environment.")
     register(
         id=ENV_ID,
         entry_point='ofc_gym_v2:OfcEnvV2',
     )
else:
     print(f"{ENV_ID} environment already registered.")


# --- Кастомный Feature Extractor для SB3 ---
class SB3OFCFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict, features_dim_override: Optional[int] = None):
        if 'game_state' not in observation_space.spaces:
            raise ValueError("Observation space must contain 'game_state'")
        game_state_dim = observation_space['game_state'].shape[0]

        # --- СНАЧАЛА вычисляем размерность, которая нужна для super().__init__ ---
        # Создаем временный экстрактор ТОЛЬКО для вычисления размерности
        # Это не оптимально, но необходимо, чтобы знать features_dim для super()
        temp_feature_extractor = OFCFeatureExtractor(game_state_dim=game_state_dim)
        calculated_features_dim = temp_feature_extractor.feature_dim
        del temp_feature_extractor # Удаляем временный объект

        actual_features_dim = features_dim_override if features_dim_override is not None else calculated_features_dim

        # --- ТЕПЕРЬ вызываем super().__init__ ---
        super().__init__(observation_space, features_dim=actual_features_dim)
        print(f"SB3 Feature Extractor Wrapper Initialized. Output dim: {self.features_dim}")

        # --- ПОСЛЕ super() создаем и присваиваем основной экстрактор ---
        self.ofc_feature_extractor = OFCFeatureExtractor(game_state_dim=game_state_dim)

        # Проверка согласованности размерностей (на всякий случай)
        if self.features_dim != self.ofc_feature_extractor.feature_dim:
             print(f"Warning: Initialized features_dim ({self.features_dim}) differs from internal extractor dim ({self.ofc_feature_extractor.feature_dim}). This might happen if features_dim_override was used.")
        if features_dim_override is not None and features_dim_override != calculated_features_dim:
             print(f"Warning: features_dim_override ({features_dim_override}) does not match calculated feature dim ({calculated_features_dim}).")

    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        # Убираем action_mask перед передачей в экстрактор, если он там есть
        extractor_input = {k: v for k, v in observations.items() if k != 'action_mask'}
        if not extractor_input:
             # Если observations пуст после удаления маски, возможно, это проблема
             # Попытаемся вернуть нулевой тензор нужной размерности?
             batch_size = list(observations.values())[0].shape[0] # Получаем batch_size из исходного словаря
             device = list(observations.values())[0].device
             print(f"Warning: Observations dict became empty in forward. Returning zeros tensor of shape ({batch_size}, {self.features_dim})")
             return torch.zeros((batch_size, self.features_dim), device=device)
             # raise ValueError("Observations dict became empty after removing 'action_mask'.")

        return self.ofc_feature_extractor(extractor_input)

# --- Функция для ActionMasker ---
# Эта функция будет вызываться ActionMasker'ом для каждой среды в VecEnv
# Он передаст одну среду (не векторную) как аргумент
def mask_fn(env: gym.Env) -> np.ndarray:
    return env.unwrapped.get_action_mask()

# --- Настройка среды ---
print("--- Environment Setup ---")
# Создаем базовую среду
# НЕ оборачиваем в ActionMasker здесь
env = gym.make(ENV_ID)
# Сначала оборачиваем в ActionMasker, потом в DummyVecEnv
env = ActionMasker(env, mask_fn)
vec_env = DummyVecEnv([lambda: env])

# тестирование перед обучением
obs = env.unwrapped.reset()

print("Observation Space (after wrappers):", vec_env.observation_space)
# Примечание: ActionMasker может удалить 'action_mask' из observation_space, т.к. он обрабатывает ее сам
if 'action_mask' in vec_env.observation_space.spaces: # type: ignore
     print("Warning: 'action_mask' key still present in observation space after ActionMasker.")
else:
     print("'action_mask' key correctly removed by ActionMasker.")

print("Action Space:", vec_env.action_space)
print("--------------------------")


# --- Настройка модели MaskablePPO ---
net_arch_config = dict(pi=[512, 256], vf=[512, 256])

policy_kwargs = dict(
    features_extractor_class=SB3OFCFeaturesExtractor,
    net_arch=net_arch_config
)

# Папки для логов и моделей
log_dir = "./ofc_logs_v2_tune1/"
model_save_path = os.path.join(log_dir, "ppo_ofc_v2_model")
tensorboard_log_path = os.path.join(log_dir, "tb_logs/")
eval_log_path = os.path.join(log_dir, "eval_logs/")
os.makedirs(log_dir, exist_ok=True)
os.makedirs(tensorboard_log_path, exist_ok=True)
os.makedirs(eval_log_path, exist_ok=True)

learning_rate = get_linear_fn(
    start=3e-4,
    end=1e-5,
    end_fraction=0.9
)
# Настройки PPO
ppo_params = dict(
    policy="MultiInputPolicy", # Используем строку "MultiInputPolicy"
    env=vec_env, # Передаем УЖЕ обернутую среду
    policy_kwargs=policy_kwargs,
    verbose=1,
    tensorboard_log=tensorboard_log_path,
    learning_rate=learning_rate,
    n_steps=2048,
    batch_size=128,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01,
    vf_coef=1.0,
    max_grad_norm=0.5,
    seed=42
)

model = MaskablePPO(**ppo_params)

print("--- Model Setup ---")
print("Policy:", model.policy)
# Проверим, удалился ли ключ маски из observation_space, видимого политикой
if 'action_mask' in model.observation_space.spaces: # type: ignore
    print("ERROR: 'action_mask' still in model's observation space!")
print("-------------------")


# --- Коллбэк для проверки легальности действий ---
class ActionValidatorCallback(BaseCallback):
    def _on_step(self):
        if self.n_calls % 1000 == 0: # Проверять еще реже
            try:
                # Попытка получить текущую маску для информации
                base_env = self.training_env.envs[0]
                while hasattr(base_env, "env") and not isinstance(base_env, OfcEnvV2):
                     if isinstance(base_env, ActionMasker): base_env = base_env.env
                     elif hasattr(base_env, "unwrapped"): base_env = base_env.unwrapped
                     else: break

                if isinstance(base_env, OfcEnvV2):
                     current_mask = mask_fn(base_env)
                     action = self.locals["actions"][0]
                     phase = base_env.current_turn_phase
                     active_idx = base_env.active_card_idx
                     # Просто выводим информацию без проверки легальности
                     print(f"--- Callback Info (Step {self.n_calls}) ---")
                     print(f"Action chosen (for prev state): {action}")
                     print(f"Current Mask (for next state): {np.where(current_mask)[0]}")
                     print(f"Current Env State: Phase={phase}, ActiveCardIdx={active_idx}")
                     print(f"---------------------------------------")

            except Exception as e:
                 # print(f"Warning: Could not get info for validation callback: {e}")
                 pass # Просто пропускаем при ошибке
        return True

# --- Коллбэк для оценки и сохранения лучшей модели ---
# Создаем отдельную среду для оценки (тоже оборачиваем)
eval_env = gym.make(ENV_ID)
eval_env = ActionMasker(eval_env, mask_fn) # <--- Применяем ActionMasker и здесь
eval_vec_env = DummyVecEnv([lambda: eval_env])

eval_callback = MaskableEvalCallback(
    eval_vec_env, # Используем обернутую векторную среду
    best_model_save_path=eval_log_path,
    log_path=eval_log_path,
    eval_freq=max(10000 // vec_env.num_envs, 1),
    n_eval_episodes=20,
    deterministic=True,
    render=False,
)

# --- Обучение ---
TOTAL_TIMESTEPS = 1_000_000

print(f"Starting training for {TOTAL_TIMESTEPS} timesteps...")
try:
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=[ActionValidatorCallback(verbose=0), eval_callback],
        tb_log_name="MaskablePPO_OFC_v2",
        progress_bar=True,
        reset_num_timesteps=False
    )
    model.save(model_save_path + "_final")
    print("Training finished. Final model saved.")

except KeyboardInterrupt:
    print("Training interrupted by user.")
    model.save(model_save_path + "_interrupted")
    print("Model saved.")
except Exception as e:
    print(f"An error occurred during training: {e}")
    import traceback
    traceback.print_exc()
    model.save(model_save_path + "_error")
    print("Model saved after error.")


# --- Оценка лучшей модели ---
print("\nLoading best model for evaluation...")
try:
    best_model_path = os.path.join(log_dir, "ppo_ofc_v2_model_final")
    # best_model_path = os.path.join(eval_log_path, "best_model")
    # Убедимся, что файл существует
    if not os.path.exists(best_model_path + ".zip"):
         print(f"Best model not found at {best_model_path}. Evaluating final model instead.")
         best_model_path = model_save_path + "_final" # Или прерванную/ошибочную
         if not os.path.exists(best_model_path + ".zip"):
             print("No model found to evaluate.")
             raise FileNotFoundError("No model saved.")

    # Пересоздаем среду для оценки с нуля, чтобы не было конфликтов состояния
    eval_env_final_instance = gym.make(ENV_ID)
    eval_env_final_instance = Monitor(eval_env_final_instance) # <--- Добавить Monitor
    eval_env_final_instance = ActionMasker(eval_env_final_instance, mask_fn)
    eval_vec_env_final = DummyVecEnv([lambda: eval_env_final_instance]) # <--- Передать функцию

    # Загружаем модель
    model_to_evaluate = MaskablePPO.load(best_model_path, env=eval_vec_env_final)
    print(f"Loaded model from {best_model_path}")

    print("Evaluating model...")
    total_reward = 0
    num_episodes = 50

    obs = eval_vec_env_final.reset()
    episodes_completed = 0
    while episodes_completed < num_episodes:
        legal_act = obs['action_mask']
        action, _states = model_to_evaluate.predict(obs, deterministic=True, action_masks=legal_act)
        obs, reward, done, info = eval_vec_env_final.step(action)
        if done[0]:
            episode_reward = info[0].get('episode', {}).get('r', 0) # Получаем награду из info, если есть Monitor
            if episode_reward == 0 and 'raw_reward' in info[0]: # Если нет Monitor, берем последнюю
                episode_reward = info[0]['raw_reward']

            print(f"Eval Episode {episodes_completed + 1} finished. Reward: {episode_reward:.2f}")
            total_reward += episode_reward
            episodes_completed += 1
            # Сброс происходит автоматически в VecEnv

    print(f"\nAverage reward over {episodes_completed} eval episodes: {total_reward / episodes_completed:.2f}")

except FileNotFoundError as e:
    print(e)
except Exception as e:
    print(f"An error occurred during evaluation: {e}")
    import traceback
    traceback.print_exc()

# Закрываем среды
vec_env.close()
# eval_vec_env.close() # Закрывается внутри EvalCallback? Проверить.
if 'eval_vec_env_final' in locals(): eval_vec_env_final.close()
print("Environments closed.")