import os
import gymnasium as gym
import numpy as np
import torch
from gymnasium import register
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import obs_as_tensor
from stable_baselines3.common.vec_env import DummyVecEnv
from torch.backends.cudnn import deterministic

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


# Загружаем модель PPO (обученную без маски)
ppo_model_path = "./ofc_logs_v2_nomask_mln/ppo_ofc_v2_model_final" # Путь к модели PPO
if not os.path.exists(ppo_model_path + ".zip"):
    print(f"PPO model not found at {ppo_model_path}")
else:
    print(f"Loading PPO model from {ppo_model_path}")
    # Создаем среду для оценки PPO модели
    eval_env_ppo = gym.make(ENV_ID)
    eval_env_ppo = Monitor(eval_env_ppo)
    # НЕ оборачиваем в ActionMasker здесь, так как модель - обычный PPO
    eval_vec_env_ppo = DummyVecEnv([lambda: eval_env_ppo])

    # Загружаем как обычный PPO
    from stable_baselines3 import PPO # Импортируем обычный PPO
    model_to_evaluate_ppo = PPO.load(ppo_model_path, env=eval_vec_env_ppo) # Передаем policy_kwargs, если экстрактор кастомный

    print("Evaluating PPO model WITH MASKING during predict...")
    total_reward_ppo = 0
    num_episodes = 50
    obs = eval_vec_env_ppo.reset()
    episodes_completed = 0
    while episodes_completed < num_episodes:
        # --- ПОЛУЧАЕМ МАСКУ ВРУЧНУЮ ---
        # Нужно получить маску из базовой среды
        # Используем env_method или получаем из obs, если ключ есть
        action_masks = eval_vec_env_ppo.env_method("get_action_mask")[0] # Пример

        # Примерная идея (может потребовать адаптации под MultiInputPolicy)
        obs_tensor = obs_as_tensor(obs, model_to_evaluate_ppo.device)
        # Нужно вызвать метод, возвращающий логиты или распределение
        # Например, через forward политики или evaluate_actions
        # Это может быть не так просто для загруженной модели PPO
        dist = model_to_evaluate_ppo.policy.get_distribution(obs_tensor) # Примерный вызов
        action_logits = dist.distribution.logits # Примерный доступ к логитам

        # action_logits - тензор логитов из шага 1
        # action_masks - numpy маска из шага 2
        action_masks_tensor = torch.tensor(action_masks, device=action_logits.device).bool()
        masked_logits = torch.where(action_masks_tensor, action_logits,
                                    torch.tensor(-float('inf'), device=action_logits.device))

        from torch.distributions import Categorical

        masked_prob_dist = Categorical(logits=masked_logits)
        if deterministic:
            action = torch.argmax(masked_logits, dim=1)
        else:
            action = masked_prob_dist.sample()
        action = action.cpu().numpy()  # Преобразовать в numpy для env.step



        obs, reward, done, info = eval_vec_env_ppo.step(action)
        if done[0]:
            episode_reward = info[0].get('episode', {}).get('r', 0)
            print(f"Eval PPO (masked predict) Episode {episodes_completed + 1} finished. Reward: {episode_reward:.2f}")
            total_reward_ppo += episode_reward
            episodes_completed += 1

    print(f"\nAverage reward for PPO model (masked predict) over {episodes_completed} episodes: {total_reward_ppo / episodes_completed:.2f}")
    eval_vec_env_ppo.close()