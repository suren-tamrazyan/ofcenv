import os
import gymnasium as gym
import numpy as np
from gymnasium import register
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

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

# Папки для логов и моделей
log_dir = "./ofc_logs_v2/"
model_save_path = os.path.join(log_dir, "ppo_ofc_v2_model")
tensorboard_log_path = os.path.join(log_dir, "tb_logs/")
eval_log_path = os.path.join(log_dir, "eval_logs/")
os.makedirs(log_dir, exist_ok=True)
os.makedirs(tensorboard_log_path, exist_ok=True)
os.makedirs(eval_log_path, exist_ok=True)

# --- Оценка лучшей модели ---
def mask_fn(env: gym.Env) -> np.ndarray:
    return env.unwrapped.get_action_mask()

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
    eval_env_final_instance = gym.make(ENV_ID)#, render_mode="human"
    eval_env_final_instance = Monitor(eval_env_final_instance) # <--- Добавить Monitor
    eval_env_final_instance = ActionMasker(eval_env_final_instance, mask_fn)
    eval_vec_env_final = DummyVecEnv([lambda: eval_env_final_instance]) # <--- Передать функцию

    # Загружаем модель
    # model_to_evaluate = MaskablePPO.load(best_model_path, env=eval_vec_env_final)
    print(f"Loaded model from {best_model_path}")

    print("Evaluating model...")
    total_reward = 0
    num_episodes = 20
    rewards_list = []

    obs = eval_vec_env_final.reset()
    episodes_completed = 0
    while episodes_completed < num_episodes:
        legal_act = obs['action_mask']
        # action, _states = model_to_evaluate.predict(obs, deterministic=True, action_masks=legal_act)
        action = np.array([np.random.choice(np.where(legal_act == 1)[1])])
        obs, reward, done, info = eval_vec_env_final.step(action)
        if done[0]:
            episode_reward = info[0].get('episode', {}).get('r', 0) # Получаем награду из info, если есть Monitor
            if episode_reward == 0 and 'raw_reward' in info[0]: # Если нет Monitor, берем последнюю
                episode_reward = info[0]['raw_reward']

            print(f"Eval Episode {episodes_completed + 1} finished. Reward: {episode_reward:.2f}")
            total_reward += episode_reward
            episodes_completed += 1
            rewards_list.append(episode_reward)
            # if episode_reward > 1:
            #     eval_vec_env_final.render(mode="human")
            # Сброс происходит автоматически в VecEnv

    print(f"\nAverage reward over {episodes_completed} eval episodes: {total_reward / episodes_completed:.2f}")
    print(f"Reward std: {np.std(rewards_list):.2f}, Min: {np.min(rewards_list):.2f}, Max: {np.max(rewards_list):.2f}")

except FileNotFoundError as e:
    print(e)
except Exception as e:
    print(f"An error occurred during evaluation: {e}")
    import traceback
    traceback.print_exc()

# Закрываем среды
# eval_vec_env.close() # Закрывается внутри EvalCallback? Проверить.
if 'eval_vec_env_final' in locals(): eval_vec_env_final.close()
print("Environments closed.")