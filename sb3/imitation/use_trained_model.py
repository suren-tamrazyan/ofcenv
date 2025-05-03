import numpy as np
from stable_baselines3.common.evaluation import evaluate_policy
from gymnasium.envs.registration import register
from gym_env.ofc_gym import OfcEnv
from cnn_policy import CNNPolicy
import torch as th

register(
    id='ofc-v0',
    entry_point='gym_env.ofc_gym:OfcEnv',
)

def load_and_use_model(model_path: str):
    # Создаем окружение
    env = OfcEnv()
    
    # Создаем новый экземпляр политики
    policy = CNNPolicy(
        observation_space=env.observation_space,
        action_space=env.action_space,
        lr_schedule=lambda _: 3e-4
    )
    
    # Загружаем веса модели
    policy.load_state_dict(th.load(model_path))
    policy.eval()  # Переключаем в режим оценки
    
    # Пример использования модели для одного эпизода
    obs, _ = env.reset()
    done = False
    total_reward = 0
    
    while not done:
        # Получаем действие от модели
        with th.no_grad():
            action, _states = policy.predict(obs, deterministic=True)
        
        # Делаем шаг в окружении
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
        
        # Опционально: визуализация
        env.render()
    
    print(f"Episode finished with reward: {total_reward}")
    return policy

def evaluate_trained_model(model_path: str, n_eval_episodes: int = 10):
    # Создаем окружение
    env = OfcEnv()
    
    # Создаем и загружаем модель
    policy = CNNPolicy(
        observation_space=env.observation_space,
        action_space=env.action_space,
        lr_schedule=lambda _: 3e-4
    )
    policy.load_state_dict(th.load(model_path))
    policy.eval()
    
    # Оцениваем модель
    mean_reward, std_reward = evaluate_policy(
        policy,
        env,
        n_eval_episodes=n_eval_episodes,
        deterministic=True
    )
    
    print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")
    return mean_reward, std_reward

if __name__ == "__main__":
    model_path = "./expert_hh/model_saves/ofc_bc_cnn_policy"
    
    # Загружаем и тестируем модель на одном эпизоде
    policy = load_and_use_model(model_path)
    
    # Оцениваем модель на нескольких эпизодах
    evaluate_trained_model(model_path)
    
    # Пример использования модели для получения действия
    env = OfcEnv()
    obs, _ = env.reset()
    
    # Получаем действие для конкретного состояния
    with th.no_grad():
        action, _states = policy.predict(obs, deterministic=True)
    print(f"Predicted action for current state: {action}") 