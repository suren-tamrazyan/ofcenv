import numpy as np
from imitation.algorithms import bc
from imitation.data import rollout
from imitation.util.util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from gym.ofc_gym import OfcEnv
from cnn_policy import CNNPolicy, CNNExtractor
from sb3.imitation.expert_data_loader import load_expert_trajectories
from gymnasium.envs.registration import register
import torch as th

register(
    id='ofc-v0',
    entry_point='gym.ofc_gym:OfcEnv',
)

def train_bc():
    # Загружаем экспертные данные
    expert_trajectories = load_expert_trajectories("./expert_hh/first.json")
    
    # Создаем векторизованное окружение
    env = make_vec_env(
        'ofc-v0',
        n_envs=1,
        rng=np.random.default_rng()
    )
    
    # Преобразуем траектории в transitions
    transitions = rollout.flatten_trajectories(expert_trajectories)
    
    # Создаем экземпляр политики
    policy = CNNPolicy(
        observation_space=env.observation_space,
        action_space=env.action_space,
        lr_schedule=lambda _: 5e-2  # Простой планировщик с постоянной скоростью обучения
    )
    
    # Создаем BC тренер с CNN политикой
    bc_trainer = bc.BC(
        observation_space=env.observation_space,
        action_space=env.action_space,
        demonstrations=transitions,
        policy=policy,
        rng=np.random.default_rng(),
        batch_size=128
    )
    
    # Оцениваем до обучения
    reward_before_training, _ = evaluate_policy(bc_trainer.policy, env, 10)
    print(f"Reward before training: {reward_before_training}")
    
    # Обучаем модель
    bc_trainer.train(n_epochs=10, progress_bar=True)
    
    # Оцениваем после обучения
    reward_after_training, _ = evaluate_policy(bc_trainer.policy, env, 10, render=True)
    print(f"Reward after training: {reward_after_training}")
    
    # Сохраняем модель
    bc_trainer.policy.save("./expert_hh/model_saves/ofc_bc_cnn_policy")
    
    # Также можно сохранить только политику
    #th.save(bc_trainer.policy.state_dict(), "ofc_bc_cnn_policy_state_dict.pth")
    
    return bc_trainer

if __name__ == "__main__":
    trained_model = train_bc() 