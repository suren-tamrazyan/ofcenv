import numpy as np
from imitation.algorithms.adversarial.gail import GAIL
from imitation.rewards.reward_nets import BasicRewardNet
from imitation.util.networks import RunningNorm
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from imitation.util.util import make_vec_env
from gym.ofc_gym import OfcEnv
from cnn_policy import CNNPolicy

def train_gail():
    # Загружаем экспертные данные
    expert_trajectories = load_expert_trajectories("expert_hh/bit.json")
    
    # Создаем векторизованное окружение
    venv = make_vec_env(
        OfcEnv,
        n_envs=8,
        rng=np.random.default_rng()
    )
    
    # Создаем PPO агента с CNN политикой
    learner = PPO(
        policy=CNNPolicy,
        env=venv,
        batch_size=64,
        ent_coef=0.01,
        learning_rate=3e-4,
        n_epochs=10,
        policy_kwargs={
            "features_extractor_class": CNNExtractor,
            "features_extractor_kwargs": {"features_dim": 512},
        }
    )
    
    # Создаем reward network
    reward_net = BasicRewardNet(
        observation_space=venv.observation_space,
        action_space=venv.action_space,
        normalize_input_layer=RunningNorm
    )
    
    # Создаем GAIL тренер
    gail_trainer = GAIL(
        demonstrations=expert_trajectories,
        demo_batch_size=1024,
        gen_replay_buffer_capacity=2048,
        n_disc_updates_per_round=4,
        venv=venv,
        gen_algo=learner,
        reward_net=reward_net
    )
    
    # Оцениваем до обучения
    reward_before_training, _ = evaluate_policy(learner, venv, 10)
    print(f"Reward before training: {reward_before_training}")
    
    # Обучаем модель
    gail_trainer.train(total_timesteps=100000)
    
    # Оцениваем после обучения
    reward_after_training, _ = evaluate_policy(learner, venv, 10)
    print(f"Reward after training: {reward_after_training}")
    
    # Сохраняем модель
    learner.save("ofc_gail_cnn_policy")
    
    return gail_trainer

if __name__ == "__main__":
    trained_model = train_gail() 