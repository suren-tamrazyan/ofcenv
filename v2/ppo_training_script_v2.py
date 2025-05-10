import os
from typing import Optional, Dict, Callable, Union, List  # Добавим Callable

import gymnasium as gym
import numpy as np
import torch
from gymnasium.envs.registration import register
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback  # Используем MaskableEvalCallback
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.utils import get_linear_fn
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

# Импортируем новую среду и компоненты архитектуры
from ofc_gym_v2 import OfcEnvV2
from ofc_neural_network_architecture import OFCFeatureExtractor, ACTION_SPACE_DIM  # Для get_last_mask

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
        temp_feature_extractor = OFCFeatureExtractor(game_state_dim=game_state_dim)
        calculated_features_dim = temp_feature_extractor.feature_dim
        del temp_feature_extractor
        actual_features_dim = features_dim_override if features_dim_override is not None else calculated_features_dim
        super().__init__(observation_space, features_dim=actual_features_dim)
        # print(f"SB3 Feature Extractor Wrapper Initialized. Output dim: {self.features_dim}")
        self.ofc_feature_extractor = OFCFeatureExtractor(game_state_dim=game_state_dim)
        if self.features_dim != self.ofc_feature_extractor.feature_dim:
             print(f"Warning: Initialized features_dim ({self.features_dim}) differs from internal extractor dim ({self.ofc_feature_extractor.feature_dim}).")
        if features_dim_override is not None and features_dim_override != calculated_features_dim:
             print(f"Warning: features_dim_override ({features_dim_override}) does not match calculated feature dim ({calculated_features_dim}).")

    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        extractor_input = {k: v for k, v in observations.items() if k != 'action_mask'}
        if not extractor_input:
             first_val = next(iter(observations.values())) # Получаем первый элемент, чтобы узнать batch_size и device
             batch_size = first_val.shape[0]
             device = first_val.device
             # print(f"Warning: Observations dict became empty in forward. Returning zeros tensor of shape ({batch_size}, {self.features_dim})")
             return torch.zeros((batch_size, self.features_dim), device=device)
        return self.ofc_feature_extractor(extractor_input)

# --- Функция для ActionMasker ---
def mask_fn(env: gym.Env) -> np.ndarray:
    # Предполагаем, что у OfcEnvV2 есть метод get_action_mask()
    unwrapped_env = env
    while hasattr(unwrapped_env, "env") and not isinstance(unwrapped_env, OfcEnvV2):
        if isinstance(unwrapped_env, ActionMasker): unwrapped_env = unwrapped_env.env
        elif hasattr(unwrapped_env, "unwrapped"): unwrapped_env = unwrapped_env.unwrapped
        else: break
    if isinstance(unwrapped_env, OfcEnvV2):
        return unwrapped_env.get_action_mask()
    else:
        print(f"Warning: mask_fn could not find OfcEnvV2. Found: {type(unwrapped_env)}")
        # Возвращаем дефолтную маску (все действия запрещены), чтобы вызвать ошибку если что-то не так
        return np.zeros(ACTION_SPACE_DIM, dtype=bool)


# --- Коллбэк для проверки (упрощенный) ---
class InfoCallback(BaseCallback): # Переименован для ясности
    def _on_step(self):
        if False and self.n_calls % 20000 == 0: # Выводить реже
            try:
                base_env = self.training_env.envs[0]
                while hasattr(base_env, "env") and not isinstance(base_env, OfcEnvV2):
                     if isinstance(base_env, ActionMasker): base_env = base_env.env
                     elif hasattr(base_env, "unwrapped"): base_env = base_env.unwrapped
                     else: break

                if isinstance(base_env, OfcEnvV2):
                     current_mask_arr = mask_fn(base_env)
                     action = self.locals["actions"][0]
                     phase = base_env.current_turn_phase
                     active_idx = base_env.active_card_idx
                     # print(f"--- Callback Info (Step {self.n_calls}) ---")
                     # print(f"Action chosen (for prev state): {action}")
                     # print(f"Current Mask (for next state): {np.where(current_mask_arr)[0]}")
                     # print(f"Current Env State: Phase={phase}, ActiveCardIdx={active_idx}")
                     # print(f"---------------------------------------")
            except Exception:
                 pass # Молча пропускаем, если не удалось получить инфо
        return True

# --- Основная функция обучения ---
def train_ofc_agent(
    total_timesteps: int = 1_000_000,
    log_dir_base: str = "./ofc_runs/",
    run_name: str = "MaskablePPO_OFC_v2",
    learning_rate_start: float = 3e-4,
    learning_rate_end: float = 1e-5,
    learning_rate_end_fraction: float = 0.9,
    ent_coef_val: float = 0.01,
    vf_coef_val: float = 0.5, # Вернул к стандартному
    net_arch_pi: Optional[List[int]] = None,
    net_arch_vf: Optional[List[int]] = None,
    activation_fn_str: str = "ReLU", # "Tanh" или "ReLU"
    n_steps_val: int = 2048,
    batch_size_val: int = 128,
    n_epochs_val: int = 10,
    load_model_path: Optional[str] = None,
    seed_val: int = 42,
    n_eval_episodes: int = 20,
    eval_freq_factor: int = 10000,
    # --- НОВЫЕ ПАРАМЕТРЫ для VecEnv ---
    vec_env_type: str = "dummy",  # "dummy" или "subproc"
    n_envs: int = 1  # Количество окружений (для subproc > 1)
    ):
    """
    Функция для обучения или дообучения агента OFC.
    """
    print(f"--- Starting/Resuming Training Run: {run_name} ---")
    print(f"Total Timesteps: {total_timesteps}")
    print(f"Using VecEnv type: {vec_env_type} with {n_envs} environments.")
    if load_model_path:
        print(f"Attempting to load model from: {load_model_path}")

    # Папки для логов и моделей
    log_dir = os.path.join(log_dir_base, run_name)
    model_save_path = os.path.join(log_dir, "ppo_ofc_model") # Убрал _v2
    tensorboard_log_path = os.path.join(log_dir, "tb_logs/")
    eval_log_path = os.path.join(log_dir, "eval_logs/")
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(tensorboard_log_path, exist_ok=True)
    os.makedirs(eval_log_path, exist_ok=True)

    # --- Настройка среды ---
    # def make_env_fn():
    #     env = gym.make(ENV_ID)
    #     env = Monitor(env) # Оборачиваем в Monitor для корректной статистики
    #     env = ActionMasker(env, mask_fn)
    #     return env
    # vec_env = DummyVecEnv([make_env_fn])
    def make_env_fn_for_vec(rank: int, seed: int = 0):
        """
        Утилитарная функция для создания окружений для VecEnv.
        :param rank: индекс окружения
        :param seed: начальное число для генератора случайных чисел
        """
        def _init():
            env = gym.make(ENV_ID)
            # Важно: Monitor и другие обертки, не зависящие от rank, лучше применять к базовой среде
            env = Monitor(env)
            env = ActionMasker(env, mask_fn)
            # Устанавливаем seed для каждого окружения, если нужно (для воспроизводимости в SubprocVecEnv)
            # env.reset(seed=seed + rank) # Gymnasium reset принимает seed
            # Однако SB3 VecEnv сам управляет сидами, поэтому явный reset здесь может быть не нужен
            return env
        # Устанавливаем seed для make_vec_env, если используется SubprocVecEnv
        # set_random_seed(seed) # SB3 set_random_seed не нужен здесь, т.к. seed передается в PPO
        return _init

    # --- ВЫБОР ТИПА VEC_ENV ---
    if vec_env_type.lower() == "subproc" and n_envs > 1:
        print(f"Creating SubprocVecEnv with {n_envs} environments.")
        # Для SubprocVecEnv каждая среда должна быть функцией, возвращающей среду
        vec_env = SubprocVecEnv([make_env_fn_for_vec(i, seed_val) for i in range(n_envs)])
    elif vec_env_type.lower() == "dummy" or n_envs == 1:
        if vec_env_type.lower() == "subproc" and n_envs == 1:
            print("Warning: n_envs=1 for SubprocVecEnv, defaulting to DummyVecEnv.")
        print(f"Creating DummyVecEnv with {n_envs} environment(s).")
        # Для DummyVecEnv можно передать список функций или одну функцию, если n_envs=1
        vec_env = DummyVecEnv([make_env_fn_for_vec(i, seed_val) for i in range(n_envs)])
    else:
        raise ValueError(f"Unsupported vec_env_type: {vec_env_type} or invalid n_envs: {n_envs}")
    # --- КОНЕЦ ВЫБОРА ТИПА VEC_ENV ---

    # --- Настройка политики ---
    if net_arch_pi is None: net_arch_pi = [512, 256]
    if net_arch_vf is None: net_arch_vf = [512, 256]
    net_arch_config = dict(pi=net_arch_pi, vf=net_arch_vf)

    activation_fn_map = {"ReLU": torch.nn.ReLU, "Tanh": torch.nn.Tanh}
    selected_activation_fn = activation_fn_map.get(activation_fn_str, torch.nn.Tanh)

    policy_kwargs = dict(
        features_extractor_class=SB3OFCFeaturesExtractor,
        net_arch=net_arch_config,
        activation_fn=selected_activation_fn
    )

    # Learning rate schedule
    effective_learning_rate: Union[float, Callable[[float], float]]
    if learning_rate_start == learning_rate_end:
        effective_learning_rate = learning_rate_start
    else:
        effective_learning_rate = get_linear_fn(
            start=learning_rate_start,
            end=learning_rate_end,
            end_fraction=learning_rate_end_fraction
        )

    ppo_params_dict = dict(
        policy="MultiInputPolicy",
        env=vec_env,
        policy_kwargs=policy_kwargs,
        verbose=1,
        tensorboard_log=tensorboard_log_path,
        learning_rate=effective_learning_rate,
        n_steps=n_steps_val,
        batch_size=batch_size_val,
        n_epochs=n_epochs_val,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=ent_coef_val,
        vf_coef=vf_coef_val,
        max_grad_norm=0.5,
        seed=seed_val
    )

    if load_model_path and os.path.exists(load_model_path + ".zip"):
        print(f"Loading model from {load_model_path}...")
        model = MaskablePPO.load(
            load_model_path,
            env=vec_env, # Передаем новую среду (это важно для SB3)
            custom_objects={"learning_rate": 0.0, "lr_schedule": lambda _:0.0, "clip_range": lambda _:0.0}, # Заглушки для lr и clip_range, если они были schedule
            # НЕ ПЕРЕДАЕМ policy_kwargs здесь, если не хотим их менять
            # SB3 попытается восстановить policy_kwargs из сохраненного файла
        )
        print("Model loaded. Setting new training parameters if needed.")

        # Явно устанавливаем параметры обучения, которые мы МОЖЕМ и ХОТИМ изменить для продолжения
        # SB3 load() может восстановить optimizer state, но мы можем захотеть новый schedule.
        model.learning_rate = effective_learning_rate # effective_learning_rate должно быть определено ранее
        model.ent_coef = ent_coef_val
        model.vf_coef = vf_coef_val
        # model.clip_range = ... # Тоже нужно установить, если clip_range - schedule
        if isinstance(ppo_params_dict["clip_range"], Callable): # ppo_params_dict определен ранее
            model.clip_range = ppo_params_dict["clip_range"]
        else:
            # Для константы или если хотим новый schedule для константы
            model.clip_range = get_linear_fn(ppo_params_dict["clip_range"], ppo_params_dict["clip_range"], 1.0)


        print("Model loaded. Continuing training.")
        # Устанавливаем num_timesteps для learn, чтобы он знал, сколько еще учиться
        # SB3 автоматически продолжит с total_timesteps загруженной модели, если reset_num_timesteps=False
    else:
        if load_model_path:
            print(f"Warning: Specified load_model_path '{load_model_path}' not found. Training new model.")
        model = MaskablePPO(**ppo_params_dict)
        print("New model created.")

    print("--- Model Setup ---")
    # print("Policy:", model.policy) # Можно раскомментировать для детального вывода
    if 'action_mask' in model.observation_space.spaces:
        print("ERROR: 'action_mask' still in model's observation space!")
    print("-------------------")

    # --- Коллбэк для оценки ---
    # eval_env_instance = gym.make(ENV_ID)
    # eval_env_instance = Monitor(eval_env_instance)
    # eval_env_instance = ActionMasker(eval_env_instance, mask_fn)
    # eval_vec_env = DummyVecEnv([lambda: eval_env_instance])
    # Создаем eval_vec_env с таким же типом и количеством, как и основной vec_env
    if vec_env_type.lower() == "subproc" and n_envs > 1:
        eval_vec_env = SubprocVecEnv([make_env_fn_for_vec(i, seed_val + n_envs + i) for i in range(n_envs)]) # Разные сиды для eval
    else:
        eval_vec_env = DummyVecEnv([make_env_fn_for_vec(i, seed_val + n_envs + i) for i in range(n_envs)])

    eval_callback = MaskableEvalCallback( # Используем MaskableEvalCallback
        eval_vec_env,
        best_model_save_path=eval_log_path, # Сохраняет как best_model.zip
        log_path=eval_log_path,
        eval_freq=max(eval_freq_factor // n_envs, 1), # Делим на n_envs
        n_eval_episodes=n_eval_episodes,
        deterministic=True, # Оцениваем детерминированно
        # render=False, # render убран, т.к. MaskableEvalCallback его не поддерживает напрямую
        use_masking=True # ВАЖНО: Указываем использовать маску при оценке
    )

    print(f"Starting/Continuing training for {total_timesteps} total timesteps...")
    try:
        model.learn(
            total_timesteps=total_timesteps, # Общее количество шагов с начала
            callback=[InfoCallback(verbose=0), eval_callback],
            tb_log_name=run_name, # Используем имя запуска для логов
            progress_bar=True,
            reset_num_timesteps=(load_model_path is None) # Сбрасывать, только если новая модель
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
    finally:
        vec_env.close()
        eval_vec_env.close()

# --- Функция для оценки модели ---

def evaluate_ofc_agent(
    model_path: str,
    n_episodes: int = 50,
    deterministic: bool = True,
    use_masking_in_predict: bool = True # Оставляем эту опцию для явного контроля
    ):
    print(f"\n--- Evaluating Model: {model_path} ---")
    if not os.path.exists(model_path + ".zip"):
        print(f"Model not found at {model_path}")
        return

    # Создаем среду для оценки
    eval_env_instance = gym.make(ENV_ID)
    eval_env_instance = Monitor(eval_env_instance)
    # ActionMasker нужен, если use_masking_in_predict=True и мы хотим, чтобы obs содержал 'action_mask'
    # Или если сама модель ожидает obs без маски, а мы ее извлекаем из среды
    # Для чистоты, если predict сам берет маску из obs, то ActionMasker для среды не нужен
    # НО! Если мы хотим ПОЛУЧИТЬ маску из среды, то ActionMasker не нужен, нужен доступ к get_action_mask
    # Оставим ActionMasker, так как он не повредит, а obs['action_mask'] может быть удобен
    eval_env_instance = ActionMasker(eval_env_instance, mask_fn)
    eval_vec_env = DummyVecEnv([lambda: eval_env_instance])

    # Загружаем модель
    # НЕ передаем policy_kwargs, SB3 должен сам восстановить экстрактор
    try:
        model_to_evaluate = MaskablePPO.load(model_path, env=eval_vec_env)
        print(f"Loaded model from {model_path}")
        # Проверим, что экстрактор правильный (опционально, для отладки)
        loaded_extractor = model_to_evaluate.policy.features_extractor
        if hasattr(loaded_extractor, 'ofc_feature_extractor') and \
                loaded_extractor.__class__.__name__ == SB3OFCFeaturesExtractor.__name__:  # Сравниваем имена классов
            print("Correct custom feature extractor type was loaded based on name and attribute.")
        elif loaded_extractor is not None:
            print(
                f"Warning: Loaded model has a feature extractor of type '{type(loaded_extractor).__name__}', expected 'SB3OFCFeaturesExtractor'. Structure might still be correct.")
            # Можно добавить вывод структуры для визуальной проверки, если нужно
            print("Loaded extractor structure:", loaded_extractor)
        else:
            print("Error: Feature extractor not found in loaded model.")
    except Exception as e:
        print(f"Error loading model: {e}")
        import traceback
        traceback.print_exc()
        eval_vec_env.close()
        return


    total_reward = 0
    total_length = 0
    rewards_list = []

    # Сбрасываем среду перед циклом оценки
    obs = eval_vec_env.reset() # obs теперь будет словарём тензоров, если VecEnv обернут в SB3

    for episode in range(n_episodes):
        terminated = False
        truncated = False # В Gymnasium done разделен на terminated и truncated
        episode_reward = 0
        episode_length = 0
        while not terminated and not truncated:
            action_masks_for_predict = None
            if use_masking_in_predict:
                # obs[0] так как reset() для VecEnv возвращает список/кортеж наблюдений
                # и мы берем наблюдение для первой (и единственной) среды
                # Если obs уже словарь (например, после первого step), то просто obs['action_mask']
                current_obs_dict = obs[0] if isinstance(obs, (list, tuple)) else obs

                if 'action_mask' not in current_obs_dict:
                    # Если маски нет в наблюдении, получаем ее из среды
                    action_masks_for_predict = eval_vec_env.env_method("get_action_mask")[0]
                else:
                    action_masks_for_predict = current_obs_dict['action_mask']

            action, _states = model_to_evaluate.predict(
                obs, # Передаем все наблюдения (SB3 сам разберется с батчем)
                deterministic=deterministic,
                action_masks=action_masks_for_predict if use_masking_in_predict else None
            )
            obs, reward, done, info = eval_vec_env.step(action) # done - это массив булевых
            episode_reward += reward[0]
            episode_length += 1

            terminated = done[0] # Используем done[0] для VecEnv
            # truncated для VecEnv обычно тоже в done[0] или в info[0].get("TimeLimit.truncated", False)
            truncated = info[0].get("TimeLimit.truncated", False)


            if terminated or truncated:
                monitor_ep_info = info[0].get('episode')
                if monitor_ep_info:
                    actual_ep_reward = monitor_ep_info['r']
                    actual_ep_length = monitor_ep_info['l']
                    print(f"Eval Episode {episode + 1}/{n_episodes} finished. Reward: {actual_ep_reward:.2f}, Length: {actual_ep_length}")
                    rewards_list.append(actual_ep_reward)
                    total_reward += actual_ep_reward
                    total_length += actual_ep_length
                else:
                    print(f"Eval Episode {episode + 1}/{n_episodes} finished (no Monitor info). Step Reward: {reward[0]:.2f}, Manual Ep Reward: {episode_reward:.2f}, Length: {episode_length}")
                    rewards_list.append(episode_reward)
                    total_reward += episode_reward
                    total_length += episode_length
                # obs = eval_vec_env.reset() # Сброс происходит автоматически в VecEnv, если эпизод завершен
                break # Выход из while для текущего эпизода

    avg_reward = total_reward / n_episodes if n_episodes > 0 else 0
    avg_length = total_length / n_episodes if n_episodes > 0 else 0
    print(f"\nAverage reward over {n_episodes} eval episodes: {avg_reward:.2f}")
    print(f"Average length over {n_episodes} eval episodes: {avg_length:.2f}")
    if rewards_list:
        print(f"Reward std: {np.std(rewards_list):.2f}, Min: {np.min(rewards_list):.2f}, Max: {np.max(rewards_list):.2f}")

    eval_vec_env.close()


# --- Пример использования (можно закомментировать при импорте) ---
if __name__ == "__main__":
    # Настройки для первого запуска или продолжения
    config = {
        "total_timesteps": 15_000, # Уменьшил для быстрого теста
        "log_dir_base": "./ofc_colab_runs/",
        "run_name": "PPO_Run1",
        "learning_rate_start": 3e-4,
        "learning_rate_end": 1e-5,
        "ent_coef_val": 0.01,
        "vf_coef_val": 0.5,
        "net_arch_pi": [512, 256],
        "net_arch_vf": [512, 256],
        "n_steps_val": 2048,
        "batch_size_val": 128,
        "load_model_path": "./ofc_colab_runs/PPO_Run1/ppo_ofc_model_final", # None, # "./ofc_colab_runs/PPO_Run1/ppo_ofc_model_final" # Пример для продолжения
        "eval_freq_factor": 5000, # Оценивать каждые ~5k шагов
        "n_eval_episodes": 10, # Меньше эпизодов для быстрой оценки
        "vec_env_type": "subproc",
        "n_envs": 4
    }

    # Запуск обучения
    train_ofc_agent(**config)

    # Оценка финальной модели
    final_model_to_eval = os.path.join(config["log_dir_base"], config["run_name"], "ppo_ofc_model_final")
    evaluate_ofc_agent(final_model_to_eval, n_episodes=50, use_masking_in_predict=True)

    # Оценка лучшей модели (если была сохранена EvalCallback)
    best_model_to_eval = os.path.join(config["log_dir_base"], config["run_name"], "eval_logs/best_model")
    if os.path.exists(best_model_to_eval + ".zip"):
        evaluate_ofc_agent(best_model_to_eval, n_episodes=50, use_masking_in_predict=True)
    else:
        print(f"Best model not found at {best_model_to_eval}, skipping its evaluation.")