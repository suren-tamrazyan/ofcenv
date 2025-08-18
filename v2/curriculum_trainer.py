import os
from typing import Optional, List, Dict, Any
import numpy as np
import torch
import gymnasium as gym

# Импортируем только то, что нужно
from ppo_training_script_v2 import train_ofc_agent, evaluate_ofc_agent
from hh_parser import HHParser
# Убираем лишние импорты, так как они теперь внутри ppo_training_script_v2

# --- Настройки Curriculum ---
HH_FILES_DIR_TRAIN = "D:\\develop\\temp\\poker\\Eureka\\hh\\train\\" # <<--- ПУТЬ К ОБУЧАЮЩИМ HH
HH_FILES_DIR_VALIDATION = "D:\\develop\\temp\\poker\\Eureka\\hh\\validation\\" # <<--- ПУТЬ К ВАЛИДАЦИОННЫМ HH
LOG_DIR_BASE_CURRICULUM = "./ofc_curriculum_runs/"
N_ENVS_CURRICULUM = 4 # Количество параллельных сред для обучения на каждом этапе

# Игровые раунды (1-5)
# Напоминание: HHParser.get_states_for_round ожидает игровой раунд (1, 2, 3, 4, 5)
CURRICULUM_STAGES = [
    {"placement_round_hh": 5, "total_timesteps": 50_000, "run_name_suffix": "_stage_r5"}, # Последний раунд
    {"placement_round_hh": 4, "total_timesteps": 100_000, "run_name_suffix": "_stage_r4"},
    #{"placement_round_hh": 3, "total_timesteps": 1_000_000, "run_name_suffix": "_stage_r3"},
    #{"placement_round_hh": 2, "total_timesteps": 1_500_000, "run_name_suffix": "_stage_r2"},
    # {"placement_round_hh": 1, "total_timesteps": 2_000_000, "run_name_suffix": "_stage_r1"}, # Полная игра (без curriculum)
]

# Общие параметры для train_ofc_agent
COMMON_TRAINING_CONFIG = {
    "log_dir_base": LOG_DIR_BASE_CURRICULUM,
    "learning_rate_start": 1e-4,
    "learning_rate_end": 5e-6,
    "learning_rate_end_fraction": 0.95,
    "ent_coef_val": 0.005,
    "vf_coef_val": 2.0, #0.7,
    "net_arch_pi": [512, 256, 128],
    "net_arch_vf": [512, 256, 128],
    "activation_fn_str": "ReLU",
    "n_steps_val": 2048,
    "batch_size_val": 256,
    "n_epochs_val": 10,
    "seed_val": 42,
    "n_eval_episodes": 50,
    "eval_freq_factor": 20000,
    "vec_env_type": "subproc" if N_ENVS_CURRICULUM > 1 else "dummy",
    "n_envs": N_ENVS_CURRICULUM
}

if __name__ == "__main__":
    if torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")

    # 1. Парсим все HH один раз
    print("--- Parsing Hand History files ---")
    print(f"Parsing TRAINING data from: {HH_FILES_DIR_TRAIN}")
    train_hh_parser = HHParser(hh_files_directory=HH_FILES_DIR_TRAIN)
    train_hh_parser.parse_files()
    if not train_hh_parser.parsed_hands:
        print(f"Error: No training hands parsed from {HH_FILES_DIR_TRAIN}. Exiting.")
        exit()

    print(f"Parsing VALIDATION data from: {HH_FILES_DIR_VALIDATION}")
    validation_hh_parser = HHParser(hh_files_directory=HH_FILES_DIR_VALIDATION)
    validation_hh_parser.parse_files()
    if not validation_hh_parser.parsed_hands:
        print(f"Warning: No validation hands parsed from {HH_FILES_DIR_VALIDATION}. Evaluation will be less reliable.")
        # Можно сделать так, чтобы валидационный парсер использовал те же данные, что и обучающий,
        # если валидационная папка пуста, но это менее идеально.
        validation_hh_parser.parsed_hands = train_hh_parser.parsed_hands[:1000] # Например, взять часть из train для валидации, если нет отдельного набора

    previous_stage_model_path = None

    for stage_idx, stage_config in enumerate(CURRICULUM_STAGES):
        game_round = stage_config["placement_round_hh"]
        stage_total_timesteps = stage_config["total_timesteps"]
        run_name = f"Curriculum{stage_config['run_name_suffix']}"

        print(f"\n--- Starting Curriculum Stage {stage_idx + 1}/{len(CURRICULUM_STAGES)} ---")
        print(f"Target Game Round: {game_round}")
        print(f"Run Name: {run_name}")

        # 2. Генерируем начальные состояния для текущего этапа
        initial_states_for_training = None
        initial_states_for_validation = None
        # Стадия 1 (полная игра) не требует начальных состояний из HH
        if game_round > 1:
            print(f"Generating states for game round {game_round}...")
            initial_states_for_training = train_hh_parser.get_states_for_round(game_round)
            initial_states_for_validation = validation_hh_parser.get_states_for_round(game_round)

            if not initial_states_for_training:
                print(f"Warning: No TRAINING states found for game round {game_round}. Skipping stage.")
                continue
            if not initial_states_for_validation:
                print(
                    f"Warning: No VALIDATION states found for game round {game_round}. Using training states for validation.")
                initial_states_for_validation = initial_states_for_training  # Запасной вариант

            print(
                f"Generated {len(initial_states_for_training)} training states and {len(initial_states_for_validation)} validation states.")
        else:
            print("This is the full game stage, training will use standard reset.")
            # Для финальной стадии оценка тоже будет на полной игре
            initial_states_for_validation = None  # Указываем, что оценка на полной игре

        # 3. Обновляем конфигурацию обучения для текущего этапа
        current_stage_training_config = COMMON_TRAINING_CONFIG.copy()
        current_stage_training_config["total_timesteps"] = stage_total_timesteps
        current_stage_training_config["run_name"] = run_name
        if previous_stage_model_path:
            current_stage_training_config["load_model_path"] = previous_stage_model_path
            print(f"Continuing training from: {previous_stage_model_path}")
        else:
            current_stage_training_config["load_model_path"] = None
            print("Starting training new model for the first stage.")
        # --- ДОБАВЛЯЕМ СПИСОК СОСТОЯНИЙ В КОНФИГУРАЦИЮ ---
        current_stage_training_config["initial_states_for_curriculum"] = initial_states_for_training

        # --- ЗАПУСКАЕМ ОБУЧЕНИЕ для текущего этапа ---
        train_ofc_agent(**current_stage_training_config)

        # Сохраняем путь к модели, обученной на этом этапе, для следующего
        # Сохраняем путь к модели
        current_model_path = os.path.join(
            current_stage_training_config["log_dir_base"],
            current_stage_training_config["run_name"],
            "ppo_ofc_model_final"
        )
        previous_stage_model_path = current_model_path

        # --- ОЦЕНКА НА ЗАДАЧЕ ТЕКУЩЕЙ СТАДИИ (используя ФИКСИРОВАННЫЙ валидационный набор) ---
        print(
            f"\n--- Evaluating Stage {stage_idx + 1} Model on STAGE-SPECIFIC validation task (Round {game_round}) ---")
        if os.path.exists(current_model_path + ".zip"):
            evaluate_ofc_agent(
                model_path=current_model_path,
                n_episodes=50,
                use_masking_in_predict=True,
                initial_states_for_eval=initial_states_for_validation
            )
        else:
            print(f"Model not found at {current_model_path}, cannot evaluate on stage task.")

    print("\n--- Curriculum Learning Finished ---")