import os
import argparse
import numpy as np
import torch

# Импортируем
from ppo_training_script_v2 import train_ofc_agent, evaluate_ofc_agent
from hh_parser import HHParser

if __name__ == "__main__":
    # --- Настройка аргументов командной строки ---
    parser = argparse.ArgumentParser(description="Run a single stage of Curriculum Learning for OFC.")
    parser.add_argument("--target_round", type=int, required=True,
                        help="Game round to train on (e.g., 5 for the last round, 1 for the full game).")
    parser.add_argument("--total_timesteps", type=int, required=True, help="Total timesteps for this training stage.")
    parser.add_argument("--run_name", type=str, required=True,
                        help="Unique name for this training run (e.g., 'Curriculum_Stage5_Run1').")
    parser.add_argument("--load_model_path", type=str, default=None,
                        help="Path to a pre-trained model to continue training from (e.g., from a previous stage).")

    # Пути
    parser.add_argument("--train_hh_dir", type=str, required=True, help="Directory with training Hand History files.")
    parser.add_argument("--validation_hh_dir", type=str, required=True,
                        help="Directory with validation Hand History files.")
    parser.add_argument("--log_dir_base", type=str, default="./ofc_curriculum_runs/",
                        help="Base directory for logs and models.")

    # Гиперпараметры (примеры)
    parser.add_argument("--lr_start", type=float, default=1e-4)
    parser.add_argument("--lr_end", type=float, default=5e-6)
    parser.add_argument("--ent_coef", type=float, default=0.005)
    parser.add_argument("--vf_coef", type=float, default=0.7)
    parser.add_argument("--n_envs", type=int, default=8)
    parser.add_argument("--n_steps", type=int, default=2048)
    parser.add_argument("--batch_size", type=int, default=256)

    args = parser.parse_args()

    # --- Начало основной логики ---
    if torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")

    # Генерация состояний для целевого раунда
    game_round = args.target_round
    all_train_states_for_stage = []
    all_validation_states_for_stage = []

    if game_round > 1:
        # Парсинг HH для валидации (для финальной оценки)
        print(f"--- Parsing VALIDATION data from: {args.validation_hh_dir} ---")
        validation_hh_parser = HHParser(hh_files_directory=args.validation_hh_dir)
        validation_hh_parser.parse_files()
        if not validation_hh_parser.parsed_hands:
            print("Warning: No validation hands parsed. Final evaluation will be less reliable.")

        # Парсинг HH для обучения
        print(f"--- Parsing TRAINING data from: {args.train_hh_dir} ---")
        train_hh_parser = HHParser(hh_files_directory=args.train_hh_dir)
        train_hh_parser.parse_files()
        if not train_hh_parser.parsed_hands:
            print("Error: No training hands found. Exiting.")
            exit()

        print(f"Generating states for game round {game_round}...")
        all_train_states_for_stage = train_hh_parser.get_states_for_round(game_round)
        all_validation_states_for_stage = validation_hh_parser.get_states_for_round(game_round)
        if not all_train_states_for_stage:
            print(f"Error: No TRAINING states found for game round {game_round}. Exiting.")
            exit()
    else:
        print("Target round is 1 (full game). Training will use standard environment reset.")

    # --- ЛОГИКА ОБУЧЕНИЯ ПО ПОРЦИЯМ (СУБ-ЭПОХИ) ---
    SUB_EPOCH_TIMESTEPS = 250_000  # Сколько шагов учиться на одной порции
    CURRICULUM_SUBSET_SIZE = 50000  # Размер одной порции

    # Определяем, сколько всего суб-эпох нужно, чтобы пройти все timesteps
    num_sub_epochs = max(1, args.total_timesteps // SUB_EPOCH_TIMESTEPS)
    timesteps_per_sub_epoch = args.total_timesteps // num_sub_epochs

    print(f"\n--- Starting Training Stage for Round {game_round} ---")
    print(f"Run Name: {args.run_name}")
    print(f"Training will be in {num_sub_epochs} sub-epochs of ~{timesteps_per_sub_epoch} timesteps each.")

    # Путь к модели, которая будет дообучаться. Начинаем с того, что передали.
    current_model_path = args.load_model_path

    # Перемешиваем индексы ОДИН РАЗ, чтобы проходить порции в случайном порядке
    if all_train_states_for_stage:
        state_indices = np.arange(len(all_train_states_for_stage))
        np.random.shuffle(state_indices)
        all_train_states_for_stage = [all_train_states_for_stage[i] for i in state_indices]

    for sub_epoch in range(num_sub_epochs):
        print(f"\n--- Sub-epoch {sub_epoch + 1}/{num_sub_epochs} for Stage (Round {game_round}) ---")

        current_training_subset = None
        if all_train_states_for_stage:
            # --- ПОСЛЕДОВАТЕЛЬНЫЙ ВЫБОР ПОРЦИИ ---
            start_idx = (sub_epoch * CURRICULUM_SUBSET_SIZE) % len(all_train_states_for_stage)
            end_idx = start_idx + CURRICULUM_SUBSET_SIZE
            # Берем порцию, обеспечивая цикличность (если дошли до конца, начинаем сначала)
            subset_indices = np.arange(start_idx, end_idx) % len(all_train_states_for_stage)
            current_training_subset = [all_train_states_for_stage[i] for i in subset_indices]
            print(
                f"Using a subset of {len(current_training_subset)} training states (indices from {start_idx} to {end_idx}).")
            # --- КОНЕЦ ПОСЛЕДОВАТЕЛЬНОГО ВЫБОРА ---

        # Конфигурация для этого запуска train_ofc_agent
        current_config = {
            "total_timesteps": timesteps_per_sub_epoch,
            "log_dir_base": args.log_dir_base,
            "run_name": args.run_name,
            "learning_rate_start": args.lr_start,
            "learning_rate_end": args.lr_end,
            "ent_coef_val": args.ent_coef,
            "vf_coef_val": args.vf_coef,
            "net_arch_pi": [512, 256, 128],  # Можно вынести в argparse
            "net_arch_vf": [512, 256, 128],
            "n_steps_val": args.n_steps,
            "batch_size_val": args.batch_size,
            "load_model_path": current_model_path,  # Загружаем модель с предыдущего шага
            "initial_states_for_curriculum": current_training_subset,
            "use_eval_callback": False,  # <--- ОТКЛЮЧАЕМ EvalCallback
            "n_envs": args.n_envs,
            "vec_env_type": "subproc" if args.n_envs > 1 else "dummy",
            # ... можно добавить другие параметры из argparse
        }

        # ЗАПУСКАЕМ ОБУЧЕНИЕ НА ПОРЦИИ
        trained_model = train_ofc_agent(**current_config)

        # Путь для сохранения модели после этой суб-эпохи
        model_save_dir = os.path.join(args.log_dir_base, args.run_name)
        current_model_path_to_save = os.path.join(model_save_dir, "ppo_ofc_model")

        if trained_model:
            trained_model.save(current_model_path_to_save)
            # Обновляем путь для следующей суб-эпохи
            current_model_path = current_model_path_to_save
            print(f"  Sub-epoch finished. Model saved to: {current_model_path}.zip")
        else:
            print(f"  ERROR: Training for sub-epoch {sub_epoch + 1} did not return a model. Stopping stage.")
            break

    print(f"\n--- Finished all sub-epochs for Stage (Round {game_round}) ---")

    # ФИНАЛЬНАЯ ОЦЕНКА после всех суб-эпох
    final_model_path = current_model_path  # Модель, обученная на последней суб-эпохе
    if final_model_path and os.path.exists(final_model_path + ".zip"):
        print(f"\n--- Final Evaluation for Round {game_round} ---")
        evaluate_ofc_agent(
            model_path=final_model_path,
            n_episodes=len(all_validation_states_for_stage) if len(all_validation_states_for_stage) > 0 else 200,  # Больше эпизодов для надежной итоговой оценки
            use_masking_in_predict=True,
            initial_states_for_eval=all_validation_states_for_stage
        )
    else:
        print("No final model to evaluate.")