import gymnasium as gym
import numpy as np
import os
from typing import Dict, Any, List

# Импортируем ваши классы
from ofc_gym_v2 import OfcEnvV2 # Ваша среда
from hh_parser import HHParser # Ваш парсер HH (убедитесь, что он доступен)
from ofc_neural_network_architecture import ACTION_SPACE_DIM # Для ACTION_SPACE_DIM, если нужно

# --- Константы и Настройки для Теста ---
ENV_ID_CURRICULUM_TEST = 'ofc-curriculum-test-v2'
HH_FILES_DIR_TEST = "D:\\develop\\temp\\poker\\Eureka\\1"
NUM_EPISODES_PER_STAGE = 1500  # Сколько эпизодов прогнать для каждого этапа curriculum
MAX_STEPS_PER_EPISODE = 50 # Ограничение на случай зацикливания

# Раунды размещения в терминах HH-парсера (0-4)
# Мы будем тестировать, начиная с более поздних раундов
TEST_PLACEMENT_ROUNDS_HH = [1]#[5, 4, 2, 1] # Например, последний, предпредпоследний, второй, первый

# --- Регистрация среды (если еще не сделано) ---
try:
    gym.spec(ENV_ID_CURRICULUM_TEST)
except gym.error.NameNotFound:
    gym.register(
        id=ENV_ID_CURRICULUM_TEST,
        entry_point='ofc_gym_v2:OfcEnvV2', # Убедитесь, что ofc_gym_v2.py в PYTHONPATH
    )

def run_episode_from_state(env: OfcEnvV2, initial_state: Dict[str, Any], episode_num: int, placement_round: int) -> bool:
    """
    Запускает один эпизод из заданного начального состояния, используя случайные легальные действия.
    Возвращает True, если эпизод успешно завершился.
    """
    print(f"\n--- Episode {episode_num} (Starting from placement round {placement_round} in HH terms) ---")
    try:
        # Сначала стандартный reset, чтобы удовлетворить OrderEnforcing
        # Он может быть не нужен, если reset_to_state вызывает super().reset() правильно
        # Но попробуем его добавить, чтобы исключить эту причину
        # Этот reset установит случайное начальное состояние, которое мы тут же переопределим
        try:
            # env.max_player = len(initial_state["players"])
            _ = env.reset(seed=episode_num * 100) # Даем другой сид, чтобы не конфликтовать
        except Exception as e:
            pass;

        obs, info = env.reset_to_state(initial_state_snapshot=initial_state, seed=episode_num)
        # print("Initial Observation:", obs) # Для отладки
        # print("Initial Info:", info)
    except Exception as e:
        print(f"ERROR during env.reset_to_state: {e}")
        import traceback
        traceback.print_exc()
        return False

    terminated = False
    truncated = False
    total_reward = 0
    step_count = 0

    while not terminated and not truncated and step_count < MAX_STEPS_PER_EPISODE:
        action_mask = obs.get('action_mask')
        if action_mask is None:
            print("ERROR: Action mask not found in observation!")
            return False

        legal_actions = np.where(action_mask)[0]

        if len(legal_actions) == 0:
            print(f"Step {step_count}: No legal actions available. Current phase: {info.get('phase')}")
            # Проверяем, действительно ли игра должна быть окончена или это ошибка
            if not env.unwrapped.game.is_game_over():
                print("ERROR: No legal actions, but game is not over according to OfcGame!")
                print(env.unwrapped.game) # Выводим состояние игры для отладки
                return False
            else:
                print("Game is over (no legal actions).")
                break # Выходим из цикла, если игра окончена

        action = np.random.choice(legal_actions)
        # print(f"Step {step_count}: Player {env.unwrapped.game.current_player_ind}, Phase {info.get('phase')}, Action Mask {legal_actions}, Chosen Action: {action}")

        try:
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            step_count += 1
        except Exception as e:
            print(f"ERROR during env.step({action}): {e}")
            import traceback
            traceback.print_exc()
            print("Current game state before error:")
            if env.unwrapped.game: print(env.unwrapped.game)
            return False

    if step_count >= MAX_STEPS_PER_EPISODE:
        print(f"Episode truncated after {MAX_STEPS_PER_EPISODE} steps (possible loop or very long game).")
        return False # Считаем такой эпизод неудачным для теста

    print(f"Episode finished in {step_count} steps. Final reward: {total_reward:.2f}")
    # Дополнительная проверка, что игра действительно завершена
    if not env.unwrapped.game.is_game_over():
        print("ERROR: Episode terminated by env, but OfcGame does not consider game over!")
        print(env.unwrapped.game)
        return False

    return True


if __name__ == "__main__":
    print("Starting Curriculum Reset Test...")

    # 1. Инициализируем парсер HH
    if not os.path.isdir(HH_FILES_DIR_TEST) or not os.listdir(HH_FILES_DIR_TEST):
        print(f"ERROR: Hand History directory '{HH_FILES_DIR_TEST}' is empty or does not exist.")
        print("Please set HH_FILES_DIR_TEST to the correct path.")
        exit()

    hh_parser = HHParser(hh_files_directory=HH_FILES_DIR_TEST)
    print(f"Parsing HH files from: {HH_FILES_DIR_TEST}")
    hh_parser.parse_files()

    if not hh_parser.parsed_hands:
        print("No hands were parsed from the HH files. Cannot proceed with the test.")
        exit()

    # 2. Создаем экземпляр среды
    # Важно: НЕ оборачиваем в ActionMasker или Monitor здесь, так как reset_to_state - это метод базовой среды
    # и мы хотим тестировать именно его. Обертки могут перехватывать reset.
    try:
        test_env = gym.make(ENV_ID_CURRICULUM_TEST) # type: OfcEnvV2
        # Убедимся, что это действительно наш класс и у него есть нужный метод
        if not isinstance(test_env.unwrapped, OfcEnvV2) or not hasattr(test_env.unwrapped, 'reset_to_state'):
            print(f"ERROR: Environment {ENV_ID_CURRICULUM_TEST} is not of type OfcEnvV2 or missing reset_to_state.")
            exit()
    except Exception as e:
        print(f"ERROR creating environment {ENV_ID_CURRICULUM_TEST}: {e}")
        exit()


    # 3. Прогоняем тесты для разных стадий curriculum
    all_stages_successful = True
    for placement_round_hh in TEST_PLACEMENT_ROUNDS_HH:
        print(f"\n===== Testing Curriculum Stage: Placement Round (HH) {placement_round_hh} =====")
        initial_states = hh_parser.get_states_for_round(placement_round_hh)

        if not initial_states:
            print(f"No initial states generated for placement round {placement_round_hh}. Skipping.")
            continue

        print(f"Generated {len(initial_states)} initial states for this stage.")
        successful_episodes_this_stage = 0
        for i in range(min(NUM_EPISODES_PER_STAGE, len(initial_states))): # Берем не больше, чем есть состояний
            initial_state = initial_states[i] # Берем по порядку или np.random.choice(initial_states)
            if run_episode_from_state(test_env, initial_state, episode_num=i+1, placement_round=placement_round_hh):
                successful_episodes_this_stage += 1
            else:
                all_stages_successful = False # Отмечаем, что хотя бы один эпизод провалился
                print(f"FAILURE in episode {i+1} for placement round {placement_round_hh}.")
                # Можно добавить break здесь, если одна ошибка на стадии достаточна для провала стадии

        print(f"--- Stage Summary (Placement Round HH {placement_round_hh}) ---")
        print(f"Successfully completed {successful_episodes_this_stage}/{min(NUM_EPISODES_PER_STAGE, len(initial_states))} episodes.")
        if successful_episodes_this_stage < min(NUM_EPISODES_PER_STAGE, len(initial_states)):
             print(f"!!! STAGE FAILED to complete all configured episodes !!!")


    test_env.close()

    print("\n===== Curriculum Reset Test Finished =====")
    if all_stages_successful and NUM_EPISODES_PER_STAGE > 0 :
        print("All configured episodes across all tested stages completed successfully (may include truncations).")
    elif NUM_EPISODES_PER_STAGE == 0:
        print("No episodes were configured to run (NUM_EPISODES_PER_STAGE = 0).")
    else:
        print("Some episodes FAILED or were truncated. Check logs above.")