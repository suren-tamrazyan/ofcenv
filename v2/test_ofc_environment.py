import gymnasium as gym
import numpy as np
import pytest # Используем pytest для удобства тестирования
from gymnasium.utils.env_checker import check_env # Стандартный чекер Gym
from gymnasium.envs.registration import register

# Импортируем нашу среду и компоненты
from ofc_gym_v2 import OfcEnvV2
from ofc_neural_network_architecture import (
    ACTION_SPACE_DIM, TURN_PHASE_START_ROUND_1, TURN_PHASE_PLACE_1,
    TURN_PHASE_PLACE_2, TURN_PHASE_PLACE_3, TURN_PHASE_PLACE_4,
    TURN_PHASE_PLACE_5, TURN_PHASE_DISCARD, OFCActionEncoder
)
# Добавить метод в ofc_gym_v2.py:
# class OfcEnvV2(...):
#     ...
#     def get_internal_state(self):
#         """Возвращает внутреннее состояние для тестирования."""
#         return {
#             "phase": self.current_turn_phase,
#             "active_idx": self.active_card_idx,
#             "placed_count": self.cards_placed_this_turn,
#             "current_player": self.game.current_player_ind if self.game else -1,
#             "round": self.game.round if self.game else 0
#         }

# --- Регистрация среды (если еще не сделано) ---
ENV_ID = 'ofc-test-v2'
try:
    gym.spec(ENV_ID)
except gym.error.NameNotFound:
    register(
        id=ENV_ID,
        entry_point='ofc_gym_v2:OfcEnvV2',
    )

# --- Фикстура для создания окружения ---
@pytest.fixture
def env() -> OfcEnvV2:
    """Создает инстанс среды для тестов."""
    environment = gym.make(ENV_ID)
    # environment = gym_env.make(ENV_ID, render_mode='human')
    # Добавляем метод для тестов, если его нет
    if not hasattr(environment.unwrapped, 'get_internal_state'):
         def _get_internal_state(self):
             return {
                 "phase": self.current_turn_phase, "active_idx": self.active_card_idx,
                 "placed_count": self.cards_placed_this_turn,
                 "current_player": self.game.current_player_ind if self.game else -1,
                 "round": self.game.round if self.game else 0
             }
         environment.unwrapped.get_internal_state = _get_internal_state.__get__(environment.unwrapped, OfcEnvV2)
    return environment

# --- Тесты ---

def test_gym_api_compliance(env):
    """Проверяет соответствие среды API Gymnasium."""
    # check_env(env.unwrapped) # Проверяем базовую среду
    print(f"\n--- Running Gym API Compliance Check for {ENV_ID} ---")
    try:
        check_env(env.unwrapped, skip_render_check=True) # Пропускаем рендер для простоты
        print("Gym API Compliance Check PASSED.")
    except Exception as e:
        pytest.fail(f"Gym API Compliance Check FAILED: {e}")

def test_reset(env):
    """Проверяет функцию reset."""
    print("\n--- Testing env.reset() ---")
    obs, info = env.reset()
    assert isinstance(obs, dict), "Observation should be a dict"
    assert isinstance(info, dict), "Info should be a dict"
    # Проверяем наличие всех ключей и маски
    assert 'action_mask' in obs, "Observation must contain 'action_mask'"
    assert obs['action_mask'].shape == (ACTION_SPACE_DIM,), f"Action mask shape mismatch, expected ({ACTION_SPACE_DIM},)"
    # Проверяем, что игра началась и фаза корректна
    internal_state = env.unwrapped.get_internal_state()
    assert internal_state['round'] == 1, "Game should start in round 1"
    assert internal_state['phase'] == TURN_PHASE_PLACE_1, "Initial phase should be PLACE_1"
    assert internal_state['active_idx'] == 0, "Initial active card index should be 0"
    print("env.reset() test PASSED.")

def test_action_mask_phases(env):
    """Проверяет корректность маски в разных фазах."""
    print("\n--- Testing Action Mask Generation ---")
    encoder = OFCActionEncoder()

    # Раунд 1: Фаза размещения
    obs, info = env.reset()
    mask_r1_place = obs['action_mask']
    assert np.sum(mask_r1_place[:encoder.placement_action_offset]) == 0, "Discard actions should be masked in placement phase"
    assert np.sum(mask_r1_place[encoder.placement_action_offset:]) > 0, "Placement actions should be available in placement phase"
    print("Round 1 Placement Mask: OK")

    # Симулируем переход ко 2 раунду и фазе сброса
    # (нужно выполнить 5 легальных действий размещения)
    # Это сложно симулировать точно, поэтому делаем вручную установку фазы
    env.reset() # Начнем заново
    env.unwrapped.game.round = 2 # Устанавливаем раунд вручную
    env.unwrapped.current_turn_phase = TURN_PHASE_DISCARD # Устанавливаем фазу
    env.unwrapped.active_card_idx = -1
    # Убедимся, что у игрока есть 3 карты (иначе маска сброса будет неверной)
    env.unwrapped.game.players[env.unwrapped.hero_idx].to_play = [1, 2, 3] # Фиктивные карты

    obs = env.unwrapped._get_obs() # Получаем новое наблюдение с новой фазой
    mask_r2_discard = obs['action_mask']
    assert np.all(mask_r2_discard[:encoder.placement_action_offset]), "Discard actions should be unmasked in discard phase (assuming 3 cards)"
    assert np.sum(mask_r2_discard[encoder.placement_action_offset:]) == 0, "Placement actions should be masked in discard phase"
    print("Round 2 Discard Mask: OK")

    # Симулируем фазу размещения после сброса
    env.unwrapped.current_turn_phase = TURN_PHASE_PLACE_1
    env.unwrapped.active_card_idx = 0
    env.unwrapped.game.players[env.unwrapped.hero_idx].to_play = [1, 2] # Осталось 2 карты
    obs = env.unwrapped._get_obs()
    mask_r2_place = obs['action_mask']
    assert np.sum(mask_r2_place[:encoder.placement_action_offset]) == 0, "Discard actions should be masked after discard"
    assert np.sum(mask_r2_place[encoder.placement_action_offset:]) > 0, "Placement actions should be available after discard"
    print("Round 2 Placement Mask (after discard): OK")
    print("Action Mask Generation test PASSED.")


def test_step_legal_action(env):
    """Проверяет шаг с легальным действием."""
    print("\n--- Testing env.step() with Legal Action ---")
    obs, info = env.reset()
    initial_state = env.unwrapped.get_internal_state()
    legal_mask = obs['action_mask']
    legal_actions = np.where(legal_mask)[0]

    if len(legal_actions) == 0:
         pytest.skip("No legal actions available at start, cannot test step.")

    action = legal_actions[0] # Берем первое легальное действие
    print(f"Taking legal action: {action}")
    new_obs, reward, terminated, truncated, new_info = env.step(action)
    new_state = env.unwrapped.get_internal_state()

    assert isinstance(new_obs, dict), "New observation should be a dict"
    assert isinstance(reward, float), "Reward should be float"
    assert isinstance(terminated, bool), "Terminated flag should be bool"
    assert isinstance(truncated, bool), "Truncated flag should be bool"
    # Проверяем, что фаза изменилась (если это не последний шаг хода)
    # Или что ход перешел к оппоненту
    is_round_1 = initial_state['round'] == 1
    placed_count_before = initial_state['placed_count']
    placed_count_after = new_state['placed_count']
    current_player_after = new_state['current_player']

    if not terminated:
        if (is_round_1 and placed_count_before < 4) or (not is_round_1 and placed_count_before < 1):
             assert placed_count_after == placed_count_before + 1, "Placed count should increment"
             assert current_player_after == env.unwrapped.hero_idx, "Player should remain hero during placement steps"
        else: # Последний шаг размещения в ходе героя
             assert current_player_after != env.unwrapped.hero_idx, "Turn should pass to opponent after completing placements"
    print("env.step() with Legal Action test PASSED.")

def test_step_illegal_action(env):
    """Проверяет шаг с нелегальным действием."""
    print("\n--- Testing env.step() with Illegal Action ---")
    obs, info = env.reset()
    legal_mask = obs['action_mask']
    illegal_actions = np.where(~legal_mask)[0]

    if len(illegal_actions) == 0:
        pytest.skip("No illegal actions available to test.")

    action = illegal_actions[0]
    print(f"Taking illegal action: {action}")
    new_obs, reward, terminated, truncated, new_info = env.step(action)

    assert terminated, "Episode should terminate on illegal action"
    assert reward < 0, "Reward should be negative (penalty) for illegal action"
    assert 'error' in new_info and 'Illegal action' in new_info['error'], "Info should contain error message"
    print("env.step() with Illegal Action test PASSED.")

def test_full_game_run(env):
    """Проверяет прогон полного эпизода со случайными легальными действиями."""
    print("\n--- Testing Full Game Run (Random Agent) ---")
    obs, info = env.reset()
    terminated = False
    truncated = False
    step_count = 0
    max_steps = 200 # Ограничение на всякий случай
    total_reward = 0

    while not terminated and not truncated and step_count < max_steps:
        mask = obs['action_mask']
        legal_actions = np.where(mask)[0]
        if len(legal_actions) == 0:
             # Это может случиться, если оппонент сделал ход, и игра закончилась
             internal_state = env.unwrapped.get_internal_state()
             print(f"No legal actions available for hero at step {step_count}. State: {internal_state}")
             assert env.unwrapped.game.is_game_over(), "No legal actions but game not over?"
             break # Завершаем цикл

        # Выбираем случайное легальное действие
        action = env.np_random.choice(legal_actions)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        step_count += 1
        # print(f"Step {step_count}, Action: {action}, Reward: {reward}, Terminated: {terminated}")

    assert terminated or truncated or step_count == max_steps, "Game loop finished unexpectedly"
    if step_count < max_steps:
         assert env.unwrapped.game.is_game_over(), "Game should be over if terminated naturally"
         print(f"Full Game Run PASSED in {step_count} steps. Final Reward: {total_reward}")
    else:
         pytest.fail(f"Game did not terminate within {max_steps} steps.")