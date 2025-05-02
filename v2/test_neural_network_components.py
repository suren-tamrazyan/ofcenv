import torch
import pytest
import gymnasium as gym
import numpy as np
from gymnasium.envs.registration import register
from typing import Dict, Optional # Добавим Optional

# Импортируем компоненты сети и функцию подготовки данных
from ofc_neural_network_architecture import (
    OFCFeatureExtractor, OFCPolicyNetwork, state_to_tensors,
    ACTION_SPACE_DIM, NUM_CARDS, MAX_OPPONENTS, MAX_CARDS_IN_ROW,
    TURN_PHASE_DISCARD, MAX_CARDS_TO_PLAY_NN
)
# Импортируем среду для получения observation_space
from ofc_gym_v2 import OfcEnvV2 # Импортируем вашу среду

# --- Регистрация среды (если еще не сделано) ---
ENV_ID_NN = 'ofc-nn-test-v2'
try:
    gym.spec(ENV_ID_NN)
except gym.error.NameNotFound:
    register(id=ENV_ID_NN, entry_point='ofc_gym_v2:OfcEnvV2')

# --- Фикстура для создания observation space ---
@pytest.fixture
def observation_space() -> gym.spaces.Dict:
    """Создает observation space из среды."""
    env = gym.make(ENV_ID_NN)
    try:
        obs_space = env.observation_space
        assert 'game_state' in obs_space.spaces, "game_state missing in observation space"
    finally:
        env.close()
    return obs_space

# --- Фикстура для создания батча фиктивных данных ---
@pytest.fixture
def sample_batch(observation_space) -> Dict[str, torch.Tensor]:
    """Создает батч фиктивных наблюдений."""
    batch_size = 4
    batch = {}
    for key, space in observation_space.spaces.items():
         if isinstance(space, gym.spaces.Box):
             low = space.low
             high = space.high
             shape = (batch_size, *space.shape)
             dtype = space.dtype

             # --- ИСПРАВЛЕННАЯ ОБРАБОТКА ГРАНИЦ ---
             # Проверяем конечность первого элемента границы (предполагаем однородность)
             low_is_finite = np.isfinite(low).all() if isinstance(low, np.ndarray) else np.isfinite(low)
             high_is_finite = np.isfinite(high).all() if isinstance(high, np.ndarray) else np.isfinite(high)

             # Получаем скалярные значения границ
             low_val = low.item(0) if isinstance(low, np.ndarray) else low
             high_val = high.item(0) if isinstance(high, np.ndarray) else high
             # --- КОНЕЦ ИСПРАВЛЕНИЙ ---

             # Корректная генерация для индексов карт (int)
             if 'player' in key or 'opp' in key or 'to_play' in key:
                 low_int = max(0, int(low_val)) if low_is_finite else 0
                 # +1 потому что high в randint не включается
                 high_int = min(NUM_CARDS + 1, int(high_val) + 1) if high_is_finite else NUM_CARDS + 1
                 # Убедимся, что low < high для randint
                 if high_int <= low_int: high_int = low_int + 1
                 sample_np = np.random.randint(low=low_int, high=high_int, size=shape, dtype=dtype)

             # Корректная генерация для game_state (float)
             elif key == 'game_state':
                 test_low = -10.0
                 test_high = 10.0
                 sample_np = np.random.uniform(low=test_low, high=test_high, size=shape).astype(dtype)

             else: # Для других Box
                 test_low = low_val if low_is_finite else -1.0
                 test_high = high_val if high_is_finite else 1.0
                 # Убедимся, что low < high для uniform
                 if test_high <= test_low: test_high = test_low + 1.0
                 sample_np = np.random.uniform(low=test_low, high=test_high, size=shape).astype(dtype)

             batch[key] = torch.from_numpy(sample_np)

         elif isinstance(space, gym.spaces.MultiBinary):
             shape = (batch_size, *space.shape)
             dtype = space.dtype
             torch_dtype = torch.long if dtype.kind in ('i', 'u') else torch.float32
             sample_np = np.random.randint(low=0, high=2, size=shape, dtype=dtype)
             if key == 'action_mask':
                 batch[key] = torch.from_numpy(sample_np).bool()
             else:
                 batch[key] = torch.from_numpy(sample_np).to(torch_dtype)
         else:
             print(f"Warning: Skipping space type {type(space)} for key '{key}' in sample_batch fixture.")
             pass

    required_keys = list(observation_space.spaces.keys())
    extractor_required_keys = [k for k in required_keys if k != 'action_mask']
    assert all(k in batch for k in extractor_required_keys), f"Missing required keys in sample batch. Expected: {extractor_required_keys}, Got: {list(batch.keys())}"

    return batch



# --- Тесты ---

def test_feature_extractor_creation(observation_space):
    """Проверяет создание OFCFeatureExtractor."""
    print("\n--- Testing OFCFeatureExtractor Creation ---")
    game_state_dim = observation_space['game_state'].shape[0]
    try:
        extractor = OFCFeatureExtractor(game_state_dim=game_state_dim)
        assert isinstance(extractor, OFCFeatureExtractor)
        print("OFCFeatureExtractor Creation PASSED.")
    except Exception as e:
        pytest.fail(f"OFCFeatureExtractor Creation FAILED: {e}")

def test_feature_extractor_forward_pass(observation_space, sample_batch):
    """Проверяет прямой проход через OFCFeatureExtractor."""
    print("\n--- Testing OFCFeatureExtractor Forward Pass ---")
    game_state_dim = observation_space['game_state'].shape[0]
    extractor = OFCFeatureExtractor(game_state_dim=game_state_dim)
    batch_size = list(sample_batch.values())[0].size(0) # Получаем batch_size из первого тензора

    # Убираем action_mask, т.к. экстрактор ее не ожидает
    if 'action_mask' in sample_batch:
        extractor_input_batch = {k: v for k, v in sample_batch.items() if k != 'action_mask'}
    else:
        extractor_input_batch = sample_batch

    try:
        features = extractor(extractor_input_batch)
        assert isinstance(features, torch.Tensor), "Output should be a Tensor"
        assert features.dim() == 2, "Output tensor should have 2 dimensions (batch, features)"
        assert features.size(0) == batch_size, "Output batch size should match input"
        assert features.size(1) == extractor.feature_dim, "Output feature dimension should match extractor.feature_dim"
        print(f"Forward Pass PASSED. Output shape: {features.shape}")
    except Exception as e:
        pytest.fail(f"OFCFeatureExtractor Forward Pass FAILED: {e}")

def test_policy_network_creation(observation_space):
    """Проверяет создание OFCPolicyNetwork."""
    print("\n--- Testing OFCPolicyNetwork Creation ---")
    # Сначала нужен экстрактор, чтобы узнать feature_dim
    game_state_dim = observation_space['game_state'].shape[0]
    extractor = OFCFeatureExtractor(game_state_dim=game_state_dim)
    feature_dim = extractor.feature_dim
    try:
        policy_net = OFCPolicyNetwork(feature_dim=feature_dim, action_dim=ACTION_SPACE_DIM)
        assert isinstance(policy_net, OFCPolicyNetwork)
        print("OFCPolicyNetwork Creation PASSED.")
    except Exception as e:
        pytest.fail(f"OFCPolicyNetwork Creation FAILED: {e}")


def test_policy_network_forward_pass(observation_space, sample_batch):
    """Проверяет прямой проход через OFCPolicyNetwork."""
    print("\n--- Testing OFCPolicyNetwork Forward Pass ---")
    game_state_dim = observation_space['game_state'].shape[0]
    extractor = OFCFeatureExtractor(game_state_dim=game_state_dim)
    policy_net = OFCPolicyNetwork(feature_dim=extractor.feature_dim, action_dim=ACTION_SPACE_DIM)
    batch_size = list(sample_batch.values())[0].size(0)

    # Убираем action_mask
    if 'action_mask' in sample_batch:
        extractor_input_batch = {k: v for k, v in sample_batch.items() if k != 'action_mask'}
    else:
        extractor_input_batch = sample_batch

    try:
        # 1. Получаем признаки
        features = extractor(extractor_input_batch)
        # 2. Пропускаем признаки через Actor/Critic
        action_logits, value_preds = policy_net(features)

        assert isinstance(action_logits, torch.Tensor), "Action logits should be a Tensor"
        assert action_logits.dim() == 2, "Action logits should have 2 dimensions (batch, actions)"
        assert action_logits.size(0) == batch_size, "Action logits batch size mismatch"
        assert action_logits.size(1) == ACTION_SPACE_DIM, f"Action logits dim mismatch, expected {ACTION_SPACE_DIM}"

        assert isinstance(value_preds, torch.Tensor), "Value predictions should be a Tensor"
        assert value_preds.dim() == 2, "Value predictions should have 2 dimensions (batch, 1)"
        assert value_preds.size(0) == batch_size, "Value predictions batch size mismatch"
        assert value_preds.size(1) == 1, "Value predictions should have dimension 1"

        print(f"Policy Network Forward Pass PASSED. Logits shape: {action_logits.shape}, Values shape: {value_preds.shape}")
    except Exception as e:
        pytest.fail(f"OFCPolicyNetwork Forward Pass FAILED: {e}")

# Дополнительно можно протестировать state_to_tensors с фиктивным объектом game,
# но это требует создания мок-объекта, имитирующего OfcGame и OfcPlayer.