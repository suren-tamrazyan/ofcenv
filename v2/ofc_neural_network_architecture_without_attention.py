import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from treys import Card, Deck  # Для создания card_int_map

# --- Константы ---
NUM_CARDS = 52
CARD_PAD_IDX = 0  # Индекс для паддинга/пустого слота в nn.Embedding
MAX_OPPONENTS = 2 # Максимальное количество оппонентов для обработки
MAX_CARDS_IN_ROW = {'front': 3, 'middle': 5, 'back': 5}
# Макс карт на руке для обработки сетью (паддинг до 5)
MAX_CARDS_TO_PLAY_NN = 5
TOTAL_SLOTS = MAX_CARDS_IN_ROW['front'] + MAX_CARDS_IN_ROW['middle'] + MAX_CARDS_IN_ROW['back'] # 13

# Новое пространство действий: 3 для сброса + 3 для размещения в ряд
ACTION_SPACE_DIM = 3 + 3 # 6
DISCARD_ACTION_OFFSET = 0
PLACEMENT_ACTION_OFFSET = 3

# Индексы для фаз хода (добавляются в game_state)
TURN_PHASE_START_ROUND_1 = 0.0 # Начало раунда 1 (размещение 1 из 5)
TURN_PHASE_PLACE_1 = 1.0 # Размещение 1-й карты (после discard или 1-я из 5)
TURN_PHASE_PLACE_2 = 2.0 # Размещение 2-й карты (после discard или 2-я из 5)
TURN_PHASE_PLACE_3 = 3.0 # Размещение 3-й из 5
TURN_PHASE_PLACE_4 = 4.0 # Размещение 4-й из 5
TURN_PHASE_PLACE_5 = 5.0 # Размещение 5-й из 5
TURN_PHASE_DISCARD = 6.0 # Фаза сброса (раунды 2+)

# --- Отображение карт ---
# Создаем словарь: treys_card_int -> nn_embedding_index (1-52)
# treys использует битовые маски, нам нужны последовательные индексы
_CARD_RANK_STR = '23456789TJQKA'
_CARD_SUIT_STR = 'shdc' # Spades, Hearts, Diamonds, Clubs по treys

def get_treys_card_int_map():
    """Создает словарь отображения int карты treys в индекс 1-52."""
    card_map = {}
    deck = [Card.new(r + s) for r in _CARD_RANK_STR for s in _CARD_SUIT_STR]
    for i, card_int in enumerate(deck):
        card_map[card_int] = i + 1

    if len(card_map) != NUM_CARDS:
         print(f"Warning: Generated card map has {len(card_map)} cards, expected {NUM_CARDS}. Check treys library.")
         # Простой запасной вариант, если treys выдает что-то странное
         card_map = {card_int: i + 1 for i, card_int in enumerate(deck[:NUM_CARDS])}

    return card_map

TREYS_INT_TO_NN_IDX = get_treys_card_int_map()

def card_to_nn_idx(card_int: Optional[int]) -> int:
    """Преобразует int карты treys в индекс для эмбеддинга (1-52) или 0 для None/паддинга."""
    if card_int is None:
        return CARD_PAD_IDX
    return TREYS_INT_TO_NN_IDX.get(card_int, CARD_PAD_IDX) # Возвращает 0 если карта не найдена


class OFCFeatureExtractor(nn.Module):
    """
    Модуль для преобразования состояния игры OFC в векторное представление (признаки).
    Версия 3.1: Адаптирована под последовательные решения в ходе.
    """
    def __init__(self, embedding_dim: int = 32, conv_out_channels: int = 64, game_state_dim: int = 63, game_state_fc_out: int = 128): # game_state_dim уточнена
        super(OFCFeatureExtractor, self).__init__()
        self.embedding_dim = embedding_dim
        self.conv_out_channels = conv_out_channels
        self.game_state_fc_out = game_state_fc_out

        self.card_embeddings = nn.Embedding(NUM_CARDS + 1, embedding_dim, padding_idx=CARD_PAD_IDX)

        self.conv_front = nn.Conv1d(embedding_dim, conv_out_channels, kernel_size=MAX_CARDS_IN_ROW['front'], padding=0)
        self.conv_middle = nn.Conv1d(embedding_dim, conv_out_channels, kernel_size=MAX_CARDS_IN_ROW['middle'], padding=0)
        self.conv_back = nn.Conv1d(embedding_dim, conv_out_channels, kernel_size=MAX_CARDS_IN_ROW['back'], padding=0)

        # Обработка to_play с паддингом до MAX_CARDS_TO_PLAY_NN
        self.to_play_fc = nn.Linear(MAX_CARDS_TO_PLAY_NN * embedding_dim, conv_out_channels * 2)
        self.game_state_fc = nn.Linear(game_state_dim, game_state_fc_out)

        # Расчет итоговой размерности признаков
        self._feature_dim = (3 * conv_out_channels) + \
                            (conv_out_channels * 2) + \
                            (MAX_OPPONENTS * 3 * conv_out_channels) + \
                            game_state_fc_out
        # print(f"Feature extractor initialized. Feature dim: {self._feature_dim}")

    def _process_board(self, cards_tensor: torch.Tensor, row_name: str) -> torch.Tensor:
        """Обрабатывает один ряд карт (уже тензор индексов)."""
        # --- ПРЕОБРАЗОВАНИЕ ТИПА ---
        # Убедимся, что входной тензор имеет тип Long
        if cards_tensor.dtype != torch.long:
            cards_tensor = cards_tensor.long()
        # --- КОНЕЦ ПРЕОБРАЗОВАНИЯ ---

        emb = self.card_embeddings(cards_tensor) # Теперь cards_tensor точно Long
        emb_p = emb.permute(0, 2, 1)

        if row_name == 'front': conv_layer = self.conv_front
        elif row_name == 'middle': conv_layer = self.conv_middle
        elif row_name == 'back': conv_layer = self.conv_back
        else: raise ValueError(f"Unknown row name: {row_name}")

        features = F.relu(conv_layer(emb_p)).squeeze(2)
        return features

    def forward(self, obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Выполняет прямое распространение для извлечения признаков из словаря наблюдений.
        """
        batch_size = obs['game_state'].size(0)

        # 1. Обработка карт игрока
        player_front_feat = self._process_board(obs['player_front'], 'front')
        player_middle_feat = self._process_board(obs['player_middle'], 'middle')
        player_back_feat = self._process_board(obs['player_back'], 'back')
        player_board_features = torch.cat([player_front_feat, player_middle_feat, player_back_feat], dim=1)

        # 2. Обработка карт 'to_play'
        # --- ПРЕОБРАЗОВАНИЕ ТИПА ---
        to_play_tensor = obs['to_play']
        if to_play_tensor.dtype != torch.long:
            to_play_tensor = to_play_tensor.long()
        # --- КОНЕЦ ПРЕОБРАЗОВАНИЯ ---
        to_play_emb = self.card_embeddings(to_play_tensor) # Теперь to_play_tensor точно Long
        to_play_flat = to_play_emb.view(batch_size, -1)
        to_play_features = F.relu(self.to_play_fc(to_play_flat))

        # 3. Обработка карт оппонентов
        opponent_features_list = []
        for i in range(MAX_OPPONENTS):
            # Передаем тензоры в _process_board, который сам преобразует тип
            opp_f = self._process_board(obs[f'opp{i}_front'], 'front')
            opp_m = self._process_board(obs[f'opp{i}_middle'], 'middle')
            opp_b = self._process_board(obs[f'opp{i}_back'], 'back')
            opp_board_features = torch.cat([opp_f, opp_m, opp_b], dim=1)
            opponent_features_list.append(opp_board_features)
        all_opponent_features = torch.cat(opponent_features_list, dim=1)

        # 4. Обработка состояния игры (остается float)
        game_state_features = F.relu(self.game_state_fc(obs['game_state']))

        # 5. Объединение всех признаков
        combined_features = torch.cat([
            player_board_features,
            to_play_features,
            all_opponent_features,
            game_state_features
        ], dim=1)

        return combined_features

    @property
    def feature_dim(self) -> int:
        return self._feature_dim


class OFCPolicyNetwork(nn.Module):
    """
    Содержит Actor и Critic сети, работающие поверх извлеченных признаков.
    Используется внутри кастомной политики SB3.
    """
    def __init__(self, feature_dim: int, action_dim: int = ACTION_SPACE_DIM):
        super().__init__()
        self.action_dim = action_dim

        # Actor (политика) - определяет вероятность выбора действия (0-5)
        self.actor_net = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim) # Выход - логиты для 6 действий
        )

        # Critic (оценка состояния) - оценивает текущее состояние игры
        self.critic_net = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1) # Выход - одно число (оценка)
        )

    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Возвращает логиты действий и оценку состояния."""
        return self.actor_net(features), self.critic_net(features)

    def forward_actor(self, features: torch.Tensor) -> torch.Tensor:
         """Возвращает только логиты действий."""
         return self.actor_net(features)

    def forward_critic(self, features: torch.Tensor) -> torch.Tensor:
         """Возвращает только оценку состояния."""
         return self.critic_net(features)


class OFCActionEncoder:
    """
    Класс для кодирования/декодирования действий в OFC (Версия 3.1).
    Действия: 0-2 (discard card index), 3 (place Front), 4 (place Middle), 5 (place Back).
    """
    def __init__(self):
        self.action_space_dim = ACTION_SPACE_DIM # 6
        self.placement_action_offset = PLACEMENT_ACTION_OFFSET # 3
        self.row_map = { # Соответствие индекса действия (смещенного на 3) имени ряда
            0: 'front',
            1: 'middle',
            2: 'back'
        }
        self.row_to_action_idx = {v: k + self.placement_action_offset for k, v in self.row_map.items()} # {'front': 3, 'middle': 4, 'back': 5}


    def encode_discard(self, card_index_to_discard: int) -> int:
        """Кодирует действие сброса карты (индексы 0, 1, 2)."""
        if 0 <= card_index_to_discard < self.placement_action_offset:
            return card_index_to_discard
        else:
            raise ValueError(f"Некорректный индекс карты для сброса: {card_index_to_discard}")

    def encode_placement(self, row: str) -> int:
        """Кодирует действие размещения в ряд ('f', 'm', 'b' -> 3, 4, 5)."""
        # Используем 'f', 'm', 'b' как принято в OfcGame
        row_short = row.lower()[0] # 'front' -> 'f'
        internal_row_map = {'f': 'front', 'm': 'middle', 'b': 'back'}
        row_long = internal_row_map.get(row_short)

        if row_long not in self.row_to_action_idx:
            raise ValueError(f"Некорректное имя ряда: {row}")
        return self.row_to_action_idx[row_long]

    def decode_action(self, action_idx: int) -> Dict[str, Any]:
        """
        Декодирует индекс действия (0-5) в информацию о действии.
        Returns:
            Словарь вида:
            {'type': 'discard', 'card_index': int} или
            {'type': 'placement', 'row': str ('front', 'middle', 'back')}
        """
        if not (0 <= action_idx < self.action_space_dim):
             raise ValueError(f"Некорректный индекс действия: {action_idx}")

        if action_idx < self.placement_action_offset: # Действия 0, 1, 2
            return {'type': 'discard', 'card_index': action_idx}
        else: # Действия 3, 4, 5
            row_idx_offset = action_idx - self.placement_action_offset # Получаем индекс ряда 0, 1, 2
            if row_idx_offset not in self.row_map:
                 raise ValueError(f"Ошибка декодирования индекса ряда из action_idx={action_idx}")
            return {'type': 'placement', 'row': self.row_map[row_idx_offset]}


# --- Функция подготовки данных (Версия 3.1) ---

def state_to_tensors(game: Any, # Тип вашего объекта OfcGame
                     player_idx: int,
                     current_turn_phase: float,
                     active_card_idx: int = -1
                    ) -> Dict[str, torch.Tensor]:
    """
    Преобразует состояние игры в словарь тензоров для нейронной сети (Версия 3.1).

    Args:
        game: Объект игры OfcGame.
        player_idx: Индекс игрока.
        current_turn_phase: Флаг текущей фазы (см. константы TURN_PHASE_*).
        active_card_idx: Индекс активной карты в player.to_play (для фаз размещения).

    Returns:
        Словарь тензоров, готовый для подачи в OFCFeatureExtractor.
        Включает 'action_mask' для использования средой Gym.
    """
    player = game.players[player_idx]
    opponents = [p for i, p in enumerate(game.players) if i != player_idx]
    obs_dict = {}

    # --- 1. Карты игрока на доске ---
    for row_name, max_len in MAX_CARDS_IN_ROW.items():
        player_row_cards = getattr(player, row_name, [])
        # Преобразуем int карты treys в nn_idx (1-52) или 0 для паддинга
        card_idxs = [card_to_nn_idx(card) for card in player_row_cards]
        padded_cards = card_idxs + [CARD_PAD_IDX] * (max_len - len(card_idxs))
        obs_dict[f'player_{row_name}'] = torch.tensor(padded_cards[:max_len], dtype=torch.long)

    # --- 2. Карты оппонентов на доске (с паддингом до MAX_OPPONENTS) ---
    for i in range(MAX_OPPONENTS):
        if i < len(opponents):
            opp = opponents[i]
            for row_name, max_len in MAX_CARDS_IN_ROW.items():
                opp_row_cards = getattr(opp, row_name, [])
                card_idxs = [card_to_nn_idx(card) for card in opp_row_cards]
                padded_cards = card_idxs + [CARD_PAD_IDX] * (max_len - len(card_idxs))
                obs_dict[f'opp{i}_{row_name}'] = torch.tensor(padded_cards[:max_len], dtype=torch.long)
        else:
            # Паддинг для отсутствующего оппонента
            for row_name, max_len in MAX_CARDS_IN_ROW.items():
                 obs_dict[f'opp{i}_{row_name}'] = torch.full((max_len,), CARD_PAD_IDX, dtype=torch.long)


    # --- 3. Карты для розыгрыша ('to_play') ---
    to_play_cards_list = getattr(player, 'to_play', [])
    card_idxs = [card_to_nn_idx(card) for card in to_play_cards_list]
    # Паддинг до MAX_CARDS_TO_PLAY_NN (5)
    padded_to_play = card_idxs + [CARD_PAD_IDX] * (MAX_CARDS_TO_PLAY_NN - len(card_idxs))
    obs_dict['to_play'] = torch.tensor(padded_to_play[:MAX_CARDS_TO_PLAY_NN], dtype=torch.long)

    # --- 4. Общее состояние игры (game_state) ---
    state_list = []
    state_list.append(float(getattr(game, 'round', 1))) # Используем 1-based индексацию раундов?
    state_list.append(float(game.current_player_ind == player_idx))
    state_list.append(float(len(to_play_cards_list))) # Реальное кол-во карт на руке

    # Статус фантазии и очки
    state_list.append(float(getattr(player, 'fantasy', False)))
    state_list.append(float(getattr(player, 'stack', 0.0))) # Используем stack или score? У вас stack
    for i in range(MAX_OPPONENTS):
        if i < len(opponents):
             opp = opponents[i];
             state_list.append(float(getattr(opp, 'fantasy', False)))
             state_list.append(float(getattr(opp, 'stack', 0.0)))
        else: state_list.append(0.0); state_list.append(0.0)

    # Оставшаяся колода (бинарный вектор 52)
    # !! Нужен метод game.get_remaining_deck() или аналогичный !!
    # Предположим он возвращает список int карт treys
    # opened_cards = game.opened_cards() # Этот метод у вас есть
    all_player_cards = []
    for p in game.players:
        all_player_cards.extend(getattr(p, 'front', []))
        all_player_cards.extend(getattr(p, 'middle', []))
        all_player_cards.extend(getattr(p, 'back', []))
        all_player_cards.extend(getattr(p, 'to_play', []))
        all_player_cards.extend(getattr(p, 'dead', [])) # Учитываем и dead

    remaining_deck_vector = [1.0] * NUM_CARDS # Индексы 0..51 для вектора
    for card_int in all_player_cards:
        nn_idx = card_to_nn_idx(card_int) # Получаем индекс 1..52
        if 1 <= nn_idx <= NUM_CARDS:
            remaining_deck_vector[nn_idx - 1] = 0.0 # Ставим 0.0 по индексу 0..51

    state_list.extend(remaining_deck_vector)

    # Фаза хода и активная карта
    state_list.append(current_turn_phase)
    state_list.append(float(active_card_idx))

    obs_dict['game_state'] = torch.tensor(state_list, dtype=torch.float32)

    # Проверка размерности game_state
    # 1(round) + 1(is_turn) + 1(n_to_play) + 1(p_fantasy) + 1(p_stack) + M*(1(o_fantasy)+1(o_stack)) + 52(deck) + 1(phase) + 1(active_idx)
    # = 3 + 2 + MAX_OPPONENTS*2 + 52 + 2 = 5 + 4 + 52 + 2 = 63 (для M=2)
    expected_dim = 5 + MAX_OPPONENTS*2 + NUM_CARDS + 2
    actual_dim = obs_dict['game_state'].shape[0]
    assert actual_dim == expected_dim, f"Размерность game_state ({actual_dim}) не совпадает с ожидаемой ({expected_dim})!"

    # --- 5. Маска действий (добавляется здесь для удобства, но используется средой) ---
    # obs_dict['action_mask'] - будет добавлена в OfcEnv._get_obs()

    return obs_dict