import gymnasium as gym
import numpy as np
from typing import Optional, Tuple, Dict, Any, List

# Импортируем из ваших файлов и новой архитектуры
from ofcgame import OfcGame, OfcPlayer # Убедитесь, что импорты OfcGame корректны
from ofc_agent import OfcRandomAgent # Ваш случайный агент
from ofc_neural_network_architecture import (
    state_to_tensors, OFCActionEncoder, ACTION_SPACE_DIM, MAX_CARDS_IN_ROW,
    TURN_PHASE_START_ROUND_1, TURN_PHASE_PLACE_1, TURN_PHASE_PLACE_2,
    TURN_PHASE_PLACE_3, TURN_PHASE_PLACE_4, TURN_PHASE_PLACE_5,
    TURN_PHASE_DISCARD, MAX_OPPONENTS, CARD_PAD_IDX, MAX_CARDS_TO_PLAY_NN, NUM_CARDS, DISCARD_ACTION_OFFSET
)
# from treys import Card # Если нужен где-то еще

class OfcEnvV2(gym.Env):
    """
    Версия среды OFC Gym с последовательным принятием решений и
    пространством действий 6 (3 discard + 3 placement).
    """
    metadata = {'render_modes': ['human', 'ansi'], 'render_fps': 1}

    def __init__(self, max_player=2, render_mode=None):
        super().__init__()

        self.max_player = max_player
        # Hero всегда игрок 0 для простоты
        self.hero_idx = 0
        # Кнопка ходит последней, например, если 2 игрока, кнопка 1
        self.button_ind = max_player - 1

        self.game: Optional[OfcGame] = None
        self.opponent_agents = [OfcRandomAgent() for _ in range(max_player - 1)]
        self.action_encoder = OFCActionEncoder()

        # --- Состояние среды для управления под-шагами ---
        self.current_turn_phase: float = TURN_PHASE_START_ROUND_1
        self.active_card_idx: int = -1 # Индекс карты в player.to_play для размещения
        self.cards_placed_this_turn: int = 0

        # --- Определяем пространства ---
        self.action_space = gym.spaces.Discrete(ACTION_SPACE_DIM) # 6 действий

        # Определяем observation_space на основе вывода state_to_tensors
        # Используем Box для тензоров и MultiBinary для маски
        obs_space_dict = {}
        # Карты игрока
        for row, max_len in MAX_CARDS_IN_ROW.items():
            obs_space_dict[f'player_{row}'] = gym.spaces.Box(low=0, high=NUM_CARDS + 1, shape=(max_len,), dtype=np.int64)
        # Карты оппонентов (с паддингом)
        for i in range(MAX_OPPONENTS):
            for row, max_len in MAX_CARDS_IN_ROW.items():
                 obs_space_dict[f'opp{i}_{row}'] = gym.spaces.Box(low=0, high=NUM_CARDS + 1, shape=(max_len,), dtype=np.int64)
        # Карты для игры
        obs_space_dict['to_play'] = gym.spaces.Box(low=0, high=NUM_CARDS + 1, shape=(MAX_CARDS_TO_PLAY_NN,), dtype=np.int64)
        # Состояние игры
        game_state_dim = 5 + MAX_OPPONENTS*2 + NUM_CARDS + 2 # Перепроверьте это число!
        obs_space_dict['game_state'] = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(game_state_dim,), dtype=np.float32)
        # Маска действий
        obs_space_dict['action_mask'] = gym.spaces.MultiBinary(ACTION_SPACE_DIM)

        self.observation_space = gym.spaces.Dict(obs_space_dict)

        # Рендеринг
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

    def _get_obs(self) -> Dict[str, np.ndarray]:
        """Собирает наблюдение для текущего состояния."""
        if self.game is None:
            # Случай до первого reset
             return {k: np.zeros(s.shape, dtype=s.dtype) for k, s in self.observation_space.spaces.items() if k != 'action_mask'}

        # Получаем словарь тензоров PyTorch
        obs_torch_dict = state_to_tensors(
            self.game,
            self.hero_idx,
            self.current_turn_phase,
            self.active_card_idx
        )
        # Конвертируем в NumPy и добавляем маску
        obs_numpy_dict = {k: v.numpy() for k, v in obs_torch_dict.items()}
        obs_numpy_dict['action_mask'] = self._get_action_mask()
        return obs_numpy_dict

    def _get_info(self) -> Dict[str, Any]:
        """Возвращает дополнительную информацию (пока не используется)."""
        return {'round': self.game.round if self.game else 0, 'phase': self.current_turn_phase}

    def _get_action_mask(self) -> np.ndarray:
        """Генерирует маску легальных действий (размер 6)."""
        mask = np.zeros(ACTION_SPACE_DIM, dtype=bool)
        if self.game is None or self.game.is_game_over() or self.game.current_player_ind != self.hero_idx:
            return mask # Нет доступных действий

        player = self.game.current_player()

        if self.current_turn_phase == TURN_PHASE_DISCARD:
            # Доступны только действия сброса 0, 1, 2
            num_to_play = len(player.to_play)
            if num_to_play == 3: # Убедимся, что есть 3 карты для выбора
                 mask[DISCARD_ACTION_OFFSET:DISCARD_ACTION_OFFSET + 3] = True
        else: # Фазы размещения (PLACE_1..5 или START_ROUND_1)
            # Доступны только действия размещения 3, 4, 5
            # Проверяем валидность каждого ряда
            can_place_front = len(player.front) < MAX_CARDS_IN_ROW['front']
            can_place_middle = len(player.middle) < MAX_CARDS_IN_ROW['middle']
            can_place_back = len(player.back) < MAX_CARDS_IN_ROW['back']

            # !! ВАЖНО: Добавить проверку правил порядка рядов, если это последний ход !!
            # Например, если это 13-я карта, нужно проверить player.is_foul() после гипотетического размещения
            # Для простоты пока опустим эту сложную проверку, но для сильного AI она нужна.

            if can_place_front:
                mask[self.action_encoder.row_to_action_idx['front']] = True
            if can_place_middle:
                mask[self.action_encoder.row_to_action_idx['middle']] = True
            if can_place_back:
                mask[self.action_encoder.row_to_action_idx['back']] = True

        return mask

    def reset(self, seed=None, options=None) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        super().reset(seed=seed) # Важно для воспроизводимости

        self.game = OfcGame(game_id=0, max_player=self.max_player, button=self.button_ind, hero=self.hero_idx, seed=seed)

        # Начальное состояние для героя (раунд 1)
        self.current_turn_phase = TURN_PHASE_PLACE_1 # Начинаем с размещения 1-й карты
        self.active_card_idx = 0 # Размещаем первую карту из player.to_play
        self.cards_placed_this_turn = 0

        # Если первый ход не у героя, проиграть ходы оппонентов
        self._play_opponent_turns_if_needed()

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def step(self, action: int) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        if self.game is None:
             raise RuntimeError("Cannot call step() before reset()")

        terminated = False
        truncated = False
        reward = 0.0
        info = {} # Инициализируем инфо

        # --- Проверка: Сейчас ход героя? ---
        if self.game.current_player_ind != self.hero_idx:
             # ... (обработка ошибки или возврат 0 reward) ...
             print(f"Warning: step() called when it's not hero's turn ({self.game.current_player_ind})")
             obs = self._get_obs() # Получаем актуальное наблюдение
             return obs, 0.0, terminated, truncated, info

        # --- Проверка и установка начальной фазы (если нужно) ---
        player = self.game.players[self.hero_idx]
        if self.active_card_idx == -1 and self.current_turn_phase != TURN_PHASE_DISCARD:
            # ... (логика определения начальной фазы PLACE_1 или DISCARD) ...
            # Эта логика теперь должна вызываться реже, только если ход действительно ТОЛЬКО ЧТО перешел
            pass # Убрал код отсюда, он будет ниже

        # --- 1. Проверяем легальность действия ---
        current_mask = self._get_action_mask() # Получаем маску для ТЕКУЩЕЙ фазы
        if not current_mask[action]:
            # ... (обработка нелегального действия) ...
            reward = -10.0; terminated = True
            info['error'] = f'Illegal action {action}. Mask: {np.where(current_mask)[0]}'
            # ВАЖНО: Возвращаем НАБЛЮДЕНИЕ ДО НЕПРАВИЛЬНОГО ДЕЙСТВИЯ
            # Нужно получить obs ДО изменения состояния
            # obs_before_illegal_action = self._get_obs() # Получаем до ошибки
            # Либо просто возвращаем последнее валидное obs? Это сложно отследить.
            # Правильнее всего, чтобы SB3 сам обработал эту ситуацию, вернув obs из пред. шага.
            # Поэтому просто возвращаем текущее obs, хотя оно может быть некорректным.
            return self._get_obs(), reward, terminated, truncated, info # Возвращаем obs на момент ошибки


        # --- 2. Выполняем действие героя ---
        decoded_action = self.action_encoder.decode_action(action)
        is_hero_turn_complete = False

        if decoded_action['type'] == 'discard':
            # ... (выполняем сброс, обновляем player.to_play, player.dead) ...
            card_index = decoded_action['card_index']
            discarded_card = player.to_play.pop(card_index)
            if hasattr(player, 'dead'): player.dead.append(discarded_card)

            # --- ОБНОВЛЯЕМ ФАЗУ СРАЗУ ---
            self.current_turn_phase = TURN_PHASE_PLACE_1
            self.active_card_idx = 0 # Будем размещать первую из оставшихся
            self.cards_placed_this_turn = 0

        elif decoded_action['type'] == 'placement':
            # ... (выполняем размещение, обновляем ряды, player.to_play) ...
            target_row_name = decoded_action['row']
            active_card = player.to_play.pop(self.active_card_idx) # active_card_idx ДОЛЖЕН быть корректным здесь
            target_row_list = getattr(player, target_row_name)
            target_row_list.append(active_card)
            self.cards_placed_this_turn += 1

            # Проверяем, завершен ли полный ход героя
            current_round = self.game.round
            if current_round == 1:
                if self.cards_placed_this_turn == 5: is_hero_turn_complete = True
            else:
                if self.cards_placed_this_turn == 2: is_hero_turn_complete = True

            # --- ОБНОВЛЯЕМ ФАЗУ СРАЗУ ---
            if not is_hero_turn_complete:
                self.current_turn_phase += 1 # PLACE_1 -> PLACE_2, etc.
                self.active_card_idx = 0 # Следующая карта всегда первая в оставшемся списке
            else:
                # Ход героя завершен, сбрасываем под-шаги
                self.cards_placed_this_turn = 0
                self.active_card_idx = -1
                self.current_turn_phase = -1.0 # Неопределенная фаза до проверки в след. step

        # --- 3. Если полный ход героя завершен, передаем ход и играем оппонентов ---
        if is_hero_turn_complete:
            # --- Логика передачи хода и игры оппонентов ---
            # ВАЖНО: Эта логика должна корректно обновить self.game.current_player_ind
            # и, возможно, self.game.round, а также раздать карты, если нужно.
            self.game._next_player() # Предполагаем, что этот метод передает ход

            # Проверяем переход раунда
            if self.hero_idx == self.game.button_ind:
                if hasattr(self.game, '_next_round'):
                    # Проверка на макс. раунд (например, 9 размещений = 5 раундов в HU)
                    num_rounds = 1 + (13 - 5) // 2 # 1 + 8 // 2 = 5 раундов для HU
                    if self.game.round < num_rounds: # Уточнить для 3-max
                         self.game._next_round() # Включает раздачу карт в OfcGame
                         print(f"DEBUG: Advanced to round {self.game.round}")
                else:
                     print("Warning: OfcGame missing _next_round.")

            # Играем оппонентов
            if not self.game.is_game_over():
                self._play_opponent_turns_if_needed() # Этот метод тоже должен вызывать _next_player/_next_round

            # --- ПОСЛЕ ходов оппонентов, определяем НАЧАЛЬНУЮ фазу для СЛЕДУЮЩЕГО хода героя ---
            # Эта логика нужна здесь, чтобы observation в конце step был корректным
            if not self.game.is_game_over() and self.game.current_player_ind == self.hero_idx:
                player = self.game.players[self.hero_idx] # Обновляем ссылку на игрока
                num_to_play = len(player.to_play)
                if num_to_play == 5: # Новый раунд 1 (не должно быть, но на всякий случай)
                     self.current_turn_phase = TURN_PHASE_PLACE_1
                     self.active_card_idx = 0
                     self.cards_placed_this_turn = 0
                elif num_to_play == 3: # Новый раунд 2+
                     self.current_turn_phase = TURN_PHASE_DISCARD
                     self.active_card_idx = -1
                     self.cards_placed_this_turn = 0
                elif num_to_play == 0: # Игра окончена для игрока? Или ошибка?
                     # Если игра не окончена глобально, это ошибка
                     if not self.game.is_game_over():
                         print(f"Error: Hero's turn but no cards to play after opponent turns!")
                         terminated = True; reward = -30; info['error'] = "No cards after opponent turn"
                         # Возвращаем obs на момент ошибки
                         return self._get_obs(), reward, terminated, truncated, info
                else: # 1, 2, 4 карты - не должно быть в начале хода
                     print(f"Error: Unexpected card count ({num_to_play}) at start of hero turn after opponents.")
                     terminated = True; reward = -30; info['error'] = f"Unexpected card count {num_to_play}"
                     return self._get_obs(), reward, terminated, truncated, info
            else: # Ход не героя или игра окончена
                 self.current_turn_phase = -1.0 # Сбрасываем фазу, если ход не у героя
                 self.active_card_idx = -1


        # --- 4. Проверяем, закончилась ли игра ГЛОБАЛЬНО ---
        if self.game.is_game_over():
            terminated = True
            # ... (расчет финальной награды) ...
            if hasattr(player, 'calc_score_single'):
                 reward = player.calc_score_single() if not player.is_foul() else -6.0
            else:
                 reward = self.game.calc_hero_score()
            info['final_reward'] = reward # Добавляем в инфо


        # --- 5. Получаем итоговое наблюдение (с уже обновленной фазой/индексом) ---
        observation = self._get_obs()
        info['phase'] = self.current_turn_phase # Добавляем текущую фазу в инфо
        info['active_card_idx'] = self.active_card_idx

        if self.render_mode == "human": self._render_frame()
        # elif self.render_mode == "ansi": print(self.game) # Можно добавить для отладки

        return observation, reward, terminated, truncated, info


    def _play_opponent_turns_if_needed(self):
        """Проигрывает ходы оппонентов, пока ход не вернется к герою или игра не закончится."""
        if self.game is None: return

        while self.game.current_player_ind != self.hero_idx and not self.game.is_game_over():
            opp_idx = self.game.current_player_ind
            opp_agent = self.opponent_agents[opp_idx - 1] # Индексы агентов 0..N-2 для оппонентов 1..N-1
            opponent = self.game.current_player()

            # Оппонент принимает решение (пока случайное)
            # Случайному агенту нужно передать ВСЕ возможные действия (весь ход целиком)
            # Это расхождение с нашим последовательным подходом!
            # Нужно либо переписать случайного агента под последовательные шаги,
            # либо сделать так, чтобы он возвращал полный ход за раз.
            # Пока сделаем полный ход для простоты (но это не идеально).

            # --- Начало: Логика полного хода для случайного оппонента ---
            num_to_play = len(opponent.to_play)
            if num_to_play == 5: # Раунд 1
                # Выбираем случайный *валидный* шаблон размещения из первых 232
                from ofc_encoder import legal_actions as old_legal_actions, ACTION_SPACE as OLD_ACTION_SPACE, action_to_dict as old_action_to_dict
                legal_action_indices = old_legal_actions(opponent) # Используем старый кодер
                if not legal_action_indices: # Нет легальных ходов? Маловероятно
                    print(f"Warning: Opponent {opp_idx} has no legal actions in round 1?")
                    break # Выход из цикла
                random_action_id = self.np_random.choice(legal_action_indices)
                action_dict = old_action_to_dict(random_action_id, opponent.to_play)
                self.game.play(action_dict) # Используем старый метод play

            elif num_to_play == 3: # Раунды 2+
                # Выбираем случайный *валидный* шаблон (1 сброс, 2 размещения) из последних 27
                from ofc_encoder import legal_actions as old_legal_actions, ACTION_SPACE as OLD_ACTION_SPACE, action_to_dict as old_action_to_dict
                legal_action_indices = old_legal_actions(opponent) # Используем старый кодер
                if not legal_action_indices:
                    print(f"Warning: Opponent {opp_idx} has no legal actions in round {self.game.round}?")
                    break
                random_action_id = self.np_random.choice(legal_action_indices)
                action_dict = old_action_to_dict(random_action_id, opponent.to_play)
                self.game.play(action_dict) # Используем старый метод play

            elif num_to_play == 0 and not self.game.is_game_over():
                 # Ход оппонента, но у него нет карт - значит, игра должна была раздать
                 # Или ошибка логики
                 print(f"Warning: Opponent {opp_idx}'s turn but no cards to play?")
                 # Просто передаем ход дальше, предполагая, что игра раздаст позже
                 self.game.current_player_ind = (self.game.current_player_ind + 1) % self.max_player
                 continue # Пропускаем остаток цикла
            else:
                 # Неожиданное количество карт у оппонента
                 print(f"Warning: Opponent {opp_idx} has unexpected number of cards: {num_to_play}")
                 break
            # --- Конец: Логика полного хода для случайного оппонента ---

            # После хода оппонента проверяем конец игры снова
            if self.game.is_game_over():
                break

            # Если игра не окончена, и ход все еще не у героя, продолжаем цикл

    def render(self):
        if self.render_mode == "human":
             self._render_frame()
        elif self.render_mode == "ansi":
            if self.game:
                 print(self.game)

    def _render_frame(self):
         # Реализация рендеринга (например, с Pygame или Matplotlib)
         # Пока просто выводим в консоль
         if self.game:
            print(self.game)
         else:
            print("Game not initialized.")

    def close(self):
        # Освобождение ресурсов, если они были заняты (например, окно Pygame)
        pass

    def get_internal_state(self):
        """Возвращает внутреннее состояние для тестирования."""
        return {
            "phase": self.current_turn_phase,
            "active_idx": self.active_card_idx,
            "placed_count": self.cards_placed_this_turn,
            "current_player": self.game.current_player_ind if self.game else -1,
            "round": self.game.round if self.game else 0
        }
