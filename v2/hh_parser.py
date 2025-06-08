import json
import os
from typing import List, Dict, Any, Optional, Tuple
from treys import Card # Предполагаем, что у вас есть способ конвертировать строки в объекты Card или int
_CARD_RANK_STR = '23456789TJQKA'
_CARD_SUIT_STR = 'shdc' # Spades, Hearts, Diamonds, Clubs по treys

# --- Конвертация строки карты из HH в объект/int ---
# Вам нужно будет реализовать эту функцию или адаптировать существующую
# Например, если 'As0' -> Card.new('As') или соответствующий int
def hh_string_to_card_int(card_str_with_round: str) -> Optional[int]:
    """Преобразует строку типа 'As0' в int представление карты (без раунда)."""
    if not card_str_with_round or len(card_str_with_round) < 2:
        return None
    card_part = card_str_with_round[:-1] # Убираем цифру раунда
    try:
        # Это зависит от того, как вы работаете с картами (treys.Card.new или др.)
        # Для treys, Card.new() принимает 'As', 'Td' и т.д.
        return Card.new(card_part) # Возвращает int представление из treys
    except Exception: # Более конкретное исключение для treys, если есть
        # print(f"Warning: Could not parse card string: {card_str_with_round}")
        return None

class HHParser:
    def __init__(self, hh_files_directory: str):
        self.hh_files_directory = hh_files_directory
        self.parsed_hands = []

    def _is_valid_rule(self, rule_str: str) -> bool:
        rule_str_lower = rule_str.lower()
        return "classic" in rule_str_lower or "nojokers" in rule_str_lower

    def parse_files(self):
        """Читает все .txt файлы из директории и парсит их."""
        for filename in os.listdir(self.hh_files_directory):
            if filename.endswith(".txt"):  # или другое расширение, если необходимо
                filepath = os.path.join(self.hh_files_directory, filename)
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        content = f.read()
                        # Разделяем контент на отдельные JSON-объекты
                        # Каждый JSON-объект отделен двумя переносами строк
                        json_blocks = content.strip().split('\n\n\n')

                        for block in json_blocks:
                            block = block.strip()
                            if not block:
                                continue
                            try:
                                hand_json = json.loads(block)
                                self._process_hand_json(hand_json)
                            except json.JSONDecodeError as json_err:
                                print(f"Warning: Could not decode JSON block in {filename}: {str(json_err)}")
                                print(f"Block starts with: {block[:100]}...")
                except Exception as e:
                    print(f"Error reading file {filename}: {e}")
        print(f"Parsed {len(self.parsed_hands)} valid hands for curriculum learning.")

    def _process_hand_json(self, hand_json: Dict[str, Any]):
        """Обрабатывает один JSON объект раздачи."""
        if not hand_json.get("joined", False):
            return

        if not self._is_valid_rule(hand_json.get("rules", "")):
            return

        hand_data_dict = hand_json.get("handData", {})
        if not hand_data_dict:
            return

        # handData содержит один ключ (ID раздачи) и значение - список игроков
        for hand_id, players_data in hand_data_dict.items():
            hero_player_data = None
            opponent_player_data_list = []

            for player_entry in players_data:
                if hero_player_data is None and player_entry.get("hero", False):
                    hero_player_data = player_entry
                else:
                    opponent_player_data_list.append(player_entry)

            if not hero_player_data or hero_player_data.get("inFantasy", False):
                continue # Героя нет или он в фантазии

            # Здесь у нас есть валидная раздача с героем не в фантазии
            # Нужно извлечь состояния игры для каждого раунда
            # Это самая сложная часть: нужно симулировать игру по ходам
            # или воссоздавать состояние на начало каждого хода героя.
            # Пока мы просто сохраним данные героя и оппонентов для этой раздачи.
            # Для Curriculum Learning нам нужно будет "развернуть" эту историю.
            self.parsed_hands.append({
                "hand_id": hand_id,
                "hero": hero_player_data,
                "opponents": opponent_player_data_list,
                "stakes": hand_json.get("stakes"),
                "rules": hand_json.get("rules")
            })

    def _parse_cards_from_string(self, cards_str: str) -> List[Tuple[Optional[int], int]]:
        """Парсит строку типа 'As0 Td1' в список [(card_int, round_num)]."""
        parsed = []
        if not cards_str: return parsed
        for card_item_str in cards_str.split(' '):
            card_int = hh_string_to_card_int(card_item_str)
            round_num = int(card_item_str[-1]) # Последний символ - номер раунда
            parsed.append((card_int, round_num))
        return parsed

    def get_states_for_round(self, target_placement_round: int) -> List[Dict[str, Any]]:
        """
        Генерирует начальные состояния для обучения, когда ход героя
        и он должен сделать свое размещение в `target_placement_round`.
        target_placement_round: 1 (первые 5 карт), 2 (1-я из 3), 3 (2-я из 3), ..., 5.
        (Нумерация раундов размещения, а не игровых раундов (1-5))
        """
        game_initial_states = []
        for hand in self.parsed_hands:
            # --- Воссоздание состояния игры на начало target_placement_round ---
            # Это требует симуляции или аккуратного разбора истории
            # Примерная логика:
            # 1. Определить, какие карты были на доске у героя и оппонентов *до* этого раунда.
            # 2. Определить, какие карты были сброшены героем *до* этого раунда.
            # 3. Определить, какие карты герой получает на руки *в этом* раунде.

            # Словарь для хранения состояния игры, похожего на то, что ожидает state_to_tensors
            # (но для OfcGame, а не напрямую для NN)
            current_game_snapshot = {
                "players": [],
                "round": 0, # Игровой раунд (1-5 или до 9)
                "current_player_ind": -1, # Индекс героя, если его ход
                "button_ind": -1,
                "deck_state": [], # Оставшиеся карты в колоде
                "hero_to_play_cards": [] # Карты, которые герой должен разыграть на этом шаге
            }

            # --- Логика для героя ---
            hero_data = hand["hero"]
            hero_board_at_round_start = {'front': [], 'middle': [], 'back': []}
            hero_dead_at_round_start = []

            # Разбор рядов героя
            row_strings = hero_data.get("rows", "").split('/')
            for i, row_name in enumerate(["front", "middle", "back"]):
                if i < len(row_strings):
                    cards_in_row = self._parse_cards_from_string(row_strings[i])
                    for card_int, placement_r in cards_in_row:
                        # placement_r здесь - это номер раунда размещения (0-4)
                        # target_placement_round у нас тоже 0-4 (или 1-5, нужно согласовать)
                        # Предположим target_placement_round 0-4.
                        # Раунд 0 = первые 5 карт. Раунды 1-4 = по 3 карты (2 кладутся, 1 сброс).
                        # Но в HH у вас нумерация раундов 0, 1, 2, 3, 4.
                        # Давайте считать, что target_placement_round в HH-терминах (0-4).
                        # 0 - первые 5 карт. 1 - первые 3 карты 2-го игрового раунда, и т.д.
                        if placement_r < target_placement_round:
                            if card_int is not None: hero_board_at_round_start[row_name].append(card_int)

            # Разбор сброса героя
            dead_cards_hero = self._parse_cards_from_string(hero_data.get("dead", ""))
            for card_int, placement_r in dead_cards_hero:
                 if placement_r < target_placement_round: # Сброшенные до текущего раунда
                     if card_int is not None: hero_dead_at_round_start.append(card_int)


            # Определяем карты для розыгрыша героем в target_placement_round
            # Это самая сложная часть, так как HH не говорит явно, какие карты были "на руках"
            # перед каждым решением. Он показывает, куда карта ПОШЛА и в каком раунде.
            # Нам нужно будет восстановить "руку".
            # Пример: если target_placement_round = 0 (первый ход), то "рука" - это 5 карт,
            # которые пошли в слоты с отметкой раунда 0.
            # Если target_placement_round = 1 (второй игровой раунд), то "рука" - это 3 карты:
            #   - две, которые пошли в слоты с отметкой раунда 1
            #   - одна, которая пошла в dead с отметкой раунда 1

            hero_hand_for_this_round = []
            all_hero_placed_cards_this_round = [] # Карты, размещенные героем в target_placement_round
            # Собираем все карты героя, которые он разместил или сбросил в target_placement_round
            for row_name in ["front", "middle", "back"]:
                row_str = hero_data.get("rows", "").split('/')[ ["front", "middle", "back"].index(row_name) ] if len(hero_data.get("rows", "").split('/')) > ["front", "middle", "back"].index(row_name) else ""
                for card_int, r in self._parse_cards_from_string(row_str):
                    if r == target_placement_round and card_int is not None:
                        all_hero_placed_cards_this_round.append(card_int)
            for card_int, r in dead_cards_hero:
                 if r == target_placement_round and card_int is not None:
                     all_hero_placed_cards_this_round.append(card_int) # Добавляем и сброшенную карту

            # Убедимся, что количество карт соответствует раунду
            expected_cards_in_hand = 5 if target_placement_round == 0 else 3
            if len(all_hero_placed_cards_this_round) != expected_cards_in_hand:
                print(f"Warning: Hand {hand['hand_id']}, Hero, Round {target_placement_round}: Expected {expected_cards_in_hand} cards, found {len(all_hero_placed_cards_this_round)}")
                continue # Пропускаем это состояние, если данные неполные/неконсистентные

            current_game_snapshot["hero_to_play_cards"] = all_hero_placed_cards_this_round

            # --- Аналогично для оппонентов (доски) ---
            # ... (нужно будет заполнить boards оппонентов до target_placement_round)
            opponent_boards_at_round_start = []
            for opp_data in hand["opponents"]:
                opp_board = {'front': [], 'middle': [], 'back': [], 'dead': []} # Оппоненты тоже могут сбрасывать, если это Pineapple
                # Парсим ряды оппонента аналогично герою
                row_strings_opp = opp_data.get("rows", "").split('/')
                for i, row_name in enumerate(["front", "middle", "back"]):
                    if i < len(row_strings_opp):
                        cards_in_row = self._parse_cards_from_string(row_strings_opp[i])
                        for card_int, placement_r in cards_in_row:
                            if placement_r < target_placement_round: # Карты, положенные до этого раунда
                                if card_int is not None: opp_board[row_name].append(card_int)
                # Парсим dead оппонента
                dead_cards_opp = self._parse_cards_from_string(opp_data.get("dead", ""))
                for card_int, placement_r in dead_cards_opp:
                    if placement_r < target_placement_round:
                        if card_int is not None: opp_board['dead'].append(card_int)
                opponent_boards_at_round_start.append(opp_board)


            # --- Заполнение current_game_snapshot ---
            # Игровой раунд: (0->1, 1->2, 2->3, 3->4, 4->5)
            current_game_snapshot["round"] = target_placement_round + 1
            # Индексы игроков и кнопка (нужно определить из hero_data["orderIndex"] и кол-ва игроков)
            num_players = 1 + len(hand["opponents"])
            hero_order_idx = hero_data["orderIndex"]
            # current_game_snapshot["current_player_ind"] = hero_order_idx # Если orderIndex это и есть индекс в списке players
            # current_game_snapshot["button_ind"] = ... # Нужно знать, кто баттон в этой раздаче

            # Собираем всех игроков для snapshot
            # Сначала герой
            hero_snapshot_player = {
                "name": hero_data["playerName"], "hero": True, "fantasy": False,
                "front": hero_board_at_round_start["front"],
                "middle": hero_board_at_round_start["middle"],
                "back": hero_board_at_round_start["back"],
                "dead": hero_dead_at_round_start,
                "to_play": current_game_snapshot["hero_to_play_cards"], # Карты для текущего решения
                "stack": 0 # или из HH, если есть
            }
            # Затем оппоненты
            # Порядок важен, нужно сохранить orderIndex
            all_players_snapshot_temp = [None] * num_players
            all_players_snapshot_temp[hero_order_idx] = hero_snapshot_player
            current_game_snapshot["current_player_ind"] = hero_order_idx


            for i, opp_data in enumerate(hand["opponents"]):
                opp_snapshot_player = {
                    "name": opp_data["playerName"], "hero": False, "fantasy": opp_data["inFantasy"],
                    "front": opponent_boards_at_round_start[i]["front"],
                    "middle": opponent_boards_at_round_start[i]["middle"],
                    "back": opponent_boards_at_round_start[i]["back"],
                    "dead": opponent_boards_at_round_start[i]["dead"],
                    "to_play": [], # Карты оппонента на руках нам не нужны для хода героя
                    "stack": 0
                }
                all_players_snapshot_temp[opp_data["orderIndex"]] = opp_snapshot_player

            current_game_snapshot["players"] = all_players_snapshot_temp

            # Состояние колоды: все карты минус те, что уже на досках или в сбросе
            # У ВСЕХ игроков до этого момента
            used_cards = set()
            for p_snap in current_game_snapshot["players"]:
                if p_snap: # Проверка, что слот игрока заполнен
                    for row_list in [p_snap["front"], p_snap["middle"], p_snap["back"], p_snap["dead"]]:
                        for card_int_val in row_list: used_cards.add(card_int_val)
            # Добавляем карты, которые герой держит на руках (они тоже вышли из колоды)
            for card_int_val in current_game_snapshot["hero_to_play_cards"]: used_cards.add(card_int_val)

            full_deck_ints = [Card.new(r + s) for r in _CARD_RANK_STR for s in _CARD_SUIT_STR] # Генерируем полную колоду
            current_game_snapshot["deck_state"] = [c for c in full_deck_ints if c not in used_cards]

            game_initial_states.append(current_game_snapshot)

        return game_initial_states

# Пример использования (нужно будет вызывать из основного скрипта)
if __name__ == "__main__":
    parser = HHParser(hh_files_directory="D:\\develop\\temp\\poker\\Eureka\\tmp")
    parser.parse_files()
    # Получить состояния для начала 2-го игрового раунда (когда герой получает 3 карты)
    # Это соответствует target_placement_round = 1 в терминах HH (0,1,2,3,4)
    round_2_states = parser.get_states_for_round(target_placement_round=1)
    print(f"Generated {len(round_2_states)} states for round 2 (placement round 1 in HH terms).")
    if round_2_states:
        print("Example state:", round_2_states[0])

    # Получить состояния для 5-го игрового раунда (последнее размещение 2 карт)
    # Это target_placement_round = 4
    last_round_states = parser.get_states_for_round(target_placement_round=4)
    print(f"Generated {len(last_round_states)} states for final placement round (placement round 4 in HH terms).")
    if last_round_states:
        print("Example state (final round):", last_round_states[0])