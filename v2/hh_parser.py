import json
import os
from typing import List, Dict, Any, Optional, Tuple
from treys import Card # Предполагаем, что у вас есть способ конвертировать строки в объекты Card или int

from v2.ofc_neural_network_architecture import NUM_CARDS

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
        """Читает все .txt файлы из указанной директории и всех ее поддиректорий (рекурсивно)."""
        print(f"Recursively parsing HH files from: {self.hh_files_directory}")
        # Используем os.walk для рекурсивного обхода
        for root, dirs, files in os.walk(self.hh_files_directory):
            for filename in files:
                if filename.endswith(".hh") or filename.endswith(".txt"):
                    filepath = os.path.join(root, filename)
                    try:
                        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                            content = f.read()

                            # Очистка от символа SUB и других нежелательных символов ---
                            # Символ SUB имеет код 26, или \x1a
                            content = content.replace('\x1a', '')
                            # Можно также удалить нулевые символы, которые тоже иногда вызывают проблемы
                            content = content.replace('\x00', '')
                            # --- КОНЕЦ ОЧИСТКИ ---
                            # Если после очистки контент пуст, пропускаем
                            if not content.strip():
                                continue

                            # Если JSON-объекты склеены (например, "}{"), вставляем между ними
                            # уникальный разделитель, по которому потом можно будет разделить строку.
                            # Используем очень маловероятную последовательность символов.
                            UNIQUE_SEPARATOR = "|||JSON_SEPARATOR|||"

                            # Заменяем "}{" на "}|||JSON_SEPARATOR|||{"
                            # Также обрабатываем случаи с пробелами/переносами строк: "} \n\n {"
                            import re
                            # Используем регулярное выражение для поиска '}' за которым могут следовать пробельные символы и потом '{'
                            # re.DOTALL позволяет '.' совпадать с переносом строки
                            content_with_separators = re.sub(r'}\s*{', f'}}{UNIQUE_SEPARATOR}{{', content,
                                                             flags=re.DOTALL)

                            json_blocks = content_with_separators.split(UNIQUE_SEPARATOR)
                            # --- КОНЕЦ НОВОЙ ЛОГИКИ ---

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
        print(f"Finished parsing. Total {len(self.parsed_hands)} valid hands found.")

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
            # --- НОВАЯ ПРОВЕРКА НА ФАНТАЗИЮ ---
            # Проверяем, находится ли ХОТЯ БЫ ОДИН игрок в фантазии в этой раздаче.
            is_any_player_in_fantasy = any(player.get("inFantasy", False) for player in players_data)
            if is_any_player_in_fantasy:
                # print(f"Skipping hand {hand_id}: a player is in Fantasyland.")
                continue # Пропускаем всю эту раздачу
            # --- КОНЕЦ НОВОЙ ПРОВЕРКИ ---

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

    def get_states_for_round(self, target_game_round: int) -> List[Dict[str, Any]]:
        """
        Генерирует начальные состояния для обучения, когда наступает ход героя
        в начале указанного target_game_round (1-5 или 1-9).
        Состояние отражает доски непосредственно ПЕРЕД ПЕРВЫМ ДЕЙСТВИЕМ ГЕРОЯ в этом раунде.
        """
        game_initial_states = []
        for hand_info in self.parsed_hands:  # Используем hand_info вместо hand, чтобы не путать с рукой карт
            # placement_round_hh (0-4 или 0-8) соответствует картам, размещенным В target_game_round
            # Нам нужны карты, размещенные ДО target_game_round для досок.
            # placement_r_hh в HH: 0 (первые 5 карт), 1 (карты 2-го игрового раунда), ..., 4 (карты 5-го игрового раунда)
            # target_game_round: 1 (первый), 2, ..., 5
            # Карты на доске должны быть с placement_r_hh < (target_game_round - 1)
            hh_round_marker_for_board_cards = target_game_round - 1

            current_game_snapshot = {
                "players": [],
                "round": target_game_round,
                "current_player_ind": -1,  # Будет установлен на индекс героя
                "button_ind": -1,  # TODO: Определить из HH или по соглашению
                "deck_state": [],
            }

            hero_data = hand_info["hero"]
            opponents_data = hand_info["opponents"]  # Список словарей данных оппонентов

            num_players = 1 + len(opponents_data)
            hero_order_idx = hero_data["orderIndex"]
            current_game_snapshot["current_player_ind"] = hero_order_idx

            button_index = (hero_order_idx - 1 + num_players) % num_players
            current_game_snapshot["button_ind"] = button_index

            # --- Определение, кто ходил до героя в этом раунде ---
            # Это важно для правильного состояния досок оппонентов.
            # Если target_game_round > 1, некоторые оппоненты могли уже походить.
            # Карты, размещенные в hh_round_marker_for_board_cards (т.е. target_game_round - 1),
            # это карты ТЕКУЩЕГО target_game_round.
            # Для досок оппонентов, ходивших ДО героя в этом раунде, мы должны учесть карты
            # с hh_round_marker_for_board_cards.
            # Для досок оппонентов, ходящих ПОСЛЕ героя, мы должны учесть карты
            # с placement_r_hh < hh_round_marker_for_board_cards.

            all_players_snapshot_temp = [None] * num_players
            processed_players = 0

            # --- Герой ---
            hero_board_at_turn = {'front': [], 'middle': [], 'back': []}
            hero_dead_at_turn = []

            row_strings_hero = hero_data.get("rows", "").split('/')
            for i, row_name in enumerate(["front", "middle", "back"]):
                if i < len(row_strings_hero):
                    cards_in_row = self._parse_cards_from_string(row_strings_hero[i])
                    for card_int, placement_r_hh in cards_in_row:
                        # Карты героя на доске - это те, что были положены в раундах HH *до* текущего раунда HH
                        if placement_r_hh < hh_round_marker_for_board_cards:
                            if card_int is not None: hero_board_at_turn[row_name].append(card_int)

            dead_cards_hero = self._parse_cards_from_string(hero_data.get("dead", ""))
            for card_int, placement_r_hh in dead_cards_hero:
                # Сброс героя в раундах HH *до* текущего раунда HH
                if placement_r_hh < hh_round_marker_for_board_cards:
                    if card_int is not None: hero_dead_at_turn.append(card_int)

            hero_snapshot_player = {
                "name": hero_data["playerName"], "hero": True, "fantasy": False,
                "front": hero_board_at_turn["front"], "middle": hero_board_at_turn["middle"],
                "back": hero_board_at_turn["back"], "dead": hero_dead_at_turn,
                "to_play": [],  # Будет заполнено в env.reset_to_state
                "stack": hero_data.get("stack", 0)  # Если есть инфо о стеке
            }
            if 0 <= hero_order_idx < num_players:
                all_players_snapshot_temp[hero_order_idx] = hero_snapshot_player
                processed_players += 1
            else:
                # print(f"Error: Invalid hero_order_idx {hero_order_idx} for hand {hand_info['hand_id']}")
                continue

            # --- Оппоненты ---
            for opp_data in opponents_data:
                opp_board_at_turn = {'front': [], 'middle': [], 'back': [], 'dead': []}
                opp_order_idx = opp_data["orderIndex"]

                # Определяем, ходил ли этот оппонент ДО героя в текущем target_game_round
                # Это зависит от orderIndex и button_ind (или просто порядка ходов)
                # Предположим, что порядок ходов соответствует orderIndex, начиная с (button_ind + 1) % num_players
                # Это упрощение, реальная логика может быть сложнее.
                # Для Curriculum Learning мы хотим состояние ПЕРЕД ходом героя.
                # Значит, если оппонент ходил до героя в этом раунде, его доска должна это отражать.

                # Если оппонент ходит раньше героя в этом раунде (меньший orderIndex, если герой не на баттоне и не первый)
                # или если target_game_round > 1 и это первый ход героя в раунде (тогда оппоненты с большим orderIndex, но до баттона, уже ходили)
                # Это сложная часть. Пока сделаем проще:
                # Доска оппонента на момент хода героя в target_game_round включает карты,
                # которые оппонент положил в hh_round_marker_for_board_cards (т.е. в текущем игровом раунде), ЕСЛИ он ходил раньше героя.
                # И карты, положенные в hh_round_marker < hh_round_marker_for_board_cards.

                # Упрощенная логика: если orderIndex оппонента меньше, чем у героя,
                # и герой не первый ходящий в раунде (т.е. не сразу после баттона),
                # то оппонент мог походить в текущем раунде до героя.
                # Это нужно уточнять по логам или правилам определения первого хода в раунде.

                # Пока будем собирать доску оппонента до hh_round_marker_for_board_cards,
                # а если target_game_round > 1, то для оппонентов, ходивших до героя,
                # нужно добавить карты с hh_round_marker_for_board_cards.
                # Это очень сложно точно восстановить без симуляции.

                # **УПРОЩЕННЫЙ ПОДХОД для Curriculum Learning:**
                # Мы берем состояние доски для всех игроков на момент *начала* target_game_round.
                # То есть, все карты, положенные с маркером раунда HH < (target_game_round - 1).
                # Затем среда раздает карты текущему игроку (герою).
                # Ходы оппонентов в этом target_game_round будут симулироваться средой.
                # Это означает, что если мы стартуем с середины игры, оппоненты начнут раунд "с нуля" для этого раунда.
                # Это не идеально для точного воссоздания, но проще для CL.

                max_placement_round_for_opp_board = hh_round_marker_for_board_cards

                row_strings_opp = opp_data.get("rows", "").split('/')
                for i, row_name in enumerate(["front", "middle", "back"]):
                    if i < len(row_strings_opp):
                        cards_in_row = self._parse_cards_from_string(row_strings_opp[i])
                        for card_int, placement_r_hh in cards_in_row:
                            if placement_r_hh < max_placement_round_for_opp_board:
                                if card_int is not None: opp_board_at_turn[row_name].append(card_int)

                dead_cards_opp = self._parse_cards_from_string(opp_data.get("dead", ""))
                for card_int, placement_r_hh in dead_cards_opp:
                    if placement_r_hh < max_placement_round_for_opp_board:
                        if card_int is not None: opp_board_at_turn['dead'].append(card_int)

                opp_snapshot_player = {
                    "name": opp_data["playerName"], "hero": False, "fantasy": opp_data["inFantasy"],
                    "front": opp_board_at_turn["front"], "middle": opp_board_at_turn["middle"],
                    "back": opp_board_at_turn["back"], "dead": opp_board_at_turn["dead"],
                    "to_play": [],  # Карты оппонента будут розданы средой, если их ход
                    "stack": opp_data.get("stack", 0)
                }
                if 0 <= opp_order_idx < num_players:
                    if all_players_snapshot_temp[opp_order_idx] is None:
                        all_players_snapshot_temp[opp_order_idx] = opp_snapshot_player
                        processed_players += 1
                    else:
                        # print(f"Error: Duplicate orderIndex {opp_order_idx} or slot already filled for hand {hand_info['hand_id']}")
                        # Это не должно происходить, если данные HH корректны
                        break  # Прерываем обработку этой руки
                else:
                    # print(f"Error: Invalid opp_order_idx {opp_order_idx} for hand {hand_info['hand_id']}")
                    break
            else:  # Если внутренний цикл по оппонентам завершился без break
                if processed_players != num_players:
                    # print(f"Warning: Hand {hand_info['hand_id']} player count mismatch. Expected {num_players}, processed {processed_players}. Snapshot: {all_players_snapshot_temp}")
                    continue  # Пропускаем эту руку, если не все игроки обработаны

                current_game_snapshot["players"] = all_players_snapshot_temp

                # Состояние колоды
                used_cards = set()
                for p_snap in current_game_snapshot["players"]:
                    if p_snap:
                        for row_list in [p_snap["front"], p_snap["middle"], p_snap["back"], p_snap["dead"]]:
                            for card_int_val in row_list:
                                if card_int_val is not None: used_cards.add(card_int_val)

                full_deck_ints = [Card.new(r + s) for r in _CARD_RANK_STR for s in _CARD_SUIT_STR]
                current_game_snapshot["deck_state"] = [c for c in full_deck_ints if c not in used_cards]

                # Проверка: если это начало игры (target_game_round=1), то в колоде должно быть 52 карты
                # (т.к. доски и сброс еще пустые).
                # Если это не так, значит, что-то не так с картами в HH или их парсингом.
                if target_game_round == 1 and len(current_game_snapshot["deck_state"]) != NUM_CARDS:
                    # print(f"Warning: Hand {hand_info['hand_id']}, target_game_round 1: Expected {NUM_CARDS} in deck, found {len(current_game_snapshot['deck_state'])}. Used: {used_cards}")
                    # Можно добавить более детальный вывод или пропустить это состояние
                    pass

                game_initial_states.append(current_game_snapshot)
                continue  # Переходим к следующей руке в self.parsed_hands

            # Если был break во внутреннем цикле (из-за ошибки с оппонентом), эта рука пропускается
            # print(f"Skipping hand {hand_info['hand_id']} due to opponent processing error.")

        return game_initial_states

# Пример использования (нужно будет вызывать из основного скрипта)
if __name__ == "__main__":
    # path = "D:\\develop\\temp\\poker\\Eureka\\tmp\\bad"
    path = "D:\\develop\\poker\\misc\\hh_ofc\\hh_ofc"
    parser = HHParser(hh_files_directory=path)
    parser.parse_files()
