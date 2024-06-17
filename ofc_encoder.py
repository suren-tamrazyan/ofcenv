import pickle

import numpy as np
from treys import Card

# load the action_space
with open('res/action_space.p', 'rb') as f:
    ACTION_SPACE = pickle.load(f)


def check_box(action, box_counts):
    if sum(1 for s in action if s == 'f') + box_counts[0] > 3:
        return False
    if sum(1 for s in action if s == 'm') + box_counts[1] > 5:
        return False
    if sum(1 for s in action if s == 'b') + box_counts[2] > 5:
        return False
    return True


# determine legal actions for current state of player
def legal_actions(player):
    box_counts = (len(player.front), len(player.middle), len(player.back))
    is_first_round = len(player.to_play) == 5
    if is_first_round:
        return [i for i in range(243) if check_box(ACTION_SPACE[i], box_counts)]
    else:
        return [i for i in range(243, len(ACTION_SPACE)) if check_box(ACTION_SPACE[i], box_counts)]

def is_legal_action(player, action_id):
    box_counts = (len(player.front), len(player.middle), len(player.back))
    is_first_round = len(player.to_play) == 5
    if is_first_round:
        return action_id < 243 and check_box(ACTION_SPACE[action_id], box_counts)
    else:
        return 243 <= action_id < len(ACTION_SPACE) and check_box(ACTION_SPACE[action_id], box_counts)


# transform action to dict for OfcGame.play()
def action_to_dict(action_id, to_play):
    result = {}
    for k, v in zip(ACTION_SPACE[action_id], to_play):
        result.setdefault(k, []).append(v)
    return result

def action_dict_to_pretty_str(action_dict):
    result = {}
    for key, value in action_dict.items():
        pretty_values = [Card.int_to_pretty_str(item) for item in value]
        result[key] = pretty_values
    return result

MATRIX_WIDTH = 5

def player_to_rank_matrix(player):
    front_rank = np.pad(np.array([Card.get_rank_int(card) for card in player.front], dtype="int32"), (0, MATRIX_WIDTH - len(player.front)), mode='constant', constant_values=-1)
    middle_rank = np.pad(np.array([Card.get_rank_int(card) for card in player.middle], dtype="int32"), (0, MATRIX_WIDTH - len(player.middle)), mode='constant', constant_values=-1)
    back_rank = np.pad(np.array([Card.get_rank_int(card) for card in player.back], dtype="int32"), (0, MATRIX_WIDTH - len(player.back)), mode='constant', constant_values=-1)
    to_play_rank = np.pad(np.array([Card.get_rank_int(card) for card in player.to_play], dtype="int32"), (0, MATRIX_WIDTH - len(player.to_play)), mode='constant', constant_values=-1)
    dead_rank = np.pad(np.array([Card.get_rank_int(card) for card in player.dead], dtype="int32"), (0, MATRIX_WIDTH - len(player.dead)), mode='constant', constant_values=-1)
    return front_rank, middle_rank, back_rank, to_play_rank, dead_rank

def player_to_suit_matrix(player):
    front_suit = np.pad(np.array([Card.get_suit_int(card) for card in player.front], dtype="int32"), (0, MATRIX_WIDTH - len(player.front)), mode='constant', constant_values=-1)
    middle_suit = np.pad(np.array([Card.get_suit_int(card) for card in player.middle], dtype="int32"), (0, MATRIX_WIDTH - len(player.middle)), mode='constant', constant_values=-1)
    back_suit = np.pad(np.array([Card.get_suit_int(card) for card in player.back], dtype="int32"), (0, MATRIX_WIDTH - len(player.back)), mode='constant', constant_values=-1)
    to_play_suit = np.pad(np.array([Card.get_suit_int(card) for card in player.to_play], dtype="int32"), (0, MATRIX_WIDTH - len(player.to_play)), mode='constant', constant_values=-1)
    dead_suit = np.pad(np.array([Card.get_suit_int(card) for card in player.dead], dtype="int32"), (0, MATRIX_WIDTH - len(player.dead)), mode='constant', constant_values=-1)
    return front_suit, middle_suit, back_suit, to_play_suit, dead_suit

# determine the state of game as numpy array of cards rank and suit
def game_state_to_tensor(game):
    hero = game.players[game.hero]
    # ranks
    hero_ranks = np.row_stack(player_to_rank_matrix(hero))
    opp_ranks = [player_to_rank_matrix(game.players[plInd]) for plInd in range(len(game.players)) if plInd != game.hero]
    # suits
    hero_suits = np.row_stack(player_to_suit_matrix(hero))
    opp_suits = [player_to_suit_matrix(game.players[plInd]) for plInd in range(len(game.players)) if plInd != game.hero]
    # matrices to cube
    result = np.stack([(hero_ranks, hero_suits)] + list(zip(opp_ranks, opp_suits)), axis=0)
    # result = np.array([(hero_ranks, hero_suits)] + list(zip(opp_ranks, opp_suits)))
    result = np.concatenate(result, axis=0)
    return result


def player_to_tensor_of_rank_suit(player, dense):
    # ranks
    ranks = np.row_stack(player_to_rank_matrix(player))
    if dense:
        ranks += 1
    # suits
    suits = np.row_stack(player_to_suit_matrix(player))
    if dense:
        suits = np.where(suits == -1, 0, suits)
        suits = np.where(suits == 8, 3, suits)

    # matrices to cube
    result = np.stack([(ranks, suits)], axis=0)
    result = np.concatenate(result, axis=0)
    return result


def get_suit_int(card):
    # spades: 0, hearts: 1, clubs: 2, diamonds: 3
    suit = Card.get_suit_int(card)
    if suit == 8:
        suit = 3
    suit -= 1
    return suit

def get_rank_int(card):
    # 2: 0, 3: 1, ..., A: 12
    return Card.get_rank_int(card)

def cards_to_one_hot(cards):
    one_hot_matrix = np.zeros((4, 13))
    for card in cards:
        rank_int = get_rank_int(card)
        suit_int = get_suit_int(card)
        one_hot_matrix[suit_int, rank_int] = 1
    return one_hot_matrix

def rest_cards_to_one_hot(except_cards):
    one_hot_matrix = np.ones((4, 13))
    for card in except_cards:
        rank_int = get_rank_int(card)
        suit_int = get_suit_int(card)
        one_hot_matrix[suit_int, rank_int] = 0
    return one_hot_matrix

def player_to_tensor_of_binary_card_matrix(player, is_hero):
    front_tensor = np.array([cards_to_one_hot(player.front)])
    middle_tensor = np.array([cards_to_one_hot(player.middle)])
    back_tensor = np.array([cards_to_one_hot(player.back)])
    dead_tensor = np.array([cards_to_one_hot(player.dead)])
    to_play_tensor = np.array([cards_to_one_hot(player.to_play)])
    if is_hero:
        return np.concatenate((front_tensor, middle_tensor, back_tensor, dead_tensor, to_play_tensor), axis=0)
    else:
        return np.concatenate((front_tensor, middle_tensor, back_tensor), axis=0)

def rest_cards_summary(except_cards):
    suit_count = np.full(4, 13)
    rank_count = np.full(13, 4)
    joker_count = np.full(1, 2)
    for card in except_cards:
        suit_count[get_suit_int(card)] -= 1
        rank_count[get_rank_int(card)] -= 1
    # return np.concatenate((suit_count, rank_count))
    return suit_count, rank_count

def normalize_data_0_1(data, data_min, data_max):
    return (data - data_min) / (data_max - data_min)
