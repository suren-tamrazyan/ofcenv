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


# transform action to dict for OfcGame.play()
def action_to_dict(action_id, player):
    result = {}
    for k, v in zip(ACTION_SPACE[action_id], player.to_play):
        result.setdefault(k, []).append(v)
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

