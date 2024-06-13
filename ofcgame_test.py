import time

import numpy as np
from treys import Card, Deck

import ofc_encoder
from ofc_agent import OfcRandomAgent
from ofc_encoder import game_state_to_tensor, player_to_tensor_of_rank_suit, get_rank_int, get_suit_int, \
    player_to_tensor_of_binary_card_matrix, ACTION_SPACE
from ofcgame import OfcGame


def random_test(print_game=False):
    game = OfcGame(game_id=0, max_player=2, button=1, hero=0)
    agent = OfcRandomAgent()
    while not game.is_game_over():
        for _ in range(len(game.players)):
            if print_game:
                print(game)
            action = agent.make_move(game.current_player())
            if print_game:
                print(action)
            game.play(action)
            if print_game:
                print(game)
                print('\n')

    game_state_tensor = game_state_to_tensor(game)
    hero_tensor = player_to_tensor_of_rank_suit(game.players[game.hero], False)
    hero_one_hot_matrix = player_to_tensor_of_binary_card_matrix(game.players[game.hero], True)
    opps = []
    for i in range(len(game.players)):
        if i != game.hero:
            opps.append(i)
    print('Opps:', opps)
    opp1_one_hot_matrix = player_to_tensor_of_binary_card_matrix(game.players[opps[0]], False)
    if len(opps) > 1:
        opp2_one_hot_matrix = player_to_tensor_of_binary_card_matrix(game.players[opps[1]], False)
    else:
        opp2_one_hot_matrix = np.zeros((3, 4, 13))
    if print_game:
        print(game.hero_player().calc_score_single())
        print(game_state_tensor)
        print(game_state_tensor.shape)
        print(game_state_tensor.dtype)
        print(hero_tensor)
        print(hero_tensor.shape)
        print(hero_tensor.dtype)
        print('Hero one hot matrix:')
        print(hero_one_hot_matrix)
        print('hero_one_hot_matrix shape:', hero_one_hot_matrix.shape)
        print('Opp1 one hot matrix:')
        print(opp1_one_hot_matrix)
        print('opp1_one_hot_matrix shape:', opp1_one_hot_matrix.shape)
        print('Opp2 one hot matrix:')
        print(opp2_one_hot_matrix)
        print('opp2_one_hot_matrix shape:', opp2_one_hot_matrix.shape)
        print('hh: ', game.hh)
        print('game opened_cards:', [Card.int_to_pretty_str(item) for item in game.opened_cards()])
        rest_cards_matrix = ofc_encoder.rest_cards_to_one_hot(game.opened_cards())
        print('rest cards matrix:\n', rest_cards_matrix)
        rest_cards_matrix_expanded = rest_cards_matrix[np.newaxis, :, :]
        union_encode = np.vstack((hero_one_hot_matrix, opp1_one_hot_matrix, opp2_one_hot_matrix, rest_cards_matrix_expanded))
        print('union_encode:\n', union_encode)
        print('union_encode shape:', union_encode.shape)
        print("rest_cards_summary:\n", ofc_encoder.rest_cards_summary(game.opened_cards()))
        vectorized_normalize = np.vectorize(ofc_encoder.normalize_data_0_1)
        rest_cards_suits, rest_cards_ranks = ofc_encoder.rest_cards_summary(game.opened_cards())
        rest_cards_suits_norm = vectorized_normalize(rest_cards_suits, 0, 13)
        rest_cards_ranks_norm = vectorized_normalize(rest_cards_ranks, 0, 4)
        print("rest_cards_summary_norm:\n", rest_cards_suits_norm, rest_cards_ranks_norm)
    # deck = Deck.GetFullDeck()
    # for i in deck:
    #     print(Card.get_rank_int(i))


# time_start = time.time()
# for i in range(10000):
#     random_test()
# print("--- %s seconds ---" % (time.time() - time_start))
random_test(print_game=True)


# import treys
#
# deck = treys.Deck()
# suits = set()
# ranks = set()
# cards = deck.draw(n=52)
# for card in cards:
#     ranks.add(get_rank_int(card))
#     suits.add(get_suit_int(card))
# print(ranks)
# print(suits)

