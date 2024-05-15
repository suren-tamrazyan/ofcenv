import time

import numpy as np
from treys import Card, Deck

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
    hero_one_hot_matrix = player_to_tensor_of_binary_card_matrix(game.players[game.hero])
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
        print('hh: ', game.hh)
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

