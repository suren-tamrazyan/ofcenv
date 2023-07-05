import time

import numpy as np
from treys import Card, Deck

from ofc_agent import OfcRandomAgent
from ofc_encoder import game_state_to_tensor, player_to_tensor
from ofcgame import OfcGame


def random_test(print_game=False):
    game = OfcGame(game_id=0, max_player=2, button=1, hero=1)
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
    hero_tensor = player_to_tensor(game.players[game.hero])
    if print_game:
        print(game.hero_player().calc_score_single())
        print(game_state_tensor)
        print(game_state_tensor.shape)
        print(game_state_tensor.dtype)
        print(hero_tensor)
        print(hero_tensor.shape)
        print(hero_tensor.dtype)
    # deck = Deck.GetFullDeck()
    # for i in deck:
    #     print(Card.get_rank_int(i))


# time_start = time.time()
# for i in range(10000):
#     random_test()
# print("--- %s seconds ---" % (time.time() - time_start))
random_test(print_game=True)
