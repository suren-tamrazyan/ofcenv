from typing import SupportsFloat, Any

import gymnasium as gym
import numpy as np

from ofc_encoder import player_to_tensor_of_rank_suit, action_to_dict, player_to_tensor_of_binary_card_matrix, \
    is_legal_action, rest_cards_to_one_hot
from ofcgame import OfcGame
from ofc_agent import OfcRandomAgent


class OfcEnv(gym.Env):
    metadata = {'render_modes': ['human'], 'render_fps': 1}

    def __init__(self, max_player=2):
        self.button_ind = 1
        self.max_player = max_player
        # if max_player == 3:
        #     button_ind = 2
        self.game = OfcGame(game_id=0, max_player=self.max_player, button=self.button_ind, hero=0)
        self.opponent1 = OfcRandomAgent()
        # self.opponent2 = OfcRandomAgent()
        # self.observation_space = gym.spaces.Tuple((gym.spaces.MultiDiscrete([[[14, 14, 14, 1, 1], [14, 14, 14, 14, 14], [14, 14, 14, 14, 14], [14, 14, 14, 14, 14], [14, 14, 14, 14, 14]], [[5, 5, 5, 1, 1], [5, 5, 5, 5, 5], [5, 5, 5, 5, 5], [5, 5, 5, 5, 5], [5, 5, 5, 5, 5]]]),
        #                                           gym.spaces.MultiDiscrete([[[14, 14, 14, 1, 1], [14, 14, 14, 14, 14], [14, 14, 14, 14, 14], [14, 14, 14, 14, 14], [14, 14, 14, 14, 14]], [[5, 5, 5, 1, 1], [5, 5, 5, 5, 5], [5, 5, 5, 5, 5], [5, 5, 5, 5, 5], [5, 5, 5, 5, 5]]])))  # (hero, opponent1); maybe MultiDiscrete([max_player, 2, 5, 5])?
        # self.observation_space = gym.spaces.MultiDiscrete([[[14, 14, 14, 1, 1], [14, 14, 14, 14, 14], [14, 14, 14, 14, 14], [14, 14, 14, 14, 14], [14, 14, 14, 14, 14]], [[5, 5, 5, 1, 1], [5, 5, 5, 5, 5], [5, 5, 5, 5, 5], [5, 5, 5, 5, 5], [5, 5, 5, 5, 5]]])
        one_hot_matrix_shape = (6, 4, 13)
        # self.observation_space = gym.spaces.Box(low=0, high=1, shape=one_hot_matrix_shape, dtype=np.uint8)
        self.observation_space = gym.spaces.MultiBinary(one_hot_matrix_shape)
        self.action_space = gym.spaces.Discrete(297)
        self.ILLEGAL_ACTION_PENALTY = -10
        self.render_mode = 'human'

    def _get_obs(self):
        # return player_to_tensor(self.game.hero_player(), True), player_to_tensor(self.game.players[1], True)
        # return player_to_tensor_of_rank_suit(self.game.hero_player(), True)
        hero_one_hot_matrix = player_to_tensor_of_binary_card_matrix(self.game.hero_player())
        rest_cards_matrix = rest_cards_to_one_hot(self.game.opened_cards())
        rest_cards_matrix_expanded = rest_cards_matrix[np.newaxis, :, :]
        union_encode = np.vstack((hero_one_hot_matrix, rest_cards_matrix_expanded))
        return union_encode

    def _get_info(self):
        return {}

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        self.game = OfcGame(game_id=0, max_player=self.max_player, button=self.button_ind, hero=0)
        return self._get_obs(), self._get_info()

    def step(self, action):
        if not is_legal_action(self.game.hero_player(), action):
            return self._get_obs(), self.ILLEGAL_ACTION_PENALTY, True, False, self._get_info()
        self.game.play(action_to_dict(action, self.game.hero_player()))
        if not self.game.is_game_over():
            self.game.play(self.opponent1.make_move(self.game.current_player()))
        reward = 0
        game_over = False
        if self.game.is_game_over():
            reward = self.game.hero_player().calc_score_single()
            game_over = True
        return self._get_obs(), reward, game_over, False, self._get_info()

    def render(self):
        print(self.game)
