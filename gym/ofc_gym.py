from typing import SupportsFloat, Any

import gymnasium as gym
import numpy as np

from ofc_encoder import player_to_tensor_of_rank_suit, action_to_dict, player_to_tensor_of_binary_card_matrix, \
    is_legal_action, rest_cards_to_one_hot, rest_cards_summary, normalize_data_0_1
from ofcgame import OfcGame
from ofc_agent import OfcRandomAgent


class OfcEnv(gym.Env):
    metadata = {'render_modes': ['human'], 'render_fps': 1}

    def __init__(self, max_player=2, observe_summary=False):
        self.button_ind = 1
        self.max_player = max_player
        self.observe_summary = observe_summary
        # if max_player == 3:
        #     button_ind = 2
        self.game = OfcGame(game_id=0, max_player=self.max_player, button=self.button_ind, hero=0)
        self.opponent1 = OfcRandomAgent()
        # self.opponent2 = OfcRandomAgent()
        one_hot_matrix_shape = (12, 4, 13)
        self.summary_dim = 21
        if self.observe_summary:
            self.observation_space = gym.spaces.Dict({
                'board': gym.spaces.MultiBinary(one_hot_matrix_shape),
                'summary': gym.spaces.Box(low=0, high=1, shape=(self.summary_dim,), dtype=np.float64)
            })
        else:
            self.observation_space = gym.spaces.MultiBinary(one_hot_matrix_shape)
        self.action_space = gym.spaces.Discrete(259)
        self.ILLEGAL_ACTION_PENALTY = -10
        self.render_mode = 'human'

    def _get_obs(self):
        # return player_to_tensor(self.game.hero_player(), True), player_to_tensor(self.game.players[1], True)
        # return player_to_tensor_of_rank_suit(self.game.hero_player(), True)
        hero_one_hot_matrix = player_to_tensor_of_binary_card_matrix(self.game.hero_player(), True)

        opps = []
        for i in range(len(self.game.players)):
            if i != self.game.hero:
                opps.append(i)
        opp1_one_hot_matrix = player_to_tensor_of_binary_card_matrix(self.game.players[opps[0]], False)
        if len(opps) > 1:
            opp2_one_hot_matrix = player_to_tensor_of_binary_card_matrix(self.game.players[opps[1]], False)
        else:
            opp2_one_hot_matrix = np.zeros((3, 4, 13))

        rest_cards_matrix = rest_cards_to_one_hot(self.game.opened_cards())
        rest_cards_matrix_expanded = rest_cards_matrix[np.newaxis, :, :]
        board_encode = np.vstack((hero_one_hot_matrix, opp1_one_hot_matrix, opp2_one_hot_matrix, rest_cards_matrix_expanded))
        if not self.observe_summary:
            return board_encode
        else:
            # game summary
            game_sum_norm = np.array([normalize_data_0_1(self.game.round, 1, 5), normalize_data_0_1(len(self.game.hero_player().front), 0, 3), normalize_data_0_1(len(self.game.hero_player().middle), 0, 5), normalize_data_0_1(len(self.game.hero_player().back), 0, 5)])
            vectorized_normalize = np.vectorize(normalize_data_0_1)
            rest_cards_suits, rest_cards_ranks = rest_cards_summary(self.game.opened_cards())
            rest_cards_suits_norm = vectorized_normalize(rest_cards_suits, 0, 13)
            rest_cards_ranks_norm = vectorized_normalize(rest_cards_ranks, 0, 4)
            summary_encode = np.concatenate((game_sum_norm, rest_cards_suits_norm, rest_cards_ranks_norm))
            return {
                'board': board_encode,
                'summary': summary_encode
            }

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
        self.game.play(action_to_dict(action, self.game.hero_player().to_play))
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
