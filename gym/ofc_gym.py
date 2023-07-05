from typing import SupportsFloat, Any

import gymnasium as gym

from ofc_encoder import player_to_tensor, action_to_dict
from ofcgame import OfcGame
from ofc_agent import OfcRandomAgent


class OfcEnv(gym.Env):

    metadata = {'render.modes': ['human']}

    def __init__(self, max_player=2):
        self.button_ind = 1
        self.max_player = max_player
        # if max_player == 3:
        #     button_ind = 2
        self.game = OfcGame(game_id=0, max_player=self.max_player, button=self.button_ind, hero=0)
        self.opponent1 = OfcRandomAgent()
        # self.opponent2 = OfcRandomAgent()
        self.observation_space = gym.spaces.Tuple(gym.spaces.MultiDiscrete([2, 5, 5], dtype='int32'),
                                                  gym.spaces.MultiDiscrete([2, 5, 5], dtype='int32'))  # (hero, opponent1); maybe MultiDiscrete([max_player, 2, 5, 5])?
        self.action_space = gym.spaces.Discrete(297)

    def _get_obs(self):
        return player_to_tensor(self.game.hero_player()), player_to_tensor(self.game.players[1])

    def _get_info(self):
        return ''

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        self.game = OfcGame(game_id=0, max_player=self.max_player, button=self.button_ind, hero=0)
        return self._get_obs(), self._get_info()

    def step(self, action):
        self.game.play(action_to_dict(action, self.game.hero_player()))
        self.game.play(self.opponent1.make_move(self.game.current_player()))
        reward = 0
        game_over = False
        if self.env.is_game_over():
            reward = self.env.hero_player().calc_score_single()
            game_over = True
        return self._get_obs(), reward, game_over, False, self._get_info()

    def render(self):
        print(self.env)