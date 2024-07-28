from copy import deepcopy

import numpy as np

import ofc_encoder
from ofc_encoder import player_to_tensor_of_binary_card_matrix, rest_cards_to_one_hot
from ofcgame import OfcGame


class OfcGameRLCard:
    '''The interface for RL Card framework'''
    def __init__(self, allow_step_back=False):
        self.game = None
        self.history = []
        self.num_players = 2
        self.game_id = 0
        self.button = 0
        self.hero_ind = 0
        self.allow_step_back = allow_step_back
        self.np_random = np.random.RandomState()
        self.seed = None

    def configure(self, game_config):
        ''' Specifiy some game specific parameters, such as number of players
        '''
        self.num_players = game_config['game_num_players']
        if 'seed' in game_config:
            self.seed = game_config['seed']

    def get_num_players(self):
        ''' Return the number of players in ofc

        Returns:
            number_of_player (int)
        '''
        return self.num_players

    @staticmethod
    def get_num_actions():
        ''' Return the number of applicable actions

        Returns:
            number_of_actions (int): total 259 action, first 232 for first round, last 27 for non-first round
        '''
        return 259

    def init_game(self):
        ''' Initialilze the game

        Returns:
            state (dict): the first state of the game
            player_id (int): current player's id
        '''
        self.history = []
        self.game_id += 1
        self.game = OfcGame(game_id=str(self.game_id), max_player=self.num_players, button=self.button, hero=self.hero_ind, seed=self.seed)
        return self.get_state(self.game.current_player_ind), self.game.current_player_ind

    def get_player_id(self):
        ''' Return the current player's id

        Returns:
            player_id (int): current player's id
        '''
        return self.game.current_player_ind

    def get_state(self, player_id):
        ''' Return player's state? players as hero

        Args:
            player_id (int): player id

        Returns:
            (dict): The state of the player
        '''
        hero_player = self.game.players[player_id]
        hero_one_hot_matrix = player_to_tensor_of_binary_card_matrix(hero_player, True)

        opps = []
        for i in range(len(self.game.players)):
            if i != player_id:
                opps.append(i)
        opp1_one_hot_matrix = player_to_tensor_of_binary_card_matrix(self.game.players[opps[0]], False)
        if len(opps) > 1:
            opp2_one_hot_matrix = player_to_tensor_of_binary_card_matrix(self.game.players[opps[1]], False)
        else:
            opp2_one_hot_matrix = np.zeros((3, 4, 13))

        rest_cards_matrix = rest_cards_to_one_hot(self.game.opened_cards())
        rest_cards_matrix_expanded = rest_cards_matrix[np.newaxis, :, :]
        board_encode = np.vstack((hero_one_hot_matrix, opp1_one_hot_matrix, opp2_one_hot_matrix, rest_cards_matrix_expanded))
        legal_actions = ofc_encoder.legal_actions(hero_player)
        legal_actions_str = [str(ofc_encoder.action_dict_to_pretty_str(ofc_encoder.action_to_dict(a, hero_player.to_play))) for a in legal_actions]
        return {'board_from_player': board_encode, 'legal_actions': legal_actions, 'legal_actions_str': legal_actions_str}

    def step(self, action):
        ''' Perform one action of the game

        Args:
            action (int): specific action of 259

        Returns:
            dict: next player's state
            int: next player's id
        '''
        if self.allow_step_back:
            # First snapshot the current state
            game_clone = deepcopy(self.game)
            self.history.append(game_clone)

        # perform action
        self.game.play(ofc_encoder.action_to_dict(action, self.game.current_player().to_play))

        return self.get_state(self.game.current_player_ind), self.game.current_player_ind

    def step_back(self):
        ''' Return to the previous state of the game

        Returns:
            Status (bool): check if the step back is success or not
        '''
        if len(self.history) > 0:
            self.game = self.history.pop()
            return True
        return False

    def is_over(self):
        ''' Check if the game is over

        Returns:
            status (bool): True/False
        '''
        return self.game.is_game_over()

    def get_legal_actions(self):
        return ofc_encoder.legal_actions(self.game.current_player())

    def get_payoffs(self):
        ''' Return the payoffs of the game

        Returns:
            (list): Each entry corresponds to the payoff of one player
        '''
        return self.game.get_payoffs()