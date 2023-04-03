import numpy as np

from ofc_encoder import legal_actions, action_to_dict


class OfcAgent:
    """An OFC decision maker."""

    def make_move(self, player):
        pass


class OfcRandomAgent(OfcAgent):
    """An OFC decision maker that makes random moves."""

    def make_move(self, player):
        return action_to_dict(np.random.choice(legal_actions(player)), player)
