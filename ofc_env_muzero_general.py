from ofcgame import OfcGame
import numpy as np


class OfcEnvMuzeroGeneral:
    def __init__(self):
        self.env = OfcGame(game_id=0, max_player=2, button=np.random.choice([0, 1]), hero=0)
