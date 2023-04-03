import pickle
from treys import Card, Evaluator
from treys.lookup import LookupTable


FRONT_LOOKUP = pickle.load(open("res/front_lookup.p", "rb"))


class OFCEvaluator(Evaluator):
    """treys' evaluator class extended to score an OFC Front."""
    def __init__(self):
        self.table = LookupTable()

        self.hand_size_map = {
            3: self._three,
            5: self._five,
            6: self._six,
            7: self._seven
        }

    def _three(self, cards):
        prime = Card.prime_product_from_hand(cards)
        return FRONT_LOOKUP[prime]
