from treys import Card, Deck

from ofc_encoder import action_dict_to_pretty_str
from ofc_evaluator import OFCEvaluator
from royalty_calculator import RoyaltyCalculator


class OfcException(Exception):
    pass


evaluator = OFCEvaluator()


class OfcPlayer:
    def __init__(self, name, fantasy):
        self.name = name
        self.front = []
        self.middle = []
        self.back = []
        self.dead = []
        self.to_play = []
        self.fantasy = fantasy
        self.stack = 0

    def _box_str(self, box, limit):
        return "".join(Card.int_to_pretty_str(item) for item in box) + "." * (limit - len(box))

    def __str__(self):
        return "{:<15} {:>7d} F:{:<12s} M:{:<20s} B:{:<20s} D:{:<16s} DL:{:<20s}".format(self.name, self.stack,
                                                                                         self._box_str(self.front, 3),
                                                                                         self._box_str(self.middle, 5),
                                                                                         self._box_str(self.back, 5),
                                                                                         self._box_str(self.dead, 4),
                                                                                         Card.ints_to_pretty_str(
                                                                                             self.to_play))

    def deal(self, cards):
        if self.to_play:
            raise OfcException("Can't deal cards while in play")
        self.to_play = cards

    def play(self, dict_cards):
        self.front.extend(dict_cards.get("f", []))
        self.middle.extend(dict_cards.get("m", []))
        self.back.extend(dict_cards.get("b", []))
        self.dead.extend(dict_cards.get("d", []))
        self.to_play.clear()

    def is_complete(self):
        return len(self.front) == 3 and len(self.middle) == len(self.back) == 5

    @staticmethod
    def get_rank(box):
        return evaluator.evaluate(box, [])

    def is_foul(self):
        if not self.is_complete():
            return True

        if self.get_rank(self.front) >= \
                self.get_rank(self.middle) >= \
                self.get_rank(self.back):
            return False

        return True

    def get_royalties(self, fantasy_score):
        if not self.is_complete():
            return 0

        royalty_total = \
            RoyaltyCalculator.score_front_royalties(self.front) + \
            RoyaltyCalculator.score_mid_royalties(self.middle) + \
            RoyaltyCalculator.score_back_royalties(self.back)

        # Fantasyland! (QQ2 evaluates to 3985+1)
        eval_front = self.get_rank(self.front)
        eval_back = self.get_rank(self.back)
        if not self.fantasy and eval_front < 3985 + 1:
            royalty_total += fantasy_score
        if self.fantasy and (eval_front <= 2468 or eval_back <= 166):
            royalty_total += fantasy_score

        return royalty_total

    def calc_score_single(self, fantasy_score=5, foul_penalty=-3):
        if not self.is_complete():
            raise OfcException("Can't calculate score if not full")
        if self.is_foul():
            return foul_penalty
        MAX_EVAL = 7462 + 1
        REGULARIZATION_PARAM = 3
        eval_front = self.get_rank(self.front)
        norm_kicker_front = (MAX_EVAL - eval_front) / MAX_EVAL
        eval_middle = self.get_rank(self.middle)
        norm_kicker_middle = (MAX_EVAL - eval_middle) / MAX_EVAL
        eval_back = self.get_rank(self.back)
        norm_kicker_back = (MAX_EVAL - eval_back) / MAX_EVAL
        bonus = self.get_royalties(fantasy_score)
        return bonus + (norm_kicker_back + norm_kicker_middle + norm_kicker_front) / REGULARIZATION_PARAM

    def calc_score_against(self, other_player, fantasy_score=5):
        if not self.is_complete() or not other_player.is_complete():
            raise OfcException("Can't calculate score if not full")

        self_royalties = self.get_royalties(fantasy_score)
        oppo_royalties = other_player.get_royalties(fantasy_score)

        self_foul = self.is_foul()
        oppo_foul = other_player.is_foul()

        if self_foul and oppo_foul:
            score = 0

        elif self_foul:
            score = (-1 * oppo_royalties) - 6

        elif oppo_foul:
            score = self_royalties + 6

        else:
            exch = self._calculate_scoop(other_player)
            score = exch + self_royalties - oppo_royalties

        return score

    def _calculate_scoop(self, other):
        won = 0

        won += self.calculate_street(self.front, other.front)
        won += self.calculate_street(self.middle, other.mid)
        won += self.calculate_street(self.back, other.back)

        if won in [3, -3]:  # Scoop, one way or the other
            won = won * 2

        return won

    @staticmethod
    def calculate_street(lhs_hand, rhs_hand):
        lhs_rank = OfcPlayer.get_rank(lhs_hand)
        rhs_rank = OfcPlayer.get_rank(rhs_hand)

        if lhs_rank < rhs_rank:
            return 1
        elif lhs_rank > rhs_rank:
            return -1
        return 0


class OfcGameBase:
    def __init__(self, game_id, max_player, button, hero=0):
        self.game_id = game_id
        self.players = [OfcPlayer("player" + str(i), False) for i in range(max_player)]
        self.hero = hero
        self.round = 1
        self.button_ind = button
        self.current_player_ind = button
        self._next_player()
        self.hh = []  # [{round, player, action, action2str}]

    def current_player(self):
        return self.players[self.current_player_ind]

    def hero_player(self):
        return self.players[self.hero]

    def _next_player(self):
        cur_ind = self.current_player_ind
        while True:
            self.current_player_ind = (self.current_player_ind + 1) % len(self.players)
            if not (self.current_player().fantasy and self.current_player_ind != cur_ind):
                break

    def _next_round(self):
        self.round += 1

    def is_game_over(self):
        return all(player.is_complete() for player in self.players)

    def play(self, dict_cards):
        self.hh.append((self.round, self.current_player_ind, dict_cards, action_dict_to_pretty_str(dict_cards)))
        self.current_player().play(dict_cards)

        ind = self.current_player_ind
        while True:
            need_transit_round = ind == self.button_ind
            ind = (ind + 1) % len(self.players)
            if not (self.players[ind].fantasy and not need_transit_round):
                break

        self._next_player()
        if need_transit_round and self.round < 5:
            self._next_round()

    def calc_hero_score(self):
        result = 0
        hero = self.players[self.hero]
        for player in self.players:
            if player == hero:
                continue
            result += hero.calc_score_against(player)
        return result

    def __str__(self):
        result = "---------------- Game #" + str(self.game_id) + "; round " + str(
            self.round) + "; " + (
                     "; FINISH" if self.is_game_over() else "") + " -----------------\n"
        for i in range(len(self.players)):
            player = self.players[i]
            strLabel = ("b" if i == self.button_ind else "") + (
                "h" if i == self.hero else "") + ("f" if player.fantasy else "")
            if strLabel.strip() != "":
                strLabel = "(" + strLabel + ")"
            result += ("*" if i == self.current_player_ind else " ") + "{:<5}".format(strLabel) + str(
                player) + "\n"
        return result

    def opened_cards(self):
        result = []
        for player in self.players:
            result += player.front
            result += player.middle
            result += player.back
            result += player.dead
            result += player.to_play
        return result

class OfcGame(OfcGameBase):
    def __init__(self, game_id, max_player, button, hero=0):
        super().__init__(game_id, max_player, button, hero)
        self.deck = Deck()
        for player in self.players:
            if player.fantasy:
                player.deal(self.deck.draw(14))
            else:
                player.deal(self.deck.draw(5))

    def _next_round(self):
        super()._next_round()
        for player in self.players:
            if not player.fantasy:
                player.deal(self.deck.draw(3))


def test():
    player = OfcPlayer("player1", False)
    player.front = [Card.new(x) for x in ['4d', 'Kd', 'Js']]
    player.middle = [Card.new(x) for x in ['2h', '5c', '6c', '3h', '7d']]
    player.back = [Card.new(x) for x in ['6h', '6d', '6s', 'Ac', 'As']]
    print(player.calc_score_single())

    player.front = [Card.new(x) for x in ['Ah', 'Ac', 'Kd']]
    player.middle = [Card.new(x) for x in ['2h', '2s', '2d', '2c', '3c']]
    player.back = [Card.new(x) for x in ['3h', '3s', '3d', '3c', '4c']]
    print(player.calc_score_single())

# test()
