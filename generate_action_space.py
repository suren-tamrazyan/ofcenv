import itertools
import pickle

first_round_box = ["f", "m", "b"]
non_first_round_box = ["f", "m", "b", "d"]

first_round_arrangement = itertools.product(first_round_box, repeat=5)
first_round_arrangement = list(filter(lambda t: sum(1 for s in t if s == 'f') <= 3, first_round_arrangement))
non_first_round_arrangement = itertools.product(non_first_round_box, repeat=3)
non_first_round_arrangement = list(filter(lambda t: sum(1 for s in t if s == 'd') == 1, non_first_round_arrangement))

# total 259 action, first 232 for first round, last 27 for non-first round
action_space = first_round_arrangement + non_first_round_arrangement

# for combo in action_space:
#     print(combo)

# serialize action_space to file
with open('res/action_space.p', 'wb') as f:
    pickle.dump(action_space, f)
