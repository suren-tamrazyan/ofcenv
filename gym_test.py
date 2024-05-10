import gymnasium as gym
from gymnasium.envs.registration import register

register(
    id='ofc-v0',
    entry_point='gym.ofc_gym:OfcEnv',
)

#print(gym.envs.registry)

env = gym.make('ofc-v0')
observation, info = env.reset()
print(observation)
env.render()

# import treys
# from treys import Card
#
# deck = treys.Deck()
# suits = set()
# ranks = set()
# cards = deck.draw(n=52)
# for card in cards:
#     ranks.add(Card.get_rank_int(card))
#     suits.add(Card.get_suit_int(card))
# print(ranks)
# print(suits)


# sp = gym.spaces.MultiDiscrete([[[14, 14, 14, 1, 1], [14, 14, 14, 14, 14], [14, 14, 14, 14, 14], [14, 14, 14, 14, 14], [14, 14, 14, 14, 14]], [[5, 5, 5, 1, 1], [5, 5, 5, 5, 5], [5, 5, 5, 5, 5], [5, 5, 5, 5, 5], [5, 5, 5, 5, 5]]])
# for _ in range(1):
#     print(sp.sample())