import numpy as np


class EpsilonGreedy(object):

    def __init__(self, action_space, prob_random_action=0.1):
        self._action_space = action_space
        self._prob = prob_random_action
        self.policy = None

    def get_action(self, observations):
        if np.random.uniform() <= self._prob:
            action = self._action_space.sample()
            info = 'random'
        else:
            action, _ = self.policy.get_action(observations)
            info = 'policy'
        return action, info

    def initialize(self, policy):
        self.policy = policy

    def get_actions(self):
        pass
