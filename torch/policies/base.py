
from garage.policies import Policy


class StochasticPolicy(Policy):

    @property
    def distribution(self):
        return self._dist
