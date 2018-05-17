from rllab.policies.base import Policy
from rllab.core.serializable import Serializable


class ArgmaxDiscretePolicy(Policy, Serializable):

    def __init__(self, env_spec, qf):
        super(ArgmaxDiscretePolicy, self).__init__(env_spec)
        Serializable.quick_init(self, locals())
        self._qf = qf

    def get_action(self, observation):
        q_vals = self._qf.get_qval(observation)
        return q_vals.argmax(), None
