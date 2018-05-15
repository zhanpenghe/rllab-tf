

class ArgmaxDiscretePolicy(object):

    def __init__(self, qf):
        self._qf = qf

    def get_action(self, observation):
        q_vals = self._qf.get_qval(observation)
        return q_vals.argmax(), None
