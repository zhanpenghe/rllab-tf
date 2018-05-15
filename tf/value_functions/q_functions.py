import numpy as np
import tensorflow as tf
from sandbox.zhanpeng.tf.core.networks import MLP


class MLPQFunction(object):

    def __init__(self,
                 env,
                 hidden_layer_sizes=[100, 100],
                 name='q_function',):
        super(MLPQFunction, self).__init__()
        self._observation_dim = env.observation_space.flat_dim
        self._action_dim = env.action_space.flat_dim
        self._n_action = env.action_space.n
        self._layer_size = hidden_layer_sizes+[1]
        self._name = name

        with tf.variable_scope(self._name, reuse=False):
            self._mlp = MLP(input_dim=self._observation_dim, output_dim=self._n_action, hidden_sizes=hidden_layer_sizes)
            self._observation_ph = self._mlp.get_input_ph()
            self._value_op = self._mlp.feedforward_op()

    def values_op(self, observation_ph, reuse=False):
        with tf.variable_scope(self._name, reuse=reuse):
            values_op = self._mlp.get_feedforward_op(observation_ph, return_preactivations=False, resuse=reuse)
        return values_op

    def get_observations_ph(self):
        return self._observation_ph

    def get_params_internal(self, scope=''):
        scope += '/' + self._name if scope else self._name
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)

    def get_qval(self, observation):
        feeds = {self._observation_ph: np.array([observation])} # TODO
        q_values = tf.get_default_session().run(self._value_op, feeds)
        return q_values[0]
