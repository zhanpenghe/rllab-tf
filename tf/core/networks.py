import tensorflow as tf
from sandbox.zhanpeng.tf.core.nn import mlp_feedforward_op


class MLP(object):

    def __init__(self,
                 input_dim,
                 output_dim,
                 hidden_sizes=[128, 128],
                 activation_fn=tf.nn.relu,
                 output_nonlinearity=None,
                 name='multi_layers_perceptrons'):

        self.input_ph = tf.placeholder(
            tf.float32,
            shape=[None, input_dim],
            name='mlp_input'
        )

        self.layer_sizes = hidden_sizes+[output_dim]
        self.activation_fn = activation_fn
        self.output_nonlinearity = output_nonlinearity
        self.name = name

        self._feedforward_op = self.get_feedforward_op(self.input_ph)

    #TODO rename this function sometime..
    def feedforward(self, inputs):
        feed_dict = {self.input_ph: inputs}
        result = tf.get_default_session().run(self._feedforward_op, feed_dict)
        return result

    def get_feedforward_op(self, input_phs, return_preactivations=False, resuse=False):
        with tf.variable_scope(self.name, reuse=resuse):
            preactivation = mlp_feedforward_op(
                inputs=[input_phs],
                layer_sizes=self.layer_sizes,
                activation_fn=self.activation_fn,
                output_nonlinearity=None,)
        if self.output_nonlinearity:
            output = self.output_nonlinearity(preactivation)
        else:
            output = preactivation

        if return_preactivations:
            return output, preactivation
        return output

    def get_input_ph(self):
        return self.input_ph

    def feedforward_op(self):
        return self._feedforward_op
