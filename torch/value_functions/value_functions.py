import torch.optim as optimizers
import torch.nn.functional as F

from sandbox.zhanpeng.torch.core.networks import MLP


class MLPValueFunction:

    """
    A value function that output the value of an observation
    """

    def __init__(self,
                 obs_dim,
                 hidden_sizes=(64, 64),
                 hidden_nonlinearity=F.relu,
                 output_nonlinearity=None,
                 trainable=True,
                 optimizer=optimizers.Adam,
                 lr=1e-3,
                 n_training_steps=1,):

        self.mlp = MLP(
            input_size=obs_dim,
            output_size=1,
            hidden_sizes=hidden_sizes,
            hidden_nonlinearity=hidden_nonlinearity,
            output_nonlinearity=output_nonlinearity,
        )

        self._trainable = trainable
        self._n_training_steps = n_training_steps  # TODO: Check multiple steps optimization possibility
        if trainable:
            self._optimizer = optimizer(self.mlp.parameters(), lr=lr)
            self._loss_function = None

    def values(self, observations):
        values = self.mlp.forward(input=observations)
        return values

    def fit(self, observations, values):
        assert self._loss_function, "Please set a loss function for the value function!"
        assert self._trainable, "Current value function is not trainable, please set it to True if you want to train it. "

        predicted_values = self.values(observations=observations)
        loss = self._loss(predicted_values, values)
        loss.backward()
        self._optimizer.step()
        self._optimizer.zero_grad()
