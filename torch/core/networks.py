from torch import nn
from torch.nn import functional as F

from sandbox.zhanpeng.torch.core.base import PyTorchModule
import sandbox.zhanpeng.torch.core.utils as ptu

def identity(x):
    return x


class MLP(PyTorchModule):

    def __init__(self,
                 input_size,
                 output_size,
                 hidden_sizes=(32, 32),
                 hidden_nonlinearity=F.relu,
                 output_nonlinearity=identity,
                 init_b=0.1,
                 init_w=3e-3,
                 hidden_init=ptu.fanin_init,
                 ):
        self.save_init_params(locals())
        super(MLP, self).__init__()

        self._input_size = input_size
        self._output_size = output_size
        self._hidden_nonlinearity = hidden_nonlinearity
        self._output_nonlinearity = output_nonlinearity if not None else identity

        self._fcs = list()
        in_size = input_size

        for i, next_size in enumerate(hidden_sizes):
            fc = nn.Linear(in_size, next_size)
            in_size = next_size
            hidden_init(fc.weight)
            fc.bias.data.fill_(init_b)
            self.__setattr__("fc{}".format(i), fc)
            self._fcs.append(fc)

        self.last_fc = nn.Linear(in_size, output_size)
        self.last_fc.weight.data.uniform_(-init_w, init_w)
        self.last_fc.bias.data.uniform_(-init_w, init_w)

    def forward(self, input, return_preactivations=False):
        h = input
        for i, fc in enumerate(self.fcs):
            h = fc(h)
            h = self._hidden_nonlinearity(h)
        preactivation = self.last_fc(h)
        output = self.output_activation(preactivation)
        if return_preactivations:
            return output, preactivation
        else:
            return output
