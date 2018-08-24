import torch
import torch.nn as nn
import numpy as np
from sandbox.zhanpeng.torch.policies.gaussian_mlp_policy import GaussianMLPPolicy
from sandbox.zhanpeng.torch.core.networks import MLP


class SimpleFunction(nn.Module):

    def forward(self, inputs):
        return inputs**2+1


def test_policy():
    policy = GaussianMLPPolicy(obs_dim=2, action_dim=2, adaptive_std=True)
    inputs = torch.randn(2, 2)
    mean, std = policy._get_mean_and_std(inputs)
    print(mean, std)
    action = policy.get_action(inputs)
    print(action)


def test_mlp():

    f = SimpleFunction()
    mlp = MLP(input_size=1, output_size=1)

    n_epoch = 100000
    loss = nn.MSELoss()
    optimizer = torch.optim.Adam(mlp.parameters(), lr=0.0001)

    for ep in range(n_epoch):
        inputs = torch.randn(256, 1)
        y = f(inputs)
        predicted_y = mlp(inputs)
        l = loss(predicted_y, y)

        l.backward()
        optimizer.step()
        if ep % 1000 == 0:
            print('ep #'+str(ep), 'loss:', l.data)

    for t in range(10):
        inputs = torch.randn(1, 1)
        y = f(inputs)
        predicted_y = mlp(inputs)
        l = loss(predicted_y, y)

        print('test #'+str(t), predicted_y, y, l)


if __name__ == '__main__':
    test_mlp()
    test_policy()
