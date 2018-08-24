import numpy as np
import torch

from sandbox.zhanpeng.torch.sampler.base import BaseSampler
from sandbox.zhanpeng.torch.core.utils import from_numpy


class BatchSampler(BaseSampler):

    def __init__(self, env, policy, vf, n_steps, gamma, lam):

        self.env = env
        self.policy = policy
        self.vf = vf
        self.n_steps = n_steps
        self.gamma = gamma
        self.lam = lam
        self.obs = np.zeros(1, env.observation_space.shape)
        self.obs[:] = self.env.reset()
        self.dones = False  # Single environment for now

    def obtain_samples(self, itr):
        mb_obs, mb_rewards, mb_actions, mb_values, mb_dones, mb_logp = [], [], [], [], [], []
        epinfos = []
        for _ in range(self.n_steps):
            action, infos = self.policy.get_action(self.obs)
            log_prob = infos.get('log_prob')
            values = self.vf.get_values(self.obs)

            mb_obs.append(self.obs.copy())
            mb_actions.append(action)
            mb_values.append(values)
            mb_logp.append(log_prob)
            mb_dones.append(self.dones)
            self.obs[:], rewards, self.dones, infos = self.env.step(action)

            for info in infos:
                maybeepinfo = info.get('episode')
                if maybeepinfo: epinfos.append(maybeepinfo)
            mb_rewards.append(rewards)

        # Prepare a batch
        mb_obs = np.asarray(mb_obs, dtype=self.obs.dtype)
        mb_rewards = np.asarray(mb_rewards, dtype=np.float32)
        mb_actions = np.asarray(mb_actions)
        mb_values = np.asarray(mb_values, dtype=np.float32)
        mb_logp = np.asarray(mb_logp, dtype=np.float32)
        mb_dones = np.asarray(mb_dones, dtype=np.bool)
        last_values = self.vf.get_values(self.obs)

        # discount/bootstrap off value fn
        # TODO: This part could be done in pytorch
        mb_returns = np.zeros_like(mb_rewards)
        mb_advs = np.zeros_like(mb_rewards)
        lastgaelam = 0

        for t in reversed(range(self.n_steps)):
            if t == self.n_steps - 1:
                nextnonterminal = 1.0 - self.dones
                nextvalues = last_values
            else:
                nextnonterminal = 1.0 - mb_dones[t+1]
                nextvalues = mb_values[t+1]
            delta = mb_rewards + self.gamma * nextvalues * nextnonterminal - mb_values[t]
            mb_advs[t] = lastgaelam = delta +self.gamma * self.lam * nextnonterminal * lastgaelam
        mb_returns = mb_advs + mb_values
        return mb_obs, mb_returns, mb_dones, mb_actions, mb_values, mb_logp, (epinfos,)
