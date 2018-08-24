import numpy as np

import garage.misc.logger as logger

from sandbox.zhanpeng.torch.sampler.vectorized_sampler import VectorizedSampler


class NPO:

    def __init__(self,
                 env,
                 policy,
                 vf,
                 name="NPO",
                 pg_loss='vanilla',
                 step_size=0.02,
                 start_itr=0,
                 total_timesteps=10e6,
                 n_steps=2048,
                 n_minibatch=1,
                 gamma=0.99,
                 lam=0.95,
                 optimizer=None,
                 optimizer_args=None,
                 policy_ent_coeff=1e-2,
                 n_optimization_steps=1,
                 sampler_args=None,
                 **kwargs):
        # TODO: add serialization
        self._env = env
        self._policy = policy
        self._vf = vf

        self._total_timesteps = total_timesteps
        self._n_steps = n_steps  # Number of step performed for each update
        self._n_minibatch = n_minibatch
        self._gamma = gamma
        self._start_itr = start_itr
        self.lam = lam

        self.name = name
        self._pg_loss = pg_loss
        self._step_size = step_size
        self._n_optimization_steps = n_optimization_steps

        # Only support vectorized sampler
        if sampler_args is None:
            sampler_args = dict()
        self.sampler = VectorizedSampler(self, **sampler_args)

    def train(self):
        for itr in range(self.start_itr, self.n_itr):
            with logger.prefix('itr #%d | ' % itr):
                paths = self.obtain_samples(itr)
                samples_data = self.process_samples(itr, paths)
                self.optimize(samples_data)

    def process_samples(self, itr, paths):
        pass

    def optimize(self, batch):
        pass

    def obtain_samples(self, itr):
        self.sampler.obtain_samples(itr)
