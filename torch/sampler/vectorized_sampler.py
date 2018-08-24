import itertools
import pickle

import numpy as np


class VecEnvExecutor(object):
    def __init__(self, envs, max_path_length):
        self.envs = envs
        self._action_space = envs[0].action_space
        self._observation_space = envs[0].observation_space
        self.ts = np.zeros(len(self.envs), dtype='int')
        self.max_path_length = max_path_length

    def step(self, action_n):
        all_results = [env.step(a) for (a, env) in zip(action_n, self.envs)]
        obs, rewards, dones, env_infos = list(
            map(list, list(zip(*all_results))))
        dones = np.asarray(dones)
        rewards = np.asarray(rewards)
        self.ts += 1
        if self.max_path_length is not None:
            dones[self.ts >= self.max_path_length] = True
        for (i, done) in enumerate(dones):
            if done:
                obs[i] = self.envs[i].reset()
                self.ts[i] = 0
        return obs, rewards, dones, env_infos # tensor_utils.stack_tensor_dict_list(env_infos)

    def reset(self):
        results = [env.reset() for env in self.envs]
        self.ts[:] = 0
        return results

    @property
    def num_envs(self):
        return len(self.envs)

    @property
    def action_space(self):
        return self._action_space

    @property
    def observation_space(self):
        return self._observation_space

    def close(self):
        pass


class VectorizedSampler:

    def __init__(self, algo, n_envs=None):
        self.algo = algo
        self.n_envs = n_envs

    def start_worker(self):
        n_envs = self.n_envs
        if n_envs is None:
            n_envs = int(self.algo.batch_size / self.algo.max_path_length)
            n_envs = max(1, min(n_envs, 100))

        envs = [pickle.loads(pickle.dumps(self.algo.env)) for _ in (n_envs)]
        self.vec_env = VecEnvExecutor(env=envs, max_path_length=self.algo.max_path_length)

    def obtain_samples(self, itr):

        paths = []
        n_sample = 0

        obses = self.vec_env.reset()
        dones = np.asarray([True] * self.vec_env.num_envs)
        running_paths = [None] * self.vec_env.num_envs

        policy = self.algo.policy

        while n_sample < self.algo.batch_size:
            # TODO: maybe we dont need this
            policy.reset(dones)
            actions, agent_infos = policy.get_actions(obses)

            next_obses, rewards, done, env_infos = self.vec_env.step(actions)

            if env_infos is None:
                env_infos = [dict() for _ in range(self.vec_env.num_envs)]
            if agent_infos is None:
                agent_infos = [dict() for _ in range(self.vec_env.num_envs)]

            for idx, observation, action, reward, env_info, agent_info, done in zip(itertools.count(), obses, actions, rewards, env_infos, agent_infos, dones):
                if running_paths[idx] is None:
                    running_paths[idx] = dict(
                        observations=[],
                        actions=[],
                        rewards=[],
                        env_infos=[],
                        agent_infos=[],
                    )
                running_paths[idx]["observations"].append(observation)
                running_paths[idx]["actions"].append(action)
                running_paths[idx]["rewards"].append(reward)
                running_paths[idx]["env_infos"].append(env_info)
                running_paths[idx]["agent_infos"].append(agent_info)

                if done:
                    paths.append(
                        dict(
                            observations=running_paths[idx]["observations"],
                            actions=running_paths[idx]["actions"],
                            rewards=running_paths[idx]["rewards"],
                            env_infos=running_paths[idx]["env_infos"],
                            agent_infos=running_paths[idx]["agent_infos"]
                        )
                    )
                    n_samples += len(running_paths[idx]["rewards"])
                    running_paths[idx] = None
            obses = next_obses
    return paths
