import tensorflow as tf
from sandbox.zhanpeng.tf.algos.rl_algorithms import RLAlgorithm
from sandbox.zhanpeng.tf.policies.argmax import ArgmaxDiscretePolicy
from sandbox.zhanpeng.tf.exploration_strategy.epsilon_greedy import EpsilonGreedy
from rllab.algos.ddpg import SimpleReplayPool
from rllab.misc import ext


class DQN(RLAlgorithm):

    def __init__(self,
                 env,
                 qf,
                 es=None,
                 lr=1e-3,
                 gamma=0.99,
                 batch_size=1000,
                 max_pool_size=1000000,
                 min_pool_size=1000,
                 target_update_period=1000,
                 epoch_length=500,
                 n_epochs=1000,
                 tau=1e-3,
                 epsilon=0.1,
                 max_path_length=500,):
        super(DQN, self).__init__()

        self._env = env
        self._qf = qf

        self._n_actions = self._env.action_space.n

        self._policy = ArgmaxDiscretePolicy(qf=self._qf)

        self._es = es if es else EpsilonGreedy(self._env.action_space)   # TODO change this with a default value
        self._es.initialize(policy=self._policy)

        self._lr = lr
        self._target_update_period = target_update_period
        self._tau = tau
        self._epsilon = epsilon

        self._batch_size = batch_size
        self._max_pool_size = max_pool_size
        self._min_pool_size = min_pool_size

        self._n_epochs = n_epochs
        self._epoch_length = epoch_length
        self._max_path_length = max_path_length
        self._gamma = gamma

        self._build_placeholders()
        self._build_training_op()
        self._build_target_update_op()

    def train(self, sess=None):

        if sess is None:
            sess = tf.Session()

        sess.run(tf.global_variables_initializer())

        replay_buffer = SimpleReplayPool(
            max_pool_size=self._max_pool_size,
            observation_dim=self._env.observation_space.flat_dim,
            action_dim=self._env.action_space.flat_dim,)

        path_length = 0
        with sess.as_default():
            for ep in range(self._n_epochs):
                observation = self._env.reset()
                for ep_iter in range(self._epoch_length):
                    action = self._es.get_action(observation)
                    next_observation, reward, terminal, _ = self._env.step(action)
                    replay_buffer.add_sample(observation, action, reward, terminal)

                    path_length += 1

                    observation = next_observation

                    if terminal and path_length >= self._max_path_length:
                        observation = self._env.reset()
                        path_length = 0

                    iter = ep * self._epoch_length + ep_iter
                    if replay_buffer.size > self._min_pool_size:
                        batch = replay_buffer.random_batch(self._batch_size)
                        self._do_training(iter, batch)
                    if iter % self._target_update_period == 0:
                        self._update_target()

    def _do_training(self, iter, batch):
        obs, actions, rewards, next_obs, terminals = ext.extract(
            batch,
            "observations", "actions", "rewards", "next_observations",
            "terminals"
        )
        feed_dict = {
            self._obs_ph: obs,
            self._next_obs_ph: next_obs,
            self._action_ph: actions,
            self._terminal_ph: terminals,
            self._reward_ph: rewards,
        }
        sess = tf.get_default_session()
        q_pred = sess.run(self.q_pred, feed_dict)
        q_target = sess.run(self.q_target, feed_dict)
        td_error = sess.run(self.td_error, feed_dict)
        print(q_pred.shape)
        print(td_error)
        loss = sess.run(self._training_op, feed_dict)
        # print(loss)
        return loss

    def _build_training_op(self):
        with tf.variable_scope('target'):
            next_q_values = self._qf.values_op(observation_ph=self._next_obs_ph, reuse=False)

        q_values = self._qf.values_op(observation_ph=self._obs_ph, reuse=True)
        print(self._action_ph.shape, q_values.shape)
        q_pred = tf.reduce_sum(q_values * self._action_ph, axis=1)
        print(q_pred.shape)
        self.q_pred = q_pred
        masked_next_q = (1.0 - self._terminal_ph) * tf.reduce_max(next_q_values, axis=1)
        q_target = self._reward_ph + self._gamma * masked_next_q
        self.q_target = q_target
        td_error = tf.reduce_mean(tf.square(q_pred - tf.stop_gradient(q_target)))
        self.td_error = td_error
        self._training_op = tf.train.AdamOptimizer(self._lr).minimize(
            loss=td_error, var_list=self._qf.get_params_internal())

    def _build_target_update_op(self):
        source_params = self._qf.get_params_internal()
        target_params = self._qf.get_params_internal(scope='target')

        self._update_target_op = [
            tf.assign(tgt, src)
            for tgt, src in zip(target_params, source_params)
        ]

    def _build_placeholders(self):
        self._obs_ph = self._qf.get_observations_ph()
        self._next_obs_ph = tf.placeholder(
            tf.float32,
            shape=[None, self._env.observation_space.flat_dim],
            name='next_obs_ph',
        )
        self._action_ph =tf.placeholder(
            tf.float32,
            shape=[None, self._n_actions],
            name='action_ph'
        )
        self._reward_ph = tf.placeholder(
            tf.float32,
            shape=[None],
            name='reward_ph',
        )
        self._terminal_ph = tf.placeholder(
            tf.float32,
            shape=[None],
            name='terminal_ph'
        )

    def _update_target(self):
        tf.get_default_session().run(self._update_target_op)
