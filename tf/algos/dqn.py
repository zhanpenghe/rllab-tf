import tensorflow as tf
from sandbox.zhanpeng.tf.algos.rl_algorithms import RLAlgorithm
from sandbox.zhanpeng.tf.policies.argmax import ArgmaxDiscretePolicy
from sandbox.zhanpeng.tf.exploration_strategy.epsilon_greedy import EpsilonGreedy
from sandbox.zhanpeng.tf.replay_buffers.replay_buffer import SimpleReplayBuffer


class DQN(RLAlgorithm):

    def __init__(self,
                 env,
                 qf,
                 es=None,
                 lr=1e-3,
                 gamma=0.99,
                 batch_size=32,
                 max_pool_size=1000000,
                 min_pool_size=1000,
                 target_update_period=1000,
                 epoch_length=1000,
                 n_epochs=1000,
                 epsilon=0.1,
                 max_path_length=500,):
        super(DQN, self).__init__()

        self._env = env
        self._n_actions = self._env.action_space.n
        self._qf = qf

        self._lr = lr
        self._target_update_period = target_update_period
        self._epsilon = epsilon

        self._batch_size = batch_size
        self._max_pool_size = max_pool_size
        self._min_pool_size = min_pool_size

        self._n_epochs = n_epochs
        self._epoch_length = epoch_length
        self._max_path_length = max_path_length
        self._gamma = gamma

        self._policy = ArgmaxDiscretePolicy(qf=self._qf)

        self._es = es if es else EpsilonGreedy(self._env.action_space, prob_random_action=self._epsilon)
        self._es.initialize(policy=self._policy)

        self._build_placeholders()
        self._build_training_op()
        self._build_target_update_op()

    def train(self, sess=None):

        if sess is None:
            sess = tf.Session()

        sess.run(tf.global_variables_initializer())

        replay_buffer = SimpleReplayBuffer(env_spec=self._env.spec, max_replay_buffer_size=self._max_pool_size)

        path_length = 0
        episode_rewards = 0
        observation = self._env.reset()

        with sess.as_default():
            for ep in range(self._n_epochs):
                print('EP%d' % ep)
                random_move = 0
                policy_move = 0
                mean_loss = 0
                trained_iter = 0
                for ep_iter in range(self._epoch_length):
                    self._env.render()
                    action, info = self._es.get_action(observation)
                    if info == 'random':
                        random_move += 1
                    elif info == 'policy':
                        policy_move += 1
                    next_observation, reward, terminal, _ = self._env.step(action)

                    replay_buffer.add_sample(
                        observation=observation,
                        next_observation=next_observation,
                        action=action,
                        terminal=terminal,
                        reward=reward,
                    )

                    episode_rewards += reward
                    path_length += 1

                    observation = next_observation

                    if terminal or path_length >= self._max_path_length:
                        observation = self._env.reset()
                        path_length = 0
                        print('ep_reward: %d'%episode_rewards)
                        episode_rewards = 0

                    iter = ep * self._epoch_length + ep_iter
                    if replay_buffer.size > self._min_pool_size:
                        batch = replay_buffer.random_batch(self._batch_size)
                        loss = self._do_training(iter, batch)
                        mean_loss += loss
                        trained_iter += 1

                    if iter % self._target_update_period == 0:
                        self._update_target()

                print('Random move: %d' % random_move)
                print('Policy move: %d' % policy_move)
                print('Mean loss: %f' % (mean_loss/self._epoch_length))

    def _do_training(self, iter, batch):
        feed_dict = {
            self._obs_ph: batch['observations'],
            self._next_obs_ph: batch['next_observations'],
            self._action_ph: batch['actions'],
            self._terminal_ph: batch['terminals'],
            self._reward_ph: batch['rewards'],
        }
        sess = tf.get_default_session()
        td_error = sess.run(self._td_error, feed_dict)
        sess.run(self._training_op, feed_dict)
        return td_error

    def _build_training_op(self):
        q_values = self._qf.get_values_op()
        q_pred = tf.reduce_sum(q_values * tf.one_hot(self._action_ph, self._n_actions), axis=1)

        with tf.variable_scope('target'):
            next_q_values = self._qf.values_op(observation_ph=self._next_obs_ph, reuse=False)

        masked_next_q = (1.0 - self._terminal_ph) * tf.reduce_max(next_q_values, axis=1)
        q_target = self._reward_ph + self._gamma * masked_next_q

        td_error = tf.reduce_mean(tf.square(q_pred - tf.stop_gradient(q_target)))
        self._td_error = td_error
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
            tf.int32,
            shape=[None],
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
        print('Target updated')
        tf.get_default_session().run(self._update_target_op)
