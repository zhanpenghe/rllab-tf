import tensorflow as tf
from sandbox.zhanpeng.tf.value_functions.q_functions import MLPQFunction
from sandbox.zhanpeng.tf.algos.dqn import DQN
from rllab.envs.gym_env import GymEnv
from rllab.envs.normalized_env import normalize
from rllab.misc.instrument import run_experiment_lite


def run_task(*_):
    env = normalize(GymEnv('CartPole-v0'))

    qf = MLPQFunction(env=env, hidden_layer_sizes=[64], activation_fn=tf.nn.tanh)

    dqn = DQN(
        env=env,
        qf=qf,
        lr=1e-3,
        max_pool_size=50000,
    )

    dqn.train()


run_experiment_lite(
    run_task,
    # Number of parallel workers for sampling
    n_parallel=4,
    # Only keep the snapshot parameters for the last iteration
    snapshot_mode="last",
    # Specifies the seed for the experiment. If this is not provided, a random seed
    # will be used
    seed=1,
    python_command='/home/xxx/.conda/envs/rllab3/bin/python',
    # plot=True,
)
