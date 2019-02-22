"""
This should results in an average return of ~3000 by the end of training.

Usually hits 3000 around epoch 80-100. Within a see, the performance will be
a bit noisy from one epoch to the next (occasionally dips dow to ~2000).

Note that one epoch = 5k steps, so 200 epochs = 1 million steps.
"""
import gym

import rlkit.torch.pytorch_util as ptu
from rlkit.data_management.obs_dict_replay_buffer import ObsDictRelabelingBuffer
from rlkit.exploration_strategies.base import (
    PolicyWrappedWithExplorationStrategy
)
from rlkit.exploration_strategies.gaussian_and_epsilon_strategy import (
    GaussianAndEpislonStrategy
)
from rlkit.launchers.launcher_util import setup_logger
#from rlkit.torch.her.her import HerDQN
from rlkit.torch.dqn.dqn import DQN

from rlkit.torch.networks import FlattenMlp, TanhMlpPolicy
import multiworld.envs.gridworlds
# from rlkit.exploration_strategies.base import (
#     PolicyWrappedWithExplorationStrategy
# )
# from rlkit.exploration_strategies.epsilon_greedy import EpsilonGreedy
# from rlkit.policies.argmax import ArgmaxDiscretePolicy

def experiment(variant):
    env = gym.make('GoalGridworld-Concatenated-v0')

    obs_dim = env.observation_space.low.size
    print(obs_dim)
    #goal_dim = env.observation_space.spaces['desired_goal'].low.size
    #import pdb; pdb.set_trace()
    action_dim = env.action_space.n
    qf1 = FlattenMlp(
        input_size=obs_dim,
        output_size=action_dim,
        hidden_sizes=[100, 100],
    )
#     qf2 = FlattenMlp(
#         input_size=obs_dim + goal_dim + action_dim,
#         output_size=1,
#         hidden_sizes=[400, 300],
#     )


#     replay_buffer = ObsDictRelabelingBuffer(
#         env=env,
#         **variant['replay_buffer_kwargs']
#     )
    algorithm = DQN(
#         her_kwargs=dict(
#             observation_key='observation',
#             desired_goal_key='desired_goal'
#         ),
#         dqn_kwargs = dict(
            env=env,
            qf=qf1,
            #policy = policy,
        #),
        #replay_buffer=replay_buffer,
        **variant['algo_kwargs']
    )
    algorithm.to(ptu.device)
    algorithm.train()


if __name__ == "__main__":
    variant = dict(
        algo_kwargs=dict(
            num_epochs=500,
            num_steps_per_epoch=1000,
            num_steps_per_eval=1000,
            max_path_length=50,
            batch_size=128,
            discount=0.99,
        ),
#         replay_buffer_kwargs=dict(
#             max_size=100000,
#             fraction_goals_rollout_goals=0.2,  # equal to k = 4 in HER paper
#             fraction_goals_env_goals=0.0,
#         ),
    )
    setup_logger('dqn-gridworld-experiment', variant=variant)
    experiment(variant)