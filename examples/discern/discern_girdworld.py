"""
This should results in an average return of -20 by the end of training.

Usually hits -30 around epoch 50.
Note that one epoch = 5k steps, so 200 epochs = 1 million steps.
"""
import gym
import torch
import rlkit.torch.pytorch_util as ptu
from rlkit.data_management.reward_replay_buffer import ObsDictRelabelingBuffer
from rlkit.exploration_strategies.base import (
    PolicyWrappedWithExplorationStrategy
)
from rlkit.exploration_strategies.gaussian_and_epsilon_strategy import (
    GaussianAndEpislonStrategy
)
from rlkit.launchers.launcher_util import setup_logger
from rlkit.torch.discern.discern import DISCERN
from rlkit.torch.networks import FlattenMlp, TanhMlpPolicy, ConcatComposition
import multiworld.envs.gridworlds


def experiment(variant):
    env = gym.make('GoalGridworld-v0')

    obs_dim = env.observation_space.spaces['observation'].low.size
    goal_dim = env.observation_space.spaces['desired_goal'].low.size
    #import pdb; pdb.set_trace()
    action_dim = env.action_space.n
    state_encoder = FlattenMlp(
        input_size=obs_dim ,
        output_size=64,
        hidden_sizes=[200, 300],
    )
    top_layer = FlattenMlp(
        input_size=64*2 ,
        output_size=action_dim,
        hidden_sizes=[],
    )
    qf = ConcatComposition(state_encoder, top_layer)
    Xi = FlattenMlp(
        input_size=64 ,
        output_size=64,
        hidden_sizes=[],
        output_activation=torch.tanh,
    )
    replay_buffer = ObsDictRelabelingBuffer(
        env=env,
        **variant['replay_buffer_kwargs']
    )
    algorithm = DISCERN(
        observation_key='observation',
        desired_goal_key='desired_goal',
        env=env,
        state_encoder=state_encoder,
        qf = qf,
        Xi = Xi,
        replay_buffer=replay_buffer,
        batch_size=64,
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
            discount=0.99,
        ),
        replay_buffer_kwargs=dict(
            max_size=100000,
            fraction_goals_rollout_goals=0.5,  # equal to k = 4 in HER paper
            fraction_goals_env_goals=0.0,
        ),
    )
    setup_logger('discern-gridworld-experiment', variant=variant)
    experiment(variant)
