"""
Run DQN on grid world.
"""
import gym
import numpy as np
from torch import nn as nn
import gridworld.envs
import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt

import rlkit.torch.pytorch_util as ptu
from rlkit.launchers.launcher_util import setup_logger
from rlkit.torch.dqn.dqn import DQN
from rlkit.torch.networks import Mlp
from gridworld.algorithms.models import Policy
from variant import VARIANT

def experiment(variant):
    env = gym.make('HammerWorld-EatBreadPolicy-v0')
    training_env = gym.make('HammerWorld-EatBreadPolicy-v0')
    #import pdb; pdb.set_trace()
    qf = Policy(
        # hidden_sizes=[32, 32],
        # input_size=int(np.prod(env.observation_space.shape)),
        output_size=env.action_space.n,
    )
    qf_criterion = nn.MSELoss()
    # Use this to switch to DoubleDQN
    # algorithm = DoubleDQN(
    algorithm = DQN(
        env,
        training_env=training_env,
        qf=qf,
        qf_criterion=qf_criterion,
        **variant['algo_params']
    )
    algorithm.to(ptu.device)
    algorithm.train()


if __name__ == "__main__":
    # noinspection PyTypeChecker
    variant = dict(
        algo_params=dict(
            num_epochs=500,
            num_steps_per_epoch=1000,
            num_steps_per_eval=1000,
            batch_size=128,
            max_path_length=200,
            discount=0.90,
            epsilon=0.2,
            tau=0.002,
            learning_rate=0.001,
            hard_update_period=1000,
            save_environment=True,  # Can't serialize CartPole for some reason
        ),
    )
    VARIANT['algo_params']['replay_buffer_size']= 10000
    setup_logger('oneobjeatbread_convfanin', variant=VARIANT)
    experiment(VARIANT)
