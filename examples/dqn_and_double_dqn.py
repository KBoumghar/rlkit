"""
Run DQN on grid world.
"""

import gym
import numpy as np
from torch import nn as nn

import rlkit.torch.pytorch_util as ptu
from rlkit.launchers.launcher_util import setup_logger
from rlkit.torch.dqn.dqn import DQN
from rlkit.torch.networks import ObjectMlp, Mlp

envName = 'DiscretePointmass-v1'
envName = 'CartPole-v0'
def experiment(variant):
    env = gym.make(envName)
    
    training_env = gym.make(envName)

    qf = Mlp(
        hidden_sizes=[32, 32],
        input_size=int(np.prod(env.observation_space.shape)),
        output_size=env.action_space.n,
    )
    qf_criterion = nn.MSELoss()
    # Use this to switch to DoubleDQN
    # algorithm = DoubleDQN(
    #import pdb; pdb.set_trace()
    algorithm = DQN(
        env,
        training_env=training_env,
        qf=qf,
        qf_criterion=qf_criterion,
        **variant['algo_params']
    )
    if ptu.gpu_enabled():
        algorithm.cuda()
    algorithm.train()


if __name__ == "__main__":
    # noinspection PyTypeChecker
    variant = dict(
        algo_params=dict(
            num_epochs=5000,
            num_steps_per_epoch=1000,
            num_steps_per_eval=1000,
            batch_size=128,
            max_path_length=50,
            discount=0.95,
            epsilon=0.2,
            tau=0.001,
            hard_update_period=1000,
            save_environment=True,  # Can't serialize CartPole for some reason
        ),
    )
    setup_logger('dqn-object', variant=variant)
    experiment(variant)
