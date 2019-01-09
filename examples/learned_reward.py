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
from gridworld.rewards.reward_functions import DeltaRewards
from variant import VARIANT
import torch

device = torch.device("cuda")
model_path = '/home/coline/affordance_world/affordance_world/agentcentric_model_2480056_epoch06.pt'
state_dict, epoch, num_per_epoch , _= torch.load(model_path)
delta_star = np.load('/home/coline/affordance_world/affordance_world/EatBreadPolicy_delta.npy')
delta_star =  torch.from_numpy(delta_star).to(device)
R = DeltaRewards(state_dict, device, trajectory=0, delta_m=1,delta_star=delta_star)
reward_fn = R
def experiment(variant):
    env = gym.make('HammerWorld-PlaceholderReward-EatBreadPolicy-v0')
    env.reward_function = reward_fn
    training_env = gym.make('HammerWorld-PlaceholderReward-EatBreadPolicy-v0')
    training_env.reward_function = reward_fn
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
    VARIANT['algo_params']['replay_buffer_size']= 100000
    setup_logger('eatbread-deltastar-model', variant=VARIANT)
    experiment(VARIANT)
