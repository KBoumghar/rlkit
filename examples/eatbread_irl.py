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
from rlkit.torch.irl.scalar_irl import IRL
from rlkit.torch.irl.models import F_model
#from rlkit.torch.networks import FlattenMlp, TanhMlpPolicy, ConcatComposition
import gridworld.envs
from gridworld.envs.grid_affordance import HammerWorld
from gridworld.algorithms.models import Policy
#from gridworld.rewards.reward_functions import SubstractionRewards
from variant import VARIANT
import torch
import glob
import cv2
pol = 'EatBreadPolicy'
device = torch.device("cuda")
from gridworld.algorithms.composite_dataset import IRLDataset

train_loader = torch.utils.data.DataLoader(
    IRLDataset(directory='/persistent/affordance_world/data/eatbread_irl_10x10/',
                             train=True, size=args.train_size),
    batch_size=8, shuffle=True, **kwargs)


def experiment(variant):
    env = HammerWorld(add_objects =[],res=3, visible_agent=True, use_exit=True, agent_centric=False, goal_dim=0, size=[10,10])


    obs_dim = env.observation_space.spaces['observation'].low.size
    goal_dim = env.observation_space.spaces['desired_goal'].low.size
    #import pdb; pdb.set_trace()
    action_dim = env.action_space.n
    qf =  Policy(
        # hidden_sizes=[32, 32],
        # input_size=int(np.prod(env.observation_space.shape)),
        in_channels = 3,
        output_size=action_dim,
        agent_centric = env.agent_centric
    )


    algorithm = IRL(
        env=env,
        state_encoder=state_encoder,
        qf = qf,
        Xi = Xi,
        batch_size=64,
        **variant['algo_kwargs']
    )
    algorithm.to(ptu.device)
    algorithm.train()


if __name__ == "__main__":
    variant = dict(
        algo_kwargs=dict(
            num_epochs=5000,
            num_steps_per_epoch=1000,
            num_steps_per_eval=1000,
            max_path_length=50,
            discount=0.99,
        ),
        replay_buffer_kwargs=dict(
            max_size=100000,
            fraction_goals_rollout_goals=0.2,  # equal to k = 4 in HER paper
            fraction_goals_env_goals=0.0,
        ),
    )
    setup_logger('discern-affordance-experiment', variant=variant)
    experiment(variant)
