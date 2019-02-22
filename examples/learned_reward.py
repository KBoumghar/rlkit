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
from gridworld.algorithms.models import GoalPolicy
from gridworld.rewards.reward_functions import SubstractionRewards
from variant import VARIANT
import torch
import glob
import cv2
from gridworld.algorithms.datasets import SelfContrastiveDeltaDataset
pol = 'PickupHammerPolicy'
device = torch.device("cuda")
#model_path = '/home/coline/affordance_world/affordance_world/agentcentric_model_2480056_epoch06.pt'
model_path = '/home/coline/affordance_world/affordance_world/contrastive_results/selfcontrastive_314916_epoch08.pt'
state_dict, epoch, num_per_epoch , _= torch.load(model_path)
#delta_star = np.load('/home/coline/affordance_world/affordance_world/EatBreadPolicy_delta.npy')
#delta_star =  torch.from_numpy(delta_star).to(device)
def load_path(path):
    image_files = sorted(glob.glob(path))
    images = [cv2.imread(img_name, cv2.IMREAD_COLOR) for img_name in image_files]
    return images

R = SubstractionRewards(state_dict, device)
path = '/home/coline/affordance_world/data/actions_notcentric/'+pol+'/img0000_*.png'
images = load_path(path)
RENDER_DIR = '/home/coline/Desktop/renderings/'
last_img = images[-1]
initgstar = R.get_delta_star(images[0], images[-1]).squeeze(0).cpu().detach().numpy()
R.delta_star = initgstar 
reward_fn = R


# train_loader = torch.utils.data.DataLoader(
#     DeltaDataset(directory='/home/coline/affordance_world/data/agent_centric_demonstrations/',
#                              tasks=['EatBreadPolicy','GoToHousePolicy'],
#                              train=True, tasksize=None),
#     batch_size=8, shuffle=True)

def experiment(variant):
    env = gym.make('HammerWorld-Delta-'+pol+'-v0')
    env.reward_function = reward_fn
    training_env = gym.make('HammerWorld-Delta-'+pol+'-v0')
    training_env.reward_function = reward_fn
    #import pdb; pdb.set_trace()
    qf = GoalPolicy(
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
        #reward_model=R.model,
        #reward_dataloader=train_loader,
        **variant['algo_params']
    )
    algorithm.to(ptu.device)
    algorithm.train()


if __name__ == "__main__":
    # noinspection PyTypeChecker
    VARIANT['algo_params']['replay_buffer_size']= 100000
    setup_logger(pol+'-deltapath-model', variant=VARIANT)
    experiment(VARIANT)
