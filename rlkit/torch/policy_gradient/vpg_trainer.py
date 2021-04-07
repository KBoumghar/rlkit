from collections import OrderedDict

import numpy as np
import torch.optim as optim

import railrl.torch.pytorch_util as ptu
from rlkit.core import logger
from rlkit.data_management.env_replay_buffer import VPGEnvReplayBuffer
from rlkit.torch.core import np_to_pytorch_batch
from rlkit.torch.distributions import TanhNormal
from rlkit.torch.torch_rl_algorithm import TorchBatchRLAlgorithm, TorchTrainer
import torch

class VPGTrainer(TorchTrainer):
    """
    Vanilla Policy Gradient
    """

    def __init__(
            self,
            policy,
            policy_learning_rate,
            replay_buffer
    ):
        super().__init__()
        self.policy = policy
        self.policy_learning_rate = policy_learning_rate
        self.policy_optimizer = optim.Adam(self.policy.parameters(),
                                           lr=self.policy_learning_rate)
        self.eval_statistics = {}
        self.need_to_update_eval_statistics = True
        self.replay_buffer = replay_buffer

    def end_epoch(self, epoch):
        self.replay_buffer.empty_buffer()

    def train_from_torch(self, batch):
        #batch = self.get_batch()
        obs = batch['observations']
        actions = batch['actions']
        returns = batch['returns']
        """
        Policy operations.
        """

        _, means, _, _, _, stds,_, _ = self.policy.forward(obs,)
        log_probs = TanhNormal(means, stds).log_prob(actions)
        log_probs_times_returns = log_probs * returns
        policy_loss = -1*torch.mean(log_probs_times_returns)



        """
        Update Networks
        """

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        self.eval_statistics['Policy Loss'] = ptu.get_numpy(policy_loss)

    @property
    def networks(self):
        return [
            self.policy,
        ]
