import torch
import numpy as np
import pdb

from railrl.torch.core import PyTorchModule
from railrl.policies.base import Policy
import railrl.torch.pytorch_util as ptu
import torch.distributions as tdist

class CategoricalMlpPolicy(Policy, PyTorchModule):
    def __init__(self, action_network, return_onehot=True):
        super().__init__()
        self.action_network = action_network
        self.return_onehot = return_onehot
        if not return_onehot:
            raise NotImplementedError("not validated with cartpole")
        self.action_dim = self.action_network._modules['last_fc'].out_features

    def log_prob(self, one_hot_actions, probs, none):
        """

        :param one_hot_actions: (B, A) one-hot actions
        :param probs: (B, A) per-action probabilities
        :returns: 
        :rtype: 

        """
        assert(probs.shape[-1] == self.action_dim)
        assert(one_hot_actions.shape[-1] == self.action_dim)
        # Replay buffer stores discrete actions as onehots
        return torch.log(probs[torch.arange(one_hot_actions.shape[0]), one_hot_actions.argmax(1)])

    def get_action(self, observation, argmax=False):
        action_dist = self.forward(ptu.from_numpy(observation))
        action_idx = self.rsample(*action_dist)
        if argmax: action_idx[0, 0] = torch.argmax(action_dist[0])
        action_onehot = ptu.zeros(action_dist[0].shape, dtype=torch.int64)
        action_onehot[0, action_idx[0, 0]] = 1
        action_log_prob = self.log_prob(action_onehot, *action_dist)
        agent_info = dict(action_log_prob=ptu.get_numpy(action_log_prob), action_dist=ptu.get_numpy(action_dist[0]))
        if self.return_onehot:
            return ptu.get_numpy(action_onehot).flatten().tolist(), agent_info
        else:
            return ptu.get_numpy(action_idx).ravel().item(), agent_info
        
    def entropy(self, probs, none):
        return - (probs * torch.log(probs)).sum(-1)

    def rsample(self, probs, none):
        s = tdist.Categorical(probs, validate_args=True).sample((1,))
        return s

    def forward(self, input):
        if len(input.shape) == 1:
            action_probs = self.action_network(input.view(1, -1))
        else:
            action_probs = self.action_network(input)
        return (action_probs, None)

    def kl(self, source_probs, dest_probs):
        source_log_probs = torch.log(source_probs)
        dest_log_probs = torch.log(dest_probs)
        assert(source_probs.shape[-1] == self.action_dim)
        assert(dest_probs.shape[-1] == self.action_dim)

        # These must be true for discrete action spaces.
        assert(0 <= source_probs.min() <= source_probs.max() <= 1)
        assert(0 <= dest_probs.min() <= dest_probs.max() <= 1)
        kl = (source_probs * (source_log_probs - dest_log_probs)).sum(-1)
        assert(ptu.get_numpy(kl.min()) >= -1e-5)
        return kl
