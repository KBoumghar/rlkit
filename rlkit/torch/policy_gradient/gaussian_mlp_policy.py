import torch
from railrl.torch.core import PyTorchModule
import numpy as np
from railrl.policies.base import Policy
import railrl.torch.pytorch_util as ptu
import torch.distributions as tdist

class GaussianMlpPolicy(Policy, PyTorchModule):
    def __init__(self,
                 mean_mlp,
                 log_var_mlp,
                 min_variance=1e-6,
                 ):
        super().__init__()
        if min_variance is None:
            self.log_min_variance = None
        else:
            #self.log_min_variance = float(np.log(min_variance)) #
            self.log_min_variance = ptu.from_numpy(np.log(np.array([min_variance])))

        self.mean_mlp = mean_mlp
        self.log_var_mlp = log_var_mlp

    def rsample(self, mean, log_var):
        return mean + torch.randn(mean.shape, device=ptu.device) * torch.exp(log_var)

    def log_prob(self, x, mean, log_var):
        # Assumes x is batch size by feature dim
        # Returns log_likehood for each point in batch
        zs = (x - mean) / torch.exp(log_var)

        dim = x.shape[0] if len(x.shape) == 1 else x.shape[-1]
        return -torch.sum(log_var, -1) - \
               0.5 * torch.sum(torch.pow(zs, 2), -1) -\
               0.5 * dim * np.log(2*np.pi)

    def entropy(self, mean, log_var):
        return torch.sum(log_var + np.log(np.sqrt(2 * np.pi * np.e)), -1)

    def get_action(self, observation):
        action_params = self.forward(ptu.from_numpy(observation))
        action = self.rsample(*action_params)

        action_log_prob = self.log_prob(action, *action_params )
        agent_info = dict(action_log_prob=ptu.get_numpy(action_log_prob))
        return ptu.get_numpy(action), agent_info

    def forward(self, input):
        #import ipdb; ipdb.set_trace()
        mean = self.mean_mlp(input)
        logvar = self.log_var_mlp(input)
        if self.log_min_variance is not None:
            logvar = torch.max(self.log_min_variance, logvar)
            #logvar = self.log_min_variance + torch.abs(logvar)
        return (mean, logvar)
