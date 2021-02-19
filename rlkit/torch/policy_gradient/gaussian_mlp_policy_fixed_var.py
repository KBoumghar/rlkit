import torch
import numpy as np
import pdb

from rlkit.torch.core import PyTorchModule
from rlkit.policies.base import Policy
import rlkit.torch.pytorch_util as ptu
import torch.distributions as tdist

class GaussianMlpPolicyFixedVar(Policy, PyTorchModule):
    def __init__(self,
                 mean_mlp,
                 log_var_mlp=0.2,
                 min_variance=1e-6,
                 ):
        super().__init__()
        if min_variance is None:
            self.log_min_variance = None
        else:
            #self.log_min_variance = float(np.log(min_variance)) #
            self.log_min_variance = ptu.from_numpy(np.log(np.array([min_variance])))

        self.mean_mlp = mean_mlp
        self.log_var_mlp = torch.tensor(log_var_mlp, requires_grad=False, device=ptu.device, dtype=torch.float32)

    def _rsample(self, mean, log_var):
        return mean + torch.randn(mean.shape, device=ptu.device) * torch.exp(log_var)

    def log_prob(self, x, mean, log_var):
        """Compute the log probability of an observation, parameterized by the mean and log(stddev**2) of a gaussian

        :param x: observation
        :param mean: The mean of the gaussian.
        :param log_var: The log(stddev**2) of the gaussian.
        :returns: 
        :rtype: 

        """
        # Assumes x is batch size by feature dim
        # Returns log_likehood for each point in batch
        zs = (x - mean) / torch.exp(log_var)

        dim = x.shape[0] if len(x.shape) == 1 else x.shape[-1]
        return -torch.sum(log_var, -1) - \
               0.5 * torch.sum(torch.pow(zs, 2), -1) -\
               0.5 * dim * np.log(2*np.pi)

    def entropy(self, mean, log_var):
        return torch.sum(log_var + np.log(np.sqrt(2 * np.pi * np.e)), -1)

    def get_action(self, observation, **kwargs):
        # (mean, log var)
        action_params = self.forward(ptu.from_numpy_or_pytorch(observation))

        # The argmax is the mean.
        if 'argmax' in kwargs and kwargs['argmax']:
            action = action_params[0]
        else:
            action = self._rsample(*action_params)

        action_log_prob = self.log_prob(action, *action_params)
        agent_info = dict(action_log_prob=ptu.get_numpy(action_log_prob))
        return ptu.get_numpy(action), agent_info

    def forward(self, input):
        """Predicts mean and log(stddev**2) of a diagonal gaussian

        :param input: 
        :returns: 
        :rtype: 

        """
        
        #import ipdb; ipdb.set_trace()
        mean = self.mean_mlp(input)
        logvar = self.log_var_mlp
        return (mean, logvar)
