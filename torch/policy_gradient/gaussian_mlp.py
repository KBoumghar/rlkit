import torch
from rlkit.torch.core import PyTorchModule
import numpy as np

class GaussianMlp(PyTorchModule):
    def __init__(self,
                 mean_mlp,
                 log_var_mlp,
                 min_variance=1e-3,
                 ):
        super().__init__()
        if min_variance is None:
            self.log_min_variance = None
        else:
            self.log_min_variance = float(np.log(min_variance))

        self.mean_mlp = mean_mlp
        self.log_var_mlp = log_var_mlp

    def forward(self, input):

        mean = self.mean_mlp(input)
        logvar = self.log_var_mlp(input)

        if self.log_min_variance is not None:
            logvar = self.log_min_variance + torch.abs(logvar)

        return (mean, logvar)