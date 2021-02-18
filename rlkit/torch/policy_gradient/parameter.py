import numpy as np
import torch.nn as nn
import railrl.torch.pytorch_util as ptu
import torch

class Parameter(nn.Module):
    def __init__(self, input_dim, output_dim, init):
        super(Parameter, self).__init__()
        self.output_dim = output_dim
        self.init = init
        self.param_init = ptu.from_numpy(np.zeros((output_dim)) + init).float()
        #TODO: fix this nn.Parameter(self.param_init)
        self.params_var = nn.Parameter(self.param_init)

    def forward(self, x):

        if len(x.shape) == 1:
            zeros = ptu.zeros(self.output_dim)
            return zeros + self.params_var
        else:
            zeros = ptu.zeros((x.shape[0], self.output_dim))
            return zeros + self.params_var.view(1, -1)