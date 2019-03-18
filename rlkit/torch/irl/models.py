"""
General networks for pytorch.

Algorithm-specific networks should go else-where.
"""
import torch
from torch import nn as nn
from torch.nn import functional as F

from rlkit.policies.base import Policy
from rlkit.torch import pytorch_util as ptu
from rlkit.torch.core import PyTorchModule
from rlkit.torch.data_management.normalizer import TorchFixedNormalizer
from rlkit.torch.modules import LayerNorm


def identity(x):
    return x


class Mlp(PyTorchModule):
    def __init__(
            self,
            hidden_sizes,
            output_size,
            input_size,
            init_w=3e-3,
            hidden_activation=F.relu,
            output_activation=identity,
            hidden_init=ptu.fanin_init,
            b_init_value=0.1,
            layer_norm=False,
            layer_norm_kwargs=None,
    ):
        self.save_init_params(locals())
        super().__init__()

        if layer_norm_kwargs is None:
            layer_norm_kwargs = dict()

        self.input_size = input_size
        self.output_size = output_size
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.layer_norm = layer_norm
        self.fcs = []
        self.layer_norms = []
        in_size = input_size

        for i, next_size in enumerate(hidden_sizes):
            fc = nn.Linear(in_size, next_size)
            in_size = next_size
            hidden_init(fc.weight)
            fc.bias.data.fill_(b_init_value)
            self.__setattr__("fc{}".format(i), fc)
            self.fcs.append(fc)

            if self.layer_norm:
                ln = LayerNorm(next_size)
                self.__setattr__("layer_norm{}".format(i), ln)
                self.layer_norms.append(ln)

        self.last_fc = nn.Linear(in_size, output_size)
        self.last_fc.weight.data.uniform_(-init_w, init_w)
        self.last_fc.bias.data.uniform_(-init_w, init_w)

    def forward(self, input, return_preactivations=False):
        h = input
        for i, fc in enumerate(self.fcs):
            h = fc(h)
            if self.layer_norm and i < len(self.fcs) - 1:
                h = self.layer_norms[i](h)
            h = self.hidden_activation(h)
        preactivation = self.last_fc(h)
        output = self.output_activation(preactivation)
        if return_preactivations:
            return output, preactivation
        else:
            return output
        
    import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from rlkit.torch.core import PyTorchModule
def layer_init(layer, w_scale=1.0):
    nn.init.orthogonal_(layer.weight.data)
    layer.weight.data.mul_(w_scale)
    nn.init.constant_(layer.bias.data, 0)
    return layer
from rlkit.torch import pytorch_util as ptu

class CNN(PyTorchModule):
    def __init__(self, in_channels=3, output_size=7, agent_centric=True):
        self.save_init_params(locals())
        super().__init__()
        self.agent_centric = agent_centric
        self.feature_dim = 3
        self.in_channels = in_channels
        self.output_size = output_size
        if self.agent_centric:
            self.fc_out = self.feature_dim*17*19#7*8* #+2
        else:
            self.fc_out = self.feature_dim*7*8 #+2
        self.conv1 = layer_init(nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=3))
        self.conv2 = layer_init(nn.Conv2d(in_channels, 3, kernel_size=3, stride=1))
        self.conv3 = layer_init(nn.Conv2d(3, 3, kernel_size=2, stride=1))
        self.fc_1  = nn.Linear(self.fc_out, 128)
        self.fc_2= nn.Linear(128, self.output_size)
        ptu.fanin_init(self.fc_1.weight)
        ptu.fanin_init(self.fc_2.weight)


    def forward(self,x):
        batch_size = x.shape[0]
        if self.agent_centric:
            x = torch.reshape(x, (batch_size, 33*2,30*2,3))
        elif self.in_channels == 6:
            x = torch.reshape(x, (batch_size, 2, 33,30, 3))
            x = x.permute([0, 2,3,4,1])
            x = torch.reshape(x, (batch_size, 33,30, 6))

        else:
            x = torch.reshape(x, (batch_size, 33,30,3))
        #self.size0 = x.shape

        x = x.permute( [0,3,1,2])
        y = F.relu(self.conv1(x))
        y = F.relu(self.conv2(y))
        y = F.relu(self.conv3(y))

        y = torch.reshape(y, (batch_size, -1))
        y = F.relu(self.fc_1(y))
        logpi = self.fc_2(y)
        return logpi

class F_model(PyTorchModule):
    def __init__(self, discount=0.9,agent_centric=False):
        self.save_init_params(locals())
        super().__init__()
        self.g = CNN(agent_centric=agent_centric)
        self.h = CNN(agent_centric=agent_centric, output_size=1)
        self.discount = discount
        
    def forward(self, obs, a, next_obs):
        #print("obs", obs.shape, "a", a.shape)
        r = torch.sum(self.g(obs)*a, dim=1).unsqueeze(1)
        v_next = self.h(next_obs)
        v = self.h(obs)
        return r + self.discount*v_next - v