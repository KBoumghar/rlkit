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
from rlkit.torch.conv_networks import ConvNet

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
        #print("input", input.shape)
        for i, fc in enumerate(self.fcs):
            #print("fc i", i, fc)
            
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


class ObjectMlp(Mlp):

    def __init__(
            self,
            *args,
            max_objects=10,
            **kwargs
    ):
        self.save_init_params(locals())
        super().__init__(*args, **kwargs)
        self.max_objects = max_objects
    
    def forward(self, inputs, return_preactivations=False):
        outputs = []
        shape = inputs.shape 
        inputs = inputs.view((-1,self.input_size))
        valid = inputs[:, -1]
        valid = valid.unsqueeze(1)
        outputs = super().forward(inputs, return_preactivations=False)
        #import pdb; pdb.set_trace()
        valid_outputs = outputs*valid
        dA =outputs.shape[1]
        outputs = valid_outputs.view((shape[0], self.max_objects, dA))
        #import pdb; pdb.set_trace()
        output = torch.sum(outputs, dim=1)
        if return_preactivations:
            return output, output
        return output

class RelationalObjectMlp(Mlp):

    def __init__(
            self,
            *args,
            max_objects=10,
            **kwargs
    ):
        self.save_init_params(locals())
        super().__init__(*args, **kwargs)
        self.max_objects = max_objects
        hidden_init=ptu.fanin_init
        b_init_value=0.1
        def mlp(layer_sizes, scope, last_layer_init=None):
            in_size = layer_sizes[0]
            layers = []
            for i, next_size in enumerate(layer_sizes[1:]):
                fc = nn.Linear(in_size, next_size)
                in_size = next_size
                if i == len(layer_sizes[:1])-1 and last_layer_init is not None:
                    last_layer_init(fc.weight)
                else:
                    hidden_init(fc.weight)
                fc.bias.data.fill_(b_init_value)
                layers.append(fc)
                self.__setattr__(scope+"fc{}".format(i), fc)
            return layers
        self.key_query_mlp = mlp([self.input_size, 32,20], 'key')
        self.embedding_mlp = mlp([self.input_size, 32,self.input_size] , 'embedding')
        self.action_mlp = mlp([self.input_size*2, 32, self.output_size], 'action')
    def _run_mlp(self, input, mlp, activation, last_activation):
        h = input
        for i, fc in enumerate(mlp[:-1]):
            h = fc(h)
            if self.layer_norm and i < len(self.fcs) - 1:
                h = self.layer_norms[i](h)
            h = activation(h)
        preactivation = mlp[-1](h)
        output = last_activation(preactivation)
        return output
    
    def forward(self, inputs, return_preactivations=False):
        outputs = []
        shape = inputs.shape 
        inputs = inputs.view((-1,self.input_size))
        valid = inputs[:, -1]
        valid = valid.unsqueeze(1)
        key_queries = self._run_mlp(inputs, self.key_query_mlp, F.relu, F.softmax)
        keys = (key_queries[:, :10]*valid).view((shape[0], self.max_objects, 10))
        queries = (key_queries[:, 10:]*valid).view((shape[0], self.max_objects, 10)).permute(0,2,1)
        weights = torch.bmm(keys, queries)
        values = (self._run_mlp(inputs, self.embedding_mlp, F.relu, identity)).view(shape[0], self.max_objects, self.input_size)
        total_actions = []# torch.zeros(shape[0], self.output_size)
        #import pdb; pdb.set_trace()
        # if shape[0] > 1:
        #     import pdb; pdb.set_trace()
        for i in range(self.max_objects):
            for k in range(self.max_objects):
                 flat = torch.cat((values[:, i], values[:, k]), dim=1)
                 w = (weights[:, i,k]).unsqueeze(1)
                 actions = self._run_mlp(flat, self.action_mlp,F.relu, identity)*w
                 total_actions.append(actions)
        
        #outputs = super().forward(inputs, return_preactivations=False)
        #import pdb; pdb.set_trace()
        #valid_outputs = outputs*valid
        #dA =outputs.shape[1]
        #outputs = valid_outputs.view((shape[0], self.max_objects, dA))
        #import pdb; pdb.set_trace()
        #output = torch.sum(outputs, dim=1)
        output = sum(total_actions)
        self.my_actions = output.data.numpy()#[ta.data.numpy() for ta in total_actions]
        self.my_inputs = inputs.data.numpy()
        self.my_weights = weights.data.numpy()
        if return_preactivations:
            return output, output
        return output


    

class FlattenMlp(Mlp):
    """
    Flatten inputs along dimension 1 and then pass through MLP.
    """

    def forward(self, *inputs, **kwargs):
        flat_inputs = torch.cat(inputs, dim=1)
        return super().forward(flat_inputs, **kwargs)


class MlpPolicy(Mlp, Policy):
    """
    A simpler interface for creating policies.
    """

    def __init__(
            self,
            *args,
            obs_normalizer: TorchFixedNormalizer = None,
            **kwargs
    ):
        self.save_init_params(locals())
        super().__init__(*args, **kwargs)
        self.obs_normalizer = obs_normalizer

    def forward(self, obs, **kwargs):
        if self.obs_normalizer:
            obs = self.obs_normalizer.normalize(obs)
        return super().forward(obs, **kwargs)

    def get_action(self, obs_np):
        actions = self.get_actions(obs_np[None])
        return actions[0, :], {}

    def get_actions(self, obs):
        return self.eval_np(obs)


class TanhMlpPolicy(MlpPolicy):
    """
    A helper class since most policies have a tanh output activation.
    """
    def __init__(self, *args, **kwargs):
        self.save_init_params(locals())
        super().__init__(*args, output_activation=torch.tanh, **kwargs)
