
import numpy as np
import torch
import pdb

import rlkit.torch.core as core
import rlkit.policies.base as base
import rlkit.misc.class_util as classu
import rlkit.torch.pytorch_util as ptu

class ContinuousBernoulliMlpPolicy(base.Policy, core.PyTorchModule):
    def __init__(self, mlp, action_space=None, scale=None, shift=None):
        """

        policy whose actions are in [scale*0 + shift, scale*1 + shift] == [shift, scale + shift]

        It uses a continuous bernoulli distribution, valid for inputs [0, 1], and transforms actions into that space first.

        :param mlp: network that outputs vectors in R^n
        :param action_space: action space
        :param scale: multiplicative scaling, before shift. zero-scale transformations are turned into 1s, silently.
        :param shift: additive shift, after scaling
        :returns: 
        :rtype: 

        """
        super().__init__()

        self.mlp = mlp

        if action_space is not None:
            self.shift = ptu.from_numpy(action_space.low)
            self.scale = ptu.from_numpy(action_space.high - action_space.low)
        elif scale is not None and shift is not None:
            self.scale = ptu.from_numpy(scale)
            self.shift = ptu.from_numpy(shift)
        else:
            raise ValueError(f"{self.__class__.__name__}.__init__() fail")
        
        # Replace zero-scale transformations with 1s.
        self.scale[torch.eq(self.scale, 0.0)] = 1.

    def _action_space_to_event_space(self, x):
        """

        :param x: values in [shift, scale + shift]
        :returns: values in [0, 1]
        :rtype: 

        """

        self._validate_action_space(x)
        return (x - self.shift) / self.scale

    def _event_space_to_action_space(self, x):
        self._validate_event_space(x)
        return self.scale * x + self.shift

    def _validate_action_space(self, x):
        # Equivalent to checking that it's in the action space)        
        assert(torch.ge(x, self.shift).all() and torch.le(x, self.scale + self.shift).all())

    def _validate_event_space(self, x):
        assert(torch.ge(x, 0).all() and torch.le(x, 1).all())
        
    def _dist(self, logits):
        # TODO remove validation        
        return torch.distributions.continuous_bernoulli.ContinuousBernoulli(logits=logits, validate_args=True)
    
    def log_prob(self, x, logits=None, dist=None, is_event_space=False, **kwargs):
        if dist is None: dist = self._dist(logits)
        if not is_event_space: x = self._action_space_to_event_space(x)
        # Uses naive-bayes assumption for multidimensional input        
        action_log_prob = dist.log_prob(x).sum(axis=-1)
        return action_log_prob

    def entropy(self, logits, **kwargs):
        # Uses naive-bayes assumption for multidimensional input
        # TODO this is haphazard, I haven't checked if this is mathematically correct.
        return self._dist(logits).entropy().sum(axis=-1)

    def get_action(self, observation, use_sample=True):
        dist = self._dist(self.forward(ptu.from_numpy(observation))[0])
        # Mean
        # action_event_space = dist.mean

        # Mode
        # action_event_space = torch.round(dist.mean)

        # This is probably preferred for exploration purposes.
        if use_sample:
            action_event_space = dist.sample()
        else:
            raise NotImplementedError
            action_event_space = dist.mean
        
        action_log_prob = self.log_prob(action_event_space, dist=dist, is_event_space=True)
        agent_info = dict(action_log_prob=ptu.get_numpy(action_log_prob))
        action_action_space = self._event_space_to_action_space(action_event_space)
        return ptu.get_numpy(action_action_space), agent_info

    def forward(self, input):
        """

        :param input: 
        :returns: logits 
        :rtype: len-1 tuple

        """
        return (self.mlp(input),)
