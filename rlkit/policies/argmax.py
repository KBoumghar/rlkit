"""
Torch argmax policy
"""
import numpy as np
import rlkit.torch.pytorch_util as ptu
from rlkit.policies.base import SerializablePolicy
from rlkit.torch.core import PyTorchModule
from torch.nn import functional as F


class ArgmaxDiscretePolicy(PyTorchModule, SerializablePolicy):
    def __init__(self, qf):
        self.save_init_params(locals())
        super().__init__()
        self.qf = qf

    def get_action(self, obs):
        obs = np.expand_dims(obs, axis=0)
        obs = ptu.from_numpy(obs).float()
        q_values = self.qf(obs).squeeze(0)
        q_values_np = ptu.get_numpy(q_values)
        return q_values_np.argmax(), {}

class MaxEntDiscretePolicy(PyTorchModule, SerializablePolicy):
    def __init__(self, qf):
        self.save_init_params(locals())
        super().__init__()
        self.qf = qf

    def get_action(self, obs):
        obs = np.expand_dims(obs, axis=0)
        obs = ptu.from_numpy(obs).float()
        q_values = self.qf(obs)#.squeeze(0)
        prob_a = ptu.get_numpy(F.softmax(q_values).squeeze(0))

        prob_a = prob_a.astype(np.float64)
        prob_a = prob_a /  np.sum(prob_a)
        indices = np.argmax(np.random.multinomial(1, prob_a, size=1), axis=1)[0]
        #import pdb; pdb. set_trace()
        return indices, {}
