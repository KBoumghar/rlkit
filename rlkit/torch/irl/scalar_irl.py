import numpy as np
import torch
from torch import nn as nn
from torch.nn import functional as F
import torch.optim as optim
from torch import nn as nn

import rlkit.torch.pytorch_util as ptu
from rlkit.exploration_strategies.base import (
    PolicyWrappedWithExplorationStrategy
)
from rlkit.samplers.rollout_functions import multitask_rollout
from rlkit.data_management.path_builder import PathBuilder
from rlkit.torch.ddpg.ddpg import DDPG
#from rlkit.torch.her.her_replay_buffer import RelabelingReplayBuffer
# from rlkit.torch.sac.sac import SoftActorCritic
# from rlkit.torch.sac.twin_sac import TwinSAC
# from rlkit.torch.td3.td3 import TD3
# from rlkit.torch.dqn.dqn import DQN
from gridworld.algorithms.datasets import OnlineDeltaDataset

from rlkit.torch.torch_rl_algorithm import TorchRLAlgorithm
from rlkit.exploration_strategies.epsilon_greedy import EpsilonGreedy
from rlkit.core.eval_util import create_stats_ordered_dict
from rlkit.policies.argmax import ArgmaxDiscretePolicy, MaxEntDiscretePolicy
from rlkit.torch.torch_rl_algorithm import TorchRLAlgorithm

#from rlkit.data_management.reward_replay_buffer import RewardRelabelingBuffer

class IRL(TorchRLAlgorithm):
    """
    """

    def __init__(
            self,
            env,
            qf,
            f,
            dataset,
            policy=None,
            learning_rate=1e-3,
            use_hard_updates=False,
            hard_update_period=1000,
            tau=0.001,
            epsilon=0.1,
            qf_criterion=None,
            observation_key=None,
            desired_goal_key=None,
            replay_buffer_kwargs=None,
            batch_size=1024,
            **kwargs
            
    ):
        """
        TODO: set reward = 0 for all t < T
        TODO: Add (sin(2pi t/T), cos(2pi t/T)) to obs.
        TODO: Add LSTM
        :param env: Env.
        :param qf: QFunction. Maps from state to action Q-values.
        :param learning_rate: Learning rate for qf. Adam is used.
        :param use_hard_updates: Use a hard rather than soft update.
        :param hard_update_period: How many gradient steps before copying the
        parameters over. Used if `use_hard_updates` is True.
        :param tau: Soft target tau to update target QF. Used if
        `use_hard_updates` is False.
        :param epsilon: Probability of taking a random action.
        :param kwargs: kwargs to pass onto TorchRLAlgorithm
        """
        exploration_strategy = EpsilonGreedy(
            action_space=env.action_space,
            prob_random_action=epsilon,
        )
        self.policy = policy or MaxEntDiscretePolicy(qf)
        exploration_policy = PolicyWrappedWithExplorationStrategy(
            exploration_strategy=exploration_strategy,
            policy=self.policy,
        )
        super().__init__(
            env, exploration_policy, eval_policy=self.policy, batch_size=batch_size,**kwargs
        )
        self.should_cotrain = False
        self.qf = qf
        self.target_qf = self.qf.copy()
        self.f = f
        self.dataset = dataset
        #self.labels = torch.arange(0,batch_size, dtype=torch.long)
        self.learning_rate = learning_rate
        self.use_hard_updates = use_hard_updates
        self.hard_update_period = hard_update_period
        self.tau = tau
        self.qf_optimizer = optim.Adam(
            self.qf.parameters(),
            lr=self.learning_rate,
        )
        self.f_optimizer = optim.Adam(
            self.f.parameters(),
            lr=self.learning_rate,
        )
        self.qf_criterion = qf_criterion or nn.MSELoss()
        self.observation_key = observation_key
        self.desired_goal_key = desired_goal_key
        
    def _do_training(self):
        batch = self.get_batch()
        rewards = batch['rewards']
        terminals = batch['terminals']
        obs = batch['observations']
        actions = batch['actions']
        next_obs = batch['next_observations']
        #goals = batch['resampled_goals']
        with torch.no_grad():
            rewards = self.get_rewards(obs, actions, next_obs)

        """
        Compute loss
        """
        target_q_values = self.target_qf(next_obs).detach().max(
            1, keepdim=True
        )[0]
        y_target = rewards + (1. - terminals) * self.discount * target_q_values
        y_target = y_target.detach()
        # actions is a one-hot vector
        q_values = self.qf(goal_obs)
        
        y_pred = torch.sum(q_values* actions, dim=1, keepdim=True)
        qf_loss = self.qf_criterion(y_pred, y_target) 
        """
        Update networks
        """
        self.qf_optimizer.zero_grad()
        qf_loss.backward()
        self.qf_optimizer.step()
        self._update_target_network()
        #self.Xi_optimizer.zero_grad()
        #xi_loss.backward()
        #import pdb; pdb. set_trace()
        #self.Xi_optimizer.step()
        self._did_training += 1
        """
        Update Xi
        """
        if self._did_training > 4:
            self._did_training = 0
            device = torch.device("cuda")
            test_paths = self.get_eval_paths()
            train_loader = torch.utils.data.DataLoader(
                OnlineDeltaDataset(test_paths),
                batch_size=8, shuffle=True)
            train_loss = 0
            train_l2 = 0
            #import pdb; pdb.set_trace()
            #print('model', self.reward_model.f_encoder.conv1.weight[0])
            for batch_idx, poldata in enumerate(train_loader):
                try:
                    exp_data= next(self.dataloader_iterator)
                except StopIteration:
                    self.dataloader_iterator = iter(self.reward_dataloader)
                    exp_data= next(self.dataloader_iterator)
                self.f_optimizer.zero_grad()
                loss = get_irl_loss(pol_data, exp_data)
                loss.backward()
                self.f_optimizer.step()
            
        """
        Save some statistics for eval using just one batch.
        """
        if self.need_to_update_eval_statistics:
            self.need_to_update_eval_statistics = False
            self.eval_statistics['QF Loss'] = np.mean(ptu.get_numpy(qf_loss))
            self.eval_statistics['QF Entropy Loss'] = np.mean(ptu.get_numpy(-1*neg_entropy))
            print("q_values", q_values)
            self.eval_statistics['Xi Loss'] = np.mean(ptu.get_numpy(xi_loss))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Y Predictions',
                ptu.get_numpy(y_pred),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Recomputed Rewards',
                ptu.get_numpy(rewards),
            ))
            

    
    def _update_target_network(self):
        if self.use_hard_updates:
            if self._n_train_steps_total % self.hard_update_period == 0:
                ptu.copy_model_params_from_to(self.qf, self.target_qf)
        else:
            ptu.soft_update_from_to(self.qf, self.target_qf, self.tau)

    def offline_evaluate(self, epoch):
        raise NotImplementedError()

    @property
    def networks(self):
        return [
            self.qf,
            self.target_qf,
            self.state_encoder,
        ]
    def to(self, device=None):
        if device is None:
            device = ptu.device
        for net in self.networks:
            net.to(device)
        self.labels.to(device)




#     def get_batch(self):
#         batch = super().get_batch()
#         obs = batch['observations']
#         next_obs = batch['next_observations']
#         goals = batch['resampled_goals']
#         return batch



    def get_rewards(self, obs, a, next_obs):
        d_values = self.Disc(obs, a, next_obs)
        r = torch.log(d_values) - torch.log(1-d_values) 
        return r
    
    def get_irl_loss(pol_data, exp_data):
        exp_obs, exp_a, exp_next_obs = self.process_expert_data(data, device)
        pol_obs, pol_a, pol_next_obs = self.process_poldata(poldata, device)
        exp_f = self.f(exp_obs, exp_a, exp_next_obs)
        pol_f = self.Disc(pol_obs, pol_a, pol_next_obs)
        exp_log_pq = torch.logsumexp([exp_f, self.qf(exp_obs)[a]], dim=0)
        exp_loss = -1*(torch.sum(exp_f-exp_log_pq))
        pol_log_pq = torch.logsumexp([pol_f, self.qf(pol_obs)[a]], dim=0)
        pol_loss = (torch.sum(pol_f-pol_log_pq))
        return pol_loss + exp_loss

    def Disc(self,obs, a, next_obs):
        f_values = self.f(obs, a, next_obs)
#         numerator = torch.exp(f_values)
#         denominator = numerator+ self.qf(obs)[a]
        log_pq = torch.logsumexp([f_values, self.qf(obs)[a]], dim=0)
        output = torch.exp(f_values-log_pq)
        return output
    
