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
#from gridworld.algorithms.datasets import OnlineDeltaDataset

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
            epsilon=0.4,
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
        self.dataloader_iterator = iter(self.dataset)
        #self.labels = torch.arange(0,batch_size, dtype=torch.long)
        self.learning_rate = learning_rate
        self.use_hard_updates = use_hard_updates
        self.hard_update_period = hard_update_period
        self.tau = tau
        self.qf_optimizer = optim.Adam(
            self.qf.parameters(),
            lr=self.learning_rate,
        )
        self.f_optimizer = optim.SGD(
            self.f.parameters(),
            lr=self.learning_rate/10,
        )
        self.qf_criterion = qf_criterion or nn.MSELoss()
        self._did_training = 0
        self.score_mean = 0
        self.score_std = 1
        
        
    def _do_irl_training(self):
        batch = self.get_batch()
        rewards = batch['rewards']
        terminals = batch['terminals']
        obs = batch['observations']
        actions = batch['actions']
        next_obs = batch['next_observations']
        #goals = batch['resampled_goals']
        
        """
        Update F
        """
        irl_loss = 0
        if True:# self._did_training > :
            device = torch.device("cuda")
            #import pdb; pdb.set_trace()
            #print('model', self.reward_model.f_encoder.conv1.weight[0])
            try:
                exp_data= next(self.dataloader_iterator)
            except StopIteration:
                self.dataloader_iterator = iter(self.dataset)
                exp_data= next(self.dataloader_iterator)
            self.f_optimizer.zero_grad()
            loss, accs = self.get_irl_loss((obs, actions, next_obs), exp_data)
            irl_loss = loss.item()
            loss.backward()
            self.f_optimizer.step()
            with torch.no_grad():
                scores = self.get_rewards(obs, actions, next_obs, centered=False)
            #import pdb; pdb.set_trace()
            self.score_std = np.std(scores)
            self.score_mean = np.mean(scores)
#             if accs[1] > 0.8:
#                 self.f_optimizer.defaults['lr'] = self.learning_rate/100
#             elif accs[1] < 0.2 :
#                 self.f_optimizer.defaults['lr'] = self.learning_rate/10
            

        #if self.need_to_update_eval_statistics:
        self.eval_statistics['IRL Loss'] =irl_loss
        self.eval_statistics['ScoreMean'] = self.score_mean
        self.eval_statistics['ScoreStd'] =self.score_std
        self.eval_statistics['exp_acc'] = ptu.get_numpy(accs[0])
        self.eval_statistics['pol_acc'] = ptu.get_numpy(accs[1])

    def _do_training(self):
        batch = self.get_batch()
        rewards = batch['rewards']
        terminals = batch['terminals']
        obs = batch['observations']
        actions = batch['actions']
        next_obs = batch['next_observations']
        #goals = batch['resampled_goals']
        
        """
        Update F
        """ 
        with torch.no_grad():
            rewards = self.get_rewards(obs, actions, next_obs, centered=True)

        """
        Compute loss
        """
        target_q_values = self.target_qf(next_obs).detach().max(
            1, keepdim=True
        )[0]
        y_target = rewards + (1. - terminals) * self.discount * target_q_values
        y_target = y_target.detach()
        # actions is a one-hot vector
        q_values = self.qf(obs)
        
        y_pred = torch.sum(q_values* actions, dim=1, keepdim=True)
        qf_loss = self.qf_criterion(y_pred, y_target) 
        """
        Update networks
        """
        self.qf_optimizer.zero_grad()
        qf_loss.backward()
        self.qf_optimizer.step()
        self._update_target_network()

        self._did_training += 1

        """
        Save some statistics for eval using just one batch.
        """
        if self.need_to_update_eval_statistics:
            self.need_to_update_eval_statistics = False
            self.eval_statistics['QF Loss'] = np.mean(ptu.get_numpy(qf_loss))
            
            #print("q_values", q_values)
            self.eval_statistics.update(create_stats_ordered_dict(
                'Y Predictions',
                ptu.get_numpy(y_pred),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'LearnedReward',
                rewards,
            ))
            
            #self.eval_statistics['IRL Loss'] = np.mean(ptu.get_numpy(irl_loss))
    
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
            self.f,
        ]
    def to(self, device=None):
        if device is None:
            device = ptu.device
        for net in self.networks:
            net.to(device)
        self.device = device

#     def get_batch(self):
#         batch = super().get_batch()
#         obs = batch['observations']
#         next_obs = batch['next_observations']
#         goals = batch['resampled_goals']
#         return batch



    def get_rewards(self, obs, a, next_obs, centered=True):
        d_values = self.Disc(obs, a, next_obs)
        r = (torch.log(d_values) - torch.log(1-d_values)).detach().numpy()
        if centered:
            r = np.clip((r - self.score_mean) / self.score_std, -3, 3)
        return r
    
    def process_expert_data(self, data):
        image = data['image'].to(self.device)
        next_image = data['next_image'].to(self.device)
        action =  data['action'].to(self.device) 
        return image, action, next_image
    
    def get_irl_loss(self, pol_data, exp_data): 
        exp_log_p_tau, exp_log_pq, exp_log_q_tau = self.get_intermediates(*self.process_expert_data(exp_data))
        exp_loss = -1*(torch.sum(exp_log_p_tau-exp_log_pq))
        pol_log_p_tau, pol_log_pq, pol_log_q_tau = self.get_intermediates(*pol_data)
        pol_loss = -1*(torch.sum(pol_log_q_tau-pol_log_pq))
        exp_acc =  torch.mean((torch.exp(exp_log_p_tau-exp_log_pq) > 0.5).type(torch.FloatTensor))
        pol_acc = torch.mean((torch.exp(pol_log_p_tau-pol_log_pq) < 0.5).type(torch.FloatTensor))
        return pol_loss + exp_loss, [exp_acc, pol_acc]

    def get_intermediates(self,obs, a, next_obs ):
        log_p_tau = self.f(obs, a, next_obs)
        log_q_tau = torch.sum(self.qf(obs)*a, dim=1).unsqueeze(1)
        log_pq = torch.logsumexp(torch.stack([log_p_tau, log_q_tau]), dim=0)
        return log_p_tau, log_pq, log_q_tau
    
    def Disc(self,obs, a, next_obs):
        log_p_tau, log_pq, log_q_tau = self.get_intermediates(obs, a, next_obs )
        output = torch.exp(log_p_tau-log_pq)
        return output
    
    
