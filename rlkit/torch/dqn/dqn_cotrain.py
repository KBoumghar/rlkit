from collections import OrderedDict

import numpy as np
import torch
import torch.optim as optim
from torch import nn as nn

import rlkit.torch.pytorch_util as ptu
from rlkit.exploration_strategies.base import (
    PolicyWrappedWithExplorationStrategy
)
from rlkit.exploration_strategies.epsilon_greedy import EpsilonGreedy
from rlkit.core.eval_util import create_stats_ordered_dict
from rlkit.policies.argmax import ArgmaxDiscretePolicy
from rlkit.torch.torch_rl_algorithm import TorchRLAlgorithm
from gridworld.algorithms.loss import npairs_loss
from gridworld.algorithms.datasets import OnlineDeltaDataset

from gridworld.algorithms.models import FCNDeltaModel as DeltaModel
class DQN(TorchRLAlgorithm):
    def __init__(
            self,
            env,
            qf,
            policy=None,
            learning_rate=1e-3,
            use_hard_updates=False,
            hard_update_period=1000,
            tau=0.001,
            epsilon=0.1,
            qf_criterion=None,
            target_qf=None,
            reward_model=None,
            reward_dataloader=None,
            **kwargs
    ):
        """

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
        self.policy = policy or ArgmaxDiscretePolicy(qf)
        exploration_policy = PolicyWrappedWithExplorationStrategy(
            exploration_strategy=exploration_strategy,
            policy=self.policy,
        )
        super().__init__(
            env, exploration_policy, eval_policy=self.policy, **kwargs
        )
        self.qf = qf
        if target_qf is None:
            self.target_qf = self.qf.copy()
        else:
            self.target_qf = target_qf
        self.learning_rate = learning_rate
        self.use_hard_updates = use_hard_updates
        self.hard_update_period = hard_update_period
        self.tau = tau
        self.qf_optimizer = optim.Adam(
            self.qf.parameters(),
            lr=self.learning_rate,
        )
        self.should_cotrain = True
        self.qf_criterion = qf_criterion or nn.MSELoss()

        self.reward_model = reward_model
        self.reward_optimizer = optim.Adam(self.reward_model.parameters(), lr=1e-4)
        self.reward_dataloader = reward_dataloader
        self.dataloader_iterator = iter(self.reward_dataloader)

    def reward_loss_function(self, deltas, agent_features, device):
        #fwd = triplet_loss(deltas, agent_features, prev_features)
        #bwd = triplet_loss(agent_features, deltas, prev_deltas)
        batch_size = deltas.shape[0]
        labels = torch.arange(0,batch_size, dtype=torch.long).to(device)
        fwd, xent, l2 = npairs_loss(deltas, agent_features, labels)
        bwd, xent2, l22 = npairs_loss(agent_features, deltas, labels)
        return fwd+bwd, l2+l22
    
    def process_reward_data(self, data, device):
        
        image = data['image'].to(device)
        post_image = data['post_image'].to(device)
        pre_image = data['pre_image'].to(device)
        #pos_x =  data['agent_pos'][0].to(device)
        #pos_y =  data['agent_pos'][1].to(device)
        
        #image = image.to(device)
        #last_image = last_image.to(device)
        return image, pre_image, post_image#, (pos_x, pos_y)
    
    def _cotrain(self):
        #train_loss = 0
        #pol_loss = 0
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
                self.reward_optimizer.zero_grad()
                data= next(self.dataloader_iterator)
                image, pre_image, post_image = self.process_reward_data(data, device)
                polimage, polpre_image, polpost_image = self.process_reward_data(poldata, device)
                images = torch.cat((image, polimage), dim=0)
                pre_images = torch.cat((pre_image, polpre_image), dim=0)
                post_images = torch.cat((post_image, polpost_image), dim=0)
                deltas, agent_features = self.reward_model.forward(images, pre_images, post_images)
                loss, l2 = self.reward_loss_function(deltas, agent_features, device)
                train_loss += loss.item()/(len(images)+len(polimage))
                loss.backward()
                #train_loss += loss.item() /len(image)
                train_l2 += l2.item() /(len(images)+len(polimage))
                self.reward_optimizer.step()
                
            except StopIteration:
                self.dataloader_iterator = iter(self.reward_dataloader)
                #data, target = next(self.ataloader_iterator)
        if self.need_to_update_eval_statistics:
            self.eval_statistics['Delta Loss'] = train_loss/batch_idx
            self.eval_statistics['Delta L2'] = train_l2/batch_idx
    
    def _do_training(self):
        
        batch = self.get_batch()
        rewards = batch['rewards']
        terminals = batch['terminals']
        obs = batch['observations']
        actions = batch['actions']
        next_obs = batch['next_observations']
        #import pdb; pdb.set_trace()
        # Update rewards with new model:
        #print("rewards", rewards.shape)
        rewards = self.env.reward_function(obs).astype(np.float32).reshape(rewards.shape)

        """
        Compute loss
        """
        
        target_q_values = self.target_qf(next_obs).detach().max(
            1, keepdim=True
        )[0]
        y_target = rewards + (1. - terminals) * self.discount * target_q_values
        y_target = y_target.detach()
        # actions is a one-hot vector
        y_pred = torch.sum(self.qf(obs) * actions, dim=1, keepdim=True)
        #import pdb; pdb.set_trace()
        qf_loss = self.qf_criterion(y_pred, y_target)

        """
        Update networks
        """
        self.qf_optimizer.zero_grad()
        qf_loss.backward()
        self.qf_optimizer.step()
        self._update_target_network()

        """
        Save some statistics for eval using just one batch.
        """
        if self.need_to_update_eval_statistics:
            self.need_to_update_eval_statistics = False
            self.eval_statistics['QF Loss'] = np.mean(ptu.get_numpy(qf_loss))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Y Predictions',
                ptu.get_numpy(y_pred),
            ))

    def _update_target_network(self):
        if self.use_hard_updates:
            if self._n_train_steps_total % self.hard_update_period == 0:
                ptu.copy_model_params_from_to(self.qf, self.target_qf)
        else:
            ptu.soft_update_from_to(self.qf, self.target_qf, self.tau)

    def offline_evaluate(self, epoch):
        raise NotImplementedError()

    def get_epoch_snapshot(self, epoch):
        snapshot = super().get_epoch_snapshot(epoch)
        snapshot.update(
            exploration_policy=self.exploration_policy,
            policy=self.policy,
            optimizer_state=self.qf_optimizer.state_dict,
            qf=self.qf,
            target_qf=self.target_qf,
        )
        return snapshot

    @property
    def networks(self):
        return [
            self.qf,
            self.target_qf,
        ]
