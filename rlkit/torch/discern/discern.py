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
from rlkit.torch.torch_rl_algorithm import TorchRLAlgorithm
from rlkit.exploration_strategies.epsilon_greedy import EpsilonGreedy
from rlkit.core.eval_util import create_stats_ordered_dict
from rlkit.policies.argmax import ArgmaxDiscretePolicy, MaxEntDiscretePolicy
from rlkit.torch.torch_rl_algorithm import TorchRLAlgorithm

#from rlkit.data_management.reward_replay_buffer import RewardRelabelingBuffer

class DISCERN(TorchRLAlgorithm):
    """
    Note: this assumes the env will sample the goal when reset() is called,
    i.e. use a "silent" env.

    Hindsight Experience Replay

    This is a template class that should be the first sub-class, i.e.[

    ```
    class HerDdpg(HER, DDPG):
    ```

    and not

    ```
    class HerDdpg(DDPG, HER):
    ```

    Or if you really want to make DDPG the first subclass, do alternatively:
    ```
    class HerDdpg(DDPG, HER):
        def get_batch(self):
            return HER.get_batch(self)
    ```
    for each function defined below.
    """

    def __init__(
            self,
            env,
            qf,
            state_encoder,
            Xi,
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
        self.state_encoder = state_encoder
        self.Xi = Xi
        print("batch_size", batch_size)
        self.labels = torch.arange(0,batch_size, dtype=torch.long)
        self.target_qf = self.qf.copy()
        self.learning_rate = learning_rate
        self.use_hard_updates = use_hard_updates
        self.hard_update_period = hard_update_period
        self.tau = tau
        self.qf_optimizer = optim.Adam(
            self.qf.parameters(),
            lr=self.learning_rate,
        )
        self.Xi_optimizer = optim.Adam(
            self.Xi.parameters(),
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
        goals = batch['resampled_goals']
        embedded_obs = self.get_Xis(obs)
        embedded_goal = self.get_Xis(goals)
        rewards = self.recompute_rewards(embedded_obs, embedded_goal).unsqueeze(1)

        """
        Compute loss
        """
        next_goal_obs = torch.cat((next_obs.unsqueeze(1), goals.unsqueeze(1)), dim=1)
        goal_obs = torch.cat((obs.unsqueeze(1), goals.unsqueeze(1)), dim=1)
        target_q_values = self.target_qf(next_goal_obs).detach().max(
            1, keepdim=True
        )[0]
        y_target = rewards + (1. - terminals) * self.discount * target_q_values
        y_target = y_target.detach()
        # actions is a one-hot vector
        q_values = self.qf(goal_obs)
        
        y_pred = torch.sum(q_values* actions, dim=1, keepdim=True)
        neg_entropy = torch.sum(q_values*torch.exp(q_values))
        #import pdb; pdb.set_trace()
        qf_loss = self.qf_criterion(y_pred, y_target) +neg_entropy/1000

        xi_loss = self.Xi_loss_function(embedded_obs, embedded_goal, terminals)
        #xi_loss = self.Xi_loss_function(embedded_obs, embedded_goal, terminals)
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

    def get_epoch_snapshot(self, epoch):
        snapshot = super().get_epoch_snapshot(epoch)
        snapshot.update(
            exploration_policy=self.exploration_policy,
            policy=self.policy,
        )
        return snapshot

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


    def _handle_step(
            self,
            observation,
            action,
            reward,
            next_observation,
            terminal,
            agent_info,
            env_info,
    ):
        self._current_path_builder.add_all(
            observations=observation,
            actions=action,
            rewards=reward,
            next_observations=next_observation,
            terminals=terminal,
            agent_infos=agent_info,
            env_infos=env_info,
        )

    def _handle_path(self, path):
        self._n_rollouts_total += 1
        self.replay_buffer.add_path(path)
        self._exploration_paths.append(path)

#     def get_batch(self):
#         batch = super().get_batch()
#         obs = batch['observations']
#         next_obs = batch['next_observations']
#         goals = batch['resampled_goals']
#         return batch

    def _handle_rollout_ending(self):
        self._n_rollouts_total += 1
        if len(self._current_path_builder) > 0:
            path = self._current_path_builder.get_all_stacked()
            self.replay_buffer.add_path(path)
            self._exploration_paths.append(path)
            self._current_path_builder = PathBuilder()

    def _get_action_and_info(self, observation):
        """
        Get an action to take in the environment.
        :param observation:
        :return:
        """
        self.exploration_policy.set_num_steps_total(self._n_env_steps_total)
        new_obs = np.concatenate([
            observation[self.observation_key],
            observation[self.desired_goal_key]], axis = 0,
        ).reshape( 2, -1)
        return self.exploration_policy.get_action(new_obs)

    def get_eval_paths(self):
        paths = []
        n_steps_total = 0
        while n_steps_total <= self.num_steps_per_eval:
            path = self.eval_multitask_rollout()
            paths.append(path)
            n_steps_total += len(path['observations'])
        return paths

    def eval_multitask_rollout(self):
        return multitask_rollout(
            self.env,
            self.policy,
            self.max_path_length,
            observation_key=self.observation_key,
            desired_goal_key=self.desired_goal_key,
            goal_stacker =  lambda obs, goal: np.concatenate([obs,goal] ,axis = 0).reshape( 2, -1)
        )
    def npairs_loss(self, embeddings_anchor, embeddings_positive):
        batch_size = embeddings_anchor.shape[0]
        #l2_loss = 0.25*reg_lambda*((torch.sum(embeddings_positive**2 )) )/batch_size
        similarity_matrix = torch.matmul(
            embeddings_anchor, torch.t(embeddings_positive))
        #print("CHECK DIRECTION OF THIS SOFTMAX!!!")
        #import pdb; pdb.set_trace()
        
        xent_loss = F.cross_entropy(similarity_matrix,self.labels, reduction='none')
#         print("achieved_goals shape", embeddings_anchor.shape, "desired_goals shape", embeddings_positive.shape, "sim shape should be ",
#               embeddings_anchor.shape[0], embeddings_positive.shape[0], "is", similarity_matrix.shape
#              )
        return  xent_loss


    def Xi_loss_function(self, embedded_obs, embedded_goals, terminals):
        beta = embedded_goals.shape[0]+1
        batch_size =  beta-1
        embedded_goals *= beta
        #labels = torch.arange(0,batch_size, dtype=torch.long).to(self.device)
        
        xent = self.npairs_loss(embedded_obs, embedded_goals)
        #import pdb; pdb.set_trace()
        loss = torch.mean(xent)
        #loss = torch.sum(xent*terminals[:,0])
#         num_eps = torch.sum(terminals)
#         if num_eps.item() == 0:
#             return loss
#         else:
#             loss = loss/num_eps
        return loss

    def get_Xis(self, observations):
        embedded_obs = self.Xi(self.state_encoder(observations).detach())
        embedded_obs_l2 = torch.sqrt(torch.sum(embedded_obs**2, dim=1,keepdim=True))
        #import pdb; pdb.set_trace()
        embedded_obs = embedded_obs/embedded_obs_l2
        return embedded_obs
    
    def recompute_rewards(self, embedded_obs, embedded_goals):
        similarity =  torch.sum(embedded_obs*embedded_goals, dim=1)
        similarity = F.relu(similarity)
        return similarity