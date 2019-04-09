import numpy as np
import torch

from rlkit.samplers.rollout_functions import multitask_rollout
from rlkit.data_management.obs_dict_replay_buffer import ObsDictRelabelingBuffer
from rlkit.data_management.path_builder import PathBuilder
from rlkit.torch.ddpg.ddpg import DDPG
from rlkit.torch.her.her_replay_buffer import RelabelingReplayBuffer
from rlkit.torch.sac.sac import SoftActorCritic
from rlkit.torch.sac.twin_sac import TwinSAC
from rlkit.torch.td3.td3 import TD3
from rlkit.torch.dqn.dqn import DQN
from rlkit.torch.torch_rl_algorithm import TorchRLAlgorithm
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
from torch.nn import functional as F


class IBC(TorchRLAlgorithm):
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
            learning_rate=1e-3,
            epsilon=0.1,
            observation_key=None,
            desired_goal_key=None,
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
        self.observation_key = observation_key
        self.desired_goal_key = desired_goal_key
        self.policy = ArgmaxDiscretePolicy(qf)
        exploration_policy = PolicyWrappedWithExplorationStrategy(
            exploration_strategy=exploration_strategy,
            policy=self.policy,
        )
        super().__init__(
            env, exploration_policy, eval_policy=self.policy, **kwargs
        )
        self.qf = qf
        #self.target_qf = self.qf.copy()
        self.learning_rate = learning_rate
        #self.hard_update_period = hard_update_period
        #self.tau = tau
        self.qf_optimizer = optim.Adam(
            self.qf.parameters(),
            lr=self.learning_rate,
        )
        #self.qf_criterion = qf_criterion or nn.MSELoss()
        

    def _do_training(self):
        batch = self.get_batch()
        #rewards = batch['rewards']
        #terminals = batch['terminals']
        obs = batch['observations']
        actions = batch['actions']
        #next_obs = batch['next_observations']

        """
        Compute loss
        """
        y_pred = self.qf(obs)
        _, targets = actions.max(dim=1)
        #import pdb; pdb.set_trace()
        qf_loss = F.cross_entropy(y_pred, targets)
        #qf_loss = self.qf_criterion(y_pred, y_target)

        """
        Update networks
        """
        self.qf_optimizer.zero_grad()
        qf_loss.backward()
        self.qf_optimizer.step()
        #self._update_target_network()

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

    def get_batch(self):
        batch = super().get_batch()
        obs = batch['observations']
        next_obs = batch['next_observations']
        goals = batch['resampled_goals']
        batch['observations'] = torch.cat((
            obs,
            goals
        ), dim=1)
        batch['next_observations'] = torch.cat((
            next_obs,
            goals
        ), dim=1)
        return batch

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
        new_obs = np.hstack((
            observation[self.observation_key],
            observation[self.desired_goal_key],
        ))
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
            train=False
        )
    @property
    def networks(self):
        return [
            self.qf,
            #self.target_qf,
        ]
