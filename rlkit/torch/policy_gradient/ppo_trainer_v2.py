
import logging
import os
import torch.optim as optim
import torch.nn
import numpy as np

import rlkit.torch.pytorch_util as ptu
import rlkit.core.pylogging
from rlkit.torch.torch_rl_algorithm import TorchTrainer
import torch
import pdb
log = logging.getLogger(os.path.basename(__name__))

class PPOTrainerV2(TorchTrainer):
    """
    Proximal Policy Optimization
    """

    def __init__(
            self,
            policy,
            policy_learning_rate,
            value_learning_rate,
            value_f,
            replay_buffer,
            clip_param=0.2,
            env_info_observation_key='',
            vf_loss_coeff=1.0,
            entropy_coeff=0.02,
            vf_clip_param=0.1,
            kl_coeff=0.2,
            allow_large_updates=False,
            debug=False,
            reset_replay_buffer=True,
            **kwargs
    ):
        super().__init__()
        self.policy = policy
        self.value_f = value_f
        self.policy_learning_rate = policy_learning_rate
        self.parameters_list = list(self.value_f.parameters()) + list(self.policy.parameters())
        self.policy_params = list(self.policy.parameters())
        self.value_params = list(self.value_f.parameters())
        self.value_lr = value_learning_rate
        self.policy_lr = policy_learning_rate
        self.optimizer_p = optim.Adam(self.policy_params, lr=self.policy_lr)
        self.optimizer_v = optim.Adam(self.value_params, lr=self.value_lr)
        self.eval_statistics = {}
        self.need_to_update_eval_statistics = True
        self.replay_buffer = replay_buffer
        self.clip_param = clip_param
        self.env_info_observation_key = env_info_observation_key
        self.debug = debug

        self.vf_loss_coeff = vf_loss_coeff
        self.entropy_coeff = entropy_coeff
        self.vf_clip_param = vf_clip_param
        self.kl_coeff = kl_coeff
        self.grad_clip_norm = 1000
        
        self.allow_large_updates = allow_large_updates
        self.initial_policy_state = self.policy.state_dict()
        self.initial_optimizer_state = self.optimizer_p.state_dict()
        self.reset_replay_buffer = reset_replay_buffer

    def __repr__(self):
        return (f"{self.__class__.__name__}(vf_loss_coeff={self.vf_loss_coeff:.3f}, " +
                f"value_lr={self.value_lr:.2g}, " +
                f"policy_lr={self.policy_lr:.2g}, " +
                f"kl_coeff={self.kl_coeff:.3f}, " +
                f"entropy_coeff={self.entropy_coeff:.3f}, " +
                f"vf_clip_param={self.vf_clip_param:.3f})")
        
    def reset_optimizers(self):
        self.optimizer.load_state_dict(self.initial_optimizer_state)

    def reset_policy(self):
        self.policy.load_state_dict(self.initial_policy_state)

    def end_epoch(self, epoch):
        if self.reset_replay_buffer:
            log.info("Resetting policy replay buffer")
            self.replay_buffer.empty_buffer()
        else:
            log.info("Not resetting policy replay buffer")

    def save_policy_and_optimizer(self, output_directory, basename):
        """Saves the policy and its optimizer to the output directory with the given basename"""
        policy_fn = os.path.join(output_directory, basename + '.pt')
        optim_fn = os.path.join(output_directory, basename + '_optim.pt')
        policy_state = dict(policy=self.policy.state_dict(), value_f=self.value_f.state_dict())
        optim_state = dict(optimizer=self.optimizer.state_dict())
        torch.save(policy_state, policy_fn)
        torch.save(optim_state, optim_fn)

    def load_policy_and_optimizer(self, policy_params_filename):
        """Loads the policy params and the optimizer params, using the path to the serialized policy parameters

        :param policy_params_filename: filename of the policy params 
        :returns: 
        :rtype: 

        """
        log.info("Loading policy and optimizer. Policy filename: {}".format(policy_params_filename))
        policy_params_filename = os.path.realpath(policy_params_filename)
        assert(os.path.isfile(policy_params_filename))
        # /foo/bar/policy.pt -> '/foo/bar/policy', '.pt'
        policy_name, policy_ext = os.path.splitext(policy_params_filename)
        # '/foo/bar/'
        directory = os.path.dirname(policy_name)
        # 'policy'
        basename = os.path.basename(policy_name)
        policy_fn = os.path.join(directory, basename + policy_ext)
        # Sanity check.
        assert(policy_fn == policy_params_filename)
        optim_fn = os.path.join(directory, basename + '_optim' + policy_ext)
        if not os.path.isfile(optim_fn):
            raise FileNotFoundError("Policy filename {filename} not found".format(filename=optim_fn))

        policy_state = torch.load(policy_fn)
        optim_state = torch.load(optim_fn)
        self.policy.load_state_dict(policy_state['policy'])
        self.value_f.load_state_dict(policy_state['value_f'])
        self.optimizer.load_state_dict(optim_state['optimizer'])

    def train_from_torch(self, batch):
        # Batch is whole replay buffer in this case
        if self.env_info_observation_key != '':
            obs = batch[self.env_info_observation_key]
            log.trace("Using alternate obs: {}".format(self.env_info_observation_key))
            next_obs = batch['next_' + self.env_info_observation_key]
        else:
            obs = batch['observations']
            next_obs = batch['next_observations']
            
        actions = batch['actions']
        value_targets = batch['returns']
        previous_value_predictions = batch['vf_preds']
        old_action_log_probs = batch['action_log_prob']
        adv_targ = batch['advs']

        assert(torch.isfinite(adv_targ).all())
        action_dist = self.policy.forward(obs)
        action_log_probs = self.policy.log_prob(actions, *action_dist).view(-1, 1)
        value_fn = self.value_f.forward(obs)

        # TOOD hardcoded discount
        # TODO doesn't account for terminal states.
        # td_target = batch['rewards'] + 0.95 * previous_value_predictions
        # td1_error = torch.pow(td_target.detach() - value_fn, 2.)

        # Clamp extremely large values in log space before exponentiating (i.e. the ratio p/q has exploded)
        ratio = torch.exp(torch.clamp(action_log_probs - old_action_log_probs, np.log(-1e8), np.log(1e5)))
        surr1 = ratio * adv_targ
        surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv_targ
        ratio_mean = ptu.get_numpy(ratio).mean()
        
        # PPO's pessimistic surrogate (L^CLIP)
        surrogate_loss = torch.min(surr1, surr2).mean()

        dist_entropy = self.policy.entropy(*action_dist).mean()
        if not np.isclose(self.kl_coeff, 0.0):
            old_action_dist = batch['action_dist']
            action_kl = torch.max(self.policy.kl(source_probs=old_action_dist, dest_probs=action_dist).mean(), 
                                  torch.zeros((), device=ptu.device))
        else:
            action_kl = torch.zeros(())

        vf_loss1 = torch.pow(value_fn - value_targets, 2.0)
        # NB vf_clip_param should depend on reward scaling.
        vf_clipped = previous_value_predictions + torch.clamp(value_fn - previous_value_predictions, -self.vf_clip_param, self.vf_clip_param)
        vf_loss2 = torch.pow(vf_clipped - value_targets, 2.0)
        vf_loss = torch.max(vf_loss1, vf_loss2).mean()
        # vf_loss = torch.clamp(vf_loss1.mean(), -self.vf_clip_param, self.vf_clip_param)

        # vf_loss = td1_error.mean()
        # vf_loss = torch.pow(value_fn - value_targets, 2.0).mean()

        total_loss = (-1 * surrogate_loss +
                      # self.kl_coeff * action_kl + 
                      # self.vf_loss_coeff * vf_loss +
                      -1 * self.entropy_coeff * dist_entropy)

        assert(torch.isfinite(total_loss))
        
        # with torch.autograd.detect_anomaly():
        self.optimizer_p.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_params, self.grad_clip_norm, norm_type=2)
        self.optimizer_p.step()

        self.optimizer_v.zero_grad()
        vf_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.value_params, self.grad_clip_norm, norm_type=2)
        self.optimizer_v.step()

        self.eval_statistics['ppo_action_loss'] = ptu.get_numpy(surrogate_loss)
        self.eval_statistics['ppo_value_loss'] = ptu.get_numpy(vf_loss)
        # self.eval_statistics['ppo_deltas_loss'] = ptu.get_numpy(deltas)
        self.eval_statistics['ppo_total_loss'] = ptu.get_numpy(total_loss)
        self.eval_statistics['ppo_action_entropy'] = ptu.get_numpy(dist_entropy)
        self.eval_statistics['ppo_action_kl'] = ptu.get_numpy(action_kl)
        self.eval_statistics['ppo_importance_ratio'] = ratio_mean

    def get_diagnostics(self):
        return self.eval_statistics

    @property
    def networks(self):
        return [
            self.policy,
            self.value_f
        ]
        
    def get_snapshot(self):
        return dict(policy=self.policy.state_dict(), baseline=self.value_f.state_dict())
