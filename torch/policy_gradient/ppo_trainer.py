
import logging
import os
import torch.optim as optim

import rlkit.torch.pytorch_util as ptu
import rlkit.core.pylogging
from rlkit.torch.torch_rl_algorithm import TorchTrainer
import torch

log = logging.getLogger(os.path.basename(__name__))

class PPOTrainer(TorchTrainer):
    """
    Proximal Policy Optimization
    """

    def __init__(
            self,
            policy,
            policy_learning_rate,
            value_f,
            replay_buffer,
            clip_param=0.2,
            entropy_bonus=0,
            env_info_observation_key='',
            discount_factor=0.99,
            et_factor=0.2,
            allow_large_updates=False,
            debug=False
    ):
        super().__init__()
        self.policy = policy
#         self.target_policy = self.policy.copy()
        self.value_f = value_f
        self.policy_learning_rate = policy_learning_rate
        self.optimizer_v = optim.Adam( list(self.value_f.parameters()),
                                           lr=self.policy_learning_rate* 10)
        
        self.optimizer_p = optim.Adam(list(self.policy.parameters()),
                                   lr=self.policy_learning_rate)
        self.eval_statistics = {}
        self.need_to_update_eval_statistics = True
        self.replay_buffer = replay_buffer
        self.clip_param = clip_param
        self.entropy_bonus = entropy_bonus
        self.env_info_observation_key = env_info_observation_key
        self.debug = debug
        self.discount_factor = discount_factor

        self.et_factor = et_factor
        self.allow_large_updates = allow_large_updates
        self.initial_policy_state = self.policy.state_dict()
        self.initial_optimizer_states = [self.optimizer_v.state_dict(), self.optimizer_p.state_dict()]

    def reset_optimizers(self):
        self.optimizer_v.load_state_dict(self.initial_optimizer_states[0])
        self.optimizer_p.load_state_dict(self.initial_optimizer_states[1])

    def reset_policy(self):
        self.policy.load_state_dict(self.initial_policy_state)

    def end_epoch(self, epoch):
        self.replay_buffer.empty_buffer()

    def save_policy_and_optimizer(self, output_directory, basename):
        """Saves the policy and its optimizer to the output directory with the given basename"""
        policy_fn = os.path.join(output_directory, basename + '.pt')
        optim_fn = os.path.join(output_directory, basename + '_optim.pt')
        policy_state = dict(policy=self.policy.state_dict(), value_f=self.value_f.state_dict())
        optim_state = dict(optimizer_v=self.optimizer_v.state_dict(), optimizer_p=self.optimizer_p.state_dict())
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
        self.optimizer_v.load_state_dict(optim_state['optimizer_v'])
        self.optimizer_p.load_state_dict(optim_state['optimizer_p'])

    def train_from_torch(self, batch):
        # Batch is whole replay buffer in this case
        if self.env_info_observation_key != '':
            obs = batch[self.env_info_observation_key]
            log.trace("Using alternate obs: {}".format(self.env_info_observation_key))
        else:
            obs = batch['observations']
            
        actions = batch['actions']
        returns = batch['returns']
        old_action_log_probs = batch['action_log_prob']
        adv_targ = batch['advs']
        adv_targ = (adv_targ - torch.mean(adv_targ)) / torch.std(adv_targ)
        assert(torch.isfinite(adv_targ).all())

#         pdb.set_trace()
        """
        Policy operations.
        """

        action_dist_params = self.policy.forward(obs)
        action_log_probs = self.policy.log_prob(actions, *action_dist_params).view(-1, 1)

        ratio = torch.exp(action_log_probs - old_action_log_probs)
        surr1 = ratio * adv_targ
        surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv_targ
        # PPO's pessimistic surrogate (L^CLIP)
        action_loss = -torch.min(surr1, surr2).mean()  

        dist_entropy = self.policy.entropy(*action_dist_params).mean()
#         total_loss = (value_loss + action_loss - dist_entropy * self.entropy_bonus)
#         total_loss = (value_loss + action_loss)
        total_loss = (action_loss)
        
        """
        Update Networks
        """
        from torch.utils.data import Dataset, TensorDataset, DataLoader
        dataset = TensorDataset(obs, returns)
        train_loader = DataLoader(dataset=dataset, batch_size=64)
        for obs_batch, return_batch in train_loader:
            values = self.value_f.forward(obs_batch)
            ### I like to scale this be the inverse discount factor
            value_loss = 0.5*(((return_batch) - values) * ((1-self.discount_factor)/1.0)).pow(2).mean()
            self.optimizer_v.zero_grad()
            if self.debug:
                # Slows things down.
                with torch.autograd.detect_anomaly():
                    value_loss.backward()
            else:
                value_loss.backward()
            self.optimizer_v.step()
            
        r_ = 1 - torch.mean(ratio)
        et_factor = 0.2
        small_update = (r_ < (et_factor)) and ( r_ > (-et_factor))

        if small_update or self.allow_large_updates:  ### update not to large )
            self.optimizer_p.zero_grad()
            if self.debug:
                # Slows things down.
                with torch.autograd.detect_anomaly():
                    action_loss.backward()
            else:
                action_loss.backward()
            self.optimizer_p.step()
        else:
            log.warning(f"Not taking large update with factor {r_:.2f}")

        self.eval_statistics['ppo_action_loss'] = ptu.get_numpy(action_loss)
        self.eval_statistics['ppo_value_loss'] = ptu.get_numpy(value_loss)
        self.eval_statistics['ppo_total_loss'] = ptu.get_numpy(total_loss)
        self.eval_statistics['ppo_action_entropy'] = ptu.get_numpy(dist_entropy)
        self.eval_statistics['ppo_importance_ratio'] = ptu.get_numpy(r_)

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
