from gym.spaces import Discrete

from rlkit.data_management.simple_replay_buffer import SimpleReplayBuffer
from rlkit.envs.env_utils import get_dim
import numpy as np


class EnvReplayBuffer(SimpleReplayBuffer):
    def __init__(
            self,
            max_replay_buffer_size,
            env,
            env_info_sizes=None,
            dtype="float32"
    ):
        """
        :param max_replay_buffer_size:
        :param env:
        """
        self.env = env
        self._ob_space = env.observation_space
        self._action_space = env.action_space

        if env_info_sizes is None:
            if hasattr(env, 'info_sizes'):
                env_info_sizes = env.info_sizes
            else:
                env_info_sizes = dict()

        super().__init__(
            max_replay_buffer_size=max_replay_buffer_size,
            observation_dim=get_dim(self._ob_space),
            action_dim=get_dim(self._action_space),
            env_info_sizes=env_info_sizes,
            dtype=dtype
        )

    def add_sample(self, observation, action, reward, terminal,
                   next_observation, **kwargs):
        if isinstance(self._action_space, Discrete):
            new_action = np.zeros(self._action_dim)
            new_action[action] = 1
        else:
            new_action = action
        return super().add_sample(
            observation=observation,
            action=new_action,
            reward=reward,
            next_observation=next_observation,
            terminal=terminal,
            **kwargs
        )
        
        
class PPOEnvReplayBuffer(EnvReplayBuffer):
    def __init__(
            self,
            max_replay_buffer_size,
            env,
            discount_factor,
            value_f,
            use_gae=False,
            gae_discount=0.95,
            **kwargs
    ):
        super().__init__(max_replay_buffer_size, env, **kwargs)
        self._returns = np.zeros((max_replay_buffer_size, 1))
        self.current_trajectory_rewards = np.zeros((max_replay_buffer_size, 1))
        self._max_replay_buffer_size = max_replay_buffer_size
        self.discount_factor = discount_factor
        self.value_f = value_f
        self.use_gae = use_gae
        self.gae_discount = gae_discount
        self._bottom = 0
        self._values = np.zeros((max_replay_buffer_size, 1))
        self._advs = np.zeros((max_replay_buffer_size, 1))

    def discounted_rewards(self, rewards, discount_factor):
        import scipy
        from scipy import signal, misc
        """
        computes discounted sums along 0th dimension of x.
        inputs
        ------
        rewards: ndarray
        discount_factor: float
        outputs
        -------
        y: ndarray with same shape as x, satisfying
            y[t] = x[t] + gamma*x[t+1] + gamma^2*x[t+2] + ... + gamma^k x[t+k],
                    where k = len(x) - t - 1
        """
        assert rewards.ndim >= 1
        return scipy.signal.lfilter([1],[1,-discount_factor],rewards[::-1], axis=0)[::-1]
    
    def terminate_episode(self):
        returns = []
        observations = self._observations[self._bottom:self._top]
        self._values[self._bottom:self._top] = ptu.get_numpy(self.value_f(ptu.from_numpy(observations)))
                                                                          
#         b1 = np.append(self._values[self._bottom:self._top], 0)
        ### THe proper way to terminate the episode
        b1 = np.append(self._values[self._bottom:self._top], 0 if self._terminals[self._top-1] else self._values[self._top-1])
#         b1 = np.append(self._values[self._bottom:self._top], self._values[self._top-1])
        b1 = np.reshape(b1, (-1,1))
        deltas = self._rewards[self._bottom:self._top] + self.discount_factor*b1[1:] - b1[:-1]
        self._advs[self._bottom:self._top] = self.discounted_rewards(deltas, self.discount_factor * self.gae_discount)
        self._returns[self._bottom:self._top] = self.discounted_rewards(self._rewards[self._bottom:self._top], self.discount_factor)

        self._bottom = self._top

    def add_sample(self, observation, action, reward, terminal,
                   next_observation, env_info=None, agent_info=None):
        if self._top == self._max_replay_buffer_size:
            raise EnvironmentError('Replay Buffer Overflow, please reduce the number of samples added!')

        # This could catch onehot vs. integer representation differences.
        assert(self._actions.shape[-1] == action.size)
        self._observations[self._top] = observation
        self._actions[self._top] = action
        self._rewards[self._top] = reward
        self._terminals[self._top] = terminal
        self._next_obs[self._top] = next_observation

        for key in self._env_info_keys:
            self._env_infos[key][self._top] = env_info[key]

        for key in self._agent_info_keys:
            self._agent_infos[key][self._top] = agent_info[key]

        self._advance()

    def add_paths(self, paths):
        log.trace("Adding {} new paths. First path length: {}".format(len(paths), paths[0]['actions'].shape[0]))
        for path in paths:
            self.add_path(path)
        # process samples after adding paths
        self.process_samples(self.value_f)

    def process_samples(self, value_f):
        # Compute value for all states
        pass
#         self._advs[:] = self._returns - self._values

        # Center adv
#         advs = self._advs[:self._top]
#         self._advs[:self._top] = (advs - advs.mean()) / (advs.std() + 1e-5)

    def random_batch(self, batch_size):
        indices = np.random.randint(0, self._size, batch_size)
        batch = dict(
            observations=self._observations[indices],
            actions=self._actions[indices],
            rewards=self._rewards[indices],
            terminals=self._terminals[indices],
            next_observations=self._next_obs[indices],
            returns=self._returns[indices],
            advs=self._advs[indices],
            vf_preds=self._values[indices]
        )
        for key in self._env_info_keys:
            assert key not in batch.keys()
            batch[key] = self._env_infos[key][indices]
        for key in self._agent_info_keys:
            assert key not in batch.keys()
            batch[key] = self._agent_infos[key][indices]
        return batch

    def all_batch_windows(self, window_len, skip=1, return_env_info=False):
        # Will return (bs, batch_len, dim)

        start_indices = np.arange(0, self._size - window_len, skip)
        terminal_sums = [self._terminals[i0:i0+window_len].sum() for i0 in start_indices]

        # NB first mask should always be True for current start_indices.
        valid_start_mask = np.logical_and(start_indices + window_len < self._size, np.equal(terminal_sums, 0))
        valid_start_indices = start_indices[valid_start_mask]

        batch = dict(
            observations=np.stack([self._observations[i:i+window_len] for i in valid_start_indices]),
            actions=np.stack([self._actions[i:i+window_len] for i in valid_start_indices]),
            rewards=np.stack([self._rewards[i:i+window_len] for i in valid_start_indices]),
            terminals=np.stack([self._terminals[i:i+window_len] for i in valid_start_indices]),
            buffer_idx=valid_start_indices,
        )
        if return_env_info:
            env_info_batch = {}
            for k, v in self._env_infos.items():
                env_info_batch[k] = np.stack([v[i:i+window_len] for i in valid_start_indices])
            batch.update(env_info_batch)
        return batch

    def relabel_rewards(self, rewards):
        # Ensure the updated rewards match the size of all of the data currently in the buffer.
        assert(rewards.shape == (self._size,))
        # Assumes the rewards correspond to the most-recently-added data to the buffer.
        #   I'm pretty sure this assumption is valid, because otherwise self._size would have to
        #   be larger than rewards.shape[0].
        self._rewards[self._top - self._size:self._size] = rewards[:, None]

