import torch as T

from ppg.gae import calculate_advantages
from utils.network_utils import normalize


class Trajectory(T.utils.data.Dataset):
    def __init__(self, config):
        max_len = config['rollout_length']
        self.states = []
        self.states = T.zeros(max_len, config['stacked_frames'],
                              config['height'], config['width'])
        self.next_states = T.zeros(max_len, config['stacked_frames'],
                                   config['height'], config['width'])
        self.actions = T.zeros(max_len, dtype=T.long)
        self.rewards = T.zeros(max_len)
        self.log_probs = T.zeros(max_len)
        self.expected_returns = T.zeros(max_len)
        self.dones = T.zeros(max_len)
        self.advantages = T.zeros(max_len)
        self.aux_state_values = T.zeros(max_len)
        self.log_dists = T.zeros(max_len, config['action_dim'])
        self.state_vals = T.zeros(max_len)
        self.aux_rets = T.zeros(max_len)
        self.is_aux_epoch = False
        self.is_critic_epoch = False

    def __len__(self):
        return len(self.states)

    def append_step(self, state, action, next_state, done,
                    log_prob, aux_val, log_dist, idx):
        self.states[idx] = state
        self.actions[idx] = action
        self.next_states[idx] = next_state
        self.dones[idx] = T.from_numpy(done).float()
        self.log_probs[idx] = log_prob
        self.aux_state_values[idx] = aux_val
        self.log_dists[idx] = log_dist

    def extend_state_vals(self, state_vals):
        self.state_vals.extend(state_vals)

    def data_to_tensors(self):
        self.states = T.cat(self.states)
        self.actions = T.tensor(self.actions, dtype=T.long)
        self.dones = T.tensor(self.dones, dtype=T.int)
        self.log_probs = T.tensor(self.log_probs, dtype=T.float)
        self.log_dists = T.stack(self.log_dists)

    def calc_advantages(self, config):
        advantages = calculate_advantages(
            self.rewards,
            self.state_vals,
            self.dones,
            config['discount_factor'],
            config['gae_lambda']
        )
        expected_returns = self.state_vals + advantages
        advantages = normalize(advantages)

        aux_advantages = calculate_advantages(
            self.rewards,
            self.aux_state_values,
            self.dones,
            config['discount_factor'],
            config['gae_lambda']
        )

        aux_returns = self.aux_state_values + aux_advantages

        self.expected_returns = expected_returns
        self.advantages = advantages
        self.aux_rets = aux_returns

    def __getitem__(self, index):
        state = self.states[index]
        expected_return = self.expected_returns[index]
        log_dist = self.log_dists[index].squeeze()
        advantage = self.advantages[index]
        # done = self.dones[index] not required by any loop

        if self.is_aux_epoch:
            aux_ret = self.aux_rets[index]
            return state, expected_return, aux_ret, log_dist
        else:
            action = self.actions[index]
            log_prob = self.log_probs[index]
            return state, action, expected_return, advantage, log_prob
