import torch as T

from ppg.gae import calculate_advantages
from utils.network_utils import normalize


class Trajectory(T.utils.data.Dataset):
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.expected_returns = []
        self.dones = []
        self.advantages = []
        self.aux_state_values = []
        self.log_dists = []
        self.state_vals = []
        self.aux_rets = []
        self.is_aux_epoch = False

    def __len__(self):
        return len(self.states)

    def append_step(self, state, state_val, action, done,
                    log_prob, aux_val, log_dist):
        self.states.append(state)
        self.actions.append(action)
        self.dones.append(done)
        self.log_probs.append(log_prob)
        self.aux_state_values.append(aux_val)
        self.state_vals.append(state_val)
        self.log_dists.append(log_dist)

    def append_reward(self, reward):
        self.rewards.append(reward)

    def extend_rewards(self, rewards):
        self.rewards.extend(rewards)

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
        expected_returns = T.tensor(self.state_vals, dtype=T.float) + advantages
        advantages = normalize(advantages)

        aux_advatages = calculate_advantages(
            self.rewards,
            self.aux_state_values,
            self.dones,
            config['discount_factor'],
            config['gae_lambda']
        )

        aux_returns = T.tensor(self.aux_state_values,
                               dtype=T.float) + aux_advatages

        self.expected_returns.extend(expected_returns)
        self.advantages.extend(advantages)
        self.aux_rets.extend(aux_returns)

    def __getitem__(self, index):
        state = self.states[index]
        expected_return = self.expected_returns[index]
        log_dist = self.log_dists[index].squeeze()
        state_val = self.state_vals[index]
        advantage = self.advantages[index]
        # done = self.dones[index] not required by any loop

        if self.is_aux_epoch:
            aux_ret = self.aux_rets[index]
            return state, expected_return, aux_ret, state_val, log_dist
        else:
            action = self.actions[index]
            log_prob = self.log_probs[index]
            return state, action, expected_return, state_val, advantage, log_prob
