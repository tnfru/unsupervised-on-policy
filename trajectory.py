import torch as T


class Trajectory(T.utils.data.Dataset):
    def __init__(self):
        self.states = []
        self.log_probs = []
        self.actions = []
        self.expected_returns = []
        self.dones = []
        self.advantages = []
        self.aux_state_values = []
        self.log_dists = []
        self.is_aux_epoch = False

    def __len__(self):
        return len(self.states)

    def append_timesteps(self, states, actions, expected_returns, dones,
                         log_probs, advantages, aux_vals, log_dists):
        self.states.extend(states)
        self.actions.extend(actions)
        self.expected_returns.extend(expected_returns)
        self.dones.extend(dones)
        self.log_probs.extend(log_probs)
        self.advantages.extend(advantages)
        self.aux_state_values.extend(aux_vals)
        self.log_dists.extend(log_dists)

    def fix_datatypes(self):
        self.states = T.stack(self.states)
        self.actions = T.tensor(self.actions, dtype=T.long)
        self.dones = T.tensor(self.dones, dtype=T.int)
        self.log_probs = T.tensor(self.log_probs, dtype=T.float)
        self.log_dists = T.stack(self.log_dists)

    def clear_memory(self):
        self.states = []
        self.log_probs = []
        self.actions = []
        self.expected_returns = []
        self.dones = []
        self.advantages = []
        self.log_dists = []

    def __getitem__(self, index):
        state = self.states[index]
        action = self.actions[index]
        expected_return = self.expected_returns[index]
        # done = self.dones[index] not required by any loop
        log_prob = self.log_probs[index]
        advantage = self.advantages[index]
        aux_val = self.aux_state_values[index]
        log_dist = self.log_dists[index]

        if self.is_aux_epoch:
            return state, expected_return, log_dist, aux_val
        else:
            return state, action, expected_return, log_prob, advantage
