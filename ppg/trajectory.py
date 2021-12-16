import torch as T


class Trajectory(T.utils.data.Dataset):
    def __init__(self):
        #TODO call super
        self.states = []
        self.log_probs = []
        self.actions = []
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

    def append_timesteps(self, states, actions, expected_returns, dones,
                         log_probs, advantages, aux_vals, log_dists,
                         state_vals, aux_rets):
        self.states.extend(states)
        self.actions.extend(actions)
        self.expected_returns.extend(expected_returns)
        self.dones.extend(dones)
        self.log_probs.extend(log_probs)
        self.advantages.extend(advantages)
        self.aux_state_values.extend(aux_vals)
        self.log_dists.extend(log_dists)
        self.state_vals.extend(state_vals)
        self.aux_rets.extend(aux_rets)

    def fix_datatypes(self):
        self.states = T.cat(self.states)
        self.actions = T.tensor(self.actions, dtype=T.long)
        self.dones = T.tensor(self.dones, dtype=T.int)
        self.log_probs = T.tensor(self.log_probs, dtype=T.float)
        self.log_dists = T.stack(self.log_dists)

    def __getitem__(self, index):
        state = self.states[index]
        expected_return = self.expected_returns[index]
        log_dist = self.log_dists[index]
        state_val = self.state_vals[index]
        advantage = self.advantages[index]
        # done = self.dones[index] not required by any loop

        if self.is_aux_epoch:
            aux_val = self.aux_state_values[index]
            aux_ret = self.aux_rets[index]
            return state, expected_return, aux_ret, state_val, aux_val, \
                   log_dist
        else:
            action = self.actions[index]
            log_prob = self.log_probs[index]
            return state, action, expected_return, state_val, advantage, log_prob
