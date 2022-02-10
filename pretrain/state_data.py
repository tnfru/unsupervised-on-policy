import torch as T
from collections import deque


class StateData(T.utils.data.Dataset):
    def __init__(self, states):
        self.states = []
        self.append_states(states)

    def __len__(self):
        return len(self.states)

    def append_states(self, states):
        self.states.extend(states)

    def fix_datatypes(self):
        self.states = T.cat(self.states)

    def __getitem__(self, index):
        state = self.states[index]

        return state


class RepresentationData(T.utils.data.Dataset):
    def __init__(self, config):
        self.states = deque(maxlen=config['replay_buffer_size'])

    def __len__(self):
        return len(self.states)

    def append_state(self, state):
        self.states.append(state)

    def __getitem__(self, index):
        state = self.states[index]

        return state
