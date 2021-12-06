import torch as T


class StateData(T.utils.data.Dataset):
    def __init__(self):
        self.states = []

    def __len__(self):
        return len(self.states)

    def append_states(self, states):
        self.states.extend(states)

    def fix_datatypes(self):
        self.states = T.cat(self.states)

    def __getitem__(self, index):
        state = self.states[index]

        return state
