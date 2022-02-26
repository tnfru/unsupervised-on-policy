import torch as T
from einops import rearrange


def append_task_reward(agent, reward, idx):
    normalization_factor = 10 if agent.config['reward_clip'] else 1
    if not agent.config['is_pretrain']:
        agent.trajectory.rewards[idx] = T.from_numpy(
            reward).float() / normalization_factor


def is_repr_learn_phase(config, steps_done):
    return config['is_pretrain'] and steps_done >= config[
        'steps_before_repr_learning'] / config['num_envs']


def is_training_step(config, steps_done):
    steps_until_train = config['rollout_length'] / config['num_envs']

    return (steps_done + 1) % steps_until_train == 0


def fetch_terminal_state(next_state, num_envs, done, info):
    terminal_state = next_state.cpu().clone()
    for i in range(num_envs):
        if done[i]:
            term = T.from_numpy(info[i]['terminal_observation']).float()
            terminal_state[i] = rearrange(term, 'h w c -> c h w')
    return terminal_state


def get_idx(agent, total_steps_done, replay_buffer=False):
    num_envs = agent.config['num_envs']
    size = agent.config['replay_buffer_size'] if replay_buffer else \
        agent.config['rollout_length']
    idx = total_steps_done % (size / num_envs)
    idx *= num_envs
    idx = T.arange(idx, idx + num_envs).long()

    return idx
