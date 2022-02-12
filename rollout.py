import torch as T
import numpy as np
from einops import rearrange

from pretrain.reward import calc_pretrain_rewards
from utils.logger import log_episode
from utils.logger import log_steps_done, \
    log_ppo_env_steps
from pretrain.contrastive_training import train_contrastive_batch
from utils.rollout_utils import append_task_reward
from utils.rollout_utils import fetch_terminal_state
from utils.rollout_utils import get_idx
from utils.rollout_utils import is_repr_learn_phase
from utils.rollout_utils import is_training_step


def run_timesteps(agent: T.nn.Module, num_timesteps: int, pretrain: bool):
    """
    Runs one episode in the set environment
    Args:
        agent: agent handling the episode
        num_timesteps: steps to train for
        pretrain: if reward is given by environment or calculated in a self
        supervised fashion

    Returns: new number of steps done

    """
    total_steps_done = 0
    num_envs = agent.config['num_envs']
    rewards = np.zeros(num_envs).astype(float)
    eps_lengths = np.zeros(num_envs).astype(int)

    state = T.from_numpy(agent.env.reset()).to(agent.device).float()
    state = rearrange(state, 'envs h w c -> envs c h w')

    while total_steps_done < num_timesteps:
        action, log_prob, aux_val, log_dist = agent.get_action(state)
        next_state, reward, done, info = agent.env.step(action)
        next_state = T.from_numpy(next_state).to(agent.device).float()
        next_state = rearrange(next_state, 'envs h w c -> envs c h w')

        rewards = rewards + reward
        state = state.cpu()
        agent.append_to_replay_buffer(state, total_steps_done)

        if is_repr_learn_phase(agent.config, total_steps_done):
            train_contrastive_batch(agent, total_steps_done)

        if is_training_step(agent.config, total_steps_done):
            with T.no_grad():
                states = agent.trajectory.states.to(agent.device)

                agent.trajectory.state_vals = agent.critic(
                    states).squeeze().detach().cpu()

            if pretrain:
                calc_pretrain_rewards(agent)

            agent.learn(total_steps_done)

        idx = get_idx(agent, total_steps_done)
        if done.any():
            log_episode(agent, rewards, eps_lengths, total_steps_done, done,
                        info)
            terminal_state = fetch_terminal_state(next_state, num_envs, done,
                                                  info)

            agent.trajectory.append_step(state, action, terminal_state, done,
                                         log_prob, aux_val, log_dist, idx)
        else:
            agent.trajectory.append_step(state, action, next_state.cpu(), done,
                                         log_prob, aux_val, log_dist, idx)
        append_task_reward(agent, reward, idx)
        state = next_state
        total_steps_done += 1
        eps_lengths = eps_lengths + 1

    return total_steps_done
