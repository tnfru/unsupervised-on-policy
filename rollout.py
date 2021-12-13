import numpy as np
import torch as T
import wandb
from einops import rearrange

from preprocessing import normalize
from utils import calculate_advantages
from logger import log_episode_length, log_particle_reward
from logger import log_running_estimates
from pretrain.reward import calc_pretrain_rewards
from pretrain.state_data import StateData


def run_episode(agent, trajectory, pretrain):
    states = []
    actions = []
    rewards = []
    dones = []
    log_probs = []
    state_vals = []
    aux_vals = []
    log_dists = []

    state = agent.env.reset()
    done = False

    max_lives = agent.env.unwrapped.ale.lives()

    lives = max_lives

    while not (lives == 0 and done):
        state = T.tensor(state, dtype=T.float, device=agent.device)
        state = rearrange(state, 'h w c -> 1 c h w')
        action, log_prob, aux_val, log_dist = agent.get_action(state)
        state_val = agent.critic(state).squeeze().item()
        state = state.cpu()
        next_state, reward, done, _ = agent.env.step(action)

        states.append(state)
        state_vals.append(state_val)
        actions.append(action)
        rewards.append(reward)
        dones.append(done)
        log_probs.append(log_prob)
        aux_vals.append(aux_val)
        log_dists.append(log_dist)

        state = next_state
        lives = agent.env.unwrapped.ale.lives()

    if agent.use_wandb:
        # TODO log reward per life
        wandb.log({'reward': np.sum(rewards)})

    if pretrain:
        state_dset = StateData(states)
        state_dset.fix_datatypes()
        rewards = calc_pretrain_rewards(agent, state_dset)

        if agent.use_wandb:
            mean = agent.reward_function.mean
            var = agent.reward_function.var
            log_particle_reward(rewards, var)
            log_running_estimates(mean, var)

    config = agent.config

    advantages = calculate_advantages(rewards,
                                      state_vals,
                                      dones,
                                      config['discount_factor'],
                                      config['gae_lambda'])
    expected_returns = T.tensor(state_vals, dtype=T.float) + advantages
    aux_advantages = calculate_advantages(rewards,
                                          aux_vals,
                                          dones,
                                          config['discount_factor'],
                                          config['gae_lambda'])
    aux_rets = T.tensor(aux_vals, dtype=T.float) + aux_advantages
    advantages = normalize(advantages)

    trajectory.append_timesteps(states=states,
                                actions=actions,
                                expected_returns=expected_returns,
                                dones=dones,
                                log_probs=log_probs,
                                advantages=advantages,
                                aux_vals=aux_vals,
                                log_dists=log_dists,
                                state_vals=state_vals,
                                aux_rets=aux_rets)
    return trajectory


def run_timesteps(agent, num_timesteps, is_pretrain):
    timestep = 0
    agent.forget()

    while timestep < num_timesteps:
        agent.trajectory = run_episode(agent, agent.trajectory, is_pretrain)

        if len(agent.trajectory) >= agent.config['rollout_length']:
            timestep += len(agent.trajectory)

            if agent.config['use_wandb']:
                log_episode_length(len(agent.trajectory))

            agent.trajectory.fix_datatypes()
            agent.learn(is_pretrain=is_pretrain)
            agent.forget()
