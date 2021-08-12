import numpy as np
import torch as T
import wandb

from preprocessing import normalize


def run_episode(agent, trajectory=None, render=False):
    states = []
    actions = []
    rewards = []
    dones = []
    log_probs = []
    state_vals = []
    aux_vals = []
    log_dists = []

    done = False
    state = agent.env.reset()

    while not done:
        if render:
            agent.env.render()

        state = T.tensor(state, dtype=T.float, device=agent.device)
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

    if agent.use_wandb:
        wandb.log({'reward': np.sum(rewards)})

    if trajectory is not None:
        advantages = trajectory.convert_to_advantages(rewards, state_vals,
                                                      agent.discount_factor)
        expected_returns = T.tensor(state_vals, dtype=T.float) + advantages
        advantages = normalize(advantages)
        trajectory.append_timesteps(states=states,
                                    actions=actions,
                                    expected_returns=expected_returns,
                                    dones=dones,
                                    log_probs=log_probs,
                                    advantages=advantages,
                                    aux_vals=aux_vals,
                                    log_dists=log_dists)
        return trajectory


def run_timesteps(agent, num_timesteps):
    timesteps = 0
    agent.forget()

    while timesteps < num_timesteps:
        agent.trajectory = run_episode(agent, agent.trajectory)

        if len(agent.trajectory) >= agent.rollout_length:
            timesteps += len(agent.trajectory)

            agent.trajectory.fix_datatypes()
            agent.learn()
            agent.forget()
