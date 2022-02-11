import torch as T
from einops import rearrange

from pretrain.reward import calc_pretrain_rewards
from pretrain.state_data import StateData
from utils.logger import log_episode_length, log_particle_reward, \
    log_rewards, log_steps_done, log_ppo_env_steps
from utils.logger import log_running_estimates
from pretrain.contrastive_training import train_contrastive_batch


def run_episode(agent: T.nn.Module, pretrain: bool, total_steps_done: int):
    """
    Runs one episode in the set environment
    Args:
        agent: agent handling the episode
        total_steps_done: current timestep
        pretrain: if reward is given by environment or calculated in a self
        supervised fashion

    Returns: new number of steps done

    """
    state = agent.env.reset()
    rewards = []
    done = False
    lives = agent.env.unwrapped.ale.lives()

    while not (lives == 0 and done):
        state = T.tensor(state, dtype=T.float, device=agent.device)
        state = rearrange(state, 'h w c -> 1 c h w')
        action, log_prob, aux_val, log_dist = agent.get_action(state)

        if pretrain:
            state = state.cpu()

        next_state, reward, done, _ = agent.env.step(action)
        rewards.append(reward)

        total_steps_done += 1
        agent.trajectory.append_step(state, action, next_state, done,
                                     log_prob, aux_val, log_dist)
        if pretrain:
            idx = total_steps_done % agent.config['replay_buffer_size']
            agent.replay_buffer[idx] = state.squeeze()
        else:
            agent.trajectory.append_reward(reward)

        state = next_state
        lives = agent.env.unwrapped.ale.lives()

        if pretrain and total_steps_done >= agent.config[
            'steps_before_repr_learning']:
            train_contrastive_batch(agent)
            log_steps_done(agent, total_steps_done)
            agent.log_metrics()

        if len(agent.trajectory) == agent.config['rollout_length']:
            with T.no_grad():
                agent.trajectory.state_vals = agent.critic(
                    T.cat(agent.trajectory.states))
            if pretrain:
                state_dset = T.cat(agent.trajectory.next_states)
                particle_rewards = calc_pretrain_rewards(agent,
                                                         state_dset).tolist()
                agent.trajectory.extend_rewards(particle_rewards)
                log_particle_reward(agent, particle_rewards)
                log_running_estimates(agent)

            online_training(agent, total_steps_done)

    log_rewards(agent, rewards)
    log_episode_length(agent, len(rewards))
    log_steps_done(agent, total_steps_done)
    agent.log_metrics()

    return total_steps_done


def online_training(agent, total_steps_done):
    agent.trajectory.calc_advantages(agent.config)
    agent.trajectory.data_to_tensors()
    log_ppo_env_steps(agent, total_steps_done)
    log_steps_done(agent, total_steps_done)

    agent.learn()

    agent.forget()
    agent.save_model()


def run_timesteps(agent: T.nn.Module, num_timesteps: int, is_pretrain: bool):
    """
    Runs given number of timesteps
    Args:
        agent: agent handling the interactions
        num_timesteps: steps to train for
        is_pretrain: if reward is given by environment or calculated in a self
        supervised fashion
    """
    steps_done = 0
    agent.forget()

    while steps_done < num_timesteps:
        steps_done = run_episode(agent, is_pretrain, steps_done)
