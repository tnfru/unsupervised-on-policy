import torch as T
from einops import rearrange

from pretrain.reward import calc_pretrain_rewards
from utils.logger import log_episode_length, log_particle_reward, \
    log_rewards, log_steps_done, log_ppo_env_steps
from pretrain.contrastive_training import train_contrastive_batch


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
    agent.forget()
    rewards = []

    num_envs = agent.config['num_envs']
    state = T.from_numpy(agent.env.reset()).to(agent.device).float()
    state = rearrange(state, 'envs h w c -> envs c h w')

    while total_steps_done < num_timesteps:
        action, log_prob, aux_val, log_dist = agent.get_action(state)
        next_state, reward, done, _ = agent.env.step(action)
        next_state = T.from_numpy(next_state).to(agent.device).float()
        next_state = rearrange(next_state, 'envs h w c -> envs c h w')

        rewards.append(reward)
        state = state.cpu()

        if pretrain:
            idx = get_idx(agent, total_steps_done, replay_buffer=True)
            agent.replay_buffer[idx] = state

        if pretrain and total_steps_done >= agent.config[
            'steps_before_repr_learning'] / num_envs:
            train_contrastive_batch(agent, total_steps_done)

        steps_until_train = agent.config['rollout_length'] / num_envs
        if (total_steps_done + 1) % steps_until_train == 0:
            with T.no_grad():
                states = agent.trajectory.states.to(agent.device)
                agent.trajectory.state_vals = agent.critic(
                    states).squeeze().detach().cpu()

            if pretrain:
                calc_pretrain_rewards(agent)

            online_training(agent, total_steps_done)
        idx = get_idx(agent, total_steps_done)
        agent.trajectory.append_step(state, action, next_state.cpu(), done,
                                     log_prob, aux_val, log_dist, idx)
        if not pretrain:
            agent.trajectory.rewards[idx] = T.from_numpy(reward).float()
        state = next_state
        total_steps_done += 1

    log_rewards(agent, rewards)  # TODO FIX LOGG
    log_episode_length(agent, len(rewards))
    log_steps_done(agent, total_steps_done)
    agent.log_metrics()

    return total_steps_done


def get_idx(agent, total_steps_done, replay_buffer=False):
    num_envs = agent.config['num_envs']
    size = agent.config['replay_buffer_size'] if replay_buffer else \
        agent.config['rollout_length']
    idx = total_steps_done % (size / num_envs)
    idx *= num_envs
    idx = T.arange(idx, idx + num_envs).long()

    return idx


def online_training(agent, total_steps_done):
    agent.trajectory.calc_advantages(agent.config)
    # agent.trajectory.data_to_tensors()
    log_ppo_env_steps(agent, total_steps_done)
    log_steps_done(agent, total_steps_done)

    agent.learn()

    agent.forget()
    agent.save_model()
