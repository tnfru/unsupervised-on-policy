import torch as T
from einops import rearrange

from pretrain.reward import calc_pretrain_rewards
from pretrain.state_data import StateData
from utils.logger import log_episode_length, log_particle_reward, \
    log_rewards, log_steps_done
from utils.logger import log_running_estimates


def run_episode(agent, trajectory, pretrain):
    state = agent.env.reset()
    done = False
    lives = agent.env.unwrapped.ale.lives()

    while not (lives == 0 and done):
        state = T.tensor(state, dtype=T.float, device=agent.device)
        state = rearrange(state, 'h w c -> 1 c h w')
        action, log_prob, aux_val, log_dist = agent.get_action(state)
        state_val = agent.critic(state).squeeze().item()
        state = state.cpu()
        next_state, reward, done, _ = agent.env.step(action)

        trajectory.append_step(state, state_val, action, reward, done,
                               log_prob, aux_val, log_dist)

        state = next_state
        lives = agent.env.unwrapped.ale.lives()

    if agent.use_wandb:
        log_rewards(trajectory.rewards)

    if pretrain:
        state_dset = StateData(trajectory.states)
        state_dset.fix_datatypes()
        rewards = calc_pretrain_rewards(agent, state_dset)

        if agent.use_wandb:
            log_particle_reward(agent, rewards, agent.reward_function.mean)
            log_running_estimates(agent, agent.reward_function.mean,
                                  agent.reward_function.var)

    trajectory.calc_advantages(agent.config)

    return trajectory


def run_timesteps(agent, num_timesteps, is_pretrain):
    steps_done = 0
    agent.forget()

    while steps_done < num_timesteps:
        agent.trajectory = run_episode(agent, agent.trajectory, is_pretrain)

        if len(agent.trajectory) >= agent.config['rollout_length']:
            steps_done += len(agent.trajectory)

            if agent.use_wandb:
                log_episode_length(agent, agent.trajectory)
                log_steps_done(agent, steps_done)

            agent.trajectory.data_to_tensors()
            agent.learn(is_pretrain=is_pretrain)

            if agent.use_wandb:
                agent.log_metrics()

            agent.forget()
            agent.save_model()
