import wandb
import warnings
import numpy as np


def init_logging(config, agent, prefix):
    # initialize logging
    wandb.init(project="EntropyPacman", config=config)
    wandb.watch(agent.actor, log="all")
    wandb.watch(agent.critic, log="all")
    wandb.watch(agent.contrast_net, log='all')
    wandb.run.name = prefix + '_' + wandb.run.name


def log_ppo(agent, entropy_loss, kl_div, kl_max):
    if kl_max is None or kl_max < kl_div:
        exceeded = 0
    else:
        exceeded = 1
    agent.metrics.update({'entropy loss': entropy_loss,
                          'entropy': -entropy_loss / agent.entropy_coeff,
                          'kl div': kl_div.mean(),
                          'kl_exceeded': exceeded})


def log_aux(agent, aux_values, aux_loss, kl_div, kl_max):
    if kl_max is None or kl_max < kl_div:
        agent.metrics.update({'aux state value': aux_values.mean(),
                              'aux loss': aux_loss.mean(),
                              'aux kl_div': kl_div,
                              'kl_exceeded': 0})
    else:
        agent.metrics.update({'aux kl_div': kl_div,
                              'kl_exceeded': 1})


def log_critic(agent, critic_loss, state_values):
    agent.metrics.update({'critic loss': critic_loss.mean(),
                          'critic state value': state_values.mean()})


def warn_about_aux_loss_scaling(aux_loss):
    warnings.warn(f'Aux Loss has value {aux_loss}. Consider '
                  f'scaling val_coeff down to not disrupt policy '
                  f'learning')


def log_episode_length(agent, episode_length):
    agent.metrics.update({
        'episode_length': episode_length
    })


def log_contrast_loss_batch(agent, loss):
    agent.metrics.update({'contrastive loss batch': loss})


def log_contrast_loss_epoch(agent, loss):
    agent.metrics.update({'contrast loss epoch': loss})


def log_rewards(agent, rewards):
    agent.metrics.update({'reward': np.sum(rewards)})


def log_steps_done(agent, steps):
    agent.metrics.update({'million env steps': steps / 1e6})


def log_nan_aux():
    wandb.log({'nan during aux': 1})


def log_particle_reward(agent, rewards):
    mean = agent.reward_function.mean
    agent.metrics.update({
        'particle reward sum': np.sum(rewards),
        'particle reward mean': np.mean(rewards),
        'particle reward unnormalized': mean * np.mean(rewards),
        'particle reward sum unnormalized': mean * np.sum(rewards)
    })


def log_running_estimates(agent):
    agent.metrics.update({
        'running mean': agent.reward_function.mean,
        'running var': agent.reward_function.var
    })


def log_entropy_coeff(agent):
    agent.metrics.update({
        'entropy_coeff': agent.entropy_coeff
    })


def log_ppo_env_steps(agent, steps):
    agent.metrics.update({'mil env steps': steps / 1e6})
