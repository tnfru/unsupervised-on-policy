import wandb
import warnings
import torch as T


def init_logging(config, actor, critic, prefix):
    # initialize logging
    wandb.init(project="EntropyPacman", config=config)
    wandb.watch(actor, log="all")
    wandb.watch(critic, log="all")
    wandb.run.name = prefix + '_' + wandb.run.name


def log_ppo(entropy_loss, kl_div, kl_max):
    exceeded = 1 if kl_div > kl_max else 0
    wandb.log({'entropy loss': entropy_loss,
               'kl div': kl_div.mean(),
               'kl_exceeded': exceeded})


def log_aux(aux_values, aux_loss, kl_div, kl_max):
    exceeded = 1 if kl_div > kl_max else 0
    if exceeded == 1:
        wandb.log({'aux kl_div': kl_div,
                   'kl_exceeded': exceeded})
    else:
        wandb.log({'aux state value': aux_values.mean(),
                   'aux loss': aux_loss.mean(),
                   'aux kl_div': kl_div,
                   'kl_exceeded': exceeded})


def log_critic(critic_loss, state_values):
    wandb.log({'critic loss': critic_loss.mean(),
               'critic state value': state_values.mean()})


def warn_about_aux_loss_scaling(aux_loss):
    warnings.warn(f'Aux Loss has value {aux_loss}. Consider '
                  f'scaling val_coeff down to not disrupt policy '
                  f'learning')


def log_episode_length(timesteps):
    wandb.log({
        'episode_length': timesteps
    })


def log_contrast_loss(loss):
    wandb.log({'contrastive loss': loss})


def log_particle_reward(rewards):
    wandb.log({'particle reward sum': T.sum(rewards)})
    wandb.log({'particle reward mean': T.mean(rewards)})
