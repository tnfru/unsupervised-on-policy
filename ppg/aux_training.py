from torch.distributions.categorical import Categorical

from ppg.losses import value_loss_fun
from ppg.critic_training import train_critic_batch, train_critic_epoch
from utils.network_utils import data_to_device, approx_kl_div, do_gradient_step
from utils.network_utils import do_accumulated_gradient_step
from utils.logger import warn_about_aux_loss_scaling, log_aux


def train_aux_epoch(agent, loader):
    num_batches = len(loader)

    for batch_idx, rollout_data in enumerate(loader):
        states, expected_returns, aux_returns, state_values, aux_values, \
        log_dists = data_to_device(rollout_data, agent.device)
        expected_returns = expected_returns.unsqueeze(1)
        aux_returns = aux_returns.unsqueeze(1)

        train_aux_batch(agent, states, aux_returns, log_dists, aux_values,
                        batch_idx, num_batches)
        # train_critic_batch(agent, states, expected_returns, state_values,
        #                   batch_idx, num_batches)


def train_aux_batch(agent, states, expected_returns, old_log_probs,
                    old_aux_value, batch_idx, num_batches):
    config = agent.config
    action_logits, aux_values = agent.actor(states)
    action_dist = Categorical(logits=action_logits)
    log_probs = action_dist.probs.log()
    kl_div = approx_kl_div(log_probs, old_log_probs, is_aux=True)

    aux_value_loss = value_loss_fun(state_values=aux_values,
                                    old_state_values=old_aux_value,
                                    expected_returns=expected_returns,
                                    is_aux_epoch=True,
                                    value_clip=config['value_clip'])
    aux_value_loss = aux_value_loss * config['val_coeff']

    if aux_value_loss > agent.AUX_WARN_THRESHOLD:
        warn_about_aux_loss_scaling(aux_value_loss)

    aux_loss = aux_value_loss + kl_div * config['beta']

    if config['kl_max'] is None or kl_div < config['kl_max']:
        # If KL divergence is too big we don't take gradient steps
        # do_gradient_step(agent.actor, agent.actor_opt, aux_loss,
        #                 grad_norm=config['grad_norm'])
        do_accumulated_gradient_step(agent.actor, agent.actor_opt, aux_loss,
                                     config, batch_idx, num_batches)

    if agent.use_wandb:
        log_aux(agent, aux_values, aux_loss, kl_div, config['kl_max'])
