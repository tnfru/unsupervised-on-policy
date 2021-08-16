from torch.distributions.categorical import Categorical
from utils import data_to_device, approx_kl_div, do_gradient_step
from objectives import value_loss_fun
from logger import warn_about_aux_loss_scaling, log_aux
from critic_train import train_critic


def train_aux_epoch(agent, loader):
    for rollout_data in loader:
        states, expected_returns, aux_returns, state_values, aux_values, \
        log_dists = data_to_device(rollout_data, agent.device)
        expected_returns = expected_returns.unsqueeze(1)
        aux_returns = aux_returns.unsqueeze(1)

        train_aux(agent, states, aux_returns, log_dists, aux_values)
        train_critic(agent, states, expected_returns, state_values)


def train_aux(agent, states, expected_returns, old_log_probs,
              old_aux_value):
    config = agent.config
    action_probs, aux_values = agent.actor(states)
    action_dist = Categorical(logits=action_probs)
    log_probs = action_dist.probs.log()
    kl_div = approx_kl_div(log_probs, old_log_probs)

    aux_value_loss = value_loss_fun(state_values=aux_values,
                                    old_state_values=old_aux_value,
                                    expected_returns=expected_returns,
                                    is_aux_epoch=True,
                                    value_clip=config['value_clip'])
    aux_value_loss *= config['val_coeff']

    if aux_value_loss > agent.AUX_WARN_THRESHOLD:
        warn_about_aux_loss_scaling(aux_value_loss)

    aux_loss = aux_value_loss + kl_div * config['beta']

    if kl_div < config['kl_max']:
        # If KL divergence is too big we don't take gradient steps
        do_gradient_step(agent.actor, agent.actor_opt, aux_loss,
                         grad_norm=config['grad_norm'])

    if agent.use_wandb:
        log_aux(aux_values, aux_loss, kl_div, config['kl_max'])
