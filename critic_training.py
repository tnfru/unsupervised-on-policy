from losses import value_loss_fun
from logger import log_critic

from utils import do_gradient_step


def train_critic_batch(agent, states, expected_returns, old_state_values):
    config = agent.config
    state_values = agent.critic(states)
    critic_loss = value_loss_fun(state_values=state_values,
                                 old_state_values=old_state_values,
                                 expected_returns=expected_returns,
                                 is_aux_epoch=agent.trajectory.is_aux_epoch,
                                 value_clip=config['value_clip'])

    do_gradient_step(agent.critic, agent.critic_opt, critic_loss, config[
        'grad_norm'])

    if agent.use_wandb:
        log_critic(critic_loss, state_values)
