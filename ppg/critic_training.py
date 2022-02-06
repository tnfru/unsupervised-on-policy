import torch as T
import torch.nn.functional as F
from utils.logger import log_critic

from utils.network_utils import do_accumulated_gradient_step
from utils.network_utils import data_to_device


def train_critic_epoch(agent: T.nn.Module, loader: T.utils.data.DataLoader,
                       is_aux=False):
    """
    Trains the critic for one epoch on the score function (MSE usually)
    Args:
        agent: agent to train
        loader: contains the data to train on
        is_aux: if it is an auxiliary epoch or not
    """
    num_batches = len(loader)
    for batch_idx, rollout_data in enumerate(loader):
        if is_aux:
            states, expected_returns, aux_returns, state_values, aux_values, \
            log_dists = data_to_device(rollout_data, agent.device)
        else:
            states, actions, expected_returns, state_values, advantages, \
            log_probs = data_to_device(rollout_data, agent.device)
        expected_returns = expected_returns.unsqueeze(1)
        train_critic_batch(agent=agent, states=states,
                           expected_returns=expected_returns,
                           batch_idx=batch_idx, num_batches=num_batches)


def train_critic_batch(agent, states: T.tensor, expected_returns: T.tensor,
                       batch_idx, num_batches):
    """
    Trains the critic for one batch on the score function (MSE usually)
    Args:
        states: representations of the environment
        expected_returns: state values + advantages
    """
    config = agent.config
    state_values = agent.critic(states)

    critic_loss = F.mse_loss(state_values, expected_returns)

    do_accumulated_gradient_step(agent.critic, agent.critic_opt, critic_loss,
                                 config, batch_idx, num_batches)

    if agent.use_wandb:
        log_critic(agent, critic_loss, state_values)
