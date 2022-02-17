import torch as T
import torch.nn.functional as F
from torch.distributions.categorical import Categorical

from utils.network_utils import data_to_device, approx_kl_div
from utils.network_utils import do_accumulated_gradient_step
from utils.logger import warn_about_aux_loss_scaling, log_aux, log_nan_aux


def train_aux_epoch(agent: T.nn.Module, loader: T.utils.data.DataLoader):
    """
    Trains actor on PPG auxiliary objective for one epoch
    Args:
        agent: agent to be trained
        loader: includes the data to be trained on
    """
    num_batches = len(loader)

    for batch_idx, rollout_data in enumerate(loader):
        states, _, aux_returns, log_dists = data_to_device(rollout_data,
                                                           agent.device)
        aux_returns = aux_returns.unsqueeze(1)

        train_aux_batch(agent=agent, states=states,
                        expected_returns=aux_returns, old_log_probs=log_dists,
                        batch_idx=batch_idx, num_batches=num_batches)


def train_aux_batch(agent: T.nn.Module, states: T.tensor,
                    expected_returns: T.tensor,
                    old_log_probs: T.tensor,
                    batch_idx, num_batches):
    """
    Trains actor on PPG auxiliary objective for one batch
    Args:
        states: representation of the environment
        expected_returns: aux state values + aux advantages
        old_log_probs: log probs at the time of action selection
    """
    config = agent.config
    action_logits, aux_values = agent.actor(states)

    if T.isnan(action_logits).any():
        log_nan_aux(agent)
        return

    action_dist = Categorical(logits=action_logits)
    log_probs = action_dist.probs.log()
    kl_div = approx_kl_div(log_probs, old_log_probs, is_aux=True)

    aux_value_loss = F.mse_loss(aux_values, expected_returns)
    aux_value_loss = aux_value_loss * config['val_coeff']

    if aux_value_loss > agent.AUX_WARN_THRESHOLD:
        warn_about_aux_loss_scaling(aux_value_loss)

    aux_loss = aux_value_loss + kl_div * config['beta']

    if config['kl_max_aux'] is None or kl_div < config['kl_max_aux']:
        # If KL divergence is too big we don't take gradient steps
        do_accumulated_gradient_step(agent.actor, agent.actor_opt, aux_loss,
                                     config, batch_idx, num_batches)

    if agent.use_wandb:
        log_aux(agent, aux_values, aux_loss, kl_div, config['kl_max'])
