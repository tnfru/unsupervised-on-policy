import torch as T
from torch.distributions.categorical import Categorical

from utils.network_utils import data_to_device, approx_kl_div
from utils.network_utils import do_accumulated_gradient_step
from utils.logger import log_ppo


def train_ppo_epoch(agent: T.nn.Module, loader: T.utils.data.DataLoader):
    """
    Trains actor on PPO surrogate objective for one epoch
    Args:
        agent: agent to be trained
        loader: includes the data to be trained on
    """
    num_batches = len(loader)

    for batch_idx, rollout_data in enumerate(loader):
        states, actions, expected_returns, state_values, advantages, \
        log_probs = data_to_device(rollout_data, agent.device)

        train_ppo_batch(agent, states, actions, log_probs, advantages,
                        batch_idx, num_batches)


def train_ppo_batch(agent, states: T.tensor, actions: T.tensor,
                    old_log_probs: T.tensor, advantages: T.tensor,
                    batch_idx, num_batches):
    """
    Trains actor on PPO surrogate objective for one batch
    Args:
        states: representation of the environment
        actions: selected action for the respective states
        advantages: advantages of the selected action in their states
        old_log_probs: log probs at the time of action selection
            old_aux_value: aux state value
    """
    config = agent.config
    action_probs, _ = agent.actor(states)
    action_dist = Categorical(logits=action_probs)
    log_probs = action_dist.log_prob(actions)

    # entropy for exploration
    entropy_loss = entropy_objective(action_dist, agent.entropy_coeff)

    # log trick for efficient computational graph during backprop
    ratio = T.exp(log_probs - old_log_probs)
    ppo_loss = ppo_objective(advantages, ratio, config['policy_clip'])

    objective = ppo_loss + entropy_loss

    kl_div = approx_kl_div(log_probs, old_log_probs, is_aux=False)

    if config['kl_max'] is None or kl_div < config['kl_max']:
        # If KL divergence is too big we don't take gradient steps
        do_accumulated_gradient_step(agent.actor, agent.actor_opt, objective,
                                     config, batch_idx, num_batches,
                                     retain_graph=True)

    if agent.use_wandb:
        log_ppo(agent, entropy_loss, kl_div, config['kl_max'])


# Both objectives flip the sign to turn their objective into a loss
# This is because we want to do gradient ascent on the objectives, but
# optimizers in PyTorch generally do gradient descent

def ppo_objective(advantages, ratio, policy_clip):
    """
    Calculates the PPO CLIP objective as loss
    Args:
        advantages: advantages of the selected actions
        ratio: importance sampling ratio between new and old policy
        policy_clip: maximum allowed change from old policy
    """
    weighted_objective = ratio * advantages
    clipped_objective = ratio.clamp(1 - policy_clip,
                                    1 + policy_clip) * advantages
    ppo_loss = -T.min(weighted_objective, clipped_objective).mean()

    return ppo_loss


def entropy_objective(
        action_distribution: T.distributions.categorical.Categorical,
        entropy_coeff: float):
    """
    Entropy for exploration on PPO objective
    Args:
        action_distribution: distribution over the actions
        entropy_coeff: scaling factor

    Returns:

    """
    entropy_loss = -action_distribution.entropy().mean() * entropy_coeff

    return entropy_loss
