import torch as T
from torch.distributions.categorical import Categorical

from utils import data_to_device, approx_kl_div, do_gradient_step
from objectives import ppo_objective, entropy_objective
from logger import log_ppo
from critic_train import train_critic


def train_ppo_epoch(agent, loader):
    for rollout_data in loader:
        states, actions, expected_returns, state_values, advantages, \
        log_probs = data_to_device(rollout_data, agent.device)
        expected_returns = expected_returns.unsqueeze(1)

        train_ppo(agent, states, actions, log_probs, advantages)
        train_critic(agent, states, expected_returns, state_values)


def train_ppo(agent, states, actions, old_log_probs, advantages):
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

    kl_div = approx_kl_div(log_probs, old_log_probs, ratio)
    if kl_div < config['kl_max']:
        # If KL divergence is too big we don't take gradient steps
        do_gradient_step(agent.actor, agent.actor_opt, objective,
                         grad_norm=config['grad_norm'], retain_graph=True)

    if agent.use_wandb:
        log_ppo(entropy_loss, kl_div, config['kl_max'])
