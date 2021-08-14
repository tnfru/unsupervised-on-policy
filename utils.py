import torch as T
from torch.nn import functional as F


def clipped_value_loss(state_values, old_state_values, expected_returns, clip):
    # TODO explain case in which this is used
    value_clipped = old_state_values + (state_values - old_state_values).clamp(
        -clip, clip)
    clipped_loss = T.square(value_clipped - expected_returns)
    mse_loss = T.square(expected_returns - expected_returns)

    loss = T.max(clipped_loss, mse_loss).mean()

    return loss


def calculate_advantages(rewards, state_vals, discount_factor, gae_lambda):
    advantages = []
    advantage = 0
    next_state_value = 0

    for reward, state_val in zip(reversed(rewards), reversed(state_vals)):
        td_error = reward + discount_factor * next_state_value - state_val
        advantage = td_error + discount_factor * gae_lambda * advantage
        advantages.insert(0, advantage)
        next_state_value = state_val

    return T.tensor(advantages, dtype=T.float)


def approx_kl_div(log_probs, old_log_probs, ratio=None):
    with T.no_grad():
        if ratio is None:
            ratio = T.exp(log_probs - old_log_probs)

        log_ratio = log_probs - old_log_probs
        kl_div = ratio - 1 - log_ratio

        return kl_div.mean()
