import torch as T
import torch.nn.functional as F
from torch.distributions.categorical import Categorical


def clipped_value_loss(state_values, old_state_values, expected_returns, clip):
    clipped_difference = (state_values - old_state_values).clamp(-clip, clip)
    value_clipped = old_state_values + clipped_difference

    clipped_loss = T.square(value_clipped - expected_returns)
    mse_loss = T.square(state_values - expected_returns)

    loss = T.max(clipped_loss, mse_loss).mean()

    return loss


def value_loss_fun(state_values, old_state_values, expected_returns,
                   is_aux_epoch, value_clip):
    if value_clip is not None and is_aux_epoch:
        # clip value loss in aux episodes to reduce overfitting
        loss = clipped_value_loss(state_values, old_state_values,
                                  expected_returns, value_clip)
    else:
        loss = F.mse_loss(state_values, expected_returns)

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
    # See Josh Schulamns Blogpost http://joschu.net/blog/kl-approx.html
    with T.no_grad():
        if ratio is None:
            ratio = T.exp(log_probs - old_log_probs)

        log_ratio = log_probs - old_log_probs
        kl_div = ratio - 1 - log_ratio

        return kl_div.mean()


def data_to_device(rollout_data, device):
    data_on_device = []
    for data in rollout_data:
        data = data.to(device)
        data_on_device.append(data)

    return tuple(data_on_device)
