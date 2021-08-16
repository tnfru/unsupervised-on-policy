import torch as T
from torch.nn import functional as F


def ppo_objective(advantages, ratio, policy_clip):
    weighted_objective = ratio * advantages
    clipped_objective = ratio.clamp(1 - policy_clip,
                                    1 + policy_clip) * advantages
    loss = -T.min(weighted_objective, clipped_objective).mean()

    return loss


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


def entropy_objective(action_distribution, entropy_coeff):
    entropy_loss = -action_distribution.entropy().mean() * entropy_coeff
    return entropy_loss
