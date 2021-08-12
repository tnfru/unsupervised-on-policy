import torch as T
from torch.nn import functional as F


def clipped_value_loss(values, expected_returns, old_values, clip=0.4):
    value_clipped = old_values + T.clamp(values - old_values,
                                         -clip, clip)
    clipped_td_error = T.square(expected_returns - value_clipped)
    td_error = T.square(values- expected_returns)

    loss = T.max(clipped_td_error, td_error).mean()

    return loss

