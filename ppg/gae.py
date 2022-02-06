import torch as T


def calculate_advantages(rewards: list, state_vals: list, dones: list,
                         discount_factor: float,
                         gae_lambda: float):
    """
    Calculated Advantages via Generalized Advantage Estimation (GAE)
    https://arxiv.org/abs/1506.02438

    Args:
        rewards: reward obtained
        state_vals: predicted state values
        dones: if terminal state or not
        discount_factor: scaling factor for value of next state
        gae_lambda: hyperparameter for GAE scaling

    Returns: advantages over the given timesteps

    """
    advantages = []
    advantage = 0
    next_state_value = 0

    for reward, state_val, done in zip(reversed(rewards), reversed(state_vals),
                                       reversed(dones)):
        td_error = reward + (
                1 - done) * discount_factor * next_state_value - state_val
        advantage = td_error + discount_factor * gae_lambda * advantage
        advantages.insert(0, advantage)
        next_state_value = state_val

    return T.tensor(advantages, dtype=T.float)
