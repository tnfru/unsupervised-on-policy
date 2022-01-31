import torch as T


def calculate_advantages(rewards, state_vals, dones, discount_factor,
                         gae_lambda):
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
