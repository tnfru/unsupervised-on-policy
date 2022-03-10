import torch as T
from einops import rearrange


def calculate_advantages(rewards: T.tensor, state_vals: T.tensor,
                         dones: T.tensor, last_state_val: T.tensor,
                         config: dict):
    """
    Calculated Advantages via Generalized Advantage Estimation (GAE)
    https://arxiv.org/abs/1506.02438

    Args:
        rewards: reward obtained
        state_vals: predicted state values
        last_state_val: value of the next state of the last step
        dones: if terminal state or not
        config: configuration file

    Returns: advantages over the given timesteps

    """
    discount_factor = config['discount_factor']
    gae_lambda = config['gae_lambda']

    advantages = []
    advantage = 0
    next_state_value = last_state_val
    num_envs = config['num_envs']

    rewards = rearrange(rewards, '(step env) -> step env ', env=num_envs)
    state_vals = rearrange(state_vals, '(step env) -> step env ', env=num_envs)
    dones = rearrange(dones, '(step env) -> step env ', env=num_envs)

    for reward, state_val, done in zip(reversed(rewards), reversed(state_vals),
                                       reversed(dones)):
        td_error = reward + (
                1 - done) * discount_factor * next_state_value - state_val
        advantage = td_error + discount_factor * gae_lambda * advantage
        advantages.insert(0, advantage)
        next_state_value = state_val

    advantages = T.stack(advantages)
    advantages = rearrange(advantages, 'step env -> (step env) ', env=num_envs)

    return advantages
