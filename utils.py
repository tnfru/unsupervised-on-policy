import torch as T


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


def approx_kl_div(log_probs, old_log_probs, ratio=None, is_aux=False):
    # See Josh Schulmans Blogpost http://joschu.net/blog/kl-approx.html
    # We only calculate the gradient during the behavior cloning loss
    # calculation of the aux epochs.

    if is_aux:
        ratio = T.exp(log_probs - old_log_probs)
        log_ratio = log_probs - old_log_probs
        kl_div = ratio - 1 - log_ratio

        return kl_div.mean()

    else:
        with T.no_grad():
            log_ratio = log_probs - old_log_probs
            kl_div = ratio - 1 - log_ratio

        return kl_div.mean()


def do_gradient_step(network, optimizer, objective, grad_norm,
                     retain_graph=False):
    optimizer.zero_grad()
    objective.backward(retain_graph=retain_graph)
    if grad_norm is not None:
        T.nn.utils.clip_grad_norm_(network.parameters(), grad_norm)
    optimizer.step()


def data_to_device(rollout_data, device):
    data_on_device = []
    for data in rollout_data:
        data = data.to(device)
        data_on_device.append(data)

    return tuple(data_on_device)
