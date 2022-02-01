import torch as T
from torch.utils.data import DataLoader


def do_gradient_step(network, optimizer, objective, grad_norm,
                     retain_graph=False):
    optimizer.zero_grad()
    objective.backward(retain_graph=retain_graph)
    if grad_norm is not None:
        T.nn.utils.clip_grad_norm_(network.parameters(), grad_norm)
    optimizer.step()


def do_accumulated_gradient_step(network, optimizer, objective, config,
                                 batch_idx, num_batches, retain_graph=False):
    batches_to_acc = int(config['target_batch_size'] / config['batch_size'])
    batches_until_step = batch_idx % batches_to_acc
    is_last_batch = batch_idx == num_batches - 1

    if batches_until_step == 0 or is_last_batch:
        do_gradient_step(network, optimizer, objective, grad_norm=config[
            'grad_norm'], retain_graph=retain_graph)


def data_to_device(rollout_data, device):
    data_on_device = []
    for data in rollout_data:
        data = data.to(device)
        data_on_device.append(data)

    return tuple(data_on_device)


def normalize(x):
    if T.isnan(x.std()):
        return x - x.mean(0)

    return (x - x.mean(0)) / (x.std(0) + 1e-8)


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


def get_loader(dset, config, drop_last=False):
    return DataLoader(dset, batch_size=config['batch_size'],
                      shuffle=True, pin_memory=True, drop_last=drop_last)
