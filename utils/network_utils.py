import torch as T
from torch.utils.data import DataLoader


def do_accumulated_gradient_step(network: T.nn.Module,
                                 optimizer: T.optim.Optimizer,
                                 objective: T.tensor, config: dict,
                                 batch_idx: int, num_batches: int,
                                 retain_graph=False):
    """
    If target batch size is bigger than the batch size, it will
    accumulate gradient and do a gradient step when target batch size is
    reached.
    Args:
        network: the network to perform the gradient step on
        optimizer: the optimizer that performs the update
        objective: loss or score function to calculate gradients
        config: configuration of the agent
        batch_idx: current batch idx
        num_batches: total number of batches
        retain_graph: retain graph for further backprop
    """
    objective.backward(retain_graph=retain_graph)

    batches_to_acc = int(config['target_batch_size'] / config['batch_size'])
    batches_until_step = (batch_idx + 1) % batches_to_acc
    is_last_batch = batch_idx == num_batches - 1

    if batches_until_step == 0 or is_last_batch:
        if config['grad_norm'] is not None:
            T.nn.utils.clip_grad_norm_(network.parameters(), config[
                'grad_norm'])
        optimizer.step()
        optimizer.zero_grad()


def data_to_device(rollout_data: tuple, device: T.device):
    """Loads data to given device (GPU)"""
    data_on_device = []
    for data in rollout_data:
        data = data.to(device)
        data_on_device.append(data)

    return tuple(data_on_device)


def normalize(x: T.tensor):
    """
    Normalizes given input, special case if only one element
    Args:
        x: input to normalize

    Returns: normalized version of x

    """
    if T.isnan(x.std()):
        return x - x.mean(0)

    return (x - x.mean(0)) / (x.std(0) + 1e-8)


def approx_kl_div(log_probs: T.tensor, old_log_probs: T.tensor,
                  is_aux=False):
    """
    Efficient version to calculate kl divergence
    See Josh Schulmans Blogpost http://joschu.net/blog/kl-approx.html
    We only calculate the gradient during the behavior cloning loss
    calculation of the aux epochs.
    Args:
        log_probs: current log probs of actions
        old_log_probs: log probs of actions at the time of action selection
        ratio: importance sampling ratio between new and old policy
        is_aux: if call is from within aux epoch

    Returns: torch tensor, approximation of KL divergence

    """

    if is_aux:
        loss = T.nn.KLDivLoss(log_target=True, reduction='batchmean')
        return loss(log_probs, old_log_probs)

    else:
        with T.no_grad():
            loss = T.nn.KLDivLoss(log_target=True, reduction='batchmean')
            return loss(log_probs, old_log_probs)


def get_loader(dset, config, drop_last=False):
    return DataLoader(dset, batch_size=config['batch_size'],
                      shuffle=True, pin_memory=True, drop_last=drop_last)
