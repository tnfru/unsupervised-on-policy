import torch as T
from operator import itemgetter

from utils.logger import log_contrast_loss_epoch
from utils.network_utils import do_gradient_step


def train_contrastive_batch(agent):
    """ Trains the encoder on the NT-Xent loss from SimCLR"""
    batch_size = agent.config['batch_size']

    indices = T.randperm(len(agent.replay_buffer))[:batch_size]

    state_batch = T.stack(itemgetter(*indices)(agent.replay_buffer))
    state_batch = state_batch.to(agent.device)

    view_1 = agent.data_aug(state_batch)
    view_2 = agent.data_aug(state_batch)

    projection_1 = agent.contrast_net.project(view_1)
    projection_2 = agent.contrast_net.project(view_2)

    loss = agent.contrast_loss(projection_1, projection_2)

    do_gradient_step(agent.contrast_net, agent.contrast_opt, loss, agent.config)

    if agent.use_wandb:
        log_contrast_loss_epoch(agent, loss.item())
