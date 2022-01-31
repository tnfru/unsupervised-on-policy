import torch as T
import torch.optim as optim
import os
import wandb

from torch.distributions.categorical import Categorical

from ppg.networks import PPG, CriticNet
from utils.logger import init_logging, log_contrast_loss_batch, \
    log_contrast_loss_epoch
from ppg.trajectory import Trajectory
from ppg.aux_training import train_aux_epoch
from ppg.ppo_training import train_ppo_epoch
from pretrain.reward import ParticleReward
from pretrain.data_augmentation import DataAugment
from pretrain.contrastive_learning import ContrastiveLearner, ContrastiveLoss
from pretrain.state_data import StateData
from utils.network_utils import get_loader, do_gradient_step, \
    do_accumulated_gradient_step


class Agent(T.nn.Module):
    def __init__(self, env, action_dim, config, load=False,
                 load_new_config=False):
        super().__init__()
        self.env = env
        self.metrics = {}

        self.actor = PPG(action_dim, config['stacked_frames'])
        self.actor_opt = optim.Adam(
            self.actor.parameters(),
            lr=config['actor_lr']
        )
        self.critic = CriticNet(config)
        self.critic_opt = optim.Adam(
            self.critic.parameters(),
            lr=config['critic_lr']
        )
        self.contrast_net = ContrastiveLearner(config)
        self.contrast_opt = optim.Adam(
            self.contrast_net.parameters(),
            lr=config['contrast_lr']
        )
        self.contrast_loss = ContrastiveLoss(config)
        self.data_aug = DataAugment(config)
        self.reward_function = ParticleReward()
        self.trajectory = Trajectory()

        self.config = config
        self.entropy_coeff = config['entropy_coeff']
        self.use_wandb = config['use_wandb']
        self.AUX_WARN_THRESHOLD = 100
        self.steps = 0
        self.path = './saved_models'

        self.device = T.device(
            'cuda' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

        if self.use_wandb:
            prefix = config['prefix']
            init_logging(config, self, prefix)

        if load:
            self.load_model()

            if load_new_config:
                self.config = config

    @T.no_grad()
    def get_action(self, state):
        action_probs, aux_value = self.actor(state)

        action_dist = Categorical(logits=action_probs)
        action = action_dist.sample()
        log_prob = action_dist.log_prob(action).item()
        log_dist = action_dist.probs.log().cpu().detach()

        return action.item(), log_prob, aux_value.item(), log_dist

    def learn(self, is_pretrain):
        if is_pretrain:
            self.contrast_training_phase()
        self.ppo_training_phase()
        self.steps += self.config['train_iterations']

        if self.steps >= self.config['aux_freq']:
            self.aux_training_phase()
            self.steps = 0

    def ppo_training_phase(self):
        loader = get_loader(dset=self.trajectory, config=self.config)

        for epoch in range(self.config['train_iterations']):
            train_ppo_epoch(agent=self, loader=loader)
            self.entropy_coeff *= self.config['entropy_decay']

    def aux_training_phase(self):
        self.trajectory.is_aux_epoch = True

        loader = get_loader(dset=self.trajectory, config=self.config)

        for aux_epoch in range(self.config['aux_iterations']):
            train_aux_epoch(agent=self, loader=loader)
        self.trajectory.is_aux_epoch = False

    def forget(self):
        self.trajectory = Trajectory()

    def contrast_training_phase(self):
        states = self.trajectory.states
        state_dset = StateData(states)
        loader = get_loader(dset=state_dset, config=self.config)
        total_contrast_loss = 0

        for batch_idx, state_batch in enumerate(loader):
            state_batch = state_batch.to(self.device)
            view_1 = self.data_aug(state_batch)
            view_2 = self.data_aug(state_batch)

            projection_1 = self.contrast_net.project(view_1)
            projection_2 = self.contrast_net.project(view_2)

            loss = self.contrast_loss(projection_1, projection_2)

            do_accumulated_gradient_step(self.contrast_net,
                                         self.contrast_opt, loss,
                                         self.config, batch_idx, len(loader))
            # do_gradient_step(self.contrast_net, self.contrast_opt, loss,
            #                 self.config['grad_norm'])

            if self.use_wandb:
                log_contrast_loss_batch(self, loss.item())
            total_contrast_loss += loss.item()

        total_contrast_loss /= len(loader)

        if self.use_wandb:
            log_contrast_loss_epoch(self, total_contrast_loss)

    def save_model(self):
        os.makedirs(self.path, exist_ok=True)
        PATH = self.path + '/agent_latest.pt'
        T.save(self.state_dict(), PATH)

    def load_model(self):
        PATH = self.path + '/agent_latest.pt'
        self.load_state_dict(T.load(PATH))

    def log_metrics(self):
        wandb.log(self.metrics)
        self.metrics = {}
