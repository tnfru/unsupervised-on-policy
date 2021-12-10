import torch as T
import torch.optim as optim
import gym
import wandb

from torch.utils.data import DataLoader
from torch.distributions.categorical import Categorical

from networks import PPG, CriticNet
from logger import init_logging, log_contrast_loss
from trajectory import Trajectory
from aux_training import train_aux_epoch
from ppo_training import train_ppo_epoch
from pretrain.reward import ParticleReward
from pretrain.data_augmentation import DataAugment
from pretrain.contrastive_learning import ContrastiveLearner, ContrastiveLoss
from pretrain.state_data import StateData


class Agent:
    def __init__(self, env, action_dim, config):
        self.env = env

        state_dim = config['frames_to_stack']
        self.batch_size = config['batch_size']
        self.actor = PPG(action_dim, state_dim)
        self.critic = CriticNet(state_dim)
        self.contrast_net = ContrastiveLearner(state_dim, out_dim=config[
            'contrast_head_dim'])
        self.actor_opt = optim.Adam(self.actor.parameters(),
                                    lr=config['actor_lr'])
        self.critic_opt = optim.Adam(self.critic.parameters(),
                                     lr=config['critic_lr'])
        self.contrast_opt = optim.Adam(self.contrast_net.parameters(),
                                       lr=config['contrast_lr'])
        self.contrast_loss = ContrastiveLoss(config['temperature'])
        self.device = T.device(
            'cuda' if T.cuda.is_available() else 'cpu')
        # TODO add data paralellism
        self.config = config
        self.entropy_coeff = config['entropy_coeff']
        self.trajectory = Trajectory()
        self.use_wandb = config['use_wandb']
        self.steps = 0
        self.AUX_WARN_THRESHOLD = 100
        self.reward_function = ParticleReward()
        self.data_aug = DataAugment(config['height'], config['width'])

        if self.use_wandb:
            prefix = 'Initial Runs'
            init_logging(config, self.actor, self.critic, prefix)

    @T.no_grad()
    def get_action(self, state):
        action_probs, aux_value = self.actor(state)

        action_dist = Categorical(logits=action_probs)
        action = action_dist.sample()
        log_prob = action_dist.log_prob(action).item()
        log_dist = action_dist.probs.log().cpu().detach()

        return action.item(), log_prob, aux_value.item(), log_dist

    def learn(self, is_pretrain):
        config = self.config
        if is_pretrain:
            self.contrast_training_phase()
        self.ppo_training_phase()
        self.steps += config['train_iterations']

        if self.steps >= config['aux_freq']:
            self.aux_training_phase()
            self.steps = 0

    def ppo_training_phase(self):
        config = self.config
        loader = DataLoader(self.trajectory, batch_size=config[
            'batch_size'], shuffle=True, pin_memory=True, drop_last=True)

        for epoch in range(config['train_iterations']):
            train_ppo_epoch(agent=self, loader=loader)
            self.entropy_coeff *= config['entropy_decay']

    def aux_training_phase(self):
        config = self.config
        self.trajectory.is_aux_epoch = True

        loader = DataLoader(self.trajectory, batch_size=config[
            'batch_size'], shuffle=True, pin_memory=True)

        for aux_epoch in range(config['aux_iterations']):
            train_aux_epoch(agent=self, loader=loader)
        self.trajectory.is_aux_epoch = False

    def forget(self):
        self.trajectory = Trajectory()

    def contrast_training_phase(self):
        config = self.config

        states = self.trajectory.states
        state_dset = StateData()
        state_dset.append_states(states)

        loader = DataLoader(state_dset, batch_size=config['batch_size'],
                            shuffle=True, pin_memory=True)
        total_contrast_loss = 0

        for state_batch in loader:
            state_batch = state_batch.to(self.device)
            view_1 = self.data_aug(state_batch)
            view_2 = self.data_aug(state_batch)

            representation_1 = self.contrast_net.project(view_1)
            representation_2 = self.contrast_net.project(view_2)

            loss = self.contrast_loss(representation_1, representation_2)

            self.contrast_opt.zero_grad()
            loss.backward()
            self.contrast_opt.step()

            log_contrast_loss(loss.item())
            total_contrast_loss += loss.item()
        total_contrast_loss /= len(loader)
        wandb.log({'total contrast loss': total_contrast_loss})
