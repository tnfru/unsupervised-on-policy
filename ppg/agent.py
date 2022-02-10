import torch as T
import torch.optim as optim
import os
import wandb
import gym

from torch.distributions.categorical import Categorical

from ppg.networks import PPG, CriticNet, PPG_DQN_ARCH
from utils.logger import init_logging, log_contrast_loss_batch, \
    log_contrast_loss_epoch, log_entropy_coeff
from ppg.trajectory import Trajectory
from ppg.aux_training import train_aux_epoch
from ppg.ppo_training import train_ppo_epoch
from pretrain.reward import ParticleReward
from pretrain.data_augmentation import DataAugment
from pretrain.contrastive_learning import ContrastiveLearner, ContrastiveLoss
from pretrain.state_data import StateData
from utils.network_utils import get_loader, do_accumulated_gradient_step
from ppg.critic_training import train_critic_epoch
from pretrain.state_data import RepresentationData


class Agent(T.nn.Module):
    def __init__(self, env: gym.envs, action_dim: int, config: dict, load=False,
                 load_new_config=False):
        """
        Agent class for PPG + APT
        Args:
            env: gym environment to interact with
            action_dim: number of available actions
            config: configuration data
            load: load from previous run
            load_new_config: load a new config file
        """
        super().__init__()
        self.env = env
        self.metrics = {}

        self.actor = PPG_DQN_ARCH(action_dim, config['stacked_frames'])
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
        self.replay_buffer = RepresentationData(config)

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
                entropy_coeff = self.entropy_coeff
                self.config = config
                self.entropy_coeff = entropy_coeff

    @T.no_grad()
    def get_action(self, state):
        """
        Args:
            state: torch tensor, current observation from environment

        Returns:
            action: int, selected action
            log_prob: float, log probability of the selected action
            aux value: float, state value as predicted by ppg value head 
            log_dist: torch tensor, log distribution over actions

        """
        action_probs, aux_value = self.actor(state)

        action_dist = Categorical(logits=action_probs)
        action = action_dist.sample()
        log_prob = action_dist.log_prob(action).item()
        log_dist = action_dist.probs.log().cpu().detach()

        return action.item(), log_prob, aux_value.item(), log_dist

    def learn(self, is_pretrain):
        """
        Trains the different networks on the collected trajectories
        Args:
            is_pretrain: if reward is calculated in a self supervised 
            fashion, as opposed to be given by the environment

        """
        self.ppo_training_phase()
        self.steps += self.config['train_iterations']

        if self.steps >= self.config['aux_freq']:
            self.aux_training_phase()
            self.steps = 0

        self.entropy_coeff *= self.config['entropy_decay']

        if self.use_wandb:
            log_entropy_coeff(self)
            self.log_metrics()

    def ppo_training_phase(self):
        """ Trains the actor network on the PPO Objective """
        loader = get_loader(dset=self.trajectory, config=self.config)

        for epoch in range(self.config['train_iterations']):
            train_ppo_epoch(agent=self, loader=loader)
            train_critic_epoch(agent=self, loader=loader)

    def aux_training_phase(self):
        """ Trains the actor network on the PPG auxiliary Objective """
        self.trajectory.is_aux_epoch = True

        loader = get_loader(dset=self.trajectory, config=self.config)

        for aux_epoch in range(self.config['aux_iterations']):
            train_aux_epoch(agent=self, loader=loader)
            train_critic_epoch(agent=self, loader=loader, is_aux=True)
        self.trajectory.is_aux_epoch = False

    def forget(self):
        """ Removes the collected data after training"""
        self.trajectory = Trajectory()

    def contrast_training_phase(self):
        """ Trains the encoder on the NT-Xent loss from SimCLR"""
        #states = self.trajectory.states
        #state_dset = StateData(states)
        loader = get_loader(dset=self.replay_buffer, config=self.config)
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
