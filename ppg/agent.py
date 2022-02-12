import torch as T
from torch.distributions.categorical import Categorical
import os
import wandb
import gym

from ppg.networks import CriticNet, PPG_DQN_ARCH
from ppg.trajectory import Trajectory
from ppg.aux_training import train_aux_epoch
from ppg.ppo_training import train_ppo_epoch
from ppg.critic_training import train_critic_epoch
from pretrain.reward import ParticleReward
from pretrain.data_augmentation import DataAugment
from pretrain.contrastive_learning import ContrastiveLearner, ContrastiveLoss
from utils.network_utils import get_loader
from utils.logger import init_logging, log_entropy_coeff

try:
    from apex.optimizers import FusedAdam as Adam

    print('Using APEX optimizer')
except ModuleNotFoundError:
    from torch.optim import Adam

    print('Apex Optimizers not installed, defaulting to PyTorch Optimizer')


class Agent(T.nn.Module):
    def __init__(self, env: gym.envs, config: dict, load=False,
                 load_new_config=False):
        """
        Agent class for PPG + APT
        Args:
            env: gym environment to interact with
            config: configuration data
            load: load from previous run
            load_new_config: load a new config file
        """
        super().__init__()
        self.env = env
        self.metrics = {}

        self.actor = PPG_DQN_ARCH(config['action_dim'],
                                  config['stacked_frames'])
        self.actor_opt = Adam(self.actor.parameters(),
                              lr=config['actor_lr'])
        self.critic = CriticNet(config)
        self.critic_opt = Adam(self.critic.parameters(),
                               lr=config['critic_lr'])
        self.contrast_net = ContrastiveLearner(config)
        self.contrast_opt = Adam(self.contrast_net.parameters(),
                                 lr=config['contrast_lr'])
        self.contrast_loss = ContrastiveLoss(config)
        self.data_aug = DataAugment(config)
        self.reward_function = ParticleReward()
        self.trajectory = Trajectory(config)
        if config['is_pretrain']:
            self.replay_buffer = T.zeros(config['replay_buffer_size'], config[
                'stacked_frames'], config['height'], config['width'])

        self.config = config
        self.entropy_coeff = config['entropy_coeff']
        self.use_wandb = config['use_wandb']
        self.AUX_WARN_THRESHOLD = 100
        self.steps = 0
        self.path = './saved_models'

        self.device = T.device(
            'cuda' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

        init_logging(config, self, config['prefix'])

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
        log_prob = action_dist.log_prob(action).squeeze()
        log_dist = action_dist.probs.log()

        return action.squeeze(), log_prob, aux_value.squeeze(), log_dist

    def learn(self):
        """
        Trains the different networks on the collected trajectories
        """
        self.ppo_training_phase()
        self.steps += self.config['train_iterations']

        if self.steps >= self.config['aux_freq']:
            self.aux_training_phase()
            self.steps = 0

        self.entropy_coeff *= self.config['entropy_decay']

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
        self.trajectory = Trajectory(self.config)

    def save_model(self):
        os.makedirs(self.path, exist_ok=True)
        PATH = self.path + '/agent_latest.pt'
        T.save(self.state_dict(), PATH)

    def load_model(self):
        PATH = self.path + '/agent_latest.pt'
        self.load_state_dict(T.load(PATH))

    def log_metrics(self):
        if self.use_wandb:
            wandb.log(self.metrics)
            self.metrics = {}
