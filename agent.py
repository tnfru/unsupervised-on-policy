import torch as T
import torch.optim as optim
import gym

from torch.utils.data import DataLoader
from torch.distributions.categorical import Categorical

from networks import PPG, CriticNet
from logger import init_logging
from trajectory import Trajectory
from aux_training import train_aux_epoch
from ppo_training import train_ppo_epoch


class Agent:
    def __init__(self, env, action_dim, state_dim, config):
        self.env = env
        # action_dim = gym.spaces.utils.flatdim(env.action_space)
        # discreteaction space

        self.actor = PPG(action_dim, state_dim)
        self.critic = CriticNet(state_dim)
        self.batch_size = config['batch_size']
        self.actor_opt = optim.Adam(self.actor.parameters(),
                                    lr=config['actor_lr'])
        self.critic_opt = optim.Adam(self.critic.parameters(),
                                     lr=config['critic_lr'])
        self.device = T.device(
            'cuda:0' if T.cuda.is_available() else 'cpu')
        self.config = config
        self.entropy_coeff = config['entropy_coeff']
        self.trajectory = Trajectory()
        self.use_wandb = config['use_wandb']
        self.steps = 0
        self.AUX_WARN_THRESHOLD = 100

        if self.use_wandb:
            prefix = '3 aux epochs'
            init_logging(config, self.actor, self.critic, prefix)

    @T.no_grad()
    def get_action(self, state):
        action_probs, aux_value = self.actor(state)

        action_dist = Categorical(logits=action_probs)
        action = action_dist.sample()
        log_prob = action_dist.log_prob(action).item()
        log_dist = action_dist.probs.log().cpu().detach()

        return action.item(), log_prob, aux_value.item(), log_dist

    def learn(self):
        config = self.config
        self.ppo_training_phase()
        self.steps += config['train_iterations']

        if self.steps >= config['aux_freq']:
            self.aux_training_phase()
            self.steps = 0

    def ppo_training_phase(self):
        config = self.config
        loader = DataLoader(self.trajectory, batch_size=config[
            'batch_size'], shuffle=True, pin_memory=True)

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
