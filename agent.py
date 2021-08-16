import torch as T

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.distributions.categorical import Categorical

from networks import PPG, CriticNet
from logger import *
from utils import approx_kl_div, data_to_device
from trajectory import Trajectory
from objectives import ppo_objective, value_loss_fun, entropy_objective
from aux_train import train_aux_epoch


class Agent:
    def __init__(self, env, action_dim, state_dim, config):
        self.env = env
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
            prefix = 'relu'
            init_logging(config, self.actor, self.critic, prefix)

    def get_action(self, state):
        with T.no_grad():
            action_probs, aux_value = self.actor(state)

            action_dist = Categorical(logits=action_probs)
            action = action_dist.sample()
            log_prob = action_dist.log_prob(action).item()
            log_dist = action_dist.probs.log().cpu().detach()

            return action.item(), log_prob, aux_value.item(), log_dist

    def learn(self):
        config = self.config
        for epoch in range(config['train_iterations']):
            loader = DataLoader(self.trajectory, batch_size=config[
                'batch_size'], shuffle=True, pin_memory=True)
            self.train_ppo_epoch(loader)
            self.entropy_coeff *= config['entropy_decay']
            self.steps += 1

        if self.steps == config['aux_freq']:
            self.aux_training_phase()

    def aux_training_phase(self):
        config = self.config
        self.trajectory.is_aux_epoch = True
        self.steps = 0
        loader = DataLoader(self.trajectory, batch_size=config[
            'batch_size'], shuffle=True, pin_memory=True)

        for aux_epoch in range(config['aux_iterations']):
            train_aux_epoch(agent=self, loader=loader)
        self.trajectory.is_aux_epoch = False

    def train_ppo_epoch(self, loader):
        for rollout_data in loader:
            states, actions, expected_returns, state_vals, advantages, \
            log_probs = data_to_device(rollout_data, self.device)
            expected_returns = expected_returns.unsqueeze(1)

            self.train_policy(states, actions, log_probs, advantages)
            self.train_critic(states, expected_returns, state_vals)

    def train_policy(self, states, actions, old_log_probs, advantages):
        config = self.config
        action_probs, _ = self.actor(states)
        action_dist = Categorical(logits=action_probs)
        log_probs = action_dist.log_prob(actions)

        # entropy for exploration
        entropy_loss = entropy_objective(action_dist, self.entropy_coeff)

        # log trick for efficient computational graph during backprop
        ratio = T.exp(log_probs - old_log_probs)
        ppo_loss = ppo_objective(advantages, ratio, config['policy_clip'])

        objective = ppo_loss + entropy_loss

        kl_div = approx_kl_div(log_probs, old_log_probs, ratio)
        if kl_div < config['kl_max']:
            # If KL divergence is too big we don't take gradient steps
            self.do_gradient_step(self.actor, self.actor_opt, objective,
                                  retain_graph=True)

        if self.use_wandb:
            log_ppo(entropy_loss, kl_div, config['kl_max'])

    def do_gradient_step(self, network, optimizer, objective,
                         retain_graph=False):
        config = self.config
        optimizer.zero_grad()
        if config['grad_norm'] is not None:
            nn.utils.clip_grad_norm_(network.parameters(),
                                     config['grad_norm'])
        objective.backward(retain_graph=retain_graph)
        optimizer.step()

    def train_critic(self, states, expected_returns, old_state_values):
        config = self.config
        state_values = self.critic(states)
        critic_loss = value_loss_fun(state_values=state_values,
                                     old_state_values=old_state_values,
                                     expected_returns=expected_returns,
                                     is_aux_epoch=self.trajectory.is_aux_epoch,
                                     value_clip=config['value_clip'])

        self.do_gradient_step(self.critic, self.critic_opt, critic_loss)

        if self.use_wandb:
            log_critic(critic_loss, state_values)

    def forget(self):
        self.trajectory = Trajectory()
