import gym
import torch as T
import random
from agent import Agent
from rollout import run_timesteps

if __name__ == '__main__':
    T.manual_seed(1337)
    random.seed(1337)
    num_timesteps = int(1e7)
    config = {'policy_clip': 0.25,
              'kl_max': 0.03,
              'beta': 1,
              'val_coeff': 1e-2,
              'rollout_length': 256,
              'train_iterations': 1,
              'entropy_coeff': 0.01,
              'grad_norm': 0.5,
              'critic_lr': 1e-3,
              'actor_lr': 3e-4,
              'aux_freq': 32,
              'aux_iterations': 3,
              'gae_lambda': 0.95,
              'batch_size': 64,
              # 'value_clip': 0.4,
              'value_clip': None,
              'entropy_decay': 0.999,
              'use_wandb': True,
              'discount_factor': 0.99
              }

    gym.envs.register(
        id='CartPole-v2000',
        entry_point='gym.envs.classic_control:CartPoleEnv',
        max_episode_steps=2000
    )

    env = gym.make("CartPole-v2000")
    agent = Agent(env, action_dim=2, state_dim=4, config=config)

    run_timesteps(agent, num_timesteps)
