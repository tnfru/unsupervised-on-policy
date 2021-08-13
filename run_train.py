import gym
from agent import Agent
from train import run_timesteps

if __name__ == '__main__':
    config = {'clip_ratio': 0.25,
              'kl_max': 0.06,
              'beta': 1,
              'val_coeff': 1e-2,
              'aux_iterations': 3,
              'rollout_length': 256,
              'train_iterations': 1,
              'entropy_coeff': 0.01,
              'grad_norm': 0.5,
              'critic_lr': 1e-3,
              'actor_lr': 3e-4,
              'batch_size': 64,
              'value_clip': 0.4,
              'use_wandb': True
              }

    gym.envs.register(
        id='CartPole-v2000',
        entry_point='gym.envs.classic_control:CartPoleEnv',
        max_episode_steps=2000
    )

    env = gym.make("CartPole-v2000")
    agent = Agent(env, 2, 4, config=config)

    run_timesteps(agent, 1e7)
