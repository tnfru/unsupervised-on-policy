from agent import Agent
from rollout import run_timesteps
import pretrain.environment as environment

if __name__ == '__main__':
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
              'discount_factor': 0.99,
              'height': 84,
              'width': 84,
              }

    FRAMES_TO_STACK = 4
    FRAMES_TO_SKIP = 4
    SEED = 1337
    NUM_TIMESTEPS = int(1e3)

    environment.seed_everything(SEED)
    env = environment.create_env(config['height'], config['width'])
    agent = Agent(env, action_dim=18, state_dim=FRAMES_TO_STACK, config=config)

    run_timesteps(agent, NUM_TIMESTEPS, pretrain=True)
