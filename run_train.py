from agent import Agent
from rollout import run_timesteps
import pretrain.environment as environment

if __name__ == '__main__':
    config = {'policy_clip': 0.25,
              'kl_max': 0.05,
              'beta': 1,
              'val_coeff': 1e-2,
              'rollout_length': 256,
              'train_iterations': 1,
              'entropy_coeff': 0.01,
              'grad_norm': 10,  # 0.5 alternatively
              'critic_lr': 1e-3,
              'actor_lr': 3e-4,
              'aux_freq': 32,
              'aux_iterations': 3,
              'gae_lambda': 0.95,
              'batch_size': 32,
              'value_clip': None,  # 0.4 alternatively
              'entropy_decay': 0.999,
              'use_wandb': True,
              'discount_factor': 0.99,
              'height': 84,
              'width': 84,
              'contrast_lr': 3e-3,
              'temperature': 0.1,
              'contrast_head_dim': 5,
              'frames_to_skip': 4,
              'frames_to_stack': 4
              }

    FRAMES_TO_STACK = 4
    FRAMES_TO_SKIP = 4
    SEED = 1337
    NUM_TIMESTEPS = 250_000_000

    # TODO Terminal on loss of life
    # TODO compare Adam with LARS optimizer vs AdamW
    # TODO vectorized envs
    # TODO image normalization
    # TODO save model

    environment.seed_everything(SEED)
    env = environment.create_env(config)
    agent = Agent(env, action_dim=18, config=config)

    run_timesteps(agent, NUM_TIMESTEPS, is_pretrain=True)
