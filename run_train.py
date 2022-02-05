from ppg.agent import Agent
from rollout import run_timesteps
import pretrain.environment as environment

if __name__ == '__main__':
    config = {'policy_clip': 0.25,
              'kl_max': None,  # 0.05 used previously
              'beta': 1,
              'val_coeff': 1e-2,
              'rollout_length': 512,
              'train_iterations': 1,
              'entropy_coeff': 0.01,
              'grad_norm': 10,  # 0.5 alternatively
              'critic_lr': 1e-3,
              'actor_lr': 3e-4,  # Paper val 1e-4 while pre-Training
              'aux_freq': 32,
              'aux_iterations': 3,
              'gae_lambda': 0.95,
              'batch_size': 256,  # 512 while pretraining, 32 after
              'target_batch_size': 512,
              'value_clip': None,  # 0.4 alternatively
              'entropy_decay': 0.999,
              'use_wandb': True,
              'discount_factor': 0.99,
              'height': 84,
              'width': 84,
              'contrast_lr': 1e-3,
              'temperature': 0.1,
              'frames_to_skip': 4,
              'stacked_frames': 4,
              'prefix': 'Loaded_Run'
              }

    SEED = 1337
    NUM_TIMESTEPS = 4_000_000
    act_dim = 18

    environment.seed_everything(SEED)
    env = environment.create_env(config)
    agent = Agent(env, action_dim=act_dim, config=config, load=True,
                  load_new_config=True)

    run_timesteps(agent, NUM_TIMESTEPS, is_pretrain=True)
