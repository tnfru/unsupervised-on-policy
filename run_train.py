from ppg.agent import Agent
from rollout import run_timesteps
import pretrain.environment as environment

if __name__ == '__main__':
    config = {'policy_clip': 0.25,
              'kl_max': None,  # 0.05 used previously
              'beta': 1,
              'val_coeff': 1e-2,
              'train_iterations': 1,
              'entropy_coeff': 0.01,
              'entropy_decay': 0.999,
              'grad_norm': 10,
              'grad_norm_ppg': 0.5,
              'critic_lr': 1e-3,
              'actor_lr': 3e-4,  # Paper val 1e-4 while pre-Training
              'aux_freq': 32,
              'aux_iterations': 3,
              'gae_lambda': 0.95,
              'batch_size': 32,  # 512 while pretraining, 32 after
              'target_batch_size': 32,
              'use_wandb': True,
              'discount_factor': 0.99,
              'height': 84,
              'width': 84,
              'action_dim': 18,
              'contrast_lr': 1e-3,
              'temperature': 0.1,
              'frames_to_skip': 4,
              'stacked_frames': 4,
              'is_pretrain': True,
              'steps_before_repr_learning': 1600,  # Paper value
              'replay_buffer_size': 10000,
              'num_envs': 16,  # Parallel Envs
              'prefix': 'PRE_VECTORIZED_ENVS'
              }

    if config['is_pretrain']:
        config.update({
            'batch_size': 512,
            'target_batch_size': 512,
        })

    config.update({
        'rollout_length': max(config['target_batch_size'], 256)
    })

    SEED = 1337
    NUM_TIMESTEPS = 250_000_000

    environment.seed_everything(SEED)
    env = environment.create_env(config)
    agent = Agent(env, config=config,
                  load=False,
                  load_new_config=False)

    run_timesteps(agent, NUM_TIMESTEPS, pretrain=config['is_pretrain'])
