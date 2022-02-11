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
              'grad_norm': 10,  # 0.5 alternatively
              'critic_lr': 1e-3,
              'actor_lr': 3e-4,  # Paper val 1e-4 while pre-Training
              'aux_freq': 32,
              'aux_iterations': 3,
              'gae_lambda': 0.95,
              'batch_size': 32,  # 512 while pretraining, 32 after
              'target_batch_size': 32,
              'use_wandb': False,
              'discount_factor': 0.99,
              'height': 84,
              'width': 84,
              'contrast_lr': 1e-3,
              'temperature': 0.1,
              'frames_to_skip': 4,
              'stacked_frames': 4,
              'is_pretrain': True,
              'steps_before_repr_learning': 1600,  # Paper value
              'prefix': 'PRETRAIN_REPR'
              }

    if config['is_pretrain']:
        config.update({
            'entropy_coeff': 0.001,
            'batch_size': 512,
            'target_batch_size': 512,
            'replay_buffer_size': 10000
        })

    config.update({
        'rollout_length': max(config['target_batch_size'], 256)
    })

    SEED = 1337
    NUM_TIMESTEPS = 250_000_000
    act_dim = 18

    environment.seed_everything(SEED)
    env = environment.create_env(config)
    agent = Agent(env, action_dim=act_dim, config=config, load=False,
                  load_new_config=False)

    run_timesteps(agent, NUM_TIMESTEPS, is_pretrain=config['is_pretrain'])
