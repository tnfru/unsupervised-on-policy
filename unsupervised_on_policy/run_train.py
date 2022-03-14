from ppg.agent import Agent
from unsupervised_on_policy.rollout import run_timesteps
import pretrain.environment as environment
from utils.parser import parse_args
import sys


def main():
    args = sys.argv[1:]
    args = parse_args(args)

    config = {'policy_clip': 0.25,
              'kl_max': None,
              'kl_max_aux': None,
              'clip_reward': True,
              'beta': 1,
              'val_coeff': 1e-2,
              'train_iterations': 1,
              'entropy_coeff': 0.01,
              'entropy_min': 0.01,
              'entropy_decay': 0.9999,
              'grad_norm': 10,
              'grad_norm_ppg': 0.5,
              'critic_lr': 1e-3,
              'actor_lr': 3e-4,
              'aux_freq': 32,
              'aux_iterations': 3,
              'gae_lambda': 0.95,
              'batch_size': 32,
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
              'steps_before_repr_learning': 1600,
              'replay_buffer_size': 10000,
              'is_pretrain': False if args.skip_pretrain else True,
              'num_envs': 16,
              'prefix': args.prefix,
              'path': args.model_path
              }

    if config['is_pretrain']:
        config.update({
            'batch_size': 512,
            'target_batch_size': 512,
            'entropy_min': 0.01,
            'actor_lr': 1e-4,
            'kl_max_aux': 0.01,  # stability in pretrain
        })

    config.update({
        'rollout_length': max(config['target_batch_size'], 256)
    })

    SEED = 1337
    NUM_TIMESTEPS = 250_000_000

    environment.seed_everything(SEED)
    env = environment.create_env(config)
    agent = Agent(env, config=config, load=args.load)

    run_timesteps(agent, NUM_TIMESTEPS, pretrain=config['is_pretrain'])


if __name__ == '__main__':
    main()
