import ale_py  # necessary for gym enviornment creation
import gym
import torch as T
import numpy as np
import einops
import random
from supersuit import frame_stack_v1, resize_v0, clip_reward_v0
from pretrain.contrastive_learning import ContrastiveLearner
from pretrain.data_augmentation import DataAugment
from pretrain.reward import ParticleReward


def preprocess(img):
    img = T.from_numpy(img) / 255
    # TODO do this after transformation
    # https://theaisummer.com/self-supervised-representation-learning-computer-vision/
    # TODO norm to running mean of each channel

    if len(img.shape) == 3:  # if no fourth dim, batch size is missing
        img = einops.rearrange(img, 'h w c -> c h w')

    else:
        img = einops.rearrange(img, 'b h w c -> b c h w')
    return img


def create_env(x_dim, y_dim, name='MsPacman', frames_to_skip=4,
               frames_to_stack=4, render=None):
    env = gym.make('ALE/' + name + '-v5',
                   obs_type='grayscale',  # ram | rgb | grayscale
                   frameskip=frames_to_skip,  # frame skip
                   mode=0,  # game mode, see Machado et al. 2018
                   difficulty=0,  # game difficulty, see Machado et al. 2018
                   repeat_action_probability=0.25,  # Sticky action probability
                   full_action_space=True,  # Use all actions
                   render_mode=render  # None | human | rgb_array
                   )

    env = clip_reward_v0(env, lower_bound=-1, upper_bound=1)
    env = resize_v0(env, x_dim, y_dim, linear_interp=True)
    env = frame_stack_v1(env, frames_to_stack)

    return env


def run_episode(env):
    done = False
    s = env.reset()
    states = []

    while not done:
        action = env.action_space.sample()
        s, r, done, _ = env.step(action)

        s = preprocess(s)
        states.append(s)

    env.close()
    return states


def prep_states(states, augment, contrast, cutoff=False):
    if cutoff:
        states = states[:256]
    states = T.stack(states)
    states = augment(states)
    states = contrast(states)

    return states


def seed_everything(seed, deterministic=False):
    T.manual_seed(seed)

    if deterministic:
        T.backends.cudnn.deterministic = True
        T.backends.cudnn.benchmark = False
    else:
        T.backends.cudnn.benchmark = True

    np.random.seed(seed)
    random.seed(seed)
    if T.cuda.is_available():
        T.cuda.manual_seed(seed)


if __name__ == '__main__':
    seed_everything(1337, deterministic=False)
    FRAMES_TO_STACK = 4
    FRAMES_TO_SKIP = 4

    # TODO Terminal on loss of life
    # TODO compare Adam with LARS optimizer vs AdamW

    X_DIM = 84
    Y_DIM = 84

    environment = create_env(X_DIM, Y_DIM, frames_to_skip=FRAMES_TO_SKIP)
    reward_function = ParticleReward()
    dm = DataAugment(X_DIM, Y_DIM)
    cl = ContrastiveLearner(FRAMES_TO_STACK)

    trajectory = run_episode(environment)
    trajectory = prep_states(trajectory, dm, cl, cutoff=True)

    rewards = reward_function.calculate_reward(trajectory)
    print(rewards)
