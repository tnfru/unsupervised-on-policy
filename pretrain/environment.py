import ale_py  # necessary for gym enviornment creation
import gym
import torch as T
import numpy as np
import einops
from supersuit import frame_stack_v1, resize_v0, clip_reward_v0
from contrastive_learning import ContrastiveLearner
from data_augmentation import DataAugment
from reward import ParticleReward

FRAMES_TO_STACK = 4
FRAMES_TO_SKIP = 4
PAD_SIZE = 4

# TODO Add seed
# TODO Terminal on loss of life

X_DIM = 84
Y_DIM = 84

rng = np.random.default_rng()


def preprocess(img):
    img = T.from_numpy(img) / 255

    if len(img.shape) == 3:  # if no fourth dim, batch size is missing
        img = einops.rearrange(img, 'h w c -> c h w')

    else:
        img = einops.rearrange(img, 'b h w c -> b c h w')
    return img


def create_env(name='MsPacman', render=None):
    env = gym.make('ALE/' + name + '-v5',
                   obs_type='grayscale',  # ram | rgb | grayscale
                   frameskip=FRAMES_TO_SKIP,  # frame skip
                   mode=0,  # game mode, see Machado et al. 2018
                   difficulty=0,  # game difficulty, see Machado et al. 2018
                   repeat_action_probability=0.25,  # Sticky action probability
                   full_action_space=True,  # Use all actions
                   render_mode=render  # None | human | rgb_array
                   )

    env = clip_reward_v0(env, lower_bound=-1, upper_bound=1)
    env = resize_v0(env, X_DIM, Y_DIM, linear_interp=True)
    env = frame_stack_v1(env, FRAMES_TO_STACK)

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


if __name__ == '__main__':
    pacman = create_env()
    reward_function = ParticleReward()
    dm = DataAugment(X_DIM, Y_DIM, PAD_SIZE, rng)
    cl = ContrastiveLearner(FRAMES_TO_STACK)

    trajectory = run_episode(pacman)
    trajectory = prep_states(trajectory, dm, cl, cutoff=True)

    rewards = reward_function.calculate_reward(trajectory)
    print(rewards)
