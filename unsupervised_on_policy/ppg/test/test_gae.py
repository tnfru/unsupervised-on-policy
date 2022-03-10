import torch as T
from unittest import TestCase

from ppg.gae import calculate_advantages


class Test(TestCase):
    def test_calculate_advantages_simplest(self):
        rewards = [0, 0, 0]
        state_vals = [0, 0, 10]
        dones = [False, False, True]

        discount_factor = 1
        gae_lambda = 1

        adv = calculate_advantages(rewards, state_vals, dones,
                                   discount_factor, gae_lambda)
        result = T.tensor([0, 0, -10])

        assert (adv == result).all()

    def test_calculate_advantages_simplest_with_lambda(self):
        rewards = [0, 0, 0]
        state_vals = [0, 0, 10]
        dones = [False, False, True]

        discount_factor = 1
        gae_lambda = 0.5

        adv = calculate_advantages(rewards, state_vals, dones,
                                   discount_factor, gae_lambda)

        result = T.tensor([2.5, 5, -10])

        assert (result == adv).all()

    def test_calculate_advantages_simplest_with_lamba_discount(self):
        rewards = [0, 0, 1]
        state_vals = [0, 0, 2]
        dones = [False, False, True]

        discount_factor = 0.5
        gae_lambda = 0.5

        adv = calculate_advantages(rewards, state_vals, dones,
                                   discount_factor, gae_lambda)

        result = T.tensor([0.1875, 0.75, -1])

        assert (result == adv).all()

    def test_calculate_advantages(self):
        rewards = [0, 0, -2]
        state_vals = [0, 2, 5]
        dones = [False, False, True]

        discount_factor = 1
        gae_lambda = 1

        adv = calculate_advantages(rewards, state_vals, dones,
                                   discount_factor, gae_lambda)

        result = T.tensor([-2, -4, -7])
        assert (result == adv).all()
