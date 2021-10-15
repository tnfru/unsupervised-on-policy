# Unsupervised On-Policy Reinforcement Learning

This work combines [Active Pre-Training](https://arxiv.org/abs/2103.04551) with an On-Policy alogirthm, [Phasic Policy Gradient](https://arxiv.org/abs/2009.04416).

## Active Pre-Training

Is used to pre-train a model free algorithm before defining a downstream task. It calculates the reward based on an estimatie of the particle based entropy of states.

## Phasic Policy Gradient

Improved Version of PPO which uses auxiliary epochs to train shared representations between the policy and a value network.
