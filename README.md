# Unsupervised On-Policy Reinforcement Learning

This work combines [Active Pre-Training](https://arxiv.org/abs/2103.04551) with an On-Policy algorithm, [Phasic Policy Gradient](https://arxiv.org/abs/2009.04416).

## Active Pre-Training

Is used to pre-train a model free algorithm before defining a downstream task. It calculates the reward based on an estimatie of the particle based entropy of states. This reduces the training time if you want to define various tasks - i.e. robots for a warehouse. 

## Phasic Policy Gradient

Improved Version of Proximal Policy Optimization, which uses auxiliary epochs to train shared representations between the policy and a value network.
