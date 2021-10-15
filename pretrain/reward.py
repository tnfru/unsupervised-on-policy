import torch as T


class ParticleReward:
    # TODO test original implementation
    def __init__(self, top_k=16):
        self.mean = 0
        self.samples_done = 0
        self.c = 1
        self.top_k = top_k

    def calculate_reward(self, states, normalize=True):
        particle_volumes = T.norm(states.unsqueeze(1) - states.unsqueeze(0),
                                  dim=-1)  # hypersphere volume
        top_k_rewards, _ = particle_volumes.topk(self.top_k, sorted=True,
                                                 largest=False, dim=1)

        self.update_mean_estimate(top_k_rewards.reshape(-1, 1))

        if normalize:
            # TODO test normalization
            top_k_rewards /= self.mean

        top_k_rewards = top_k_rewards.mean(dim=1)
        particle_rewards = T.log(self.c + top_k_rewards)

        return particle_rewards

    def update_mean_estimate(self, x):
        batch_size = x.size(0)
        self.samples_done += batch_size
        difference = x.mean(dim=0) - self.mean
        self.mean += difference * batch_size / self.samples_done
