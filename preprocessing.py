import torch as T


def normalize(x):
    if T.isnan(x.std()):
        return x - x.mean(0)

    return (x - x.mean(0)) / (x.std(0) + 1e-8)


def approx_kl_div(log_probs, old_log_probs, ratio=None):
    with T.no_grad():
        if ratio is None:
            ratio = T.exp(log_probs - old_log_probs)

        log_ratio = log_probs - old_log_probs
        kl_div = ratio - 1 - log_ratio

        return kl_div.mean()
