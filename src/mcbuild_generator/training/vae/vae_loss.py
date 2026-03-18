import torch.nn.functional as F
import torch


def vae_loss(outputs, targets, mu_list, logvar_list, kl_weight=1e-4):
    ce_loss = sum(
        F.cross_entropy(out, tgt) for out, tgt in zip(outputs, targets)
    ) / len(outputs)

    kl_loss = sum(
        -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        for mu, logvar in zip(mu_list, logvar_list)
    ) / len(mu_list)

    return ce_loss + kl_weight * kl_loss, ce_loss, kl_loss
