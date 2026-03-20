import torch.nn.functional as F
import torch

from mcbuild_generator.utils.fs_io import read_json
from mcbuild_generator.constants.paths import BLOCK_TO_IDX_JSON, IDX_TO_BLOCK_JSON


class VAELoss(torch.nn.Module):
    def __init__(
        self,
        block_count,
        air_index,
        air_weight=0.05,
        kl_start=0.0,
        kl_end=1.0,
        kl_anneal_steps=5000,
    ) -> None:
        super().__init__()
        ce_weights = torch.ones(block_count)
        ce_weights[air_index] = air_weight
        self.register_buffer("ce_weights", ce_weights)

        self.kl_start = kl_start
        self.kl_end = kl_end
        self.kl_anneal_steps = kl_anneal_steps
        self.global_step = 0
        self.kl_weight = kl_start

    def step(self):
        """Update KL weight for annealing"""
        self.global_step += 1
        progress = min(1.0, self.global_step / self.kl_anneal_steps)
        self.kl_weight = self.kl_start + progress * (self.kl_end - self.kl_start)

    def loss_ce(self, outputs, targets):
        return sum(
            F.cross_entropy(out, tgt, weight=self.ce_weights)  # type: ignore
            for out, tgt in zip(outputs, targets)
        ) / len(outputs)

    def loss_kl(self, mu_list, logvar_list):
        total = 0
        for mu, logvar in zip(mu_list, logvar_list):
            mu = mu.float()  # force float32
            logvar = logvar.float()  # force float32
            kl = torch.clamp(logvar, min=-10, max=10)  # to prevent gradient explosion
            kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
            total += torch.mean(kl)
        return total / len(mu_list)

    def forward(self, outputs, targets, mu_list, logvar_list):
        ce_loss = self.loss_ce(outputs, targets)
        kl_loss = self.loss_kl(mu_list, logvar_list)
        return ce_loss + self.kl_weight * kl_loss, ce_loss, kl_loss


def get_vaeloss(air_weight, kl_start, kl_end, kl_anneal_step):
    """
    Get VAELoss object by retrieving parameters values in JSON files.
    """
    idx_to_block = dict(read_json(IDX_TO_BLOCK_JSON))
    block_to_idx = dict(read_json(BLOCK_TO_IDX_JSON))

    block_count = len(idx_to_block)
    air_index = block_to_idx["minecraft:air"]

    return VAELoss(block_count, air_index, air_weight, kl_start, kl_end, kl_anneal_step)
