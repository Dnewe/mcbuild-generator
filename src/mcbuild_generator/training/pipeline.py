import torch

from mcbuild_generator.training.vae.train import train
from mcbuild_generator.training.vae.vae import get_model
from mcbuild_generator.training.vae.vae_loss import get_vaeloss
from mcbuild_generator.training.dataset import get_loaders
from mcbuild_generator.utils.args import get_config
from mcbuild_generator.utils.plots import plot_losses
from mcbuild_generator.utils.fs_io import read_json
from mcbuild_generator.constants.paths import (
    LOSSES_PLOT_FP,
    TRAIN_PROCESSED_BUILDS_DIR,
    BLOCK_TO_IDX_JSON,
    IDX_TO_BLOCK_JSON,
    MODEL_FP,
)


def pipeline_training(config):
    """
    Model training pipeline
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    block_to_idx = dict(read_json(BLOCK_TO_IDX_JSON))
    idx_to_block = dict(read_json(IDX_TO_BLOCK_JSON))

    # Dataloader
    train_loader, val_loader = get_loaders(
        TRAIN_PROCESSED_BUILDS_DIR, **config["dataset"]
    )

    # Model
    model = get_model(block_to_idx, idx_to_block, **config["model"], device=device)

    # Criterion
    criterion = get_vaeloss(block_to_idx, idx_to_block, **config["loss"])
    criterion.to(device)

    # training
    print("\nTraining...")
    train_losses, val_losses = train(
        model,
        criterion,
        train_loader,
        val_loader,
        MODEL_FP,
        **config["train"],
        device=device,
    )

    # save losses plot
    plot_losses(train_losses, val_losses, LOSSES_PLOT_FP)


if __name__ == "__main__":
    # get args
    config = get_config()

    # run
    pipeline_training(config["training"])
