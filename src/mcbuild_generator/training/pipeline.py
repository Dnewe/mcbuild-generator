import torch

from mcbuild_generator.training.vae.train import train
from mcbuild_generator.training.vae.vae import VAE
from mcbuild_generator.training.dataset import get_loaders
from mcbuild_generator.utils.args import get_config
from mcbuild_generator.utils.fs_io import read_json
from mcbuild_generator.utils.plots import plot_losses
from mcbuild_generator.constants.paths import (
    LOSSES_PLOT_FP,
    BLOCK_TO_IDX_JSON,
    MODEL_FP,
)


def pipeline_training(config):
    """
    Model training pipeline
    """
    # Dataloader
    train_loader, val_loader = get_loaders(**config["dataset"])

    # Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    block_to_idx = dict(read_json(BLOCK_TO_IDX_JSON))
    block_count = len(block_to_idx)
    air_index = block_to_idx["minecraft:air"]

    vae = VAE(block_count, air_index, **config["model"]).to(device)
    if config["use_pretrained"]:
        vae.load_state_dict(torch.load(config["pretrained_fp"], map_location=device))

    # training
    print("\nTraining...")
    train_losses, val_losses = train(
        vae, train_loader, val_loader, **config["train"], device=device
    )

    # save model
    torch.save(vae.state_dict(), MODEL_FP.replace("model", "vae"))

    # save losses plot
    plot_losses(train_losses, val_losses, LOSSES_PLOT_FP)


if __name__ == "__main__":
    # get args
    config = get_config()

    # run
    pipeline_training(config["training"])
