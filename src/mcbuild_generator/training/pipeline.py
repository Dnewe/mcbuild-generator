import os
from torch.utils.data import DataLoader, random_split
import torch

from mcbuild_generator.training.vae.train import train
from mcbuild_generator.training.vae.vae import VAE
from mcbuild_generator.training.dataset import get_loaders
from mcbuild_generator.utils.args import get_config
from mcbuild_generator.utils.fs_io import read_json
from mcbuild_generator.constants.paths import PROCESSED_BUILDS_DIR, IDX_TO_BLOCK_JSON


def pipeline_training(config):
    """
    Model training pipeline
    """
    # Dataloader
    train_loader, val_loader = get_loaders(**config['dataset'])

    # Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    block_count = len(read_json(IDX_TO_BLOCK_JSON))
    vae = VAE(block_count, **config["model"]).to(device)

    print("\nTraining...")
    train_losses, val_losses = train(vae, train_loader, val_loader, **config["train"], device=device)


if __name__ == "__main__":
    # get args
    config = get_config()

    # run
    pipeline_training(config["training"])
