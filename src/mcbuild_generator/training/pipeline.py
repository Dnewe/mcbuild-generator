import torch

from mcbuild_generator.training.vae.train import train
from mcbuild_generator.training.vae.vae import get_model
from mcbuild_generator.training.dataset import get_loaders
from mcbuild_generator.utils.args import get_config
from mcbuild_generator.utils.plots import plot_losses
from mcbuild_generator.constants.paths import (
    LOSSES_PLOT_FP,
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

    model = get_model(**config["model"], device=device)

    # training
    print("\nTraining...")
    train_losses, val_losses = train(
        model, train_loader, val_loader, **config["train"], device=device
    )

    # save model
    torch.save(model.state_dict(), MODEL_FP)

    # save losses plot
    plot_losses(train_losses, val_losses, LOSSES_PLOT_FP)


if __name__ == "__main__":
    # get args
    config = get_config()

    # run
    pipeline_training(config["training"])
