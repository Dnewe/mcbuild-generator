import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict,List

sns.set_theme()


def plot_losses(train_losses: Dict[str,List], val_losses: Dict[str,List], save_fp: str, show=True):
    """
    Plot the training and validation loss across epochs.

    - train_losses: List of training losses per epoch.
    - val_losses: List of validation losses per epoch.
    """
    plt.figure(figsize=(8, 5))
    for k,v in train_losses.items():
        plt.plot(v, label=f"Train {k}", marker="o")
    for k,v in val_losses.items():
        plt.plot(v, label=f"Val {k}", marker="o")

    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()
    plt.grid()
    plt.savefig(save_fp)
    if show:
        plt.show()
