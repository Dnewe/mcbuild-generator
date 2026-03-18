import torch.optim as optim
import torch.nn as nn
import torch
from tqdm import tqdm

from mcbuild_generator.training.vae.vae import VAE
from mcbuild_generator.training.vae.vae_loss import vae_loss


def train(model:VAE, train_loader, val_loader, epochs, lr, kl_weight, device):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_losses = []
    val_losses = []
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        tqdm_loader = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}")

        for x_list in tqdm_loader:
            x_list = [x.to(device) for x in x_list]
            recon_x_list, mu_list, logvar_list = model(x_list)

            loss, ce_loss, kl_loss = vae_loss(recon_x_list, x_list, mu_list, logvar_list, kl_weight)

            optimizer.zero_grad()
            loss.backward()  # type: ignore
            optimizer.step()

            total_loss += loss.item() # type: ignore
            tqdm_loader.set_postfix({'loss':loss.item(), 'ce_loss': ce_loss.item(), 'kl_loss': kl_loss.item()}) # type: ignore

        avg_train_loss = total_loss / len(train_loader.dataset)
        avg_val_loss = evaluate_model(model, val_loader, kl_weight, device)

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)

    return train_losses, val_losses


def evaluate_model(model, loader, kl_weight, device):
    model.eval()
    total_loss = 0
    loss_fn = nn.CrossEntropyLoss()
    with torch.no_grad():  # No need to compute gradients
        for x in tqdm(loader, desc="Evaluating"):
            x_list = [x.to(device) for x in x_list]
            recon_x_list, mu_list, logvar_list = model(x_list)

            loss, ce_loss, kl_loss = vae_loss(recon_x_list, x_list, mu_list, logvar_list, kl_weight)
            total_loss += loss.item() # type: ignore

    avg_loss = total_loss / len(loader.dataset)
    return avg_loss
