import torch.optim as optim
import torch
from tqdm import tqdm

from mcbuild_generator.training.vae.vae_loss import VAELoss
from mcbuild_generator.training.vae.vae import VAE
from mcbuild_generator.constants.paths import MODEL_FP


def train(model: VAE, criterion: VAELoss, train_loader, val_loader, epochs, lr, device):
    model.to(device)
    use_amp = torch.cuda.is_available()
    scaler = torch.GradScaler(enabled=use_amp)

    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_losses = []
    val_losses = []
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        tqdm_loader = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}")

        for x_list in tqdm_loader:
            optimizer.zero_grad(set_to_none=True)
            x_list = [x.to(device) for x in x_list]

            with torch.autocast("cuda", enabled=use_amp):
                recon_x_list, mu_list, logvar_list = model(x_list)
                loss, ce_loss, kl_loss = criterion(
                    recon_x_list, x_list, mu_list, logvar_list
                )

            scaler.scale(loss).backward()  # type: ignore
            scaler.step(optimizer)
            scaler.update()

            criterion.step()

            total_loss += loss.item()  # type: ignore
            tqdm_loader.set_postfix(
                {
                    "loss": loss.item(),  # type: ignore
                    "ce_loss": ce_loss.item(),  # type: ignore
                    "kl_loss": kl_loss.item(),  # type: ignore
                }
            )

        avg_train_loss = total_loss / len(train_loader.dataset)
        avg_val_loss = evaluate_model(model, criterion, val_loader, device, use_amp)

        print(
            f"[{epoch + 1}/{epochs}]: train_loss= {avg_train_loss:.3f} | val_loss= {avg_val_loss:.3f}"
        )

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)

        torch.save(model.state_dict(), MODEL_FP.replace("model", "vae"))

    return train_losses, val_losses


def evaluate_model(model: VAE, criterion: VAELoss, loader, device, use_amp):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for x_list in tqdm(loader, desc="Evaluating"):
            x_list = [x.to(device) for x in x_list]

            with torch.autocast("cuda", enabled=use_amp):
                recon_x_list, mu_list, logvar_list = model(x_list)
                loss, _, _ = criterion(recon_x_list, x_list, mu_list, logvar_list)

            total_loss += loss.item()  # type: ignore

    avg_loss = total_loss / len(loader.dataset)
    return avg_loss
