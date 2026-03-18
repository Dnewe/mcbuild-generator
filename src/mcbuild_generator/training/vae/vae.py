import torch
import torch.nn as nn
import torch.nn.functional as F


class VAE(nn.Module):
    def __init__(self, block_count, pad_value, embed_dim=64, latent_channels=16) -> None:
        super().__init__()

        self.pad_value = pad_value
        self.embedding = nn.Embedding(block_count, embed_dim)

        self.encoder = nn.Sequential(
            nn.Conv3d(embed_dim, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv3d(32, 64, 3, 2, 1),
            nn.ReLU(),
            nn.Conv3d(64, 128, 3, 2, 1),
            nn.ReLU(),
        )

        self.conv_mu = nn.Conv3d(128, latent_channels, 1)
        self.conv_logvar = nn.Conv3d(128, latent_channels, 1)

        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(latent_channels, 128, 3, 1, 1),
            nn.ReLU(),
            nn.ConvTranspose3d(128, 64, 3, 2, 1, 1),
            nn.ReLU(),
            nn.ConvTranspose3d(64, 32, 3, 2, 1, 1),
            nn.ReLU(),
            nn.ConvTranspose3d(32, block_count, 3, 1, 1),
        )

    def pad_to_multiple(self, x, multiple=4):
        d, h, w = x.shape[2:]
        pd = (multiple - d % multiple) % multiple
        ph = (multiple - h % multiple) % multiple
        pw = (multiple - w % multiple) % multiple
        # pad last 3 dims: (w_before, w_after, h_before, h_after, d_before, d_after)
        x = F.pad(x, (0, pw, 0, ph, 0, pd), value=self.pad_value)
        return x, (d, h, w)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode(self, x):
        h = self.encoder(x)
        return self.conv_mu(h), self.conv_logvar(h)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x_list):
        results = []
        mu_list, logvar_list = [], []

        for x in x_list:
            x = self.embedding(x).permute(0, 4, 1, 2, 3)
            x, original_shape = self.pad_to_multiple(x)
            mu, logvar = self.encode(x)
            z = self.reparameterize(mu, logvar)
            out = self.decode(z)
            w, l, h = original_shape
            out = out[:, :, :w, :l, :h]
            results.append(out)
            mu_list.append(mu)
            logvar_list.append(logvar)

        return results, mu_list, logvar_list
