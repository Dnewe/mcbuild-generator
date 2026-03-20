import torch
import torch.nn as nn
import torch.nn.functional as F

from mcbuild_generator.utils.fs_io import read_json
from mcbuild_generator.constants.paths import BLOCK_TO_IDX_JSON, IDX_TO_BLOCK_JSON


class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(channels, channels, 3, 1, 1),
            nn.LeakyReLU(0.1),
            nn.Conv3d(channels, channels, 3, 1, 1),
        )
        self.act = nn.LeakyReLU(0.1)

    def forward(self, x):
        return self.act(x + self.block(x))


class VAE(nn.Module):
    def __init__(
        self, block_count, pad_value, embed_dim=64, latent_channels=64
    ) -> None:
        super().__init__()

        self.pad_value = pad_value
        self.latent_channels = latent_channels
        self.multiple = 8 # given by dimension reduction factor caused by downsampling

        self.embedding = nn.Embedding(block_count, embed_dim)

        self.encoder = nn.Sequential(
            nn.Conv3d(embed_dim, 32, 3, 2, 1),
            nn.LeakyReLU(0.1),
            ResBlock(32),
            nn.Conv3d(32, 64, 3, 2, 1),
            nn.LeakyReLU(0.1),
            nn.Conv3d(64, 128, 3, 2, 1),
            nn.LeakyReLU(0.1),
            ResBlock(128),
            ResBlock(128)
        )
        self.encoder_norm = nn.GroupNorm(8, 128)

        self.conv_mu = nn.Conv3d(128, self.latent_channels, 1)
        self.conv_logvar = nn.Conv3d(128, self.latent_channels, 1)

        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(self.latent_channels, 128, 3, 2, 1, 1),
            nn.LeakyReLU(0.1),
            ResBlock(128),
            nn.ConvTranspose3d(128, 64, 3, 2, 1, 1),
            nn.LeakyReLU(0.1),
            nn.ConvTranspose3d(64, 32, 3, 2, 1, 1),
            nn.LeakyReLU(0.1),
            ResBlock(32),
            nn.LeakyReLU(0.1),
            nn.ConvTranspose3d(32, block_count, 1),
        )

    def pad_to_multiple(self, x):
        d, h, w = x.shape[2:]
        pd = (self.multiple - d % self.multiple) % self.multiple
        ph = (self.multiple - h % self.multiple) % self.multiple
        pw = (self.multiple - w % self.multiple) % self.multiple
        # pad last 3 dims: (w_before, w_after, h_before, h_after, d_before, d_after)
        x = F.pad(x, (0, pw, 0, ph, 0, pd), value=self.pad_value)
        return x, (d, h, w)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode(self, x):
        h = self.encoder(x)
        h = self.encoder_norm(h)
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
    
    @torch.inference_mode()
    def reconstruct(self, x, device=torch.device("cpu")) -> torch.Tensor:
        x = x.to(device)
        x = self.embedding(x).permute(0, 4, 1, 2, 3)
        x, original_shape = self.pad_to_multiple(x)

        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        out = self.decode(z)

        w, l, h = original_shape
        out = out[:, :, :w, :l, :h]
        block_ids = out.argmax(dim=1)

        return block_ids
    
    @torch.inference_mode()
    def generate(self, shape=(16,16,16), device=torch.device('cpu')) -> torch.Tensor:
        latent_h = shape[0] // self.multiple
        latent_l = shape[1] // self.multiple
        latent_w = shape[2] // self.multiple

        z = torch.randn(1, self.latent_channels, latent_h, latent_l, latent_w).to(device)

        out = self.decode(z)
        block_ids = out.argmax(dim=1)

        return block_ids


def get_model(embed_dim, latent_channels, use_pretrained, pretrained_fp, device):
    idx_to_block = dict(read_json(IDX_TO_BLOCK_JSON))
    block_to_idx = dict(read_json(BLOCK_TO_IDX_JSON))
    block_count = len(idx_to_block)
    air_index = block_to_idx["minecraft:air"]

    vae = VAE(block_count, air_index, embed_dim, latent_channels).to(device)
    if use_pretrained:
        vae.load_state_dict(torch.load(pretrained_fp, map_location=device))

    return vae