import torch
import torch.nn as nn
import torch.nn.functional as F

"""
    Variational Autoencoder for patch-wise VU-Net
"""

class VAE(nn.Module):
    def __init__(self, latent_size=2, in_size=4096, mid_size=1024, pm='border'):
        super(VAE, self).__init__()
        self._in_size_flatten = in_size
        self._mid_size = mid_size
        self._latent_size = latent_size

        # encoder
        self.conv_enc = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=1, padding_mode=pm)
        self.linear_enc = nn.Linear(in_features=self._in_size_flatten, out_features=self._mid_size)
        self.linear_enc_mu = nn.Linear(in_features=self._mid_size, out_features=self._latent_size)
        self.linear_enc_sigm = nn.Linear(in_features=self._mid_size, out_features=self._latent_size)

        # decoder
        self.linear_dec1 = nn.Linear(in_features=self._latent_size, out_features=self._mid_size)
        self.linear_dec2 = nn.Linear(in_features=self._mid_size, out_features=self._in_size_flatten)
        self.conv_dec = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=1, padding_mode=pm)

    def encode(self, x):
        out = self.conv_enc(x)
        out = F.relu(out).view(-1, self._in_size_flatten)
        out = F.relu(self.linear_enc(out))
        return self.linear_enc_mu(out), torch.clamp(self.linear_enc_sigm(out), min=0)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(logvar * 0.5)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu

    def decode(self, z, shape=(64,64,64,64)):
        # used for sampling, z are the latent variables
        z = F.relu(self.linear_dec2(F.relu(self.linear_dec1(z))))
        z = self.conv_dec(z.view(shape[0], -1, shape[2], shape[3]))
        return torch.sigmoid(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, x.shape), mu, logvar

