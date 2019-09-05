import torch
import torch.nn as nn
import torch.nn.functional as F

"""
    Variational Autoencoder for prior knowledge generation
    called encoder because only the encoded space is of interest, the recreation is just for stabilization
"""

class VAEncoder(nn.Module):
    def __init__(self, latent_size=2, pm='border'):
        super(VAEncoder, self).__init__()
        self._in_size_flatten = 64 * 64
        self._mid_size = 1024
        self._latent_size = latent_size

        # encoder
        self.conv_enc = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=1, padding_mode=pm)
        self.batch_norm = torch.nn.BatchNorm2d(num_features=1, eps=1e-05, momentum=0.1, affine=True)
        self.linear_enc = nn.Linear(in_features=self._in_size_flatten, out_features=self._mid_size)
        self.linear_enc_mu = nn.Linear(in_features=self._mid_size, out_features=self._latent_size)
        self.linear_enc_sigm = nn.Linear(in_features=self._mid_size, out_features=self._latent_size)

        # decoder
        self.linear_dec1 = nn.Linear(in_features=self._latent_size, out_features=self._mid_size)
        self.linear_dec2 = nn.Linear(in_features=self._mid_size, out_features=self._in_size_flatten)
        self.conv_dec = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=1, padding_mode=pm)

    def encode(self, x):
        out = self.batch_norm(self.conv_enc(x))
        out = F.relu(out).view(-1, self._in_size_flatten)
        out = F.relu(self.linear_enc(out))
        return self.linear_enc_mu(out), torch.clamp(self.linear_enc_sigm(out), min=0)

    def decode(self, mu, logvar, shape=(63, 1, 64, 64)):
        # used for decoding, no sampling is intended for this version of the vae
        if self.training:
            std = torch.exp(logvar * 0.5)
            eps = torch.randn_like(std)
            z = mu + eps * std
        else:
            z = mu
        z = F.relu(self.linear_dec2(F.relu(self.linear_dec1(z))))
        z = self.conv_dec(z.view(shape[0], -1, shape[2], shape[3]))
        return torch.sigmoid(z)

    def forward(self, x):
        # the "normal" use of this network is to get the latent space, therefore "forward" does not implement decoding
        mu, logvar = self.encode(x)
        return mu

