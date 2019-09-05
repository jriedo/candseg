import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE(nn.Module):
    """
    Variational Autoencoder for whole-slice VU-Net
    """
    def __init__(self, latent_size=2, in_size=4096, mid_size=1024, pm='border'):
        super(VAE, self).__init__()

        self._in_size_flatten = in_size * 4
        self._mid_size = mid_size
        self._latent_size = latent_size

        # pooling is done by bilinear interpolation
        self.pool = lambda input: F.interpolate(input, scale_factor=0.5, mode='bilinear', align_corners=True)

        # activation function
        self.activation = nn.ReLU()

        # upsampling instead of deconvolution
        self.up = lambda input: F.interpolate(input, scale_factor=2, mode='bilinear', align_corners=True)

        # encoding path with 3 conv layers and 2 linear (split to mu and sigma in last layer)
        self.conv_enc1 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1, padding_mode=pm)
        self.conv_enc2 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1, padding_mode=pm)
        self.conv_enc3 = nn.Conv2d(in_channels=256, out_channels=4, kernel_size=3, padding=1, padding_mode=pm)
        self.linear_enc = nn.Linear(in_features=self._in_size_flatten, out_features=self._mid_size)
        self.linear_enc_mu = nn.Linear(in_features=self._mid_size, out_features=self._latent_size)
        self.linear_enc_sigm = nn.Linear(in_features=self._mid_size, out_features=self._latent_size)

        # decoding path with 2 linear and 2 conv layers
        self.linear_dec1 = nn.Linear(in_features=self._latent_size, out_features=self._mid_size)
        self.linear_dec2 = nn.Linear(in_features=self._mid_size, out_features=self._in_size_flatten)
        self.conv_dec1 = nn.Conv2d(in_channels=4, out_channels=32, kernel_size=3, padding=1, padding_mode=pm)
        self.conv_dec2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1, padding_mode=pm)

    def encode(self, x):
        # using above layers for encoding mu and logvar
        out = self.pool(self.activation(self.conv_enc1(x)))
        out = self.pool(self.activation(self.conv_enc2(out)))
        out = self.activation(self.conv_enc3(out)).view(-1, self._in_size_flatten)
        out = self.activation(self.linear_enc(out))
        return self.linear_enc_mu(out), torch.clamp(self.linear_enc_sigm(out), min=0)

    def reparameterize(self, mu, logvar):
        # implementation of the reparametrization trick for backpropagation during training
        if self.training: # defined with the net.eval() net.train() commands from nn.module
            std = torch.exp(logvar * 0.5)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu

    def decode(self, z, shape=(64, 64, 64, 64)):
        # used for sampling, z are the latent variables
        z = self.activation(self.linear_dec2(self.activation(self.linear_dec1(z))))
        z = z.view(shape[0], -1, shape[2], shape[3])
        z = self.activation(self.conv_dec1(self.up(z)))
        z = self.conv_dec2(self.up(z))
        return torch.sigmoid(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, shape=(x.shape[0], 64, 64, 64)), mu, logvar
