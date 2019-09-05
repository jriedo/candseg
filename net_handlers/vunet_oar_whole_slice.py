import time
import torch
import numpy as np
import os
import pickle

from net_handlers.nets.vunet_huge import VUNet
from net_handlers.nets.conv_vae import VAE
from utils.dice_coefficient import Dice

class Oar():
    """
        Class for training and testing the VU-Net on whole OAR slices
        The whole network is trained in two steps:
         i) the U-Net with early stopping for regularization
         ii) the VAE with frozen U-Net weights
    """

    def __init__(self, path, run_name, ls=32):
        # define device and create output path
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._fullpath = os.path.join(path, run_name)
        self._path = path
        self._first = True
        self._ls = ls
        if not os.path.isdir(self._fullpath):
            os.mkdir(self._fullpath)

        # define the u-net and vae with corresponding optimizers
        self.unet = VUNet().to(device=self._device)
        self.vae = VAE(latent_size=ls).to(device=self._device)
        self.unet_optimizer = torch.optim.Adam(self.unet.parameters())
        self.vae_optimizer = torch.optim.Adam(self.vae.parameters())

        self.bceloss = torch.nn.BCELoss()
        self.dice = Dice()

    def train_unet(self, epochs):
        print('start training unet')

        # load training set
        trainData = pickle.load(open(os.path.join('data', '{}.pkl'.format('oar_data_train_fs_33')), "rb"))
        st = time.time()
        for epoch in range(epochs):  # epochs loop
            # on train set
            loss_epoch = 0
            self.unet.train() # important, as validation sets it to "eval"
            for i, batch in enumerate(trainData):
                # prepare training batch
                segms = batch['segms'].to(device=self._device)
                imgs = batch['imgs'].requires_grad_().to(device=self._device)
                self.unet_optimizer.zero_grad()

                output = self.unet(imgs)
                loss = self.bceloss(output, segms)
                loss_epoch += loss.item()
                loss.backward()
                self.unet_optimizer.step()

            print('Epoch [{0:d} /{1:d}], train loss: {2:.3f}'.format((epoch + 1), epochs, loss_epoch / (i + 1)))

            if (epoch + 1) % 10 == 0:
                self._validate_unet()
                tm = (time.time() - st) / float(epoch + 1)
                print('Training took {0:4.1f}s, {1:4.1f}s per epoch, {2:4.1f}s to go'.format(time.time() - st, tm,
                                                                                             (epochs - 1 - epoch) * tm))

    def train_vae(self, epochs):
        print('start training vae')
        self.unet.eval()

        # freeze base unet's weights
        for param in self.unet.parameters():
            param.requires_grad = False

        # load training set
        trainData = pickle.load(open(os.path.join('data', '{}.pkl'.format('oar_data_train_fs_33')), "rb"))

        st = time.time()
        for epoch in range(epochs):  # epochs loop
            # on train set
            self.vae.train() # important, as validation sets it to "eval"
            loss_epoch = 0

            for i, batch in enumerate(trainData):  # batches for training
                # prepare training batch
                imgs = batch['imgs'].to(device=self._device)

                self.vae_optimizer.zero_grad()

                feats = self.unet.get_feature_map(imgs)
                out, mu, logvar = self.vae(feats)

                loss = self._vae_loss(out, feats, mu, logvar)
                loss_epoch += loss.item()
                loss.backward()

                self.vae_optimizer.step()

            print('Epoch [{0:d} /{1:d}], train loss: {2:.3f}'.format((epoch + 1), epochs, loss_epoch / (i + 1)))

            if (epoch + 1) % 10 == 0:
                self._validate_vae()
                tm = (time.time() - st) / float(epoch + 1)
                print('Training took {0:4.1f}s, {1:4.1f}s per epoch, {2:4.1f}s to go'.format(time.time() - st, tm,
                                                                                             (epochs - 1 - epoch) * tm))

    @torch.no_grad()
    def _validate_unet(self):
        self.unet.eval()
        valData = pickle.load(open(os.path.join('data', '{}.pkl'.format('oar_data_val_fs_33')), "rb"))
        loss_epoch = 0
        dice_epoch = 0
        for i, batch in enumerate(valData):  # batches for validation
            segms = batch['segms'].to(device=self._device)
            imgs = batch['imgs'].to(device=self._device)
            output = self.unet(imgs)
            loss = self.bceloss(output, segms)
            loss_epoch += loss.item()
            output[output < 0.5] = 0  # background
            output[output >= 0.5] = 1  # foreground
            dice_epoch += np.mean(self.dice(output, segms))

        print('Validation loss: {0:.3f}, dice: {1:.3f}'.format(loss_epoch / (i + 1), dice_epoch / (i + 1)))


    @torch.no_grad()
    def _validate_vae(self):
        self.unet.eval()
        self.vae.eval()
        valData = pickle.load(open(os.path.join('data', '{}.pkl'.format('oar_data_val_fs_33')), "rb"))
        loss_epoch = 0
        dice_epoch = 0
        for i, batch in enumerate(valData):  # batches for validation
            segms = batch['segms'].to(device=self._device)
            imgs = batch['imgs'].to(device=self._device)

            enc_unet = self.unet.get_encoded(imgs)
            feats, mu, logvar = self.vae(enc_unet[0])
            output = self.unet.decode(*enc_unet, feats)
            loss = self._vae_loss(feats, enc_unet[0], mu, logvar)
            loss_epoch += loss.item()
            output[output < 0.5] = 0  # background
            output[output >= 0.5] = 1  # foreground
            dice_epoch += np.mean(self.dice(output, segms))

        print('Validation loss: {0:.3f}, dice: {1:.3f}'.format(loss_epoch / (i + 1), dice_epoch / (i + 1)))


    @torch.no_grad()
    def test_unet(self):
        print('start testing')
        self.unet.eval()
        testData = pickle.load(open(os.path.join('data', '{}.pkl'.format('oar_data_test_fs_33')), "rb"))
        loss_tot = 0
        dice_epoch = 0
        for i, batch in enumerate(testData):  # batches for testing
            imgs = batch['imgs'].to(device=self._device)
            segms = batch['segms'].to(device=self._device)

            output = self.unet(imgs)

            loss = self.bceloss(output, segms)
            loss_tot += loss.item()
            output[output < 0.5] = 0  # background
            output[output >= 0.5] = 1  # foreground
            dice_epoch += np.mean(self.dice(output, segms))
        print('Test loss: {0:.3f}, dice: {1:.3f}'.format(loss_tot / (i + 1), dice_epoch / (i + 1)))
        torch.save(self.unet.state_dict(), os.path.join(self._fullpath, 'vunet_unet_fs_weights.pth'))

    @torch.no_grad()
    def test_vae(self):
        self.unet.eval()
        self.vae.eval()
        valData = pickle.load(open(os.path.join('data', '{}.pkl'.format('oar_data_test_fs_33')), "rb"))
        loss_epoch = 0
        dice_epoch = 0
        for i, batch in enumerate(valData):  # batches for validation
            segms = batch['segms'].to(device=self._device)
            imgs = batch['imgs'].to(device=self._device)

            enc_unet = self.unet.get_encoded(imgs)
            feats, mu, logvar = self.vae(enc_unet[0])
            output = self.unet.decode(*enc_unet, feats)

            loss = self._vae_loss(feats, enc_unet[0], mu, logvar)
            loss_epoch += loss.item()
            output[output < 0.5] = 0  # background
            output[output >= 0.5] = 1  # foreground
            dice_epoch += np.mean(self.dice(output, segms))
        print('Test loss: {0:.3f}, dice: {1:.3f}'.format(loss_epoch / (i + 1), dice_epoch / (i + 1)))
        torch.save(self.vae.state_dict(), os.path.join(self._fullpath, 'vunet_vae_fs_weights.pth'))

    @staticmethod
    def _vae_loss(out, x, mu, logvar):
        criterion = torch.nn.MSELoss(reduction='sum')
        MSE = criterion(out, x)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return MSE + KLD
