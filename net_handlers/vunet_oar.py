import time
import pickle
import torch
import numpy as np
import os

from net_handlers.nets.vunet import VUNet
from net_handlers.nets.vae import VAE
from utils.dice_coefficient import Dice


class Oar():
    """
        Class for training and testing the VU-Net on OAR patches
        The whole network is trained in two steps:
         i) the U-Net with early stopping for regularization
         ii) the VAE with frozen U-Net weights
    """

    def __init__(self, path, run_name):
        # define device and create output path
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._path = path
        self._fullpath = os.path.join(path, run_name)
        if not os.path.isdir(self._fullpath):
            os.mkdir(self._fullpath)

        # define the networks with corresponding optimizers
        self.unet = VUNet(isdropout=True).to(device=self._device)
        self.unet_optimizer = torch.optim.Adam(self.unet.parameters())
        self.vae = VAE().to(device=self._device)
        self.vae_optimizer = torch.optim.Adam(self.vae.parameters())

        self.criterion = torch.nn.BCELoss()
        self.dice = Dice()

    def train_unet(self, epochs):
        print('start training base unet')
        counter = 0
        last_loss = 1e3

        # load training set
        trainData = pickle.load(open(os.path.join('data', '{}.pkl'.format('oar_data_train_63')), "rb"))
        st = time.time()
        for epoch in range(epochs):  # epochs loop
            # on train set
            loss_epoch = 0
            self.unet.train()  # important, as validation sets it to "eval"
            for i, batch in enumerate(trainData):
                # prepare training batch
                segms = batch['segms'].to(device=self._device)
                imgs = batch['imgs'].requires_grad_().to(device=self._device)

                # training step
                self.unet_optimizer.zero_grad()
                output = self.unet(imgs)
                loss = self.criterion(output, segms)
                loss_epoch += loss.item()
                loss.backward()
                self.unet_optimizer.step()

            print('Epoch [{0:d} /{1:d}], train loss: {2:.3f}'.format((epoch + 1), epochs, loss_epoch / (i + 1)))

            # early stopping
            if (epoch + 1) % 10 == 0:
                self._validate()
                tm = (time.time() - st) / float(epoch + 1)
                print('Training took {0:4.1f}s, {1:4.1f}s per epoch, {2:4.1f}s to go'.format(time.time() - st, tm,
                                                                                             (epochs - 1 - epoch) * tm))
            if np.abs(loss_epoch / (i + 1) - last_loss) < 0.0005:
                counter += 1
            else:
                counter = 0
            if counter >= 3:
                print('Training stopped after {} epochs due to convergence'.format(epoch))
                break
            last_loss = loss_epoch / (i + 1)

    def train_vae(self, epochs):
        print('start training vae')
        self.unet.eval()
        # freeze base unet's weights
        for param in self.unet.parameters():
            param.requires_grad = False

        # load training set
        trainData = pickle.load(open(os.path.join('data', '{}.pkl'.format('oar_data_train_63')), "rb"))

        st = time.time()
        for epoch in range(epochs):  # epochs loop
            # on train set
            self.vae.train() # important as validation sets it to "eval"
            loss_epoch = 0
            for i, batch in enumerate(trainData):  # batches for training
                imgs = batch['imgs'].to(device=self._device)

                # training step
                feats = self.unet.get_feature_map(imgs)
                self.vae_optimizer.zero_grad()
                out_vae = self.vae(feats)
                loss = self._vae_loss(*out_vae, feats)
                loss_epoch += loss.item()
                loss.backward()
                self.vae_optimizer.step()

            print('Epoch [{0:d} /{1:d}], train loss: {2:.3f}'.format((epoch + 1), epochs, loss_epoch / (i + 1)))
            if (epoch + 1) % 10 == 0:
                self._validate(epoch, vae_on=True)
                tm = (time.time() - st) / float(epoch + 1)
                print('Training took {0:4.1f}s, {1:4.1f}s per epoch, {2:4.1f}s to go'.format(time.time() - st, tm,
                                                                                             (epochs - 1 - epoch) * tm))

    @torch.no_grad()
    def _validate(self, vae_on=False):
        self.unet.eval()

        # load validation set
        valData = pickle.load(open(os.path.join('data', '{}.pkl'.format('oar_data_val_63')), "rb"))
        loss_epoch = 0
        dice_epoch = 0
        for i, batch in enumerate(valData):  # batches for validation
            segms = batch['segms'].to(device=self._device)
            imgs = batch['imgs'].to(device=self._device)

            if vae_on:
                enc_unet = self.unet.get_encoded(imgs)
                feats, _, _ = self.vae(enc_unet[0])
                output = self.unet.decode(*enc_unet, feats)
            else:
                output = self.unet(imgs)
            loss = self.criterion(output, segms)
            loss_epoch += loss.item()
            output[output < 0.5] = 0  # background
            output[output >= 0.5] = 1  # foreground
            dice_epoch += np.sum(self.dice(output, segms))
        print('Validation loss: {0:.3f}, dice: {1:.3f}'.format(loss_epoch / (i + 1), dice_epoch / (i + 1)))

    @torch.no_grad()
    def test(self, vae_on=False):
        print('start testing')
        self.unet.eval()
        # load testing set
        testData = pickle.load(open(os.path.join('data', '{}.pkl'.format('oar_data_test_63')), "rb"))
        loss_tot = 0
        dice_epoch = 0
        for i, batch in enumerate(testData):  # batches for testing
            segms = batch['segms'].to(device=self._device)
            imgs = batch['imgs'].to(device=self._device)

            if vae_on:
                skip_cons = self.unet.get_encoded(imgs)
                feats, _, _ = self.vae(skip_cons[0])
                output = self.unet.decode(*skip_cons, feats=feats)
                torch.save(self.vae.state_dict(), os.path.join(self._fullpath, 'vunet_vae_weights.pth'))

            else:
                output = self.unet(imgs)
                torch.save(self.unet.state_dict(), os.path.join(self._fullpath, 'vunet_unet_weights.pth'))

            loss = self.criterion(output, segms)
            loss_tot += loss.item()
            output[output < 0.5] = 0  # background
            output[output >= 0.5] = 1  # foreground
            dice_epoch += np.mean(self.dice(output, segms))

        print('Test loss: {0:.3f}, dice: {1:.3f}'.format(loss_tot / (i + 1), dice_epoch / (i + 1)))

    @staticmethod
    def _vae_loss(recon, mu, logvar, feats):
        criterion = torch.nn.MSELoss(reduction='sum')
        MSE = criterion(recon, feats)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return MSE + KLD
