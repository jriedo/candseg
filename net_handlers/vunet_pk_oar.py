import time
import torch
import pickle
import os
import numpy as np

from net_handlers.nets.vunet_padding import VUNet
from net_handlers.nets.unet import UNet
from net_handlers.nets.vae import VAE
from net_handlers.nets.vae_encoder import VAEncoder
from utils.dice_coefficient import Dice
from utils.latent_space import LatentSpace
from utils.composed_loss import ComposedLoss

class Oar():
    """
        Class for training the VU-Net with PK on OAR patches
    """
    def __init__(self, path, run_name):
        # define device and create output path
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._fullpath = os.path.join(path, run_name)
        self._path = path
        if not os.path.isdir(self._fullpath):
            os.mkdir(self._fullpath)

        # create the 4 networks needed
        self.unet = VUNet(isdropout=True).to(device=self._device)
        self.static_unet = UNet(isdropout=True).to(device=self._device)
        self.uvae = VAE(latent_size=self.latent_size).to(device=self._device) # u-net vae (internal)
        self.evae = VAEncoder(latent_size=self.latent_size).to(device=self._device) # external vae

        # define optimizer, loss, metric, latent space (ls)
        self.optimizer = torch.optim.Adam(list(self.uvae.parameters()) + list(self.evae.parameters()), lr=1e-4)
        self.dice = Dice()
        self.ls = LatentSpace(self._fullpath)
        self.loss = ComposedLoss()

        # initialize networks with weights from the VU-Net
        unet_dict = torch.load('checkpoints/static_unet_weights.pth', map_location=self._device)['unet_state_dict']
        self.static_unet.load_state_dict(unet_dict)
        self.unet.load_state_dict(unet_dict)
        vae_dict = torch.load('checkpoints/init_vae_weights.pth',  map_location=self._device)['vae_state_dict']
        self.uvae.load_state_dict(vae_dict)

    def train(self, epochs):
        print('start training')
        self.unet.eval()
        self.static_unet.eval()

        # load training set
        trainData = pickle.load(open(os.path.join('data', '{}.pkl'.format('oar_data_train_63')), "rb"))

        # freeze parameters of unets
        for param in self.unet.parameters():
            param.requires_grad = False
        for param in self.static_unet.parameters():
            param.requires_grad = False

        st = time.time()
        for epoch in range(epochs):  # epochs loop
            self.loss.zero()
            self.uvae.train() # important, as validation sets it to "eval"
            self.evae.train() # important, as validation sets it to "eval"
            for i, batch in enumerate(trainData):  # batches for training
                segms = batch['segms'].to(device=self._device)
                imgs = batch['imgs'].to(device=self._device)
                self.optimizer.zero_grad()

                out_evae = self.evae.encode(self.static_unet(imgs))
                evae_recon = self.evae.decode(*out_evae)
                feats = self.unet.get_feature_map(imgs)
                out_uvae = self.uvae(feats)
                loss = self.loss.add_batch(segms, out_uvae, out_evae, feats, evae_recon, get_batch_loss=True)

                loss.backward()
                self.optimizer.step()

            loss = self.loss.get_loss()
            print('Epoch [{0:d} /{1:d}], train loss: {2:.3f}'.format((epoch + 1), epochs, loss))

            if (epoch + 1) % 10 == 0:
                self._validate(epoch)
                tm = (time.time() - st) / float(epoch + 1)
                print('Training took {0:4.1f}s, {1:4.1f}s per epoch, {2:4.1f}s to go'.format(time.time() - st, tm,
                                                                                             (epochs - 1 - epoch) * tm))

    @torch.no_grad()
    def _validate(self, epoch):
        self.unet.eval()
        self.uvae.eval()
        self.evae.eval()
        self.valData = pickle.load(open(os.path.join('data', '{}.pkl'.format('oar_data_val_63')), "rb"))
        self.loss.zero()
        self.ls.new('val_{}'.format(epoch))
        for i, batch in enumerate(self.valData):  # batches for validation
            segms = batch['segms'].to(device=self._device)
            imgs = batch['imgs'].to(device=self._device)

            out_evae = self.evae.encode(self.static_unet(imgs))
            evae_recon = self.evae.decode(*out_evae)
            feats = self.unet.get_feature_map(imgs)
            out_uvae = self.uvae(feats)
            self.ls.add_batch(out_uvae[1], out_evae[0])
            self.loss.add_batch(segms, out_uvae, out_evae, feats, evae_recon, get_batch_loss=True)

        loss = self.loss.get_loss()
        self.ls.show()
        print('Validation loss: {0:.3f}'.format(loss))

    @torch.no_grad()
    def test(self):
        print('start testing')
        self.unet.eval()
        testData = pickle.load(open(os.path.join('data', '{}.pkl'.format('oar_data_test_63')), "rb"))
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
        torch.save(self.uvae.state_dict(), os.path.join(self._fullpath, 'vunet_pk_uvae_weights.pth'))
        torch.save(self.evae.state_dict(), os.path.join(self._fullpath, 'vunet_pk_evae_weights.pth'))


