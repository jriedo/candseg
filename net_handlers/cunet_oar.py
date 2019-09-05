import time
import torch
import numpy as np
import os
import pickle

from net_handlers.nets.cunet import UNet

class Oar():
    """
        Class for training and testing the CU-Net on OAR data
        The whole network is trained end-to-end with early stopping for regularization
    """
    def __init__(self, path, run_name):
        # define device and create output path
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._fullpath = os.path.join(path, run_name)
        if not os.path.isdir(self._fullpath):
            os.mkdir(self._fullpath)

        # define the u-net with corresponding optimizers
        self.unet = UNet(isdropout=True).to(device=self._device)
        self.criterion = torch.nn.BCELoss()
        self.optimizer = torch.optim.Adam(self.unet.parameters())

    def train(self, epochs):
        print('start training')
        self.unet.train()

        # used for early stopping
        counter = 0
        last_loss = 1e3

        # load training set
        trainData = pickle.load(open(os.path.join('data', '{}.pkl'.format('oar_data_train_63')), "rb"))

        st = time.time()
        for epoch in range(epochs):
            self.unet.train() # important, as validation sets it to "eval"
            loss_epoch = 0
            for i, batch in enumerate(trainData):
                # prepare training batch
                segms = batch['segms'].to(device=self._device)
                imgs = batch['imgs'].requires_grad_().to(device=self._device)
                labels = batch['r'].requires_grad_().to(device=self._device)

                # training step
                self.optimizer.zero_grad()
                output = self.unet(imgs, labels)
                loss = self.criterion(output, segms)
                loss_epoch += loss.item()
                loss.backward()
                self.optimizer.step()
            print('Epoch [{0:d} /{1:d}], train loss: {2:.3f}'.format((epoch + 1), epochs, loss_epoch / (i + 1)))

            # validation
            if (epoch + 1) % 10 == 0:
                self._validate()
                tm = (time.time() - st) / float(epoch + 1)
                print('Training took {0:4.1f}s, {1:4.1f}s per epoch, {2:4.1f}s to go'.format(time.time() - st, tm,
                                                                                             (epochs - 1 - epoch) * tm))
            # early stopping
            if np.abs(loss_epoch / (i + 1) - last_loss) < 0.0005:
                counter += 1
            else:
                counter = 0
            if counter >= 3:
                print('Training stoped after {} epochs due to convergence'.format(epoch))
                break
            last_loss = loss_epoch / (i + 1)

    @torch.no_grad()
    def _validate(self):
        self.unet.eval()
        valData = pickle.load(open(os.path.join('data', '{}.pkl'.format('oar_data_val_63')), "rb"))
        loss_epoch = 0
        for i, batch in enumerate(valData):
            segms = batch['segms'].to(device=self._device)
            imgs = batch['imgs'].to(device=self._device)
            labels = batch['r'].to(device=self._device)
            output = self.unet(imgs, labels)
            loss = self.criterion(output, segms)
            loss_epoch += loss.item()
        print('Validation loss: {0:.3f}'.format(loss_epoch / (i + 1)))

    @torch.no_grad()
    def test(self):
        print('start testing')
        self.unet.eval()
        testData = pickle.load(open(os.path.join('data', '{}.pkl'.format('oar_data_test_63')), "rb"))
        loss_tot = 0
        for i, batch in enumerate(testData):
            segms = batch['segms'].to(device=self._device)
            imgs = batch['imgs'].to(device=self._device)
            labels = batch['r'].to(device=self._device)
            output = self.unet(imgs, labels)
            loss = self.criterion(output, segms)
            loss_tot += loss.item()
        print('Test loss: {0:.3f}'.format(loss_tot / (i + 1)))

        torch.save(self.unet.state_dict(), os.path.join(self._fullpath, 'cunet_oar_weights.pth'))

