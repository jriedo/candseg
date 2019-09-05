import time
import torch
import os
from net_handlers.nets.cunet import UNet
import pickle


class Synthetic():
    """
    Class for training and testing the CU-Net on synthetic data
    The whole network is trained end-to-end
    """
    def __init__(self, path, run_name):
        # define device and create output path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.fullpath = os.path.join(path, run_name)
        if not os.path.isdir(self.fullpath):
            os.mkdir(self.fullpath)

        # define the u-net with corresponding optimizers
        self.unet = UNet(isdropout=True).to(device=self.device)
        self.criterion = torch.nn.BCELoss()
        self.optimizer = torch.optim.Adam(self.unet.parameters())

    def train(self, epochs):
        print('start training')
        # load training set
        trainData = pickle.load(open(os.path.join('data', '{}.pkl'.format('syn_data_train_50')), "rb"))

        st = time.time()
        for epoch in range(epochs):
            loss_epoch = 0
            self.unet.train() # important, as validation sets it to "eval"
            for i, batch in enumerate(trainData):
                # prepare training batch
                segms = batch['segms'].to(device=self.device)
                imgs = batch['imgs'].requires_grad_().to(device=self.device)
                # prepare labels (rater numbers) broadcast them to desired image shape
                labels = torch.mul(batch['r'][:, :, None, None], torch.ones((50, 1, 240, 240))).requires_grad_().to(device=self.device)

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
                self.validate()
                tm = (time.time() - st) / float(epoch + 1)
                print('Training took {0:4.1f}s, {1:4.1f}s per epoch, {2:4.1f}s to go'.format(time.time() - st, tm,
                                                                                             (epochs - 1 - epoch) * tm))

    @torch.no_grad()
    def validate(self):
        self.unet.eval()
        valData = pickle.load(open(os.path.join('data', '{}.pkl'.format('syn_data_val_50')), "rb"))
        loss_epoch = 0
        for i, batch in enumerate(valData):
            segms = batch['segms'].to(device=self.device)
            imgs = batch['imgs'].to(device=self.device)
            # prepare labels (rater numbers) broadcast them to desired image shape
            labels = torch.mul(batch['r'][:, :, None, None] + 1, torch.ones((50, 1, 240, 240))).requires_grad_().to(
                device=self.device)
            output = self.unet(imgs, labels)
            loss = self.criterion(output, segms)
            loss_epoch += loss.item()
        print('Validation loss: {0:.3f}'.format(loss_epoch / (i + 1)))

    @torch.no_grad()
    def test(self):
        print('start testing')
        self.unet.eval()
        testData = pickle.load(open(os.path.join('data', '{}.pkl'.format('syn_data_test_50')), "rb"))
        loss_tot = 0
        for i, batch in enumerate(testData):
            segms = batch['segms'].to(device=self.device)
            imgs = batch['imgs'].to(device=self.device)
            # prepare labels (rater numbers) broadcast them to desired image shape
            labels = torch.mul(batch['r'][:, :, None, None] + 1, torch.ones((50, 1, 240, 240))).to(device=self.device)
            output = self.unet(imgs, labels)
            loss = self.criterion(output, segms)
            loss_tot += loss.item()
        print('Test loss: {0:.3f}'.format(loss_tot / (i + 1)))
        torch.save(self.unet.state_dict(), os.path.join(self._fullpath, 'cunet_syn_weights.pth'))

