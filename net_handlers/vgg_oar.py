import os
import pickle
import torch
import numpy as np
import torchvision.models as models

from net_handlers.nets.vunet_padding import VUNet
from net_handlers.nets.vae import VAE

from utils.dice_coefficient import Dice


class Sampler():
    """
        Class for training and testing the VGG-16 network on estimating the OAR patch segmentations
        The VU-Net is loaded and used to generate samples.
        For each epoch and each image (21 per batch), a random sample segmentation is generated for training
    """

    def __init__(self, out, run_name):
        # define device and create output path
        self._out = out
        self._run_name = run_name
        self._fullpath = os.path.join(out, run_name)
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # define the sampling network (VU-Net) and load its weights
        self.unet = VUNet().to(device=self._device)
        self.vae = VAE().to(device=self._device)
        unet_dict = torch.load('checkpoints/static_unet_weights.pth', map_location=self._device)['unet_state_dict']
        vae_dict = torch.load('checkpoints/init_vae_weights.pth', map_location=self._device)['vae_state_dict']
        self.unet.load_state_dict(unet_dict)
        self.vae.load_state_dict(vae_dict)
        self.unet.eval()
        self.vae.eval()

        # define the regression network (vgg + linear layer) with corresponding optimizer
        self.vgg16 = models.vgg16(pretrained=True).to(self._device)
        self.final_linear = torch.nn.Linear(in_features=1000, out_features=1).to(self._device)
        self.optimizer = torch.optim.Adam(list(self.vgg16.parameters()) + list(self.final_linear.parameters()), lr=1e-4)
        self.loss = torch.nn.MSELoss()

        # load training and validation set
        self.train_data = pickle.load(open('data/oar_data_train_63.pkl', "rb"))
        self.val_data = pickle.load(open('data/oar_data_val_63.pkl', "rb"))

        self.dice = Dice()

    @torch.no_grad()
    def _generate_samples(self, imgbatch, segms_per_sample=1, deterministic=False):
        # generate the defined amount of samples for each image in imbatch
        samples = torch.empty((len(imgbatch), segms_per_sample, 64, 64)).to(self._device)
        enc_unet = self.unet.get_encoded(imgbatch)
        if deterministic: # used for comparison, here each call generates the same deterministic samples)
            out = self.unet.decode(*enc_unet)
            seg = out.squeeze()
            seg[seg < 0.5] = 0
            seg[seg >= 0.5] = 1
            samples[:, 0, :, :] = seg
        else:
            for k in range(segms_per_sample):
                z = torch.randn(len(imgbatch), 2).to(device=self._device) # actual sampling of latent variables
                feats = self.vae.decode(z, enc_unet[0].shape)
                out = self.unet.decode(*enc_unet, feats)
                seg = out.squeeze()
                seg[seg < 0.5] = 0
                seg[seg >= 0.5] = 1
                samples[:, k, :, :] = seg
        return samples

    def _upsample_64_224(self, low_res):
        return torch.nn.functional.interpolate(low_res, scale_factor=4, mode='bilinear', align_corners=True)

    def _best_dice(self, sample, segms, raters):
        # return best dice to any rater (used for estimation)
        best = np.zeros((sample.size()[0]))
        for r in raters:
            dice = np.array(self.dice(sample, segms[raters == r, :, :, :]))
            best[dice > best] = dice[dice > best]
        return best

    def _normalize(self, tensor):
        # torch outputs values and indices, therefore the [0] for the values
        maxs = torch.max(torch.max(tensor, dim=2)[0], dim=2)[0]
        maxs[maxs == 0] = 1
        tensor = tensor / maxs[:, None, None]
        return tensor

    def train(self, epochs):
        print('start_training')
        for epoch in range(epochs):
            loss_epoch = 0
            self.vgg16.train()  # important, as validation sets it to "eval"
            for i, batch in enumerate(self.train_data):
                # prepare training batch
                imgs = batch['imgs'][0:63:3].to(self._device)
                segms = batch['segms'].to(self._device)
                r = batch['r'].type(torch.ByteTensor)[:, 0, 0, 0].to(self._device)

                # generate samples and resize to 256x256 (224 comes from min size of vgg)
                samples = self._generate_samples(imgs, 1, deterministic=False)
                labels = torch.tensor(self._best_dice(samples, segms, r), dtype=torch.float32).to(self._device)
                samples = self._upsample_64_224(samples)
                imgs = self._upsample_64_224(imgs)
                # stack input image from the three parts (sample segs, images, sample segs + images)
                in_imgs = torch.stack(
                    (self._normalize(samples), self._normalize(imgs), self._normalize(samples + imgs)), dim=1).squeeze()

                # training step
                pred = self.vgg16(in_imgs)
                pred = self.final_linear(pred).squeeze()
                self.optimizer.zero_grad()
                loss = self.loss(pred, labels)
                loss_epoch += loss.item()
                loss.backward()
                self.optimizer.step()

            print('Epoch [{0:d} /{1:d}], train loss: {2:.3f}'.format((epoch + 1), epochs, loss_epoch / (i + 1)))

            # validation
            if (epoch + 1) % 10 == 0:
                self._validate(epoch)

    @torch.no_grad()
    def _validate(self, epoch):
        loss_val = 0
        corr_val = 0 # correlation coefficient

        k = 0 # counter for each round and batch
        self.vgg16.eval()
        for _ in range(10):
            for i, batch in enumerate(self.val_data):
                imgs = batch['imgs'][0:63:3].to(self._device)
                segms = batch['segms'].to(self._device)
                r = batch['r'].type(torch.ByteTensor)[:, 0, 0, 0].to(self._device)

                # generate samples and resize to 256x256 (224 comes from min size of vgg)
                samples = self._generate_samples(imgs, 1, deterministic=True)
                labels = torch.tensor(self._best_dice(samples, segms, r), dtype=torch.float32).to(self._device)
                samples = self._upsample_64_224(samples)
                imgs = self._upsample_64_224(imgs)
                in_imgs = torch.stack(
                    (self._normalize(samples), self._normalize(imgs), self._normalize(samples + imgs)),
                    dim=1).squeeze()

                pred = self.vgg16(in_imgs)
                pred = self.final_linear(pred).squeeze()
                loss = self.loss(pred, labels)
                loss_val += loss.item()

                corr_val += self.corr_coeff(pred, labels)
                k += 1

        print('Val e {0:d}: MSE loss: {1:.3f}, corr. coeff.: {2:.3f}'.format((epoch + 1), loss_val / k, corr_val / k))

    @torch.no_grad()
    def test(self):
        test_data = pickle.load(open('data/oar_data_test_63.pkl', "rb"))
        loss_test = 0
        corr_test = 0 # correlation coefficient
        k = 0
        for _ in range(10):
            for i, batch in enumerate(test_data):
                imgs = batch['imgs'][0:63:3].to(self._device)
                segms = batch['segms'].to(self._device)
                r = batch['r'].type(torch.ByteTensor)[:, 0, 0, 0].to(self._device)

                samples = self._generate_samples(imgs, 1, deterministic=True)
                labels = torch.tensor(self._best_dice(samples, segms, r), dtype=torch.float32).to(self._device)
                samples = self._upsample_64_224(samples)
                imgs = self._upsample_64_224(imgs)
                in_imgs = torch.stack(
                    (self._normalize(samples), self._normalize(imgs), self._normalize(samples + imgs)),
                    dim=1).squeeze()

                pred = self.vgg16(in_imgs)
                pred = self.final_linear(pred).squeeze()
                loss = self.loss(pred, labels)
                loss_test += loss.item()
                corr_test += self.corr_coeff(pred, labels)

                k += 1
        print('Test: MSE loss: {:.3f}, corr. coeff.: {:.3f}'.format(loss_test / k, corr_test / k))
        torch.save({'vgg16': self.vgg16.state_dict(), 'linear_1000_1': self.final_linear.state_dict()},
                   os.path.join(self._fullpath, 'vgg_oar_weights.pth'))


    def corr_coeff(self, prediction, target):
        # calculation coefficient as per definition
        prediction = prediction - torch.mean(prediction)
        target = target - torch.mean(target)
        return torch.sum(prediction * target) / (
                    torch.sqrt(torch.sum(prediction ** 2)) * torch.sqrt(torch.sum(target ** 2)))
