import time
import os

from net_handlers.vunet_oar_whole_slice import Oar

def main():
    run_name = 'vunet_oar_whole_slice'

    # define number of epochs to train
    epochs = 100

    # define the dimensionality of the latent space
    ls = 32

    # train and test networks
    st = time.time()
    vari = Oar('out', run_name, ls)
    vari.train_unet(epochs)
    vari.test_unet()
    vari.train_vae(epochs)
    vari.test_vae()
    print('took {0:.0f} seconds'.format(time.time() - st))

if __name__=='__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"
    main()