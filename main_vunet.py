import time
import os

from net_handlers.vunet_oar import Oar

def main():
    run_name = 'vunet_oar'

    # define number of epochs to train
    epochs = 1

    # train and test networks
    st = time.time()
    vari = Oar('out', run_name)
    vari.train_unet(epochs)
    vari.test(vae_on=False)
    vari.train_vae(epochs)
    vari.test(vae_on=True)
    print('took {0:.0f} seconds'.format(time.time() - st))

if __name__=='__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    main()