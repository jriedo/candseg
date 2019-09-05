import time
import os

from net_handlers.vunet_pk_oar import Oar

def main():
    run_name = 'vunet_pk'

    # define number of epochs to train
    epochs = 100

    # train and test network
    st = time.time()
    vari = Oar('out', run_name)
    # only the VAEs are trained, unets are pretrained and loaded
    vari.train(epochs)
    vari.test()
    print('took {0:.0f} seconds'.format(time.time() - st))

if __name__=='__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    main()