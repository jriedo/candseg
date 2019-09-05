import time
import os

from net_handlers.cunet_oar import Oar

def main():
    run_name = 'cunet_oar'

    # define number of epochs to train
    epochs = 100

    # train and test network
    st = time.time()
    vari = Oar('out', run_name)
    vari.train(epochs)
    vari.test()
    print('took {0:.0f} seconds'.format(time.time() - st))

if __name__=='__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    main()
