import time
import os

from net_handlers.vgg_oar import Sampler

def main():
    run_name='regression_oar'

    # define number of epochs to train
    epochs = 1

    # train and test network
    st = time.time()
    reg = Sampler('out', run_name)
    reg.train(epochs)
    reg.test()
    print('took {0:.0f} seconds'.format(time.time() - st))

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    main()

