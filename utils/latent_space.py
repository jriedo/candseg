import matplotlib.pyplot as plt

class LatentSpace():
    def __init__(self, fullpath):
        self._fullpath = fullpath
        self._z1_u = []
        self._z2_u = []
        self._z1_e = []
        self._z2_e = []
        self._tag = 'default'
        self._first_batch = True
        self._first_epoch = True

    def add_batch(self, z_u, z_e):
        [self._z1_u.append(float(z_u[k, 0].cpu())) for k in range(len(z_u[:, 0]))]
        [self._z2_u.append(float(z_u[k, 1].cpu())) for k in range(len(z_u[:, 0]))]
        [self._z1_e.append(float(z_e[k, 0].cpu())) for k in range(len(z_e[:, 0]))]
        [self._z2_e.append(float(z_e[k, 1].cpu())) for k in range(len(z_e[:, 0]))]


    def show(self):
        fig, ax = plt.subplots(figsize=(4, 4), dpi=600)

        ax.scatter(self._z1_u, self._z2_u, marker='.', label='U-Net VAE', alpha=0.5)
        ax.scatter(self._z1_e, self._z2_e, marker='.', label='prior VAE', alpha=0.5)

        ax.legend(loc='lower right')
        ax.grid(True)
        ax.set_title(self._tag)
        # plt.show()


    def new(self, tag=''):
        self._tag = tag
        self._z1_u = []
        self._z2_u = []
        self._z1_e = []
        self._z2_e = []
        self._first_batch = True
