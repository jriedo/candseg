import torch


class ComposedLoss:

    def __init__(self):
        self._composed_loss = 0
        self._bce_loss = 0
        self._mse_loss = 0
        self._kl_loss = 0
        self._batch_no = 1
        self._tag = ''

        self._get_bce_loss = torch.nn.BCELoss(reduction='sum')
        self._get_mse_loss = torch.nn.MSELoss(reduction='sum')

    def zero(self):
        self._composed_loss = 0
        self._bce_loss = 0
        self._mse_loss = 0
        self._kl_loss = 0
        self._batch_no = 1

    def add_batch(self, segms=None, uvae_recon=None, enc_evae=None, feats=None, evae_recon=None, get_batch_loss=False):
        self._batch_no += 1
        kl, mse, bce = self._get_losses(feats, segms, uvae_recon, evae_recon, enc_evae)

        self._bce_loss += bce
        self._kl_loss += kl
        self._mse_loss += mse
        self._composed_loss += (kl + mse + bce)
        if get_batch_loss:
            # for trainings backprop
            return kl + mse + bce

    def get_loss(self):
        loss = self._composed_loss / self._batch_no
        return loss

    def _get_losses(self, feats, segms, uvae_recon, evae_recon, enc_evae):
        bce = self._get_bce_loss(evae_recon, segms)  # reconstruction loss evae
        mse = self._get_mse_loss(uvae_recon[0], feats)  # reconstruction loss uvae
        kl = self._get_kl_loss(enc_evae[0], enc_evae[1], uvae_recon[1], uvae_recon[2])

        return kl, mse, bce

    @staticmethod
    def _get_kl_loss(mu_p, logvar_p, mu_q, logvar_q):
        # distribution q is matched to p

        q = torch.distributions.Normal(mu_q, torch.exp(logvar_q * 0.5))
        p = torch.distributions.Normal(mu_p, torch.exp(logvar_p * 0.5))

        loss = torch.distributions.kl.kl_divergence(p, q)
        loss = torch.sum(loss)

        return loss
