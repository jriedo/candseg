import torch
import numpy as np


class Dice():
    def __init__(self):
        self._prediction = None
        self._target = None
        self._eps = 1e-6

    def __call__(self, prediction, target):
        self._prediction = prediction.contiguous()
        self._target = target.contiguous()
        return self._dicecoeff()

    def _dicecoeff(self):
        # loss = 2 * torch.matmul(self._prediction, self._target) / (self._prediction.sum() + self._target.sum())
        loss = []

        for pred, tar in zip(self._prediction, self._target):
            num = 2 * torch.matmul(pred.view(-1), tar.view(-1)).item() + self._eps
            den = (pred.sum() + tar.sum()).item() + self._eps
            l = 1 if np.isnan(num/den) else num/den
            loss.append(l)
            if np.any(np.isnan(loss[-1])):
                print(len(loss))
        return loss
