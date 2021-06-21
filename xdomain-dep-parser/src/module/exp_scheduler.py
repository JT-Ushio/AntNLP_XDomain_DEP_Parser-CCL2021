import warnings
from torch.optim.lr_scheduler import _LRScheduler


class ExponentialLRwithWarmUp(_LRScheduler):

    def __init__(self, optimizer, gamma, decay, double_gamma_step=None, warm_step=0, last_epoch=-1):
        self.gamma = gamma
        self.decay = decay
        self.warm_step = warm_step
        self.double_gamma_step = double_gamma_step
        super(ExponentialLRwithWarmUp, self).__init__(optimizer, last_epoch=last_epoch)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", UserWarning)

        if self.last_epoch < self.warm_step:
            return [base_lr * (1+self.last_epoch) / self.warm_step
                    for base_lr in self.base_lrs]
        else:
            if self.double_gamma_step and self.last_epoch < self.double_gamma_step:
                return [base_lr * self.gamma ** ((self.last_epoch-self.warm_step)/self.decay)
                        for base_lr in self.base_lrs]
                # return [base_lr * self.gamma ** (self.last_epoch-self.warm_step)
                #         for base_lr in self.base_lrs]
            else:
                return [base_lr * self.gamma ** ((0.5*self.double_gamma_step-self.warm_step+0.5*self.last_epoch)/self.decay)
                        for base_lr in self.base_lrs]

