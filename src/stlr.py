import torch


class STLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_fraction, min_lr_ratio, t_total, last_epoch=-1):
        """
        Slanted triangular learning rates

        As described in "Universal Language Model Fine-tuning for Text Classification".
        """
        self.warmup_until = int(warmup_fraction * t_total)
        self.t_total = t_total
        self.min_lr_ratio = min_lr_ratio
        super(STLR, self).__init__(optimizer, last_epoch=last_epoch)

    def get_lr(self):
        if self.last_epoch <= self.warmup_until:
            progress = self.last_epoch / self.warmup_until
            multiplier = self.min_lr_ratio + ((1 - self.min_lr_ratio) * progress)
            return [base_lr * multiplier  for base_lr in self.base_lrs]
        else:
            progress = (self.last_epoch - self.warmup_until) / (self.t_total - self.warmup_until)
            multiplier = self.min_lr_ratio + ((1 - self.min_lr_ratio) * (1 - progress))
            return [base_lr * multiplier for base_lr in self.base_lrs]
