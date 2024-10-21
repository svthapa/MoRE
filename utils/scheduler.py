from torch.optim.lr_scheduler import _LRScheduler
import torch 

class LinearIncreaseLR(_LRScheduler):
    def __init__(self, optimizer, target_lr, total_iters, last_epoch=-1):
        self.target_lr = target_lr
        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """Compute learning rate using a linear schedule."""
        # Calculate the learning rate factor
        lr_factor = min(self._step_count / self.total_iters, 1)
        # Update the learning rate towards the target learning rate
        return [base_lr + lr_factor * (self.target_lr - base_lr) for base_lr in self.base_lrs]
    

class LinearWarmupCosineAnnealingLR(_LRScheduler):
    def __init__(self, optimizer, warmup_epochs, total_epochs, min_lr, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        self.max_lr = optimizer.param_groups[0]['lr']
        super(LinearWarmupCosineAnnealingLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            # Linear warmup
            lr = (self.max_lr - self.base_lrs[0]) / self.warmup_epochs * self.last_epoch + self.base_lrs[0]
        else:
            # Cosine annealing
            cos_epoch = self.last_epoch - self.warmup_epochs
            cos_epochs = self.total_epochs - self.warmup_epochs
            lr = self.min_lr + 0.5 * (self.max_lr - self.min_lr) * (1 + torch.cos(torch.pi * cos_epoch / cos_epochs))
        return [lr for _ in self.base_lrs]

