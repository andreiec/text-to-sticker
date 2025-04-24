import math

class KLAnnealingScheduler:
    def __init__(self, max_beta=1e-2, strategy='sigmoid', mid_epoch=10, steepness=0.25, warmup_epochs=10):
        self.max_beta = max_beta
        self.strategy = strategy
        self.mid_epoch = mid_epoch
        self.steepness = steepness
        self.warmup_epochs = warmup_epochs

    def __call__(self, epoch):
        if self.strategy == 'sigmoid':
            return self.max_beta / (1 + math.exp(-self.steepness * (epoch - self.mid_epoch)))
        elif self.strategy == 'linear':
            return min(self.max_beta, self.max_beta * epoch / self.warmup_epochs)
        else:
            raise ValueError(f"Unknown strategy '{self.strategy}' — use 'sigmoid' or 'linear'.")

    def get_beta(self, epoch):
        return self.__call__(epoch)
