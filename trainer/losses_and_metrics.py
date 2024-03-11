import time

import torch
from torchmetrics import Metric
from lightning.pytorch.callbacks import Callback

EPSILON = 1e-5

class GausInstNorm(object):
    def __init__(self, dims:tuple):
        super().__init__()
        self.dims = dims
    def forward(self, x):
        # calculate what we need
        means = torch.mean(x, dim=self.dims, keepdim=True)
        stds = torch.std(x, dim=self.dims, keepdim=True) + EPSILON
        # make the transfroms
        x = (x - means) / stds
        return x, (means, stds)
    def inverse(self, x, info:tuple):
        means, stds = info
        x = x * stds + means
        return x

class RangeInstNorm(object):
    def __init__(self, dims:tuple):
        super().__init__()
        self.dims = dims
    def forward(self, x):
        # calculate what we need
        mins = torch.minimum(x, dim=self.dims, keepdim=True)
        maxs = torch.maximum(x, dim=self.dims, keepdim=True)
        # make the transfroms
        x = (x - mins) / (maxs - mins)
        return x, (mins, maxs)
    def inverse(self, x, info:tuple):
        mins, maxs = info
        x = x * (maxs - mins) + mins
        return x

class PassInstNorm(object):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return x, ()
    def inverse(self, x, info:tuple):
        return x

class CustomMAE(Metric):
    def __init__(self):
        super().__init__()
        self.add_state("error", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")
    def update(self, pred: torch.Tensor, target: torch.Tensor):
        assert pred.shape == target.shape
        self.error += torch.sum(torch.abs(torch.subtract(pred, target)))
        self.total += target.numel()
    def compute(self):
        return self.error.float() / self.total

class LpLoss(object):
    def __init__(self, d=2, p=2, size_average=False, reduction=True):
        super(LpLoss, self).__init__()
        #Dimension and Lp-norm type are postive
        assert d > 0 and p > 0
        self.d = d
        self.p = p
        self.reduction = reduction
        self.size_average = size_average
    def abs(self, x, y):
        num_examples = x.size()[0]
        #Assume uniform mesh
        h = 1.0 / (x.size()[1] - 1.0)
        all_norms = (h**(self.d/self.p))*torch.norm(x.view(num_examples,-1) - y.view(num_examples,-1), self.p, 1)
        if self.reduction:
            if self.size_average:
                return torch.mean(all_norms)
            else:
                return torch.sum(all_norms)
        return all_norms
    def rel(self, x, y):
        # we could improve this by vectorizing this operation
        # this is an interesting loss tho
        num_examples = x.size()[0]
        diff_norms = torch.norm(x.reshape(num_examples,-1) - y.reshape(num_examples,-1), self.p, 1)
        y_norms = torch.norm(y.reshape(num_examples,-1), self.p, 1)
        if self.reduction:
            if self.size_average:
                return torch.mean(diff_norms/y_norms)
            else:
                return torch.sum(diff_norms/y_norms)
        return diff_norms/y_norms
    def __call__(self, x, y):
        loss = 0
        for step in range(x.shape[-1]):
            loss += self.rel(x[..., step], y[..., step])
        return loss

class TimingCallback(Callback):
    def __init__(self):
        super().__init__()
        self.epoch_start_time = 0
        self.epoch_total = 0
        self.epoch_count = 0

    def on_train_epoch_start(self, trainer, pl_module):
        self.epoch_start_time = time.time()

    def on_train_epoch_end(self, trainer, pl_module):
        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - self.epoch_start_time
        self.epoch_total += epoch_duration
        self.epoch_count += 1

    def _get_average_time_per_epoch(self):
        return self.epoch_total / self.epoch_count
