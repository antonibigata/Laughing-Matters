from torch.optim.lr_scheduler import _LRScheduler
import numpy as np
import math

from src import utils

log = utils.get_pylogger(__name__)


# class WarmupCosineScheduler(_LRScheduler):
#     def __init__(
#         self,
#         optimizer,
#         iter_per_epoch,
#         warmup_epochs=0,
#         num_epochs=1,
#         total_iter=1,
#         cosine_decay=True,
#         excluded_group=None,
#     ):

#         self.excluded_group = excluded_group
#         assert len(optimizer.param_groups) == 1, "WarmupCosineScheduler only supports one param group"
#         # self.base_lrs = {param_group["name"]: param_group["lr"] for param_group in optimizer.param_groups}
#         # if warmup_epochs < 0:  # if warmup_epochs is < 0, then it is a ratio
#         #     warmup_epochs = int(warmup_epochs * num_epochs)
#         self.warmup_iter = warmup_epochs * total_iter
#         self.total_iter = total_iter
#         self.optimizer = optimizer
#         self.iter = 0
#         self.current_lr = 0
#         self.cosine_decay = cosine_decay

#         self.init_lr()  # so that at first step we have the correct step size
#         # super().__init__(optimizer)

#     def get_lr(self, base_lr):
#         if self.iter < self.warmup_iter:
#             # print("warmup", base_lr * self.iter / self.warmup_iter)
#             # print(base_lr)
#             return base_lr * self.iter / self.warmup_iter
#         elif not self.cosine_decay:
#             return base_lr
#         else:
#             decay_iter = self.total_iter - self.warmup_iter
#             return 0.5 * base_lr * (1 + np.cos(np.pi * (self.iter - self.warmup_iter) / decay_iter))

#     def update_param_groups(self):
#         # for param_group in self.optimizer.param_groups:
#         #     if not self.excluded_group or not param_group["name"].startswith(self.excluded_group):
#         #         param_group["lr"] = self.get_lr(self.base_lrs[param_group["name"]])
#         for param_group in self.optimizer.param_groups:
#             print("lr: ", param_group["lr"])
#             param_group["lr"] = self.get_lr(param_group["lr"])
#             # print(param_group["lr"], self.iter, self.warmup_iter, self.total_iter)

#     def step(self):
#         self.update_param_groups()
#         self.iter += 1

#     def init_lr(self):
#         self.update_param_groups()


class WarmupCosineScheduler_bis(_LRScheduler):
    def __init__(self, optimizer, warmup, max_iters, func=None):
        self.warmup = warmup * max_iters
        self.max_num_iters = max_iters
        print("warmup: ", self.warmup)
        print("max_iters: ", self.max_num_iters)
        super().__init__(optimizer)

    def get_lr(self):
        lr_factor = self.get_lr_factor(iter_step=self.last_epoch)
        return [base_lr * lr_factor for base_lr in self.base_lrs]

    def get_lr_factor(self, iter_step):
        lr_factor = 0.5 * (1 + np.cos(np.pi * iter_step / self.max_num_iters))
        if iter_step <= self.warmup:
            lr_factor *= iter_step / self.warmup
        return lr_factor


class WarmupCosineScheduler(_LRScheduler):
    def __init__(self, optimizer, max_iters, warmup_steps=0, func=None):
        # self.base_lr_orig = base_lr
        if warmup_steps < 1:  # if warmup_epochs is < 0, then it is a ratio
            warmup_steps = int(warmup_steps * max_iters)
        print("warmup: ", warmup_steps)
        print("max_iters: ", max_iters)
        self.max_iters = max_iters
        # self.final_lr = final_lr
        self.warmup_steps = warmup_steps
        # self.warmup_begin_lr = warmup_begin_lr
        self.max_steps = self.max_iters - self.warmup_steps
        super().__init__(optimizer)

    def get_warmup_lr(self, epoch):
        # increase = (self.base_lr_orig - self.warmup_begin_lr) * float(epoch) / float(self.warmup_steps)
        factor = float(epoch) / float(self.warmup_steps)
        # return self.warmup_begin_lr + increase
        return factor

    def get_lr(self):
        epoch = self.last_epoch
        factor = 1
        if epoch < self.warmup_steps:
            return [self.get_warmup_lr(epoch) * base_lr for base_lr in self.base_lrs]
        if epoch <= self.max_iters:
            # self.base_lr = (
            #     self.final_lr
            #     + (self.base_lr_orig - self.final_lr)
            #     * (1 + math.cos(math.pi * (epoch - self.warmup_steps) / self.max_steps))
            #     * 0.5
            # )
            factor = (1 + math.cos(math.pi * (epoch - self.warmup_steps) / self.max_steps)) * 0.5
        return [base_lr * factor for base_lr in self.base_lrs]
