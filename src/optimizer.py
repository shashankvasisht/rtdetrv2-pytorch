import torch
import torch.nn as nn
import torch.distributed
import torchvision
from copy import deepcopy
import math

torchvision.disable_beta_transforms_warning()
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from .utils import de_parallel

from torch.optim.lr_scheduler import LRScheduler

SGD = optim.SGD
Adam = optim.Adam
AdamW = optim.AdamW


MultiStepLR = lr_scheduler.MultiStepLR
CosineAnnealingLR = lr_scheduler.CosineAnnealingLR
OneCycleLR = lr_scheduler.OneCycleLR
LambdaLR = lr_scheduler.LambdaLR


class ModelEMA(object):
    """
    Model Exponential Moving Average from https://github.com/rwightman/pytorch-image-models
    Keep a moving average of everything in the model state_dict (parameters and buffers).
    This is intended to allow functionality like
    https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage
    A smoothed version of the weights is necessary for some training schemes to perform well.
    This class is sensitive where it is initialized in the sequence of model init,
    GPU assignment and distributed training wrappers.
    """

    def __init__(
        self,
        model: nn.Module,
        decay: float = 0.9999,
        warmups: int = 2000,
    ):
        super().__init__()

        self.module = deepcopy(de_parallel(model)).eval()
        # if next(model.parameters()).device.type != 'cpu':
        #     self.module.half()  # FP16 EMA

        self.decay = decay
        self.warmups = warmups
        self.updates = 0  # number of EMA updates
        self.decay_fn = lambda x: decay * (
            1 - math.exp(-x / warmups)
        )  # decay exponential ramp (to help early epochs)

        for p in self.module.parameters():
            p.requires_grad_(False)

    def update(self, model: nn.Module):
        # Update EMA parameters
        with torch.no_grad():
            self.updates += 1
            d = self.decay_fn(self.updates)
            msd = de_parallel(model).state_dict()
            for k, v in self.module.state_dict().items():
                if v.dtype.is_floating_point:
                    v *= d
                    v += (1 - d) * msd[k].detach()

    def to(self, *args, **kwargs):
        self.module = self.module.to(*args, **kwargs)
        return self

    def state_dict(
        self,
    ):
        return dict(module=self.module.state_dict(), updates=self.updates)

    def load_state_dict(self, state, strict=True):
        self.module.load_state_dict(state["module"], strict=strict)
        if "updates" in state:
            self.updates = state["updates"]

    def forwad(
        self,
    ):
        raise RuntimeError("ema...")

    def extra_repr(self) -> str:
        return f"decay={self.decay}, warmups={self.warmups}"


class ExponentialMovingAverage(torch.optim.swa_utils.AveragedModel):
    """Maintains moving averages of model parameters using an exponential decay.
    ``ema_avg = decay * avg_model_param + (1 - decay) * model_param``
    `torch.optim.swa_utils.AveragedModel <https://pytorch.org/docs/stable/optim.html#custom-averaging-strategies>`_
    is used to compute the EMA.
    """

    def __init__(self, model, decay, device="cpu", use_buffers=True):

        self.decay_fn = lambda x: decay * (1 - math.exp(-x / 2000))

        def ema_avg(avg_model_param, model_param, num_averaged):
            decay = self.decay_fn(num_averaged)
            return decay * avg_model_param + (1 - decay) * model_param

        super().__init__(model, device, ema_avg, use_buffers=use_buffers)


class Warmup(object):
    def __init__(
        self, lr_scheduler: LRScheduler, warmup_duration: int, last_step: int = -1
    ) -> None:
        self.lr_scheduler = lr_scheduler
        self.warmup_end_values = [
            pg["lr"] for pg in lr_scheduler.optimizer.param_groups
        ]
        self.last_step = last_step
        self.warmup_duration = warmup_duration
        self.step()

    def state_dict(self):
        return {k: v for k, v in self.__dict__.items() if k != "lr_scheduler"}

    def load_state_dict(self, state_dict):
        self.__dict__.update(state_dict)

    def get_warmup_factor(self, step, **kwargs):
        raise NotImplementedError

    def step(
        self,
    ):
        self.last_step += 1
        if self.last_step >= self.warmup_duration:
            return
        factor = self.get_warmup_factor(self.last_step)
        for i, pg in enumerate(self.lr_scheduler.optimizer.param_groups):
            pg["lr"] = factor * self.warmup_end_values[i]

    def finished(
        self,
    ):
        if self.last_step >= self.warmup_duration:
            return True
        return False


class LinearWarmup(Warmup):
    def __init__(
        self, lr_scheduler: LRScheduler, warmup_duration: int, last_step: int = -1
    ) -> None:
        super().__init__(lr_scheduler, warmup_duration, last_step)

    def get_warmup_factor(self, step):
        return min(1.0, (step + 1) / self.warmup_duration)
