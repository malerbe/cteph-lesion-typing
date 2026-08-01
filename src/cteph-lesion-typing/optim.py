"""
author: Louca Malerba
"""

# Imports
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


######################################################################
# Losses
######################################################################
def get_loss(lossname):
    """Create a loss module.

    Args:
        lossname: either a string with the loss class name (e.g. 'CrossEntropyLoss', 'FocalLoss', 'CombinedCELoss')
                  or a dict { 'name': <str>, 'params': { ... } } to pass kwargs to the constructor.
    """
    # allow lossname to be a dict with params
    if isinstance(lossname, dict):
        name = lossname.get('name')
        params = lossname.get('params', {})
    else:
        name = lossname
        params = {}

    if not isinstance(name, str):
        raise ValueError('lossname must be a str or dict with a "name" key')

    # try PyTorch nn first
    if hasattr(nn, name):
        return getattr(nn, name)(**params)

    # then try local implementations
    if name in globals():
        return globals()[name](**params)

    raise ValueError(f"Unknown loss '{name}'")

######################################################################
# Optimizers
######################################################################
def get_optimizer(cfg, params):
    params_dict = cfg["params"]
    exec(f"global optim; optim = torch.optim.{cfg['algo']}(params, **params_dict)")
    return optim

######################################################################
# Schedulers
######################################################################
class LinearWarmupCosineWarmRestarts(torch.optim.lr_scheduler._LRScheduler):
    """Linear warmup followed by cosine annealing with periodic warm restarts.

    For the first ``warmup_epochs`` epochs the LR ramps linearly from
    ``warmup_start_factor * base_lr`` to ``base_lr``.

    After warmup, the LR follows ``CosineAnnealingWarmRestarts`` with
    period ``T_0`` (optionally multiplied by ``T_mult`` each cycle).

    Args:
        optimizer: Wrapped optimizer.
        warmup_epochs: Number of warmup epochs (linear ramp).
        T_0: First restart period *after* warmup (in epochs).
        T_mult: Multiply restart period by this after each restart.
        eta_min_ratio: Minimum LR = base_lr * eta_min_ratio.
        warmup_start_factor: Starting LR = base_lr * warmup_start_factor.
        last_epoch: Index of last epoch (for resuming).
    """

    def __init__(
        self,
        optimizer,
        warmup_epochs: int = 5,
        T_0: int = 20,
        T_mult: int = 1,
        eta_min_ratio: float = 0.01,
        warmup_start_factor: float = 0.01,
        last_epoch: int = -1,
    ):
        self.warmup_epochs = warmup_epochs
        self.T_0 = T_0
        self.T_mult = T_mult
        self.eta_min_ratio = eta_min_ratio
        self.warmup_start_factor = warmup_start_factor
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        epoch = self.last_epoch

        if epoch < self.warmup_epochs:
            # Linear warmup
            alpha = epoch / max(1, self.warmup_epochs)
            factor = self.warmup_start_factor + (1.0 - self.warmup_start_factor) * alpha
            return [base_lr * factor for base_lr in self.base_lrs]

        # Post-warmup: cosine with warm restarts
        t = epoch - self.warmup_epochs
        T_cur = self.T_0
        T_i = self.T_0

        if self.T_mult == 1:
            # Simple periodic restart
            t_in_cycle = t % T_i
        else:
            # Geometric restart periods
            cycle = 0
            t_acc = 0
            while t_acc + T_i <= t:
                t_acc += T_i
                T_i = int(T_i * self.T_mult)
                cycle += 1
            t_in_cycle = t - t_acc
            T_cur = T_i

        cos_factor = 0.5 * (1.0 + math.cos(math.pi * t_in_cycle / max(1, T_cur)))
        return [
            base_lr * (self.eta_min_ratio + (1.0 - self.eta_min_ratio) * cos_factor)
            for base_lr in self.base_lrs
        ]
    
def get_scheduler(cfg, optimizer):
    """Build a learning-rate scheduler from a config dict.

    Args:
        cfg: dict with keys:
            algo: name of a torch.optim.lr_scheduler class, or one of our
                  custom schedulers (e.g. "LinearWarmupCosineWarmRestarts")
            params: dict of kwargs passed to the scheduler constructor
        optimizer: the optimizer to wrap

    Returns:
        A scheduler instance, or None if cfg is None.
    """
    if cfg is None:
        return None
    algo = cfg["algo"]
    params = cfg.get("params", {})

    # Check local custom schedulers first
    _LOCAL_SCHEDULERS = {
        "LinearWarmupCosineWarmRestarts": LinearWarmupCosineWarmRestarts,
    }
    if algo in _LOCAL_SCHEDULERS:
        return _LOCAL_SCHEDULERS[algo](optimizer, **params)

    scheduler_cls = getattr(torch.optim.lr_scheduler, algo)
    return scheduler_cls(optimizer, **params)