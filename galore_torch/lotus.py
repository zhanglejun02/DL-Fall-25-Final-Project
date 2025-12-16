import math
import numpy as np
import warnings
from typing import Callable, Iterable, Tuple
import torch
from torch import nn
from torch.optim import Optimizer
from transformers.utils.versions import require_version
from .auto_projector import AutoProjector 

class AdamW(Optimizer):
    def __init__(
        self,
        params: Iterable[nn.parameter.Parameter],
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-6,
        weight_decay: float = 0.0,
        correct_bias: bool = True,
        scale_front: bool = False,
        disable_nl: bool = False,
        no_deprecation_warning: bool = True,
    ):
        if not no_deprecation_warning:
            warnings.warn(
                "This implementation of AdamW is deprecated and will be removed in a future version. Use the PyTorch"
                " implementation torch.optim.AdamW instead, or set `no_deprecation_warning=True` to disable this"
                " warning",
                FutureWarning,
            )
        require_version("torch>=1.5.0")  # add_ with alpha
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr} - should be >= 0.0")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter: {betas[0]} - should be in [0.0, 1.0)")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter: {betas[1]} - should be in [0.0, 1.0)")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps} - should be >= 0.0")

        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
            "correct_bias": correct_bias
        }
        super().__init__(params, defaults)

        self.scale_front = scale_front
        self.disable_nl = disable_nl

        # assign a unique seed for each param
        params_idx = 0
        for group in self.param_groups:
            for p in group["params"]:
                if p.requires_grad:
                    self.state[p]["seed"] = params_idx
                    params_idx += 1

    def _initialize_projector(self, group, state):
        return AutoProjector(
            rank=group["rank"],
            proj_type=group["proj_type"],
            scale=group["scale"],
            drift_threshold=group["drift_threshold"],
            check_interval=group["check_interval"],
            verbose=group.get("verbose", True),
            use_stable=group["use_stable"],
            method=group["proj_method"]
            )

    @torch.no_grad()
    def step(self, closure: Callable = None):
        """
        Performs a single optimization step.
        """
        loss = None
        if closure is not None:
            loss = closure()

        update_counts = {}

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("Adam does not support sparse gradients")

                state = self.state[p]
                if "step" not in state:
                    state["step"] = 0

                if "rank" in group:
                    if "projector" not in state:
                        state["projector"] = self._initialize_projector(group, state)

                    # project
                    grad = state["projector"].project(grad)

                if "exp_avg" not in state:
                    state["exp_avg"] = torch.zeros_like(grad)
                    state["exp_avg_sq"] = torch.zeros_like(grad)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = group["betas"]
                state["step"] += 1

                # update exp_avg / exp_avg_sq
                exp_avg.mul_(beta1).add_(grad, alpha=(1.0 - beta1))
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
                denom = exp_avg_sq.sqrt().add_(group["eps"])

                step_size = group["lr"]
                if group["correct_bias"]:
                    bias_correction1 = 1.0 - beta1 ** state["step"]
                    bias_correction2 = 1.0 - beta2 ** state["step"]
                    step_size *= math.sqrt(bias_correction2) / bias_correction1

                norm_grad = exp_avg / denom

                if "rank" in group:
                    norm_grad = state["projector"].project_back(norm_grad)

                # update param
                p.add_(norm_grad, alpha=-step_size)

                # Weight Decay
                if group["weight_decay"] > 0.0:
                    p.add_(p, alpha=(-group["lr"] * group["weight_decay"]))

                # ========== 记录 projector 更新次数（可选） ========== #
                if "rank" in group:
                    proj_obj = state["projector"]
                    update_counts[id(proj_obj)] = proj_obj.update_count

        return loss
