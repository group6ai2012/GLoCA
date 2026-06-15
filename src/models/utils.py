from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class SafeBatchNorm1d(nn.BatchNorm1d):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.training and input.shape[0] < 2:
            return F.batch_norm(
                input,
                self.running_mean,
                self.running_var,
                self.weight,
                self.bias,
                False,
                0.0,
                self.eps,
            )
        return super().forward(input)
