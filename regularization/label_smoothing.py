"""Label smoothing loss for classification with soft targets."""

import torch
import torch.nn.functional as F
from torch import nn


class LabelSmoothingLoss(nn.Module):
    """
    Cross-entropy loss with label smoothing for multi-class classification.

    Designed for soft targets (one-hot encoded or mixed distributions from MixUp).

    Label smoothing prevents overconfident predictions by redistributing probability mass
    from the target class to all classes uniformly:
        smoothed_target = (1 - ε) * target + ε / K
    where ε is the smoothing factor and K is the number of classes.

    Args:
        smoothing (float): Smoothing factor in [0, 1). 0 = no smoothing.
        reduction (str): Reduction method ('mean', 'sum', 'none').
    """

    smoothing: float
    reduction: str

    def __init__(self, smoothing: float = 0.1, reduction: str = "mean") -> None:
        super().__init__()
        assert 0.0 <= smoothing < 1.0, "Smoothing must be in [0, 1)"
        self.smoothing = smoothing
        self.reduction = reduction

    def forward(
        self,
        input: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute label smoothed cross-entropy loss.

        Args:
            input (torch.Tensor): Logits of shape (N, C) where N is batch size and C is number of classes.
            target (torch.Tensor): Soft targets (one-hot or mixed distributions) of shape (N, C).
                                  Expected from datasets that return one-hot encoded labels or MixUp.

        Returns:
            torch.Tensor: Loss value (scalar if reduction='mean'/'sum', tensor if 'none').
        """
        num_classes = input.size(-1)

        if num_classes < 2:
            raise ValueError(f"Number of classes must be >= 2, got {num_classes}")

        # Convert target to same dtype as input
        target_float = target.to(input.dtype)

        # Apply label smoothing to soft targets
        if self.smoothing > 0.0:
            target_smoothed = (
                target_float * (1.0 - self.smoothing) + self.smoothing / num_classes
            )
        else:
            target_smoothed = target_float

        # Compute cross-entropy with smoothed soft targets
        log_probs = F.log_softmax(input, dim=-1)
        loss = -(target_smoothed * log_probs).sum(dim=-1)

        # Apply reduction
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        elif self.reduction == "none":
            return loss
        else:
            raise ValueError(f"Unknown reduction: {self.reduction}")
