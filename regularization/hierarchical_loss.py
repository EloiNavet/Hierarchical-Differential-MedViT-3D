"""Hierarchical and risk-aware loss functions for medical image classification.

This module implements loss functions that incorporate class hierarchy and focal
weighting to improve performance on imbalanced multi-class classification tasks,
particularly for neurodegenerative disease classification from brain MRI.

Based on:
- CytoDINO (2025): Focal loss for hierarchical classification
- BBFL (2023): Balanced Binary Focal Loss for imbalanced data
"""

import torch
import torch.nn.functional as F
from torch import nn


class HierarchicalRiskAwareLoss(nn.Module):
    """Hierarchical Risk-Aware Loss combining fine and coarse classification.

    This loss function combines:
    1. Fine-grained cross-entropy loss on original classes (with optional label smoothing)
    2. Focal loss on hierarchically aggregated coarse classes

    The coarse-level focal loss penalizes confusion between high-level groups
    (e.g., Healthy vs Diseased) more heavily, helping stabilize predictions
    on dominant classes like CN while improving rare class detection.

    Inspired by CytoDINO (2025) and HCAL approaches for hierarchical classification.

    Args:
        hierarchy_map: Dict mapping fine class indices to coarse class indices.
                      Example: {0: 0, 1: 1, 2: 1, 3: 2, 4: 2, 5: 2, 6: 1}
                      Maps 7 fine classes to 3 coarse groups.
        alpha: Weight for the coarse (hierarchical) loss component. Default: 2.0
        focal_gamma: Gamma parameter for focal loss. Higher values increase
                    focus on hard examples. Default: 2.0
        label_smoothing: Label smoothing factor for fine loss. Default: 0.1
        reduction: Reduction method ('mean', 'sum', 'none'). Default: 'mean'

    Example:
        >>> # Hierarchy: CN=Healthy, AD/DLB/PSP=Group1, BV/PNFA/SD=Group2
        >>> hierarchy_map = {0: 0, 1: 1, 2: 1, 3: 2, 4: 2, 5: 2, 6: 1}
        >>> loss_fn = HierarchicalRiskAwareLoss(hierarchy_map, alpha=2.0, focal_gamma=2.0)
        >>> loss = loss_fn(logits, one_hot_targets)
    """

    def __init__(
        self,
        hierarchy_map: dict[int, int],
        alpha: float = 2.0,
        focal_gamma: float = 2.0,
        label_smoothing: float = 0.1,
        reduction: str = "mean",
    ) -> None:
        super().__init__()

        if not hierarchy_map:
            raise ValueError("hierarchy_map cannot be empty")
        if alpha < 0:
            raise ValueError(f"alpha must be non-negative, got {alpha}")
        if focal_gamma < 0:
            raise ValueError(f"focal_gamma must be non-negative, got {focal_gamma}")
        if not 0.0 <= label_smoothing < 1.0:
            raise ValueError(
                f"label_smoothing must be in [0, 1), got {label_smoothing}"
            )
        if reduction not in ("mean", "sum", "none"):
            raise ValueError(
                f"reduction must be 'mean', 'sum', or 'none', got {reduction}"
            )

        self.alpha = alpha
        self.gamma = focal_gamma
        self.label_smoothing = label_smoothing
        self.reduction = reduction

        # Store hierarchy info
        self.num_fine_classes = len(hierarchy_map)
        self.num_coarse_classes = len(set(hierarchy_map.values()))

        # Create aggregation matrix: (num_coarse, num_fine)
        # aggregation_matrix[coarse_idx, fine_idx] = 1 if fine_idx maps to coarse_idx
        aggregation_matrix = torch.zeros(
            self.num_coarse_classes, self.num_fine_classes, dtype=torch.float32
        )
        for fine_idx, coarse_idx in hierarchy_map.items():
            aggregation_matrix[coarse_idx, fine_idx] = 1.0

        self.register_buffer("aggregation_matrix", aggregation_matrix)

    def _compute_fine_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Compute fine-grained cross-entropy loss with label smoothing.

        Args:
            logits: Prediction logits of shape (N, num_fine_classes)
            targets: Soft targets (one-hot) of shape (N, num_fine_classes)

        Returns:
            Fine loss tensor
        """
        num_classes = logits.size(-1)

        # Convert target to same dtype as logits
        target_float = targets.to(logits.dtype)

        # Apply label smoothing
        if self.label_smoothing > 0.0:
            target_smoothed = (
                target_float * (1.0 - self.label_smoothing)
                + self.label_smoothing / num_classes
            )
        else:
            target_smoothed = target_float

        # Compute cross-entropy with smoothed soft targets
        log_probs = F.log_softmax(logits, dim=-1)
        loss = -(target_smoothed * log_probs).sum(dim=-1)

        return loss

    def _compute_coarse_focal_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Compute focal loss on coarse (hierarchical) classes.

        Aggregates fine-class probabilities to coarse classes, then applies
        focal loss to penalize misclassifications at the hierarchy level.

        Uses per-class focal loss formulation to properly handle soft targets
        from MixUp: L = -sum_c( y_c * (1 - p_c)^gamma * log(p_c) )

        Args:
            logits: Prediction logits of shape (N, num_fine_classes)
            targets: Soft targets (one-hot) of shape (N, num_fine_classes)

        Returns:
            Coarse focal loss tensor of shape (N,)
        """
        # Get fine-grained probabilities
        probs = F.softmax(logits, dim=-1)  # (N, num_fine)

        # Ensure aggregation matrix is on the same device as logits
        aggregation_matrix = self.aggregation_matrix
        if aggregation_matrix.device != logits.device:
            aggregation_matrix = aggregation_matrix.to(logits.device)

        # Aggregate to coarse probabilities using matrix multiplication
        # coarse_probs = probs @ aggregation_matrix.T -> (N, num_coarse)
        coarse_probs = torch.mm(probs, aggregation_matrix.t())

        # Clamp for numerical stability
        coarse_probs = torch.clamp(coarse_probs, min=1e-7, max=1.0 - 1e-7)

        # Get coarse targets by aggregating fine targets
        # coarse_targets = targets @ aggregation_matrix.T -> (N, num_coarse)
        coarse_targets = torch.mm(targets.to(logits.dtype), aggregation_matrix.t())

        # Compute absolute prediction error
        # This is the true generalization of (1-pt) for soft labels
        # If Target=1, Pred=0.9 -> Error=0.1 (Like classic FL)
        # If Target=0.5, Pred=0.9 -> Error=0.4 (Penalty maintained)
        prediction_error = torch.abs(coarse_targets - coarse_probs)

        # Focal weight based on absolute prediction error
        focal_weight = prediction_error**self.gamma

        # Weighted cross-entropy with focal weight
        # L = - sum( weight * y * log(p) )
        cross_entropy = -coarse_targets * torch.log(coarse_probs)
        focal_loss = (focal_weight * cross_entropy).sum(dim=-1)

        return focal_loss

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Compute hierarchical risk-aware loss.

        Args:
            logits: Prediction logits of shape (N, num_fine_classes)
            targets: Soft targets (one-hot encoded) of shape (N, num_fine_classes).
                    Expected from datasets that return one-hot labels or from MixUp.

        Returns:
            Combined loss value (scalar if reduction='mean'/'sum', tensor if 'none')
        """
        if logits.size(-1) != self.num_fine_classes:
            raise ValueError(
                f"Expected {self.num_fine_classes} classes in logits, "
                f"got {logits.size(-1)}"
            )
        if targets.size(-1) != self.num_fine_classes:
            raise ValueError(
                f"Expected {self.num_fine_classes} classes in targets, "
                f"got {targets.size(-1)}"
            )

        # Compute fine-grained loss
        fine_loss = self._compute_fine_loss(logits, targets)  # (N,)

        # Compute coarse focal loss
        coarse_loss = self._compute_coarse_focal_loss(logits, targets)  # (N,)

        # Combine losses
        total_loss = fine_loss + self.alpha * coarse_loss

        # For wandb logging
        self.last_fine_loss = fine_loss.mean().detach().item()
        self.last_coarse_loss = coarse_loss.mean().detach().item()

        # Apply reduction
        if self.reduction == "mean":
            return total_loss.mean()
        elif self.reduction == "sum":
            return total_loss.sum()
        else:  # 'none'
            return total_loss

    def extra_repr(self) -> str:
        return (
            f"num_fine_classes={self.num_fine_classes}, "
            f"num_coarse_classes={self.num_coarse_classes}, "
            f"alpha={self.alpha}, gamma={self.gamma}, "
            f"label_smoothing={self.label_smoothing}, reduction={self.reduction}"
        )
