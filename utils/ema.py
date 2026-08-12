"""Exponential Moving Average (EMA) for model parameters."""

import copy
from collections import deque

import torch
from torch import nn


class EMAModel:
    """Exponential Moving Average over N most recent model states.

    Parameters
    ----------
    model : nn.Module
        PyTorch model
    decay : float
        EMA decay rate (τ);
        If decay = 0, EMA weights are equal to current model weights;
        If decay = 1, EMA weights are equal to the average of all model weights.
    n_models : int
        Number of most recent models to average (default: 3)
    device : torch.device, optional
        Device to store EMA weights
    """

    def __init__(
        self,
        model: nn.Module,
        decay: float,
        n_models: int = 3,
        device: torch.device | None = None,
    ):
        self.decay = decay
        self.n_models = n_models
        self.device = device or next(model.parameters()).device

        # Initialize queue to store last N model states
        self.model_states = deque(maxlen=n_models)

        # Initialize with current model state
        initial_state = {
            k: v.clone().cpu().detach() for k, v in model.state_dict().items()
        }
        self.model_states.append(initial_state)

        # Store EMA weights on CPU to save GPU memory
        self.model_state = {k: v.clone() for k, v in initial_state.items()}

        # Add storage for original weights
        self.orig_state = None

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        """Update EMA weights using current model weights.

        Parameters
        ----------
        model : nn.Module
            Current model to get weights from
        """
        # Get current model state
        current_state = model.state_dict()

        current_state = {
            k: v.detach().cpu() if v.is_cuda else v.detach()
            for k, v in current_state.items()
        }

        # Add to queue and remove oldest if needed
        self.model_states.append(current_state)

        # Compute weighted average of last N states
        weights = [self.decay**i for i in range(len(self.model_states))]
        weights.reverse()  # Most recent model gets highest weight
        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]

        # Update EMA state
        for name, ema_tensor in self.model_state.items():
            # Skip weighted averaging for integer/bool buffers (e.g., num_batches_tracked)
            if not (ema_tensor.is_floating_point() or ema_tensor.is_complex()):
                ema_tensor.copy_(self.model_states[-1][name])
                continue

            # Compute weighted sum efficiently using in-place operations
            ema_tensor.zero_()
            for state, w in zip(self.model_states, weights):
                ema_tensor.add_(state[name], alpha=w)

    @torch.no_grad()
    def apply_to(self, model: nn.Module) -> None:
        """Apply EMA weights to given model.

        Parameters
        ----------
        model : nn.Module
            Model to apply EMA weights to
        """
        if self.model_state is not None:
            # Store original weights before applying EMA weights
            self.orig_state = {
                k: v.detach().clone() for k, v in model.state_dict().items()
            }
            # Apply EMA weights
            for name, param in model.state_dict().items():
                if name in self.model_state:
                    param.copy_(self.model_state[name])

    @torch.no_grad()
    def restore(self, model: nn.Module) -> None:
        """Restore original weights to the model.

        Parameters
        ----------
        model : nn.Module
            Model to restore original weights to
        """
        if self.orig_state is not None:
            for name, param in model.state_dict().items():
                if name in self.orig_state:
                    param.copy_(self.orig_state[name])
            self.orig_state = None

    @torch.no_grad()
    def update_bn_stats(
        self, model: nn.Module, data_loader: torch.utils.data.DataLoader
    ) -> None:
        """Update BatchNorm statistics using EMA weights.

        Parameters
        ----------
        model : nn.Module
            Model containing BN layers
        data_loader : torch.utils.data.DataLoader
            DataLoader to compute new statistics
        """
        # Check if model has any BatchNorm layers with running stats
        has_bn_layers = any(
            hasattr(m, "running_mean") and m.running_mean is not None
            for m in model.modules()
        )
        if not has_bn_layers:
            # No BatchNorm layers to update, skip entirely
            return

        # Store original model state
        original_state = copy.deepcopy(model.state_dict())

        # Apply EMA weights
        self.apply_to(model)

        # Set model to train mode (for BN stats collection)
        model.train()

        # Update BN stats using a forward pass through the data
        # Limit to a reasonable number of batches to avoid blocking
        max_batches = min(100, len(data_loader))
        with torch.no_grad():
            for i, (x, _) in enumerate(data_loader):
                if i >= max_batches:
                    break
                x = x.to(next(model.parameters()).device)
                model(x)

        # Store updated BN stats in EMA state
        for name, param in model.state_dict().items():
            if "running_" in name or "batch_norm" in name:
                self.model_state[name] = param.cpu().clone()

        # Restore original model state
        model.load_state_dict(original_state)
