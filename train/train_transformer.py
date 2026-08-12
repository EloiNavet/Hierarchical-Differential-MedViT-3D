"""Step-based DDP training for 3D medical image transformers with k-fold CV."""

import argparse
import datetime
import json
import logging
import math
import os
import queue
import random
import re
import shutil
import sys
import threading
import warnings
from collections.abc import Sequence
from contextlib import nullcontext
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

import numpy as np
import torch
import torch.distributed as dist
import wandb as w
import yaml
from monai.transforms import (
    Compose,
    Identity,
    NormalizeIntensity,
    OneOf,
    RandAdjustContrast,
    RandAffine,
    RandBiasField,
    RandCoarseDropout,
    RandGaussianSmooth,
    RandGibbsNoise,
    RandHistogramShift,
    RandKSpaceSpikeNoise,
    RandScaleIntensity,
    RandSpatialCrop,
    Resize,
)
from monai.utils import set_determinism as monai_set_determinism
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

sys.path.append(str(Path(__file__).resolve().parent.parent))
from dataset.dataset import MRIMixUp, NormalDataset
from dataset.preprocessing import DataPrepa
from regularization import HierarchicalRiskAwareLoss, LabelSmoothingLoss
from utils import (
    _MAX_UINT32,
    AdaptiveGaussianNoise,
    AdaptiveRicianNoise,
    ChannelSelectiveTransform,
    EMAModel,
    MultiChannelRandAffine,
    MultiChannelResize,
    cosine_scheduler,
    count_parameters,
    dir_path,
    file_path,
    get_model_size,
    get_params_groups,
    get_rank,
    get_train_val_test,
    get_world_size,
    init_distributed_mode,
    is_dist_avail_and_initialized,
    normalize_seed,
    seed_everything,
)

# Suppress non-deterministic warnings when using torch.use_deterministic_algorithms(warn_only=True)
_NONDETERMINISTIC_OPS = [
    "adaptive_avg_pool3d_backward_cuda",
    "grid_sampler_3d_backward_cuda",
    "upsample_linear1d_backward_out_cuda",
    "avg_pool3d_backward_cuda",
    "_histc_cuda",
]
for op in _NONDETERMINISTIC_OPS:
    warnings.filterwarnings("ignore", message=f".*{op}.*", category=UserWarning)
warnings.filterwarnings(
    "ignore", category=DeprecationWarning, module="ignite.handlers.checkpoint"
)

# TF32 kept on to mirror paper runs; disabling changes numerics on Ampere+ GPUs.
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512,expandable_segments:True"


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--training-csv-dir",
        type=dir_path,
        help="Directory containing 10 fold csv files",
    )
    parser.add_argument(
        "--intermediate-dir",
        type=str,
        help="Directory to save intermediate data",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        help="Directory to save models",
    )
    parser.add_argument(
        "--project-name",
        type=str,
        default="Transformers",
        help="Wandb project name",
    )
    parser.add_argument(
        "--runname",
        type=str,
        help="Wandb run name",
    )
    parser.add_argument(
        "--wandb-mode",
        type=str,
        default="disabled",
        help="Wandb mode",
        choices=["online", "offline", "disabled"],
    )
    parser.add_argument(
        "--checkpoint",
        type=file_path,
        help="Path to the checkpoint to load.",
        default=None,
    )
    parser.add_argument(
        "--config",
        type=file_path,
        help="Path to custom YAML configuration file that will override default config values",
        default=None,
        nargs="?",
    )
    parser.add_argument(
        "--fold",
        type=int,
        help="Fold number to train on (0-9)",
        default=None,
        nargs="?",
    )
    parser.add_argument(
        "--seed",
        type=str,
        help="Override training seed (set to 'none' or 'false' to disable deterministic seeding)",
        default=None,
        nargs="?",
    )

    args = parser.parse_args()
    return args


def compute_gradient_accumulation_steps(
    batch_size_per_gpu: int,
    effective_batch_size: int,
    world_size: int,
) -> tuple[int, int]:
    """Compute gradient accumulation steps and actual effective batch size.

    Parameters
    ----------
    batch_size_per_gpu : int
        Batch size per GPU (hardware constraint).
    effective_batch_size : int
        Desired effective batch size.
    world_size : int
        Number of GPUs/processes.

    Returns
    -------
    tuple[int, int]
        Gradient accumulation steps and actual effective batch size achieved.
    """
    per_step_batch = batch_size_per_gpu * world_size

    if effective_batch_size and effective_batch_size > 0:
        gradient_accumulation = max(1, math.ceil(effective_batch_size / per_step_batch))
    else:
        gradient_accumulation = 1

    # Calculate actual effective batch size achieved (can slightly exceed target)
    actual_effective_batch_size = per_step_batch * gradient_accumulation

    return gradient_accumulation, actual_effective_batch_size


class AsyncCheckpointSaver:
    """Asynchronous checkpoint saver that offloads I/O to a background thread."""

    def __init__(self, max_queue_size: int = 3):
        """Initialize the async checkpoint saver.

        Parameters
        ----------
        max_queue_size : int
            Maximum number of pending checkpoint saves in the queue.
        """
        self.queue = queue.Queue(maxsize=max_queue_size)
        self.worker_thread = threading.Thread(target=self._worker, daemon=False)
        self.stop_event = threading.Event()
        self.worker_thread.start()

    def _worker(self):
        """Worker thread that processes checkpoint save requests."""
        while not self.stop_event.is_set() or not self.queue.empty():
            try:
                checkpoint_gpu, save_path = self.queue.get(timeout=0.1)
                if checkpoint_gpu is None:
                    break

                checkpoint_cpu = {
                    k: v.cpu().clone() if isinstance(v, torch.Tensor) else v
                    for k, v in checkpoint_gpu.items()
                }
                torch.save(checkpoint_cpu, save_path)

                logger.info(f"[AsyncSaver] Checkpoint saved to {save_path}")
                self.queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:  # noqa: BLE001 - keep the saver thread alive
                logger.info(f"[AsyncSaver] Error saving checkpoint: {e}")
                self.queue.task_done()

    def save_checkpoint(self, checkpoint: dict, save_path: Path):
        """Queue a checkpoint for asynchronous saving.

        Parameters
        ----------
        checkpoint : dict
            The checkpoint dictionary to save.
        save_path : Path
            Path where to save the checkpoint.
        """
        self.queue.put((checkpoint, save_path), timeout=60)

    def shutdown(self, timeout: float = 300):
        """Shutdown the async saver and wait for pending saves to complete.

        Parameters
        ----------
        timeout : float
            Maximum time to wait for pending saves (in seconds).
        """
        logger.info(
            f"[AsyncSaver] Shutting down, waiting for {self.queue.qsize()} pending saves..."
        )

        # Wait for all pending saves to complete
        self.queue.join()

        # Signal the worker to stop
        self.stop_event.set()
        self.queue.put((None, None))

        self.worker_thread.join(timeout=timeout)

        if self.worker_thread.is_alive():
            logger.info("[AsyncSaver] Warning: Worker thread did not finish in time")
        else:
            logger.info("[AsyncSaver] Shutdown complete")

    def __del__(self):
        """Ensure proper cleanup on deletion."""
        if hasattr(self, "worker_thread") and self.worker_thread.is_alive():
            self.shutdown(timeout=30)


def set_hyperparams(
    optimizer: torch.optim.Optimizer,
    iterator: int,
    lr_scheduler: Sequence[float] | None = None,
    wd_scheduler: Sequence[float] | None = None,
) -> None:
    """Set learning rate and weight decay for the optimizer.

    Parameters
    ----------
    optimizer : torch.optim.Optimizer
        The optimizer for training.
    iterator : int
        The current iteration in the training loop.
    lr_scheduler : Sequence[float], optional
        Learning rate schedule to apply (default is None).
    wd_scheduler : Sequence[float], optional
        Weight decay schedule to apply (default is None).
    """

    for param_group in optimizer.param_groups:
        if lr_scheduler is not None:
            scale = param_group.get("lr_scale", 1.0)
            param_group["lr"] = lr_scheduler[iterator] * scale

        if wd_scheduler is not None:
            if param_group.get("skip_weight_decay", False):
                continue

            param_group["weight_decay"] = wd_scheduler[iterator]


def load_checkpoint(
    checkpoint_path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    strict: bool = False,
) -> tuple[nn.Module, torch.optim.Optimizer, int, float, list, int, float]:
    """Load a checkpoint and restore model/optimizer state.

    Supports two checkpoint formats:
    1. Training checkpoints: dict with 'model', 'optimizer', 'step', etc.
    2. Pretrained/SSL checkpoints: dict with 'network_weights' or direct state_dict

    Parameters
    ----------
    checkpoint_path : Path
        Path to the checkpoint file.
    model : nn.Module
        Model to load weights into.
    optimizer : torch.optim.Optimizer
        Optimizer to load state into (only for training checkpoints).
    device : torch.device
        Device to load the checkpoint onto.
    strict : bool
        Whether to strictly enforce that the keys in state_dict match.

    Returns
    -------
    tuple
        (model, optimizer, start_step, best_loss, history, sampler_epoch, best_metric_for_early_stopping)
    """
    logger.info(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Determine checkpoint format and extract model weights
    is_training_checkpoint = isinstance(checkpoint, dict) and "model" in checkpoint

    if is_training_checkpoint:
        # Standard training checkpoint format
        state_dict = checkpoint["model"]
        logger.info("Detected training checkpoint format (contains 'model' key)")
    elif isinstance(checkpoint, dict) and "network_weights" in checkpoint:
        # SSL/pretrained checkpoint format (e.g., from nnU-Net SSL)
        state_dict = checkpoint["network_weights"]
        logger.info(
            "Detected SSL/pretrained checkpoint format (contains 'network_weights' key)"
        )

        # Check for adaptation plan info
        if "nnssl_adaptation_plan" in checkpoint:
            plan = checkpoint["nnssl_adaptation_plan"]
            if "key_to_encoder" in plan:
                logger.info(f"  Encoder key mapping: {plan['key_to_encoder']}")
    elif isinstance(checkpoint, dict):
        # Assume the dict itself is the state_dict (raw weights)
        state_dict = checkpoint
        logger.info("Detected raw state_dict checkpoint format")
    else:
        raise ValueError(
            f"Unknown checkpoint format. Expected dict with 'model' or 'network_weights' key, "
            f"or a raw state_dict. Got: {type(checkpoint)}"
        )

    # Load model weights
    model_to_load = model.module if is_dist_avail_and_initialized() else model

    # Try to load with filtering for mismatched keys (e.g., classifier head)
    model_state = model_to_load.state_dict()
    filtered_state_dict = {}
    skipped_keys = []

    for k, v in state_dict.items():
        if k in model_state:
            if v.shape == model_state[k].shape:
                filtered_state_dict[k] = v
            else:
                skipped_keys.append(
                    f"{k} (shape mismatch: {v.shape} vs {model_state[k].shape})"
                )
        else:
            skipped_keys.append(f"{k} (not in model)")

    missing_keys = set(model_state.keys()) - set(filtered_state_dict.keys())

    if skipped_keys:
        logger.info(f"  Skipped {len(skipped_keys)} keys from checkpoint:")
        for key in skipped_keys[:10]:  # Show first 10
            logger.info(f"    - {key}")
        if len(skipped_keys) > 10:
            logger.info(f"    ... and {len(skipped_keys) - 10} more")

    if missing_keys:
        logger.info(
            f"  Missing {len(missing_keys)} keys in checkpoint (will use random init):"
        )
        for key in list(missing_keys)[:10]:  # Show first 10
            logger.info(f"    - {key}")
        if len(missing_keys) > 10:
            logger.info(f"    ... and {len(missing_keys) - 10} more")

    # Load the filtered state dict
    model_to_load.load_state_dict(filtered_state_dict, strict=strict)
    logger.info(f"  Loaded {len(filtered_state_dict)}/{len(model_state)} parameters")

    # For training checkpoints, also restore optimizer and training state
    if is_training_checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])

        start_step = checkpoint.get("step", 0)
        best_loss = checkpoint.get(
            "loss",
            checkpoint.get("best_loss", checkpoint.get("val_loss", float("inf"))),
        )
        history = checkpoint.get("history", [])
        sampler_epoch = checkpoint.get("sampler_epoch", 0)
        best_metric_for_early_stopping = checkpoint.get(
            "best_metric_for_early_stopping", float("inf")
        )
        logger.info(
            f"Resuming training from step {start_step} with best loss: {best_loss:.4f}"
        )
    else:
        # For pretrained checkpoints, start fresh training
        start_step = 0
        best_loss = float("inf")
        history = []
        sampler_epoch = 0
        best_metric_for_early_stopping = float("inf")
        logger.info("Starting fine-tuning from pretrained weights (step 0)")

    return (
        model,
        optimizer,
        start_step,
        best_loss,
        history,
        sampler_epoch,
        best_metric_for_early_stopping,
    )


def compute_metrics(
    preds: torch.Tensor | list[np.ndarray], gts: torch.Tensor | list[np.ndarray]
) -> tuple[float, float, float, float, float, np.ndarray, float]:
    """Optimized metric computation.

    Parameters
    ----------
    preds : torch.Tensor or List[np.ndarray]
        Predictions (probabilities)
    gts : torch.Tensor or List[np.ndarray]
        Ground truth labels

    Returns
    -------
    Tuple[float, float, float, float, float, np.ndarray, float]
        Accuracy, balanced accuracy, ROC-AUC, PR-AUC, macro F1, per-class F1, and MCC
    """
    # Handle torch tensors efficiently
    if isinstance(preds, torch.Tensor):
        # Compute argmax on GPU if available
        preds_argmax = preds.argmax(dim=1)
        gts_argmax = gts.argmax(dim=1)

        # Accuracy on GPU
        acc = (preds_argmax == gts_argmax).float().mean().item()

        # Move to CPU once for sklearn
        preds_np = preds.cpu().numpy()
        preds_argmax_np = preds_argmax.cpu().numpy()
        gts_argmax_np = gts_argmax.cpu().numpy()
    else:
        # Legacy list handling
        if isinstance(preds, list) and all(isinstance(p, np.ndarray) for p in preds):
            preds_np = np.concatenate(preds, axis=0)
        else:
            preds_np = preds

        if isinstance(gts, list) and all(isinstance(g, np.ndarray) for g in gts):
            gts_np = np.concatenate(gts, axis=0)
        else:
            gts_np = gts

        preds_argmax_np = np.argmax(preds_np, axis=1)
        gts_argmax_np = np.argmax(gts_np, axis=1)
        acc = accuracy_score(gts_argmax_np, preds_argmax_np)

    # Balanced accuracy
    bacc = balanced_accuracy_score(gts_argmax_np, preds_argmax_np)

    # Vectorized AUC computation
    num_classes = preds_np.shape[1]
    roc_auc_scores = []
    pr_auc_scores = []

    # Check for NaN values in predictions
    if np.isnan(preds_np).any():
        # If there are NaN values, replace them with uniform probabilities
        # This prevents crashes during validation while still allowing training to continue
        nan_mask = np.isnan(preds_np).any(axis=1)
        preds_np[nan_mask] = 1.0 / num_classes
        if get_rank() == 0:
            logger.info(
                f"Warning: Found {nan_mask.sum()} samples with NaN predictions, replacing with uniform probabilities"
            )

    for class_idx in range(num_classes):
        binary_gt = (gts_argmax_np == class_idx).astype(np.int32)
        n_positive = binary_gt.sum()

        # Only calculate if we have both classes
        if 0 < n_positive < len(binary_gt):
            # Additional safety check for NaN in class predictions
            class_preds = preds_np[:, class_idx]
            if not np.isnan(class_preds).any():
                roc_auc_scores.append(
                    roc_auc_score(binary_gt, class_preds, average="weighted")
                )
                pr_auc_scores.append(
                    average_precision_score(binary_gt, class_preds, average="weighted")
                )

    roc_auc = np.mean(roc_auc_scores) if roc_auc_scores else 0.5
    pr_auc = np.mean(pr_auc_scores) if pr_auc_scores else 0.5

    macro_f1 = f1_score(
        gts_argmax_np, preds_argmax_np, average="weighted", zero_division=0
    )

    per_class_f1 = f1_score(
        gts_argmax_np, preds_argmax_np, average=None, zero_division=0
    )

    mcc = matthews_corrcoef(gts_argmax_np, preds_argmax_np)

    return acc, bacc, roc_auc, pr_auc, macro_f1, per_class_f1, mcc


def _all_gather_variable_length(t: torch.Tensor) -> list[torch.Tensor]:
    """Gather tensors with potentially different leading dimensions across ranks."""
    world_size = dist.get_world_size()

    local_size = torch.tensor([t.size(0)], device=t.device, dtype=torch.long)
    sizes = [torch.zeros_like(local_size) for _ in range(world_size)]
    dist.all_gather(sizes, local_size)
    sizes = [s.item() for s in sizes]
    max_size = max(sizes) if sizes else 0

    pad = max_size - t.size(0)
    if pad > 0:
        pad_shape = (pad, *t.shape[1:])
        t = torch.cat([t, t.new_zeros(pad_shape)], dim=0)

    gathered = [t.new_zeros((max_size, *t.shape[1:])) for _ in sizes]
    dist.all_gather(gathered, t)

    return [g[:sz] for g, sz in zip(gathered, sizes)]


def compute_distributed_metrics(
    all_preds: torch.Tensor,
    all_gts: torch.Tensor,
    device: torch.device,
) -> tuple[float, float, float, float, np.ndarray, float]:
    """Compute metrics in distributed setting with memory-efficient aggregation.

    Instead of gathering all predictions/labels, we compute per-class statistics
    locally and aggregate them across ranks. For AUC, we still need to gather
    on rank 0, but with optimized communication.

    Parameters
    ----------
    all_preds : torch.Tensor
        Local predictions (N, num_classes)
    all_gts : torch.Tensor
        Local ground truth labels (N, num_classes)
    device : torch.device
        Device for tensor operations

    Returns
    -------
    Tuple[float, float, float, float, np.ndarray, float]
        Accuracy, balanced accuracy, AUC, macro F1, per-class F1, and MCC
    """
    num_classes = all_preds.shape[1]

    # Gather predictions across all ranks
    if is_dist_avail_and_initialized():
        gathered_preds = _all_gather_variable_length(all_preds)
        gathered_gts = _all_gather_variable_length(all_gts)

        # Only rank 0 computes metrics
        if get_rank() == 0:
            # Concatenate gathered tensors
            global_preds = torch.cat(gathered_preds, dim=0)
            global_gts = torch.cat(gathered_gts, dim=0)

            # Compute all metrics at once
            acc, bacc, roc_auc, pr_auc, macro_f1, per_class_f1, mcc = compute_metrics(
                global_preds, global_gts
            )
        else:
            acc = 0.0
            bacc = 0.0
            roc_auc = 0.0
            pr_auc = 0.0
            macro_f1 = 0.0
            per_class_f1 = np.zeros(num_classes)
            mcc = 0.0

        # Broadcast metrics to all ranks
        metrics_tensor = torch.tensor(
            [acc, bacc, roc_auc, pr_auc, macro_f1, mcc],
            device=device,
            dtype=torch.float32,
        )
        dist.broadcast(metrics_tensor, src=0)
        acc, bacc, roc_auc, pr_auc, macro_f1, mcc = metrics_tensor.tolist()

        # Broadcast per-class F1 with proper initialization on all ranks
        per_class_f1_tensor = torch.zeros(
            num_classes, device=device, dtype=torch.float32
        )
        if get_rank() == 0:
            per_class_f1_tensor.copy_(
                torch.tensor(per_class_f1, device=device, dtype=torch.float32)
            )
        dist.broadcast(per_class_f1_tensor, src=0)
        per_class_f1 = per_class_f1_tensor.cpu().numpy()
    else:
        # Single GPU case
        acc, bacc, roc_auc, pr_auc, macro_f1, per_class_f1, mcc = compute_metrics(
            all_preds, all_gts
        )

    return acc, bacc, roc_auc, pr_auc, macro_f1, per_class_f1, mcc


def validation(
    model: nn.Module,
    val_loader: DataLoader,
    loss_fn: nn.Module,
) -> tuple[
    float,
    float,
    float,
    float,
    float,
    float,
    np.ndarray,
    float,
    float | None,
    float | None,
]:
    """Validate the model on the validation dataset.

    Parameters
    ----------
    model : nn.Module
        The model to validate.
    val_loader : DataLoader
        DataLoader for the validation dataset.
    loss_fn : nn.Module
        The loss function to optimize.

    Returns
    -------
    Tuple[float, float, float, float, float, float, np.ndarray, float, Optional[float], Optional[float]]
        Average validation loss, accuracy, balanced accuracy, ROC-AUC, PR-AUC, macro F1,
        per-class F1, MCC, and (optional) fine/coarse hierarchical losses.
    """
    model.eval()
    num_classes = len(w.config.DISEASES)

    # Preallocate tensors for efficiency
    if is_dist_avail_and_initialized():
        local_samples_estimate = len(val_loader.dataset) // get_world_size() + 1
    else:
        local_samples_estimate = len(val_loader.dataset)

    # Preallocate tensors on GPU directly (more efficient than pinned->GPU)
    all_preds = torch.zeros(
        local_samples_estimate, num_classes, dtype=torch.float32, device=device
    )
    all_gts = torch.zeros(
        local_samples_estimate, num_classes, dtype=torch.float32, device=device
    )

    total_loss = torch.tensor(0.0, device=device, dtype=torch.float64)
    total_fine_loss = torch.tensor(0.0, device=device, dtype=torch.float64)
    total_coarse_loss = torch.tensor(0.0, device=device, dtype=torch.float64)
    track_hierarchical = isinstance(loss_fn, HierarchicalRiskAwareLoss)
    sample_idx = 0

    # Use inference_mode for better performance than no_grad (PyTorch 1.9+)
    with torch.inference_mode():
        for x_val, y_val in val_loader:
            x_val = x_val.to(
                device, memory_format=torch.channels_last_3d, non_blocking=True
            )
            y_val = y_val.to(device, non_blocking=True)
            batch_size = x_val.size(0)

            # Forward pass
            y_pred = model(x_val)
            loss = loss_fn(y_pred, y_val)

            # Apply softmax to get probabilities
            y_pred_probs = torch.nn.functional.softmax(y_pred, dim=1)

            # Store in preallocated tensors (no concatenation!)
            all_preds[sample_idx : sample_idx + batch_size] = y_pred_probs
            all_gts[sample_idx : sample_idx + batch_size] = y_val

            total_loss += loss.item() * batch_size
            if track_hierarchical:
                total_fine_loss += loss_fn.last_fine_loss * batch_size
                total_coarse_loss += loss_fn.last_coarse_loss * batch_size
            sample_idx += batch_size

    # Trim to actual size
    all_preds = all_preds[:sample_idx]
    all_gts = all_gts[:sample_idx]
    total_samples = sample_idx

    if is_dist_avail_and_initialized():
        total_samples_tensor = torch.tensor(
            total_samples, device=device, dtype=torch.int64
        )
        dist.all_reduce(total_loss)
        if track_hierarchical:
            dist.all_reduce(total_fine_loss)
            dist.all_reduce(total_coarse_loss)
        dist.all_reduce(total_samples_tensor)
        total_samples = total_samples_tensor.item()
        total_loss = total_loss.item()
        if track_hierarchical:
            total_fine_loss = total_fine_loss.item()
            total_coarse_loss = total_coarse_loss.item()

        # Use memory-efficient distributed metrics computation
        acc, bacc, roc_auc, pr_auc, macro_f1, per_class_f1, mcc = (
            compute_distributed_metrics(all_preds, all_gts, device)
        )
    else:
        # Move loss to CPU for non-distributed case
        total_loss = total_loss.item()
        if track_hierarchical:
            total_fine_loss = total_fine_loss.item()
            total_coarse_loss = total_coarse_loss.item()
        if total_samples > 0:
            acc, bacc, roc_auc, pr_auc, macro_f1, per_class_f1, mcc = compute_metrics(
                all_preds, all_gts
            )
        else:
            acc, bacc, roc_auc, pr_auc, macro_f1, mcc = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
            per_class_f1 = np.zeros(num_classes)

    val_loss = total_loss / total_samples if total_samples > 0 else 0.0
    if track_hierarchical and total_samples > 0:
        val_fine_loss = total_fine_loss / total_samples
        val_coarse_loss = total_coarse_loss / total_samples
    else:
        val_fine_loss = None
        val_coarse_loss = None

    return (
        val_loss,
        acc,
        bacc,
        roc_auc,
        pr_auc,
        macro_f1,
        per_class_f1,
        mcc,
        val_fine_loss,
        val_coarse_loss,
    )


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    history: list,
    loss: float,
    save_path: Path,
    ema: EMAModel | None = None,
    async_saver: AsyncCheckpointSaver | None = None,
    sampler_epoch: int = 0,
    best_metric_for_early_stopping: float = float("inf"),
) -> None:
    """Save a checkpoint with model weights and training state.

    Parameters
    ----------
    model : nn.Module
        The model to save weights from
    optimizer : torch.optim.Optimizer
        The optimizer to save state from
    step : int
        Current global training step (optimizer updates completed)
    history : list
        Training history
    loss : float
        Loss value to record
    save_path : Path
        Path where to save the checkpoint
    ema : EMAModel, optional
        EMA model if used
    async_saver : AsyncCheckpointSaver, optional
        Async checkpoint saver instance (required for master process)
    sampler_epoch : int
        Current epoch for distributed sampler (for shuffle consistency)
    best_metric_for_early_stopping : float
        Best metric value for early stopping logic
    """
    if get_rank() != 0:
        # Save only the model on the master process
        return

    optimizer_state = optimizer.state_dict()

    param_groups = optimizer.param_groups

    checkpoint = {
        "model": ema.model_state if ema is not None else model.state_dict(),
        "optimizer": optimizer_state,
        "weight_decay": param_groups[0]["weight_decay"],
        "lr": param_groups[0]["lr"],
        "step": step,
        "history": history,
        "loss": loss,
        "sampler_epoch": sampler_epoch,
        "best_metric_for_early_stopping": best_metric_for_early_stopping,
    }

    # Async saver is always used (required)
    async_saver.save_checkpoint(checkpoint, save_path)


def save_best_n_models(
    best_models: dict[int, float],
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    cur_metric: float,
    history: list,
    save_dir: Path,
    ema: EMAModel | None,
    async_saver: AsyncCheckpointSaver | None = None,
    sampler_epoch: int = 0,
    best_metric_for_early_stopping: float = float("inf"),
    metric_mode: str = "min",
) -> dict[int, float]:
    """
    Save the N best models based on a validation metric.

    Parameters
    ----------
    best_models : dict[int, float]
        Dictionary of the best models with step as key and metric value as value.
        Should be sorted by metric (ascending for min mode, descending for max mode).
    model : nn.Module
        The model to save.
    optimizer : torch.optim.Optimizer
        The optimizer to save.
    step : int
        Current step number.
    cur_metric : float
        Current validation metric value.
    history : list
        Training history.
    save_dir : Path
        Directory to save the best models.
    ema : EMAModel, optional
        EMA model if used.
    async_saver : AsyncCheckpointSaver, optional
        Async checkpoint saver instance (required for master process).
    sampler_epoch : int
        Current epoch for distributed sampler.
    best_metric_for_early_stopping : float
        Best metric value for early stopping.
    metric_mode : str
        'min' if lower is better (loss), 'max' if higher is better (acc, bacc, etc.).

    Returns
    -------
    dict[int, float]
        Updated dictionary of the best models, sorted by metric.
    """
    if get_rank() != 0:
        logger.info(
            f"Rank {get_rank()} is not the master process. Skipping saving best models."
        )
        return best_models

    if step in best_models:
        logger.info(
            f"Model already exists for step {step} with metric {best_models[step]}"
        )
        return best_models

    # index of the new model /!\ best_models has to be sorted
    # For min mode (loss): sort ascending (lower is better)
    # For max mode (acc, bacc, etc.): sort descending (higher is better)
    reverse_sort = metric_mode == "max"
    best_models = dict(
        sorted(best_models.items(), key=lambda item: item[1], reverse=reverse_sort)
    )

    # Find where the new model should be inserted based on metric value
    position = 0
    for idx, (_, metric_val) in enumerate(best_models.items()):
        if metric_mode == "min":
            if cur_metric < metric_val:
                break
        else:  # max mode
            if cur_metric > metric_val:
                break
        position = idx + 1

    if position == len(best_models) and position >= w.config.KEEP_BEST_N:
        return best_models

    for i in range(min(len(best_models), w.config.KEEP_BEST_N - 1), position, -1):
        src_path = save_dir / f"model_{w.run.id}_{w.config.FOLD}_best{i - 1}.pt"
        dst_path = save_dir / f"model_{w.run.id}_{w.config.FOLD}_best{i}.pt"
        if src_path.exists():
            shutil.move(src_path, dst_path)

    save_path = save_dir / f"model_{w.run.id}_{w.config.FOLD}_best{position}.pt"
    save_checkpoint(
        model,
        optimizer,
        step,
        history,
        cur_metric,
        save_path,
        ema,
        async_saver,
        sampler_epoch,
        best_metric_for_early_stopping,
    )

    best_models[step] = cur_metric
    # Sort and keep best N models
    reverse_sort = metric_mode == "max"
    best_models = dict(
        sorted(best_models.items(), key=lambda item: item[1], reverse=reverse_sort)[
            : w.config.KEEP_BEST_N
        ]
    )

    return best_models


def training_loops(
    model: nn.Module,
    loss: nn.Module,
    optimizer: torch.optim.Optimizer,
    total_steps: int,
    train_loader: DataLoader,
    val_loader: DataLoader,
    lr_scheduler: np.ndarray,
    wd_scheduler: np.ndarray,
    save_dir: Path,
    start_step: int,
    best_loss: float,
    history: list,
    sampler_epoch: int = 0,
    best_metric_for_early_stopping: float = float("inf"),
) -> None:
    """Run the training loop using global optimization steps.

    Parameters
    ----------
    model : nn.Module
        The model to train, potentially wrapped in DDP.
    loss : nn.Module
        Loss function to optimize.
    optimizer : torch.optim.Optimizer
        Optimizer instance.
    total_steps : int
        Total number of optimizer steps to execute.
    train_loader : DataLoader
        Loader for training batches.
    val_loader : DataLoader
        Loader for validation batches.
    lr_scheduler : np.ndarray
        Precomputed learning-rate schedule indexed by step.
    wd_scheduler : np.ndarray
        Precomputed weight-decay schedule indexed by step.
    save_dir : Path
        Directory where checkpoints are written.
    start_step : int
        Step to resume from (0 for fresh runs).
    best_loss : float
        Best validation loss observed so far.
    history : list
        Accumulated logging history.
    sampler_epoch : int
        Starting epoch for distributed sampler (restored from checkpoint).
    best_metric_for_early_stopping : float
        Best metric value for early stopping (restored from checkpoint).
    """
    best_models = {} if w.config.KEEP_BEST_N else None

    # Early stopping initialization
    early_stopping_enabled = (
        w.config.EARLY_STOPPING_PATIENCE and w.config.EARLY_STOPPING_PATIENCE > 0
    )
    if early_stopping_enabled:
        steps_without_improvement = 0

        # Get early stopping metric and validate it
        early_stopping_metric = w.config.EARLY_STOPPING_METRIC.lower()
        valid_metrics = ["loss", "acc", "bacc", "roc_auc", "pr_auc", "macro_f1", "mcc"]
        if early_stopping_metric not in valid_metrics:
            if get_rank() == 0:
                logger.info(
                    f"[Warning] Invalid EARLY_STOPPING_METRIC '{early_stopping_metric}', defaulting to 'loss'"
                )
            early_stopping_metric = "loss"

        # Determine if metric should be minimized (loss) or maximized (acc, bacc, roc_auc, pr_auc, macro_f1, mcc)
        metric_mode = "min" if early_stopping_metric == "loss" else "max"

        if start_step == 0:
            if metric_mode == "min":
                # For loss-based metrics: start with infinity (worse is higher)
                best_metric_for_early_stopping = float("inf")
            else:
                # For accuracy-based metrics: start with -infinity (worse is lower)
                best_metric_for_early_stopping = float("-inf")

        min_delta = w.config.EARLY_STOPPING_MIN_DELTA

        if get_rank() == 0:
            logger.info("Early stopping enabled:")
            logger.info(f"  - Metric: {early_stopping_metric} (mode: {metric_mode})")
            logger.info(f"  - Patience: {w.config.EARLY_STOPPING_PATIENCE} steps")
            logger.info(f"  - Min delta: {min_delta:.6f} (improvement threshold)")
            logger.info(f"  - Best metric so far: {best_metric_for_early_stopping:.6f}")
    else:
        steps_without_improvement = 0
        min_delta = 0.0
        early_stopping_metric = "loss"
        metric_mode = "min"

    # Best model selection metric initialization
    valid_metrics = ["loss", "acc", "bacc", "roc_auc", "pr_auc", "macro_f1", "mcc"]
    best_model_metric = getattr(w.config, "METRIC_BEST_MODEL", "loss").lower()
    if best_model_metric not in valid_metrics:
        if get_rank() == 0:
            logger.info(
                f"[Warning] Invalid METRIC_BEST_MODEL '{best_model_metric}', defaulting to 'loss'"
            )
        best_model_metric = "loss"

    # Determine if best model metric should be minimized (loss) or maximized (acc, bacc, etc.)
    best_model_metric_mode = "min" if best_model_metric == "loss" else "max"

    if get_rank() == 0:
        logger.info(
            f"Best model selection: {best_model_metric} (mode: {best_model_metric_mode})"
        )

    ema = (
        EMAModel(
            model=model.module if is_dist_avail_and_initialized() else model,
            decay=w.config.EMA_DECAY,
            device=device,
        )
        if w.config.USE_EMA
        else None
    )

    async_saver = AsyncCheckpointSaver(max_queue_size=3) if get_rank() == 0 else None

    if async_saver and get_rank() == 0:
        logger.info("[AsyncSaver] Initialized asynchronous checkpoint saving")

    grad_accum_steps = max(1, w.config.GRADIENT_ACCUMULATION)
    effective_step = int(start_step)
    # Mixed precision: use lower initial scale for better stability
    # Default is 2^16=65536 which can cause gradient overflow in early training
    # We use 2^10=1024 and let it grow automatically (growth_interval=100)
    fp16_scaler = (
        torch.amp.GradScaler("cuda", init_scale=1024.0, growth_interval=100)
        if w.config.FP16
        else None
    )

    batches_per_epoch = max(1, len(train_loader))
    micro_batches_seen = effective_step * grad_accum_steps
    sampler_epoch = micro_batches_seen // batches_per_epoch

    if effective_step >= total_steps:
        if get_rank() == 0:
            logger.info(
                f"Requested {total_steps} steps but checkpoint already passed that mark; skipping training."
            )
        return

    optimizer.zero_grad(set_to_none=True)
    train_iter = iter(train_loader)

    def set_training_epoch(epoch: int):
        """Set epoch for sampler and dataset (e.g., MRIMixUp)."""
        if hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(int(epoch))
        if hasattr(train_loader.dataset, "set_epoch"):
            train_loader.dataset.set_epoch(int(epoch))

    if is_dist_avail_and_initialized():
        set_training_epoch(sampler_epoch)

    validation_frequency = max(1, w.config.VALIDATION_FREQUENCY)

    try:
        while effective_step < total_steps:
            set_hyperparams(optimizer, effective_step, lr_scheduler, wd_scheduler)

            # Accumulate gradients from all micro-batches
            skip_step_due_to_nan = False
            for micro_step in range(grad_accum_steps):
                try:
                    x_train, y_train = next(train_iter)
                except StopIteration:
                    sampler_epoch += 1
                    if is_dist_avail_and_initialized():
                        set_training_epoch(sampler_epoch)
                    train_iter = iter(train_loader)
                    x_train, y_train = next(train_iter)

                model.train()
                x_train, y_train = (
                    x_train.to(device, non_blocking=True),
                    y_train.to(device, non_blocking=True),
                )

                if x_train.ndim == 5:
                    x_train = x_train.to(memory_format=torch.channels_last_3d)

                is_final_micro_step = micro_step == grad_accum_steps - 1
                ddp_sync_context = (
                    model.no_sync()
                    if grad_accum_steps > 1
                    and hasattr(model, "no_sync")
                    and not is_final_micro_step
                    else nullcontext()
                )

                nan_in_microbatch = False
                with ddp_sync_context:
                    with torch.amp.autocast("cuda", enabled=fp16_scaler is not None):
                        y_pred = model(x_train)
                        batch_loss = loss(y_pred, y_train) / grad_accum_steps

                    local_nan = torch.isnan(batch_loss) or torch.isnan(y_pred).any()
                    if is_dist_avail_and_initialized():
                        nan_tensor = torch.tensor(
                            [int(local_nan)],
                            device=device,
                            dtype=torch.int32,
                        )
                        dist.all_reduce(nan_tensor, op=dist.ReduceOp.MAX)
                        local_nan = nan_tensor.item() > 0

                    if local_nan:
                        nan_in_microbatch = True
                        if get_rank() == 0:
                            logger.info(
                                f"WARNING at step {effective_step}: NaN detected in batch_loss or y_pred"
                            )
                            logger.info(
                                f"batch_loss: {batch_loss.item()}, "
                                f"y_pred has NaN: {torch.isnan(y_pred).any().item()}"
                            )
                        optimizer.zero_grad(set_to_none=True)
                    else:
                        if fp16_scaler is None:
                            batch_loss.backward()
                        else:
                            fp16_scaler.scale(batch_loss).backward()

                if nan_in_microbatch:
                    skip_step_due_to_nan = True
                    break

                micro_batches_seen += 1
                if is_dist_avail_and_initialized():
                    new_epoch = micro_batches_seen // batches_per_epoch
                    if new_epoch != sampler_epoch:
                        sampler_epoch = new_epoch
                        set_training_epoch(sampler_epoch)

            if skip_step_due_to_nan:
                continue

            # Standard optimizer: unscale (if FP16), clip, then step
            if w.config.GRADIENT_CLIP:
                if fp16_scaler is None:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        max_norm=w.config.GRADIENT_CLIP,
                    )
                else:
                    fp16_scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        max_norm=w.config.GRADIENT_CLIP,
                    )

            if fp16_scaler is None:
                optimizer.step()
            else:
                fp16_scaler.step(optimizer)
                fp16_scaler.update()

            optimizer.zero_grad(set_to_none=True)

            # Update EMA model
            if ema is not None:
                ema.update(model.module if is_dist_avail_and_initialized() else model)

            effective_step += 1

            # Validation and checkpointing
            if (
                effective_step % validation_frequency != 0
                and effective_step != total_steps
            ):
                continue

            if ema is not None:
                ema.apply_to(model.module if is_dist_avail_and_initialized() else model)

            (
                val_loss,
                val_acc,
                val_bacc,
                val_roc_auc,
                val_pr_auc,
                val_macro_f1,
                val_per_class_f1,
                val_mcc,
                val_fine_loss,
                val_coarse_loss,
            ) = validation(model, val_loader, loss)

            if ema is not None:
                ema.restore(model.module if is_dist_avail_and_initialized() else model)

            if get_rank() == 0:
                logger.info(
                    f"{datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} Step {effective_step}",
                    f"\n\tVal: loss {val_loss:.4f}, acc {val_acc:.4f}, bacc {val_bacc:.4f}, roc_auc {val_roc_auc:.4f}, pr_auc {val_pr_auc:.4f}, macro_f1 {val_macro_f1:.4f}, mcc {val_mcc:.4f}",
                )

                per_class_f1_str = ", ".join(
                    [
                        f"{w.config.DISEASES[i]}: {val_per_class_f1[i]:.4f}"
                        for i in range(len(w.config.DISEASES))
                    ]
                )
                logger.info(f"\t     Per-class F1: {per_class_f1_str}")

                wandb_log = {
                    "Validation loss": val_loss,
                    "Validation accuracy": val_acc,
                    "Validation balanced accuracy": val_bacc,
                    "Validation ROC AUC": val_roc_auc,
                    "Validation PR AUC": val_pr_auc,
                    "Validation macro F1": val_macro_f1,
                    "Validation MCC": val_mcc,
                    "Learning rate": optimizer.param_groups[0]["lr"],
                    "Weight decay": optimizer.param_groups[0]["weight_decay"],
                }

                if val_fine_loss is not None and val_coarse_loss is not None:
                    wandb_log["Validation fine loss"] = val_fine_loss
                    wandb_log["Validation coarse loss"] = val_coarse_loss

                for i, disease in enumerate(w.config.DISEASES):
                    wandb_log[f"Validation F1 {disease}"] = val_per_class_f1[i]

                w.log(wandb_log, step=effective_step)
                history.append(wandb_log)

            save_checkpoint(
                model.module if is_dist_avail_and_initialized() else model,
                optimizer,
                effective_step,
                history,
                val_loss,
                save_dir / f"model_{w.run.id}_{w.config.FOLD}_last.pt",
                ema,
                async_saver,
                sampler_epoch,
                best_metric_for_early_stopping,
            )

            # Get current metric value for best model selection
            metric_values = {
                "loss": val_loss,
                "acc": val_acc,
                "bacc": val_bacc,
                "roc_auc": val_roc_auc,
                "pr_auc": val_pr_auc,
                "macro_f1": val_macro_f1,
                "mcc": val_mcc,
            }
            current_best_model_metric = metric_values[best_model_metric]

            if not w.config.KEEP_BEST_N:
                # Single best model mode
                is_better = (
                    current_best_model_metric < best_loss
                    if best_model_metric_mode == "min"
                    else current_best_model_metric > best_loss
                )
                if is_better:
                    save_checkpoint(
                        model.module if is_dist_avail_and_initialized() else model,
                        optimizer,
                        effective_step,
                        history,
                        current_best_model_metric,
                        save_dir / f"model_{w.run.id}_{w.config.FOLD}.pt",
                        ema,
                        async_saver,
                        sampler_epoch,
                        best_metric_for_early_stopping,
                    )
                    best_loss = current_best_model_metric
            elif get_rank() == 0:
                # Keep best N models mode
                best_models = save_best_n_models(
                    best_models,
                    model.module if is_dist_avail_and_initialized() else model,
                    optimizer,
                    effective_step,
                    current_best_model_metric,
                    history,
                    save_dir,
                    ema,
                    async_saver,
                    sampler_epoch,
                    best_metric_for_early_stopping,
                    best_model_metric_mode,
                )

            # Early stopping logic
            if early_stopping_enabled:
                # Get current metric value based on selected metric
                metric_values = {
                    "loss": val_loss,
                    "acc": val_acc,
                    "bacc": val_bacc,
                    "roc_auc": val_roc_auc,
                    "pr_auc": val_pr_auc,
                    "macro_f1": val_macro_f1,
                    "mcc": val_mcc,
                }
                current_metric_value = metric_values[early_stopping_metric]

                # Compute improvement based on metric mode
                if metric_mode == "min":
                    # For loss: improvement = best - current (positive means improvement)
                    improvement = best_metric_for_early_stopping - current_metric_value
                else:
                    # For acc/bacc/auc: improvement = current - best (positive means improvement)
                    improvement = current_metric_value - best_metric_for_early_stopping

                if improvement > min_delta:
                    # Significant improvement detected
                    best_metric_for_early_stopping = current_metric_value
                    steps_without_improvement = 0
                    if get_rank() == 0:
                        logger.info(
                            f"[Early Stopping] New best {early_stopping_metric}: {best_metric_for_early_stopping:.4f} "
                            f"(improvement: {improvement:.6f}, threshold: {min_delta:.6f}), resetting patience counter"
                        )
                else:
                    # No significant improvement
                    steps_without_improvement += validation_frequency
                    if get_rank() == 0:
                        if improvement > 0:
                            logger.info(
                                f"[Early Stopping] Insufficient improvement in {early_stopping_metric}: "
                                f"{improvement:.6f} < {min_delta:.6f} "
                                f"({steps_without_improvement}/{w.config.EARLY_STOPPING_PATIENCE} steps)"
                            )
                        else:
                            action_verb = (
                                "increased" if metric_mode == "min" else "decreased"
                            )
                            logger.info(
                                f"[Early Stopping] {early_stopping_metric.capitalize()} {action_verb}: {abs(improvement):.6f} "
                                f"({steps_without_improvement}/{w.config.EARLY_STOPPING_PATIENCE} steps)"
                            )

                    if steps_without_improvement >= w.config.EARLY_STOPPING_PATIENCE:
                        if get_rank() == 0:
                            logger.info(f"\n{'=' * 80}")
                            logger.info(
                                f"[Early Stopping] Stopping training at step {effective_step}"
                            )
                            logger.info(
                                f"No significant improvement in {early_stopping_metric} (threshold: {min_delta:.6f}) for {steps_without_improvement} steps"
                            )
                            logger.info(
                                f"Best {early_stopping_metric}: {best_metric_for_early_stopping:.4f}"
                            )
                            logger.info(
                                f"Current {early_stopping_metric}: {current_metric_value:.4f}"
                            )
                            logger.info(f"{'=' * 80}\n")
                        # Synchronize before breaking to ensure all ranks exit together
                        if is_dist_avail_and_initialized():
                            dist.barrier()
                        break

            # Synchronize all ranks after validation/checkpointing to prevent NCCL timeouts.
            # Rank 0 performs checkpoint saving (async I/O) and logging which can take
            # variable time. Without this barrier, other ranks may advance to the next
            # training iteration and issue collective operations (e.g., all_reduce for
            # NaN checks) before rank 0 is ready, causing a deadlock.
            if is_dist_avail_and_initialized():
                dist.barrier()

        # Update BN statistics for EMA model at the end of training
        if ema is not None and w.config.UPDATE_BN_STATS:
            ema.update_bn_stats(
                model.module if is_dist_avail_and_initialized() else model, train_loader
            )

    finally:
        # Ensure all pending checkpoint saves complete before exiting
        if async_saver is not None:
            async_saver.shutdown()


def train(save_dir: Path, fold: int) -> None:
    """Train the model for one fold of the dataset.

    Parameters
    ----------
    save_dir : Path
        The path of the directory where the model will be saved.
    fold : int
        The current fold number for cross-validation.
    """
    metadata_train, metadata_val, metadata_test, metadata_all = get_train_val_test(
        Path(args.training_csv_dir), fold, w.config.KFOLD, split=w.config.SPLIT
    )

    if get_rank() == 0:
        logger.info("\n=== Dataset Sizes ===")
        logger.info(f"In-domain Training set: {len(metadata_train)} samples")
        logger.info(f"In-domain Validation set: {len(metadata_val)} samples")
        logger.info(f"In-domain Test set: {len(metadata_test)} samples")
        logger.info(f"In-domain Total: {len(metadata_all)} samples")

        if "Diagnosis" in metadata_train.columns:
            logger.info("\n=== Training Class Distribution ===")
            class_counts = metadata_train["Diagnosis"].value_counts().sort_index()
            for cls, count in class_counts.items():
                logger.info(
                    f"  {cls}: {count} samples ({count / len(metadata_train) * 100:.1f}%)"
                )

            logger.info("\n=== Validation Class Distribution ===")
            class_counts = metadata_val["Diagnosis"].value_counts().sort_index()
            for cls, count in class_counts.items():
                logger.info(
                    f"  {cls}: {count} samples ({count / len(metadata_val) * 100:.1f}%)"
                )

            logger.info("\n=== Test Class Distribution ===")
            class_counts = metadata_test["Diagnosis"].value_counts().sort_index()
            for cls, count in class_counts.items():
                logger.info(
                    f"  {cls}: {count} samples ({count / len(metadata_test) * 100:.1f}%)"
                )

            # Calculate imbalance ratio for each split
            logger.info("\n=== Class Imbalance Ratios ===")
            train_counts = metadata_train["Diagnosis"].value_counts()
            val_counts = metadata_val["Diagnosis"].value_counts()
            test_counts = metadata_test["Diagnosis"].value_counts()

            train_imb = train_counts.max() / train_counts.min()
            val_imb = val_counts.max() / val_counts.min() if len(val_counts) > 1 else 0
            test_imb = (
                test_counts.max() / test_counts.min() if len(test_counts) > 1 else 0
            )

            logger.info(f"  Training imbalance ratio: {train_imb:.2f}:1")
            logger.info(f"  Validation imbalance ratio: {val_imb:.2f}:1")
            logger.info(f"  Test imbalance ratio: {test_imb:.2f}:1")

        logger.info("=" * 30)

        logger.info("\n=== Architecture Features ===")
        use_rope = getattr(w.config, "USE_ROPE", False)
        use_relative_pos_bias = getattr(w.config, "USE_RELATIVE_POS_BIAS", True)
        use_diff_attn = getattr(w.config, "USE_DIFF_ATTN", False)

        logger.info(
            f"RoPE (Rotary Position Embedding): {'Enabled' if use_rope else 'Disabled'}"
        )
        if use_relative_pos_bias:
            logger.info(
                f"Relative Position Bias: {'Enabled' if use_relative_pos_bias else 'Disabled'}"
            )
        logger.info(
            f"Differential Attention: {'Enabled' if use_diff_attn else 'Disabled'}"
        )

        logger.info(f"Attention heads per stage: {w.config.HEADS}")

        logger.info("=" * 30)

    preprocess_data_dir = Path(args.intermediate_dir) / "train"
    preprocess_data_dir.mkdir(parents=True, exist_ok=True)

    if all(
        (preprocess_data_dir / f"{subject}.pt").exists()
        for subject in metadata_all["Subject"]
    ):
        if get_rank() == 0:
            logger.info(
                f"All preprocessed data exists in {preprocess_data_dir}. Skipping."
            )
    else:
        if get_rank() == 0:
            logger.info("Preprocessing data...")

        # For the preprocessing step, we still want to use all data
        metadata_all = metadata_all.sort_values(by=["Subject"]).reset_index(drop=True)

        # Get world size and rank
        world_size = get_world_size()
        rank = get_rank()

        # Split the metadata among processes
        metadata_subset = np.array_split(metadata_all, world_size)[rank]

        data_prepa = DataPrepa(
            metadata_subset,
            has_seg=w.config.IN_CHANNELS > 1,
            preprocess_data_dir=preprocess_data_dir,
            device=device,
        )
        data_prepa.preprocess_data(
            crop=w.config.IMG_SIZE,
            downsample=False,
            tqdm_kwargs={
                "position": rank,
                "desc": f"Process {rank}/{world_size - 1} preprocessing",
                "dynamic_ncols": True,
            },
        )

    # Wait for all processes to finish writing before proceeding
    if is_dist_avail_and_initialized():
        dist.barrier()

    # Configure dataset loading strategy
    preload_data = w.config.PRELOAD_DATA

    if get_rank() == 0:
        logger.info("\n=== Dataset Configuration ===")
        if preload_data:
            logger.info("Data loading: PRELOAD (all data in RAM)")
        else:
            logger.info(
                "Data loading: ON-DEMAND (load from disk with DataLoader prefetching)"
            )
        logger.info("=" * 30 + "\n")

    base_seed_value = getattr(w.config, "SEED", None)
    if base_seed_value in (None, False):
        dataloader_seed = None
    else:
        dataloader_seed = (int(base_seed_value) + get_rank()) % _MAX_UINT32

    worker_init_fn = None
    train_generator: torch.Generator | None = None
    val_generator: torch.Generator | None = None

    if dataloader_seed is not None:
        if get_rank() == 0:
            logger.info(
                f"Seeding data pipeline with base seed {int(base_seed_value)} "
                f"(rank-adjusted seed {dataloader_seed})."
            )

        def _seed_worker(worker_id: int) -> None:
            worker_seed = (dataloader_seed + worker_id) % _MAX_UINT32
            np.random.seed(worker_seed)
            random.seed(worker_seed)
            torch.manual_seed(worker_seed)
            if monai_set_determinism is not None:
                monai_set_determinism(seed=worker_seed)

        worker_init_fn = _seed_worker

        train_generator = torch.Generator()
        train_generator.manual_seed(dataloader_seed)

        val_generator = torch.Generator()
        val_generator.manual_seed((dataloader_seed + 1) % _MAX_UINT32)

    include_resize = not (preload_data and w.config.RESHAPE_SIZE)

    # Build training transforms based on USE_EXTENDED_DATA_AUGMENTATION
    # Determine if multimodal (T1 + Segmentation)
    is_multimodal = w.config.IN_CHANNELS > 1

    if w.config.USE_EXTENDED_DATA_AUGMENTATION:
        # Extended augmentation: all geometric + intensity transforms
        train_transforms_list = []

        if include_resize:
            train_transforms_list.append(
                Resize(w.config.RESHAPE_SIZE)
                if w.config.RESHAPE_SIZE
                else Resize(w.config.IMG_SIZE)
            )

        train_transforms_list.append(
            MultiChannelRandAffine(
                prob=0.5,
                rotate_range=(np.deg2rad(-5), np.deg2rad(5)),
                scale_range=(-0.05, 0.05),
                translate_range=(-5, 5),
                padding_mode="border",
            )
            if is_multimodal
            else RandAffine(
                prob=0.5,
                rotate_range=(np.deg2rad(-5), np.deg2rad(5)),
                scale_range=(-0.05, 0.05),
                translate_range=(-5, 5),
                padding_mode="border",
            )
        )

        if is_multimodal:
            # For multimodal input, we apply some geometric transforms that are safe for both modalities using ChannelSelectiveTransform
            train_transforms_list.extend(
                [
                    OneOf(
                        [
                            Identity(),
                            RandSpatialCrop(
                                roi_size=tuple(
                                    int(w * 0.9) for w in w.config.RESHAPE_SIZE
                                )
                                if w.config.RESHAPE_SIZE
                                else tuple(int(w * 0.9) for w in w.config.IMG_SIZE),
                                random_center=True,
                                random_size=False,
                            ),
                        ],
                        weights=[0.3, 0.7],
                    ),
                    ChannelSelectiveTransform(
                        AdaptiveRicianNoise(prob=0.2), channels=[0]
                    ),
                    ChannelSelectiveTransform(
                        AdaptiveGaussianNoise(prob=0.2, noise_factor=0.1), channels=[0]
                    ),
                    ChannelSelectiveTransform(RandBiasField(prob=0.3), channels=[0]),
                    ChannelSelectiveTransform(
                        RandAdjustContrast(prob=0.4, gamma=(0.6, 1.4)), channels=[0]
                    ),
                    ChannelSelectiveTransform(
                        RandScaleIntensity(prob=0.3, factors=(-0.5, 1.0)), channels=[0]
                    ),
                    ChannelSelectiveTransform(
                        RandHistogramShift(prob=0.2, num_control_points=(3, 5)),
                        channels=[0],
                    ),
                    ChannelSelectiveTransform(
                        RandGibbsNoise(prob=0.3, alpha=(0.6, 0.8)), channels=[0]
                    ),
                    ChannelSelectiveTransform(
                        RandKSpaceSpikeNoise(prob=0.1, intensity_range=(13, 15)),
                        channels=[0],
                    ),
                    ChannelSelectiveTransform(
                        RandGaussianSmooth(
                            prob=0.2,
                            sigma_x=(0.5, 1.0),
                            sigma_y=(0.5, 1.0),
                            sigma_z=(0.5, 1.0),
                        ),
                        channels=[0],
                    ),
                    ChannelSelectiveTransform(
                        RandCoarseDropout(
                            prob=0.4,
                            holes=8,
                            spatial_size=(16, 16, 16),
                            fill_value=0,
                        ),
                        channels=[0],
                    ),
                    MultiChannelResize(
                        w.config.RESHAPE_SIZE
                        if w.config.RESHAPE_SIZE
                        else w.config.IMG_SIZE
                    ),
                    ChannelSelectiveTransform(
                        NormalizeIntensity(nonzero=True), channels=[0]
                    ),
                    # NOTE: do we add RandElastic3D ? RandFlip (sagittal) ?
                ]
            )

        else:
            train_transforms_list.extend(
                [
                    RandAffine(
                        prob=0.5,
                        rotate_range=(np.deg2rad(-5), np.deg2rad(5)),
                        scale_range=(-0.05, 0.05),
                        translate_range=(-5, 5),
                        padding_mode="border",
                    ),
                    OneOf(
                        [
                            Identity(),
                            RandSpatialCrop(
                                roi_size=tuple(
                                    int(w * 0.9) for w in w.config.RESHAPE_SIZE
                                )
                                if w.config.RESHAPE_SIZE
                                else tuple(int(w * 0.9) for w in w.config.IMG_SIZE),
                                random_center=True,
                                random_size=False,
                            ),
                        ],
                        weights=[0.3, 0.7],
                    ),
                    AdaptiveRicianNoise(prob=0.2),
                    AdaptiveGaussianNoise(prob=0.2, noise_factor=0.1),
                    RandBiasField(prob=0.3),
                    RandAdjustContrast(prob=0.4, gamma=(0.6, 1.4)),
                    RandScaleIntensity(prob=0.3, factors=(-0.5, 1.0)),
                    RandHistogramShift(prob=0.2, num_control_points=(3, 5)),
                    RandGibbsNoise(prob=0.3, alpha=(0.6, 0.8)),
                    RandKSpaceSpikeNoise(prob=0.1, intensity_range=(13, 15)),
                    RandGaussianSmooth(
                        prob=0.2,
                        sigma_x=(0.5, 1.0),
                        sigma_y=(0.5, 1.0),
                        sigma_z=(0.5, 1.0),
                    ),
                    RandCoarseDropout(
                        prob=0.4,
                        holes=8,
                        spatial_size=(16, 16, 16),
                        fill_value=0,
                    ),
                    Resize(w.config.RESHAPE_SIZE)
                    if w.config.RESHAPE_SIZE
                    else Resize(w.config.IMG_SIZE),
                    NormalizeIntensity(nonzero=True, channel_wise=True),
                    # NOTE: do we add RandElastic3D ? RandFlip (sagittal) ?
                ]
            )

    else:
        # Minimal augmentation: only resize + normalize (same as validation)
        train_transforms_list = []
        if include_resize:
            train_transforms_list.append(
                MultiChannelResize(
                    w.config.RESHAPE_SIZE
                    if w.config.RESHAPE_SIZE
                    else w.config.IMG_SIZE
                )
                if is_multimodal
                else (
                    Resize(w.config.RESHAPE_SIZE)
                    if w.config.RESHAPE_SIZE
                    else Resize(w.config.IMG_SIZE)
                )
            )
        train_transforms_list.append(
            ChannelSelectiveTransform(NormalizeIntensity(nonzero=True), channels=[0])
            if is_multimodal
            else NormalizeIntensity(nonzero=True, channel_wise=True)
        )

    train_da = Compose(train_transforms_list)

    # Validation transforms (always minimal: resize + normalize)
    val_transforms_list = []
    if include_resize:
        val_transforms_list.append(
            MultiChannelResize(
                w.config.RESHAPE_SIZE if w.config.RESHAPE_SIZE else w.config.IMG_SIZE
            )
            if is_multimodal
            else (
                Resize(w.config.RESHAPE_SIZE)
                if w.config.RESHAPE_SIZE
                else Resize(w.config.IMG_SIZE)
            )
        )
    val_transforms_list.append(
        ChannelSelectiveTransform(NormalizeIntensity(nonzero=True), channels=[0])
        if is_multimodal
        else NormalizeIntensity(nonzero=True, channel_wise=True)
    )

    val_da = Compose(val_transforms_list)

    # Log transform details to W&B
    def _jsonify(obj):
        try:
            json.dumps(obj)
            return obj
        except Exception:  # noqa: BLE001 - stringify any non-JSON-serializable value
            return str(obj)

    def get_transform_details(transform):
        if hasattr(transform, "transforms"):
            return [get_transform_details(t) for t in transform.transforms]
        name = transform.__class__.__name__
        params = {
            k: _jsonify(v)
            for k, v in transform.__dict__.items()
            if not k.startswith("_") and v is not None
        }
        return {name: params}

    train_transforms = get_transform_details(train_da)
    val_transforms = get_transform_details(val_da)
    w.config.update({"TRAIN_TRANSFORMS": train_transforms}, allow_val_change=True)
    w.config.update({"VAL_TRANSFORMS": val_transforms}, allow_val_change=True)

    preload_transform = None
    if preload_data and w.config.RESHAPE_SIZE:
        preload_transform = Resize(w.config.RESHAPE_SIZE)
        if get_rank() == 0:
            logger.info(f"Preload transform: Resize to {w.config.RESHAPE_SIZE}")

    # Configure dataset parameters
    dataset_kwargs = {"preload": preload_data, "preload_transform": preload_transform}
    mixup_seed = (
        (dataloader_seed + 2) % _MAX_UINT32 if dataloader_seed is not None else None
    )

    # Get the datasets: training one uses the transform train_da (defined in main) and optionally MRIMixUp
    if w.config.USE_MIXUP:
        train_set = MRIMixUp(
            NormalDataset(
                preprocess_data_dir,
                metadata_train,
                device="cpu",
                diseases=w.config.DISEASES,
                transform=None,
                **dataset_kwargs,
            ),
            num_samples=len(metadata_train),
            transform=train_da,
            alpha=w.config.MIXUP_ALPHA,
            mixup_prob=w.config.MIXUP_PROB,
            seed=mixup_seed,
        )
    else:
        train_set = NormalDataset(
            preprocess_data_dir,
            metadata_train,
            device="cpu",
            diseases=w.config.DISEASES,
            transform=train_da,
            **dataset_kwargs,
        )

    val_set = NormalDataset(
        preprocess_data_dir,
        metadata_val,
        device="cpu",
        diseases=w.config.DISEASES,
        transform=val_da,
        **dataset_kwargs,
    )

    if dataloader_seed is not None:
        sampler_seed = int(dataloader_seed)
    else:
        seed_tensor = torch.randint(0, _MAX_UINT32, (1,), device=device)
        if is_dist_avail_and_initialized():
            dist.broadcast(seed_tensor, src=0)
        sampler_seed = int(seed_tensor.item())
        if get_rank() == 0:
            logger.info(f"Generated synchronized random sampler seed: {sampler_seed}")

    if w.config.USE_BALANCED_SAMPLER:
        # Import balanced sampler utilities
        from utils.balanced_sampler import (
            DistributedWeightedSampler,
            compute_class_weights,
            compute_sample_weights,
        )

        if get_rank() == 0:
            logger.info("\n=== Balanced Sampler Configuration ===")
            logger.info("Computing class weights from training data...")

        # Compute class weights based on inverse frequency
        class_weights = compute_class_weights(
            metadata_train, diagnosis_column="Diagnosis", normalize=True
        )

        if get_rank() == 0:
            logger.info("Class weights (inverse frequency, normalized):")
            for cls, weight in sorted(class_weights.items()):
                class_count = (metadata_train["Diagnosis"] == cls).sum()
                logger.info(
                    f"  {cls}: weight={weight:.4f} (count={class_count}, "
                    f"{class_count / len(metadata_train) * 100:.1f}%)"
                )

            # Calculate expected samples per class per epoch
            expected_samples_per_class = len(metadata_train) / len(class_weights)
            logger.info(
                f"\nExpected samples per class per epoch: ~{expected_samples_per_class:.0f}"
            )
            logger.info("=" * 40 + "\n")

        # Compute sample weights for training set
        sample_weights = compute_sample_weights(
            metadata_train, class_weights, diagnosis_column="Diagnosis"
        )

        # Create distributed weighted sampler for training
        # Note: num_samples should be set to the original dataset size divided by world_size to maintain consistent epoch length across all training runs
        num_replicas = get_world_size() if is_dist_avail_and_initialized() else 1
        rank = get_rank() if is_dist_avail_and_initialized() else 0

        train_sampler = DistributedWeightedSampler(
            dataset=train_set,
            weights=sample_weights,
            num_samples=math.ceil(len(train_set) / num_replicas),
            replacement=True,
            num_replicas=num_replicas,
            rank=rank,
            seed=int(sampler_seed),
            drop_last=False,
        )

        # Validation sampler is always standard distributed sampler (no balancing needed)
        if is_dist_avail_and_initialized():
            val_sampler = DistributedSampler(
                val_set, shuffle=False, seed=int(sampler_seed)
            )
        else:
            val_sampler = None

    else:
        # Standard distributed samplers (original behavior)
        if is_dist_avail_and_initialized():
            train_sampler = DistributedSampler(
                train_set, shuffle=True, seed=int(sampler_seed)
            )
            val_sampler = DistributedSampler(
                val_set, shuffle=False, seed=int(sampler_seed)
            )
        else:
            train_sampler = None
            val_sampler = None

    dataloader_kwargs = {
        "batch_size": w.config.BATCH_SIZE,
        "num_workers": w.config.NUM_WORKERS,
        "pin_memory": True,
        "persistent_workers": w.config.NUM_WORKERS > 0,
        "prefetch_factor": (
            w.config.PREFETCH_FACTOR if w.config.NUM_WORKERS > 0 else None
        ),
        "worker_init_fn": worker_init_fn,
    }

    if device.type == "cuda" and hasattr(torch.utils.data.DataLoader, "__init__"):
        import inspect

        sig = inspect.signature(torch.utils.data.DataLoader.__init__)
        if "pin_memory_device" in sig.parameters:
            dataloader_kwargs["pin_memory_device"] = f"cuda:{local_rank}"

    train_loader = DataLoader(
        train_set,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        generator=train_generator,
        **dataloader_kwargs,
    )

    val_loader = DataLoader(
        val_set,
        sampler=val_sampler,
        shuffle=False,
        generator=val_generator,
        **dataloader_kwargs,
    )

    if w.config.ARCHITECTURE == "SwinDPL":
        from models.swin_transformer_3d_dpl import (
            SwinTransformerT as SwinTransformer,
        )

        model_without_ddp = SwinTransformer(
            img_size=(
                tuple(w.config.RESHAPE_SIZE)
                if w.config.RESHAPE_SIZE
                else tuple(w.config.IMG_SIZE)
            ),
            in_channels=w.config.IN_CHANNELS,
            patch_size=tuple(w.config.PATCH_SHAPE),
            embed_dim=w.config.EMBED_DIM,
            depths=w.config.DEPTH,
            num_heads=w.config.HEADS,
            window_size=tuple(w.config.WINDOW_SIZE),
            mlp_ratio=w.config.MLP_RATIO,
            qkv_bias=w.config.QKV_BIAS,
            dropout=w.config.DROPOUT,
            attention_dropout=w.config.ATTENTION_DROPOUT,
            stochastic_depth_prob=w.config.STOCHASTIC_DEPTH_PROB,
            num_classes=len(w.config.DISEASES),
            norm_layer=(eval(w.config.NORM_LAYER) if w.config.NORM_LAYER else None),
            patch_norm=w.config.PATCH_NORM,
            use_checkpoint=w.config.USE_CHECKPOINT,
            use_diff_attn=w.config.USE_DIFF_ATTN,
            use_rope=w.config.USE_ROPE,
            use_relative_pos_bias=w.config.USE_RELATIVE_POS_BIAS,
        ).to(device)
    elif w.config.ARCHITECTURE == "MedViT":
        from models.medvit_diff_3d import MedViT

        model_without_ddp = MedViT(
            in_channels=w.config.IN_CHANNELS,
            num_classes=len(w.config.DISEASES),
            depths=w.config.DEPTH,
            path_dropout=w.config.STOCHASTIC_DEPTH_PROB,
            attn_drop=w.config.ATTENTION_DROPOUT,
            drop=w.config.DROPOUT,
            use_checkpoint=w.config.USE_CHECKPOINT,
            use_diff_attn=w.config.USE_DIFF_ATTN,
            use_rope=w.config.USE_ROPE,
        ).to(device)
    elif w.config.ARCHITECTURE == "ResNet":
        from models.resnet_3d import ResNet3DMedical

        model_without_ddp = ResNet3DMedical(
            img_size=(
                tuple(w.config.RESHAPE_SIZE)
                if w.config.RESHAPE_SIZE
                else tuple(w.config.IMG_SIZE)
            ),
            num_classes=len(w.config.DISEASES),
            in_channels=w.config.IN_CHANNELS,
            resnet_variant="resnet50",
            shortcut_type="B",
            dropout=w.config.DROPOUT,
        ).to(device)
    else:
        raise ValueError(f"Unknown architecture: {w.config.ARCHITECTURE}")

    if is_dist_avail_and_initialized() and w.config.USE_SYNC_BN:
        model_without_ddp = nn.SyncBatchNorm.convert_sync_batchnorm(model_without_ddp)

    def _convert_rank5_to_channels_last_3d(module: nn.Module) -> nn.Module:
        for submodule in module.modules():
            for name, param in submodule.named_parameters(recurse=False):
                if param is not None and param.dim() == 5:
                    param.data = param.data.contiguous(
                        memory_format=torch.channels_last_3d
                    )
            for name, buf in submodule.named_buffers(recurse=False):
                if buf is not None and buf.dim() == 5:
                    submodule._buffers[name] = buf.contiguous(
                        memory_format=torch.channels_last_3d
                    )
        return module

    if device.type == "cuda":
        model_without_ddp = _convert_rank5_to_channels_last_3d(model_without_ddp)

    if is_dist_avail_and_initialized():
        model = DDP(
            model_without_ddp,
            device_ids=[local_rank],
            output_device=local_rank,
            gradient_as_bucket_view=True,
            static_graph=False,
            find_unused_parameters=True,
        )
    else:
        model = model_without_ddp

    if get_rank() == 0:
        # Gradient logging - use "gradients" or "all" to log, None to disable
        # Note: "all" logs histograms which triggers non-deterministic warnings with histc_cuda
        w.watch(
            model_without_ddp,
            log="gradients",  # Only log gradient norms, not histograms
            log_freq=10,  # in steps
            log_graph=False,  # Disable computational graph logging
        )

    param_groups = get_params_groups(model_without_ddp, w.config)

    # Use fused optimizer for speed (10-20% faster on CUDA)
    base_optimizer_class = torch.optim.AdamW
    optimizer_kwargs = {
        "lr": w.config.LR_BASE,
        "fused": torch.cuda.is_available(),
    }

    optimizer = base_optimizer_class(param_groups, **optimizer_kwargs)

    # Compute gradient accumulation steps and effective batch size
    world_size = get_world_size() if is_dist_avail_and_initialized() else 1
    gradient_accumulation, actual_effective_batch_size = (
        compute_gradient_accumulation_steps(
            w.config.BATCH_SIZE, w.config.EFFECTIVE_BATCH_SIZE, world_size
        )
    )

    # Update config with computed gradient accumulation
    w.config.update(
        {"GRADIENT_ACCUMULATION": gradient_accumulation}, allow_val_change=True
    )

    if get_rank() == 0:
        logger.info("\n" + "=" * 50)
        logger.info("TRAINING CONFIGURATION SUMMARY".center(50))
        logger.info("=" * 50)

        logger.info("\n=== Model Architecture ===")
        logger.info(f"Architecture: {w.config.ARCHITECTURE}")
        logger.info(f"Image size: {w.config.IMG_SIZE}")
        logger.info(f"Patch shape: {w.config.PATCH_SHAPE}")
        logger.info(f"Embedding dimension: {w.config.EMBED_DIM}")
        logger.info(f"Network depth: {w.config.DEPTH}")
        logger.info(f"Attention heads: {w.config.HEADS}")
        logger.info(f"Window size: {w.config.WINDOW_SIZE}")
        logger.info(f"MLP ratio: {w.config.MLP_RATIO}")
        logger.info(f"Normalization layer: {w.config.NORM_LAYER}")
        logger.info(f"Dropout: {w.config.DROPOUT}")
        logger.info(f"Attention dropout: {w.config.ATTENTION_DROPOUT}")
        logger.info(f"Stochastic depth probability: {w.config.STOCHASTIC_DEPTH_PROB}")
        logger.info(f"Differential Attention: {w.config.USE_DIFF_ATTN}")

        logger.info("\n=== Model Information ===")
        logger.info(
            f"Model size in MB: {get_model_size(model.module if is_dist_avail_and_initialized() else model):.2f}"
        )
        n_parameters = count_parameters(
            model.module if is_dist_avail_and_initialized() else model
        )
        logger.info(f"Number of trainable parameters: {n_parameters:,}")
        logger.info(
            f"Classification task: {len(w.config.DISEASES)} classes {w.config.DISEASES}"
        )

        logger.info("\n=== Training Hyperparameters ===")
        logger.info(f"Total training steps: {w.config.STEPS}")
        logger.info(f"Validation frequency: {w.config.VALIDATION_FREQUENCY} steps")
        logger.info(f"Batch size per GPU: {w.config.BATCH_SIZE}")
        logger.info(f"Desired effective batch size: {w.config.EFFECTIVE_BATCH_SIZE}")
        logger.info(f"Gradient accumulation steps: {gradient_accumulation}")
        logger.info(f"Actual effective batch size: {actual_effective_batch_size}")
        logger.info(
            f"Step semantics: 1 step = {actual_effective_batch_size} samples processed"
        )
        logger.info(
            f"Learning rate: {w.config.LR_BASE} → {w.config.LR_FINAL} (warmup: {w.config.LR_WARMUP} steps)"
        )
        logger.info(
            f"Weight decay: {w.config.WD_BASE} → {w.config.WD_FINAL} (warmup: {w.config.WD_WARMUP} steps)"
        )
        logger.info(
            f"Using EMA: {w.config.USE_EMA} (decay: {w.config.EMA_DECAY})"
            if w.config.USE_EMA
            else "Using EMA: False"
        )
        logger.info(f"Using mixed precision (FP16): {w.config.FP16}")
        logger.info(
            f"Label smoothing: {w.config.LABEL_SMOOTHING if w.config.LABEL_SMOOTHING else 'Disabled'}"
        )
        logger.info(
            f"Cross-validation: Fold {w.config.FOLD + 1}/{w.config.KFOLD} with split ratio {w.config.SPLIT}"
        )

        if w.config.FP16:
            logger.info("Using FP16 - Gradient Scaler active")

        logger.info(f"Number of workers: {w.config.NUM_WORKERS}")

        logger.info("\n=== Optimizer State ===")
        logger.info("Parameter groups:")
        for i, group in enumerate(optimizer.param_groups):
            logger.info(f"Group {i}:")
            logger.info(f"\tLearning rate: {group['lr']}")
            logger.info(f"\tWeight decay: {group['weight_decay']}")
            logger.info(f"\tParameters: {len(group['params'])}")

        logger.info("\n=== Hardware Configuration ===")
        logger.info(f"Device: {device}")
        logger.info(f"Number of GPUs: {get_world_size()}")

        logger.info("\n" + "=" * 50)

    if args.checkpoint:
        (
            model,
            optimizer,
            start_step,
            best_loss,
            history,
            sampler_epoch,
            best_metric_for_early_stopping,
        ) = load_checkpoint(args.checkpoint, model, optimizer, device)
    else:
        start_step = 0
        # Initialize best metric value based on METRIC_BEST_MODEL
        # For 'loss' (min mode): start with infinity (lower is better)
        # For other metrics (max mode): start with -infinity (higher is better)
        best_model_metric = getattr(w.config, "METRIC_BEST_MODEL", "loss").lower()
        if best_model_metric == "loss":
            best_loss = float("inf")
        else:
            best_loss = float("-inf")
        history = []
        sampler_epoch = 0
        best_metric_for_early_stopping = float("inf")

    # Step-based training
    total_steps = int(w.config.STEPS)

    lr_scheduler = cosine_scheduler(
        w.config.LR_BASE,
        w.config.LR_FINAL,
        total_steps,
        warmup_steps=w.config.LR_WARMUP,
    )

    wd_scheduler = cosine_scheduler(
        w.config.WD_BASE,
        w.config.WD_FINAL,
        total_steps,
        warmup_steps=w.config.WD_WARMUP,
    )

    # Initialize loss function based on config
    label_smoothing = w.config.LABEL_SMOOTHING if w.config.LABEL_SMOOTHING else 0.0
    loss_type = getattr(w.config, "LOSS", "cross_entropy")

    if loss_type == "hierarchical_risk_aware":
        # Validate hierarchy map matches number of classes
        hierarchy_map = dict(w.config.HIERARCHY_MAP)
        num_diseases = len(w.config.DISEASES)

        if len(hierarchy_map) != num_diseases:
            raise ValueError(
                f"HIERARCHY_MAP has {len(hierarchy_map)} entries but DISEASES has {num_diseases} classes. "
                f"Each disease class must have a mapping. DISEASES: {w.config.DISEASES}"
            )

        # Validate all fine class indices are present
        expected_indices = set(range(num_diseases))
        actual_indices = set(hierarchy_map.keys())
        if expected_indices != actual_indices:
            missing = expected_indices - actual_indices
            extra = actual_indices - expected_indices
            raise ValueError(
                f"HIERARCHY_MAP keys must be 0 to {num_diseases - 1}. "
                f"Missing: {missing}, Extra: {extra}"
            )

        loss = HierarchicalRiskAwareLoss(
            hierarchy_map=hierarchy_map,
            alpha=w.config.HIERARCHICAL_ALPHA,
            focal_gamma=w.config.HIERARCHICAL_GAMMA,
            label_smoothing=label_smoothing,
            reduction="mean",
        )

        if get_rank() == 0:
            logger.info("\n=== Hierarchical Risk-Aware Loss ===")
            logger.info(
                f"Fine classes: {num_diseases} -> Coarse classes: {loss.num_coarse_classes}"
            )
            logger.info("Hierarchy mapping:")
            for fine_idx, coarse_idx in sorted(hierarchy_map.items()):
                logger.info(
                    f"  {w.config.DISEASES[fine_idx]} (idx={fine_idx}) -> Group {coarse_idx}"
                )
            logger.info(f"Alpha (coarse weight): {w.config.HIERARCHICAL_ALPHA}")
            logger.info(f"Focal gamma: {w.config.HIERARCHICAL_GAMMA}")
            logger.info(f"Label smoothing: {label_smoothing}")
            logger.info("=" * 40)
    else:
        # Default: LabelSmoothingLoss (cross_entropy with optional smoothing)
        loss = LabelSmoothingLoss(smoothing=label_smoothing, reduction="mean")

        if get_rank() == 0 and label_smoothing > 0.0:
            logger.info("\n=== Label Smoothing ===")
            logger.info(f"Label smoothing enabled with factor: {label_smoothing}")
            logger.info(
                f"This will redistribute {label_smoothing * 100:.1f}% of probability mass uniformly across all classes."
            )
            logger.info("=" * 30)

    training_loops(
        model,
        loss,
        optimizer,
        total_steps,
        train_loader,
        val_loader,
        lr_scheduler,
        wd_scheduler,
        save_dir,
        start_step,
        best_loss,
        history,
        sampler_epoch,
        best_metric_for_early_stopping,
    )


if __name__ == "__main__":
    args = get_args()

    if (
        args.wandb_mode == "online"
        and os.getenv("WANDB_API_KEY") is None
        and not os.path.exists("wandb.key")
    ):
        raise ValueError(
            "WANDB_API_KEY is not set. Please set it using the command WANDB_API_KEY=<your_api_key> <training_command> (or `export WANDB_API_KEY=<your_api_key>` before training). The wandb api key can be found in your wandb account page (https://wandb.ai/authorize)."
        )

    if args.wandb_mode == "online" and not os.getenv("WANDB_API_KEY"):
        if os.path.exists("wandb.key"):
            with open("wandb.key", "r") as f:
                os.environ["WANDB_API_KEY"] = f.read().strip()
        else:
            logger.info("WANDB_API_KEY is missing. Running in offline mode.")
            args.wandb_mode = "offline"

    num_gpus = torch.cuda.device_count()
    if torch.cuda.is_available() and num_gpus > 1:
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        os.environ["WANDB_MODE"] = "disabled" if local_rank != 0 else args.wandb_mode
        distributed = True
    else:
        local_rank = 0
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        os.environ["WANDB_MODE"] = args.wandb_mode
        distributed = False

    save_dir = Path(args.save_dir) / args.runname
    save_dir.mkdir(exist_ok=True, mode=0o777, parents=True)

    if args.checkpoint is not None:
        args.checkpoint = Path(args.checkpoint)

    # Extract wandb_id from checkpoint name if it follows the format: model_{wandb_id}_{fold}_best{n}.pt
    # Otherwise, start a new run (wandb_id = None)
    wandb_id = None
    if args.checkpoint:
        checkpoint_parts = args.checkpoint.stem.split("_")
        if len(checkpoint_parts) >= 2:
            potential_id = checkpoint_parts[1]
            # Validate that it looks like a wandb run id (8 lowercase alphanumeric chars)
            if re.match(r"^[a-z0-9]{8}$", potential_id):
                wandb_id = potential_id
    w.init(
        project=args.project_name,
        dir=save_dir,
        resume="allow",
        id=wandb_id,
        settings=w.Settings(init_timeout=180),
    )
    w.run.name = save_dir.name

    w.define_metric("Validation loss", summary="min")
    w.define_metric("Validation fine loss", summary="min")
    w.define_metric("Validation coarse loss", summary="min")
    w.define_metric("Validation accuracy", summary="max")
    w.define_metric("Validation balanced accuracy", summary="max")
    w.define_metric("Validation ROC AUC", summary="max")
    w.define_metric("Validation PR AUC", summary="max")
    w.define_metric("Validation macro F1", summary="max")
    w.define_metric("Validation MCC", summary="max")
    w.define_metric("Learning rate", summary="mean")
    w.define_metric("Weight decay", summary="mean")

    wandb_config = w.config.as_dict()
    if args.config:
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)

        for key, value in config.items():
            wandb_config[key] = value.get("value", value)

        w.config.update(wandb_config, allow_val_change=True)

    if args.seed is not None:
        wandb_config["SEED"] = normalize_seed(args.seed)

    normalized_seed = normalize_seed(wandb_config.get("SEED"))
    wandb_config["SEED"] = normalized_seed
    w.config.update({"SEED": normalized_seed}, allow_val_change=True)
    global_seed = normalized_seed

    # --fold argument always takes precedence over config default
    # This is essential for resuming training on a specific fold
    if args.fold is not None and args.fold != wandb_config.get("FOLD"):
        wandb_config["FOLD"] = args.fold
        w.config.update({"FOLD": args.fold}, allow_val_change=True)
        if get_rank() == 0:
            logger.info(f"Using fold {args.fold} from command line argument.")

    # Only initialize distributed mode if more than one GPU is available
    if distributed:
        init_distributed_mode(w.config)
        logger.info(f"Initialized distributed training with {get_world_size()} GPUs")
    else:
        logger.info("Using single GPU training (non-distributed mode)")

    if global_seed is not None:
        rank_adjusted_seed = (int(global_seed) + get_rank()) % _MAX_UINT32
        seed_everything(rank_adjusted_seed)
        if get_rank() == 0:
            logger.info(
                f"Seeded all RNGs with base seed {int(global_seed)} "
                f"(rank-adjusted seed {rank_adjusted_seed})."
            )
    else:
        # When seeding is disabled, enable cudnn.benchmark for performance
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        if get_rank() == 0:
            logger.info("Global seed disabled; training will be non-deterministic.")
            logger.info("Enabled cudnn.benchmark for better performance.")

    # Save config after distributed init to ensure proper rank checking
    if get_rank() == 0:
        # w.run.dir is the official, robust way to get the run directory path
        config_save_path = Path(w.run.dir) / "config.yaml"
        with open(config_save_path, "w") as f:
            yaml.dump(wandb_config, f, default_flow_style=False)

        logger.info(f"Saved final configuration to {config_save_path}")

    logger.info(f"Training with {w.config.KFOLD} folds - fold {w.config.FOLD}")
    train(save_dir, fold=w.config.FOLD)

    # Properly close wandb to avoid blocking on exit
    w.finish()

    # Clean up distributed training resources
    if is_dist_avail_and_initialized():
        # Synchronize all processes before destroying the process group
        # This ensures all ranks have completed their work
        dist.barrier()
        dist.destroy_process_group()

    # Explicitly clear CUDA cache to release GPU memory
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
