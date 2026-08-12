"""
Balanced Sampler for Imbalanced Datasets with Distributed Training Support.

This module provides samplers that perform class-balanced sampling by weighting
samples according to the inverse of their class frequency. This helps mitigate
class imbalance during training.

Classes
-------
DistributedWeightedSampler
    A distributed sampler that performs weighted sampling with replacement.
"""

import math
from collections.abc import Iterator

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from torch.utils.data import Sampler


def compute_class_weights(
    metadata: pd.DataFrame,
    diagnosis_column: str = "Diagnosis",
    normalize: bool = True,
) -> dict[str, float]:
    """
    Compute inverse frequency class weights from metadata.

    Parameters
    ----------
    metadata : pd.DataFrame
        DataFrame containing sample metadata with diagnosis labels.
    diagnosis_column : str, optional
        Name of the column containing class labels (default: "Diagnosis").
    normalize : bool, optional
        Whether to normalize weights so the minimum weight is 1.0 (default: True).
        This improves numerical stability.

    Returns
    -------
    dict[str, float]
        Dictionary mapping class labels to their inverse frequency weights.

    Examples
    --------
    >>> metadata = pd.DataFrame({'Diagnosis': ['CN', 'CN', 'AD', 'FTD']})
    >>> weights = compute_class_weights(metadata)
    >>> print(weights)
    {'CN': 1.0, 'AD': 2.0, 'FTD': 2.0}
    """
    # Validate input
    if len(metadata) == 0:
        raise ValueError("Cannot compute class weights for empty metadata")

    if diagnosis_column not in metadata.columns:
        raise ValueError(f"Column '{diagnosis_column}' not found in metadata")

    # Count samples per class
    class_counts = metadata[diagnosis_column].value_counts()

    # Compute inverse frequency weights
    total_samples = len(metadata)
    class_weights = {}

    for cls, count in class_counts.items():
        # Weight = total_samples / (num_classes * count)
        # This ensures that expected samples per class per epoch = total_samples / num_classes
        weight = total_samples / (len(class_counts) * count)
        class_weights[cls] = weight

    if normalize:
        # Normalize so minimum weight is 1.0 for numerical stability
        min_weight = min(class_weights.values())
        class_weights = {cls: w / min_weight for cls, w in class_weights.items()}

    return class_weights


def compute_sample_weights(
    metadata: pd.DataFrame,
    class_weights: dict[str, float],
    diagnosis_column: str = "Diagnosis",
) -> np.ndarray:
    """
    Assign weights to each sample based on its class.

    Parameters
    ----------
    metadata : pd.DataFrame
        DataFrame containing sample metadata with diagnosis labels.
    class_weights : dict[str, float]
        Dictionary mapping class labels to their weights.
    diagnosis_column : str, optional
        Name of the column containing class labels (default: "Diagnosis").

    Returns
    -------
    np.ndarray
        Array of sample weights, one per sample in the metadata.

    Examples
    --------
    >>> metadata = pd.DataFrame({'Diagnosis': ['CN', 'AD', 'CN']})
    >>> class_weights = {'CN': 1.0, 'AD': 2.0}
    >>> weights = compute_sample_weights(metadata, class_weights)
    >>> print(weights)
    [1.0, 2.0, 1.0]
    """
    # Validate that all diagnoses have corresponding weights
    unique_diagnoses = set(metadata[diagnosis_column].unique())
    missing_classes = unique_diagnoses - set(class_weights.keys())
    if missing_classes:
        raise ValueError(
            f"Found diagnoses in metadata not present in class_weights: {missing_classes}"
        )

    sample_weights = np.array(
        [class_weights[diagnosis] for diagnosis in metadata[diagnosis_column]],
        dtype=np.float32,
    )
    return sample_weights


class DistributedWeightedSampler(Sampler):
    """
    Distributed sampler that performs weighted sampling with replacement.

    This sampler combines the functionality of PyTorch's WeightedRandomSampler
    with DistributedSampler to enable balanced sampling in distributed training.
    Each rank samples from its own partition of the dataset using class weights.

    Parameters
    ----------
    dataset : Dataset
        The dataset to sample from.
    weights : np.ndarray or torch.Tensor
        Weight for each sample in the dataset.
    num_samples : int, optional
        Number of samples to draw per 'chunk'. If None, defaults to len(dataset).
        Used to determine the size of the internal buffer for multinomial sampling.
    replacement : bool, optional
        Whether to sample with replacement (default: True). For balanced sampling,
        this should typically be True.
    num_replicas : int, optional
        Number of processes participating in distributed training. If None,
        retrieved from the current distributed group.
    rank : int, optional
        Rank of the current process. If None, retrieved from the current
        distributed group.
    seed : int, optional
        Random seed for reproducibility.
    drop_last : bool, optional
        Whether to drop the last incomplete batch (only relevant if infinite=False).
    infinite : bool, optional
        If True, the sampler yields indices indefinitely (for step-based training).
        If False, it stops after num_samples (for epoch-based training).
    """

    def __init__(
        self,
        dataset,
        weights: np.ndarray | torch.Tensor,
        num_samples: int | None = None,
        replacement: bool = True,
        num_replicas: int | None = None,
        rank: int | None = None,
        seed: int = 0,
        drop_last: bool = False,
        infinite: bool = True,
    ):
        # Distributed setup
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size() if dist.is_initialized() else 1

        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank() if dist.is_initialized() else 0

        if rank >= num_replicas or rank < 0:
            raise ValueError(
                f"Invalid rank {rank}, rank should be in the interval [0, {num_replicas - 1}]"
            )

        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.drop_last = drop_last
        self.replacement = replacement
        self.seed = seed
        self.infinite = infinite

        # Convert weights to tensor
        if isinstance(weights, np.ndarray):
            weights = torch.from_numpy(weights).float()
        elif isinstance(weights, torch.Tensor):
            weights = weights.float()
        else:
            raise TypeError(
                f"weights should be np.ndarray or torch.Tensor, got {type(weights)}"
            )

        if len(weights) != len(dataset):
            raise ValueError(
                f"Length of weights ({len(weights)}) must match dataset size ({len(dataset)})"
            )

        # Validate dataset is not empty
        if len(dataset) == 0:
            raise ValueError("Cannot create sampler for empty dataset")

        self.weights = weights

        # Validate weights are valid for sampling
        if torch.any(torch.isnan(self.weights)) or torch.any(torch.isinf(self.weights)):
            raise ValueError("Weights contain NaN or Inf values")

        if torch.any(self.weights < 0):
            raise ValueError("Weights must be non-negative for weighted sampling")

        if torch.sum(self.weights) == 0:
            raise ValueError(
                "Sum of weights is zero - cannot perform weighted sampling"
            )

        # Determine "chunk" size for generation
        # If infinite, this acts as the buffer refill size
        if num_samples is None:
            self.num_samples = len(self.dataset)
        else:
            self.num_samples = num_samples

        # In distributed setting, we need to ensure the total size is divisible by num_replicas
        # to ensure all ranks get the same number of samples per chunk
        if self.num_samples % self.num_replicas != 0:
            self.num_samples = (
                math.ceil(self.num_samples / self.num_replicas) * self.num_replicas
            )

        self.total_size = self.num_samples
        self.num_samples_per_rank = self.total_size // self.num_replicas

    def __iter__(self) -> Iterator[int]:
        """
        Generate sample indices.

        If infinite=True, this loops forever.
        """
        generator = torch.Generator()
        # Seed logic:
        # In infinite mode, we set the seed once at the start. The generator maintains state.
        # In finite mode, we incorporate self.epoch to allow reshuffling between calls.
        current_seed = self.seed + self.epoch
        generator.manual_seed(current_seed)

        while True:
            # 1. Generate indices for the whole world_size (conceptually)
            if self.replacement:
                indices = torch.multinomial(
                    self.weights,
                    self.total_size,
                    replacement=True,
                    generator=generator,
                ).tolist()
            else:
                # Weighted shuffling without replacement
                rand_tensor = torch.rand(len(self.weights), generator=generator)
                indices = torch.argsort(rand_tensor / self.weights, descending=True)[
                    : self.total_size
                ].tolist()

            # 2. Subsample for the current rank
            # Striding is efficient and statistically sound for randomized data
            indices = indices[self.rank : self.total_size : self.num_replicas]

            # Yield indices for this chunk
            yield from indices

            # 3. Stop or Continue
            if not self.infinite:
                break

            # If infinite, we just loop back and generate a new chunk.
            # The generator state is preserved, so the next chunk is new random data.

    def __len__(self) -> int:
        """
        Return the number of samples per rank (per chunk).
        In infinite mode, this is purely for tqdm/logging estimates, as the iterator won't stop.
        """
        return self.num_samples_per_rank

    def set_epoch(self, epoch: int):
        """
        Set the epoch for this sampler.

        This ensures deterministic shuffling across epochs when using a seed.
        Should be called at the start of each epoch.

        Parameters
        ----------
        epoch : int
            Current epoch number.
        """
        self.epoch = epoch


def create_balanced_sampler(
    dataset,
    metadata: pd.DataFrame,
    num_samples: int | None = None,
    diagnosis_column: str = "Diagnosis",
    seed: int = 0,
    num_replicas: int | None = None,
    rank: int | None = None,
    drop_last: bool = False,
    infinite: bool = True,
) -> DistributedWeightedSampler:
    """
    Factory function to create a balanced distributed sampler.

    This is a convenience function that combines class weight computation
    and sampler creation into a single call.

    Parameters
    ----------
    dataset : Dataset
        The dataset to sample from.
    metadata : pd.DataFrame
        DataFrame containing sample metadata with diagnosis labels.
    num_samples : int, optional
        Number of samples per epoch per rank. If None, defaults to
        len(dataset) // world_size.
    diagnosis_column : str, optional
        Name of the column containing class labels (default: "Diagnosis").
    seed : int, optional
        Random seed for reproducibility (default: 0).
    num_replicas : int, optional
        Number of distributed processes. If None, auto-detected.
    rank : int, optional
        Rank of current process. If None, auto-detected.
    drop_last : bool, optional
        Whether to drop the last incomplete batch (default: False).

    Returns
    -------
    DistributedWeightedSampler
        Configured balanced sampler ready for use in DataLoader.

    Examples
    --------
    >>> sampler = create_balanced_sampler(
    ...     train_dataset,
    ...     train_metadata,
    ...     seed=42
    ... )
    >>> train_loader = DataLoader(train_dataset, batch_size=32, sampler=sampler)
    """
    # Compute class weights
    class_weights = compute_class_weights(metadata, diagnosis_column=diagnosis_column)

    # Compute sample weights
    sample_weights = compute_sample_weights(
        metadata, class_weights, diagnosis_column=diagnosis_column
    )

    # Create sampler
    sampler = DistributedWeightedSampler(
        dataset=dataset,
        weights=sample_weights,
        num_samples=num_samples,
        replacement=True,  # Always use replacement for balanced sampling
        num_replicas=num_replicas,
        rank=rank,
        seed=seed,
        drop_last=drop_last,
        infinite=infinite,
    )

    return sampler
