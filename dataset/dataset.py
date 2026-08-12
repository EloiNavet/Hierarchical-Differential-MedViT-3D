import os
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.distributions.beta import Beta
from torch.utils.data import Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.seed import _MAX_UINT32


def load_metadata(
    metadata_paths: str | list[str] | pd.DataFrame | pd.Series,
    accept_datasets: list[str] | None = None,
) -> pd.DataFrame:
    """Load and concatenate metadata from CSV files or DataFrame."""
    if isinstance(metadata_paths, str):
        metadata = pd.read_csv(metadata_paths).reset_index(drop=True)
    elif isinstance(metadata_paths, list):
        metadata = [pd.read_csv(e) for e in metadata_paths]
        metadata = pd.concat(metadata, ignore_index=True).reset_index(drop=True)
    elif isinstance(metadata_paths, (pd.DataFrame, pd.Series)):
        metadata = metadata_paths.reset_index(drop=True)
    else:
        raise NotImplementedError

    if accept_datasets is not None:
        metadata = metadata[metadata.Dataset.isin(accept_datasets)].reset_index(
            drop=True
        )

    return metadata


class NormalDataset(Dataset):
    """Dataset for loading preprocessed MRI tensors with optional transforms."""

    def __init__(
        self,
        data_root: str,
        meta_data: pd.DataFrame,
        device: torch.device,
        diseases: list[str],
        transform: callable | None = None,
        preload: bool = False,
        preload_transform: callable | None = None,
    ):
        super().__init__()

        self.data_root = data_root
        self.meta_data = meta_data
        self.transform = transform
        self.preload_transform = preload_transform
        self.device = device
        self.diseases = diseases

        # Pre-create label tensors for each diagnosis (memory optimization)
        self._label_cache = {}
        for diagnosis in meta_data.Diagnosis.unique():
            label_tensor = torch.zeros(len(diseases), dtype=torch.float32)
            if diagnosis in diseases:
                label_tensor[diseases.index(diagnosis)] = 1.0
            self._label_cache[diagnosis] = label_tensor

        # Preload all data if requested (only for small datasets!)
        self.preloaded_data = None
        if preload:
            print(f"Preloading {len(meta_data)} samples into memory...")
            self._preload_all_data()
            print("Preloading complete!")

    def _preload_all_data(self):
        """Preload all data into memory. Use only for small datasets!

        If preload_transform is provided, it will be applied to each sample
        during loading. This is useful for ensuring consistent data shapes
        (e.g., applying Resize) before caching.
        """
        self.preloaded_data = {}
        for idx in range(len(self.meta_data)):
            subject = self.meta_data.Subject.iloc[idx]
            path = os.path.join(self.data_root, f"{subject}.pt")
            data = torch.load(
                path,
                map_location="cpu",
                weights_only=False,
            )
            # Apply preload transform if provided (e.g., Resize for shape consistency)
            if self.preload_transform is not None:
                data = self.preload_transform(data)
            self.preloaded_data[idx] = data

    def _load_sample(self, idx: int) -> torch.Tensor:
        """Load a sample from disk or preloaded cache."""
        # Fast path: preloaded data
        if self.preloaded_data is not None:
            return self.preloaded_data[idx]

        # Load from disk
        subject = self.meta_data.Subject.iloc[idx]
        path = os.path.join(self.data_root, f"{subject}.pt")
        return torch.load(path, map_location="cpu", weights_only=False)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = self._load_sample(idx)

        # Clone if preloaded to avoid modifying cached data
        if self.preloaded_data is not None:
            x = x.clone()

        # Apply transform if specified
        if self.transform is not None:
            x = self.transform(x)

        # Use pre-cached label tensor (memory optimization)
        diagnosis = self.meta_data.Diagnosis.iloc[idx]
        y = self._label_cache[diagnosis].clone()

        return x, y

    def __len__(self) -> int:
        """Return the total number of samples in the dataset."""
        return len(self.meta_data)


class SVMDataset(Dataset):
    """
    Dataset class for SVM feature extraction from preprocessed segmentation volumes.

    This dataset loads preprocessed segmentation volume features (from DataPrepaSVM)
    and returns both features and one-hot encoded labels for SVM training.

    Parameters
    ----------
    data_root : str
        Directory containing the preprocessed .pt files.
    meta_data : pd.DataFrame
        Metadata DataFrame containing subject information.
    diseases : list[str]
        List of disease labels (e.g., ['CN', 'AD', 'FTD']).
    device : torch.device or str
        The device to load the data onto.
    """

    def __init__(
        self,
        data_root: str,
        meta_data: pd.DataFrame,
        diseases: list[str],
        device: torch.device | str = "cpu",
    ):
        super().__init__()
        self.data_root = data_root
        self.meta_data = meta_data
        self.device = torch.device(device) if isinstance(device, str) else device
        self.diseases = diseases

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.load(
            os.path.join(self.data_root, f"{self.meta_data.Subject.iloc[idx]}.pt"),
            map_location=self.device,
            weights_only=False,
        )

        disease = self.meta_data.Diagnosis.iloc[idx]
        y = torch.zeros(len(self.diseases), device=self.device)
        y[self.diseases.index(disease)] = 1.0

        x = x.to(self.device)

        return x, y

    def __len__(self) -> int:
        return len(self.meta_data)


class MRIMixUp(Dataset):
    """MixUp augmentation for 3D MRI. Reference: https://arxiv.org/abs/1710.09412"""

    def __init__(
        self,
        dataset: Dataset,
        num_samples: int,
        alpha: float,
        mixup_prob: float,
        transform: torch.nn.Module | None = None,
        seed: int | None = None,
    ):
        super().__init__()
        assert 0 < alpha < 1, "alpha should be between 0 and 1"
        assert 0 <= mixup_prob <= 1, "mixup_prob should be between 0 and 1"
        assert num_samples > 0, "num_samples should be greater than 0"
        self.dataset = dataset
        self.num_samples = num_samples
        self.alpha = alpha
        self.mixup_prob = mixup_prob

        self.dist = Beta(torch.tensor([alpha]), torch.tensor([alpha]))
        self.transform = transform
        self.seed = int(seed) if seed is not None else None
        self._current_epoch = 0

        # Precompute indices grouped by class
        self.class_indices = {
            cls: torch.tensor(list(meta.index), dtype=torch.long)
            for cls, meta in dataset.meta_data.groupby("Diagnosis")
        }
        self.class_list = list(self.class_indices.keys())

        self._regenerate_mixup_params()

    def _regenerate_mixup_params(self):
        """Pre-generate random decisions for the epoch (lambda sampled on-the-fly)."""
        generator = None
        if self.seed is not None:
            generator = torch.Generator()
            generator.manual_seed((self.seed + self._current_epoch) % _MAX_UINT32)

        self.mixup_decisions = (
            torch.rand(self.num_samples, generator=generator) > self.mixup_prob
        )

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        if self.seed is not None:
            seed = int((self.seed + self._current_epoch + idx) % _MAX_UINT32)
            rng = np.random.RandomState(seed)
            do_skip = bool(rng.rand() > self.mixup_prob)
            if do_skip:
                sample, target = self.dataset[idx]
                if self.transform is not None:
                    sample = self.transform(sample)
                return sample, target
        else:
            if self.mixup_decisions[idx]:
                sample, target = self.dataset[idx]
                if self.transform is not None:
                    sample = self.transform(sample)
                return sample, target

        # Get first sample
        sample1, target1 = self.dataset[idx]

        # Get its class
        cls1 = self.dataset.meta_data.Diagnosis.iloc[idx]

        # Randomly sample from a different class.
        if self.seed is not None:
            # Use the same numpy RNG to pick partner and alpha deterministically
            available_classes = [cls for cls in self.class_list if cls != cls1]
            cls2_idx = int(rng.randint(0, len(available_classes)))
            cls2 = available_classes[cls2_idx]
            cls2_indices = self.class_indices[cls2]
            idx2_pos = int(rng.randint(0, len(cls2_indices)))
            idx2 = int(cls2_indices[idx2_pos].item())
            sample2, target2 = self.dataset[idx2]

            # Sample alpha from Beta using numpy RNG
            alpha = float(rng.beta(self.alpha, self.alpha))
        else:
            # Uses Python's random module (seeded via worker_init_fn in DataLoader)
            available_classes = [cls for cls in self.class_list if cls != cls1]
            cls2 = random.choice(available_classes)
            cls2_indices = self.class_indices[cls2]
            idx2 = cls2_indices[random.randint(0, len(cls2_indices) - 1)].item()
            sample2, target2 = self.dataset[idx2]

            # Sample alpha on-the-fly using torch.distributions (worker RNG)
            alpha = self.dist.sample().item()

        # Clone before in-place ops to avoid corrupting cached data in the underlying dataset.
        sample1 = sample1.clone()
        target1 = target1.clone()
        sample1.mul_(alpha).add_(sample2, alpha=(1 - alpha))
        target1.mul_(alpha).add_(target2, alpha=(1 - alpha))

        if self.transform is not None:
            sample1 = self.transform(sample1)

        return sample1, target1

    def __len__(self) -> int:
        """Return the number of samples to generate."""
        return self.num_samples

    def set_epoch(self, epoch: int):
        """Regenerate mixup parameters for a new epoch."""
        self._current_epoch = int(epoch)
        self._regenerate_mixup_params()
