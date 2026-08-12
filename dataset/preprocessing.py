import os
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import torch
from joblib import Parallel, delayed
from monai.transforms import CenterSpatialCrop, Transform
from nilearn.image import resample_img
from tqdm import tqdm

# SLANT brain segmentation labels (133 regions)
# Reference: https://github.com/MASILab/SLANTbrainSeg/blob/master/BrainColorLUT.txt
LABELS_SLANT: dict[int, str] = {
    0: "Background",
    4: "3rd-Ventricle",
    11: "4th-Ventricle",
    23: "Right-Accumbens-Area",
    30: "Left-Accumbens-Area",
    31: "Right-Amygdala",
    32: "Left-Amygdala",
    35: "Brain-Stem",
    36: "Right-Caudate",
    37: "Left-Caudate",
    38: "Right-Cerebellum-Exterior",
    39: "Left-Cerebellum-Exterior",
    40: "Right-Cerebellum-White-Matter",
    41: "Left-Cerebellum-White-Matter",
    44: "Right-Cerebral-White-Matter",
    45: "Left-Cerebral-White-Matter",
    47: "Right-Hippocampus",
    48: "Left-Hippocampus",
    49: "Right-Inf-Lat-Vent",
    50: "Left-Inf-Lat-Vent",
    51: "Right-Lateral-Ventricle",
    52: "Left-Lateral-Ventricle",
    55: "Right-Pallidum",
    56: "Left-Pallidum",
    57: "Right-Putamen",
    58: "Left-Putamen",
    59: "Right-Thalamus-Proper",
    60: "Left-Thalamus-Proper",
    61: "Right-Ventral-DC",
    62: "Left-Ventral-DC",
    71: "Cerebellar-Vermal-Lobules-I-V",
    72: "Cerebellar-Vermal-Lobules-VI-VII",
    73: "Cerebellar-Vermal-Lobules-VIII-X",
    75: "Left-Basal-Forebrain",
    76: "Right-Basal-Forebrain",
    100: "Right-ACgG--anterior-cingulate-gyrus",
    101: "Left-ACgG--anterior-cingulate-gyrus",
    102: "Right-AIns--anterior-insula",
    103: "Left-AIns--anterior-insula",
    104: "Right-AOrG--anterior-orbital-gyrus",
    105: "Left-AOrG--anterior-orbital-gyrus",
    106: "Right-AnG---angular-gyrus",
    107: "Left-AnG---angular-gyrus",
    108: "Right-Calc--calcarine-cortex",
    109: "Left-Calc--calcarine-cortex",
    112: "Right-CO----central-operculum",
    113: "Left-CO----central-operculum",
    114: "Right-Cun---cuneus",
    115: "Left-Cun---cuneus",
    116: "Right-Ent---entorhinal-area",
    117: "Left-Ent---entorhinal-area",
    118: "Right-FO----frontal-operculum",
    119: "Left-FO----frontal-operculum",
    120: "Right-FRP---frontal-pole",
    121: "Left-FRP---frontal-pole",
    122: "Right-FuG---fusiform-gyrus",
    123: "Left-FuG---fusiform-gyrus",
    124: "Right-GRe---gyrus-rectus",
    125: "Left-GRe---gyrus-rectus",
    128: "Right-IOG---inferior-occipital-gyrus",
    129: "Left-IOG---inferior-occipital-gyrus",
    132: "Right-ITG---inferior-temporal-gyrus",
    133: "Left-ITG---inferior-temporal-gyrus",
    134: "Right-LiG---lingual-gyrus",
    135: "Left-LiG---lingual-gyrus",
    136: "Right-LOrG--lateral-orbital-gyrus",
    137: "Left-LOrG--lateral-orbital-gyrus",
    138: "Right-MCgG--middle-cingulate-gyrus",
    139: "Left-MCgG--middle-cingulate-gyrus",
    140: "Right-MFC---medial-frontal-cortex",
    141: "Left-MFC---medial-frontal-cortex",
    142: "Right-MFG---middle-frontal-gyrus",
    143: "Left-MFG---middle-frontal-gyrus",
    144: "Right-MOG---middle-occipital-gyrus",
    145: "Left-MOG---middle-occipital-gyrus",
    146: "Right-MOrG--medial-orbital-gyrus",
    147: "Left-MOrG--medial-orbital-gyrus",
    148: "Right-MPoG--postcentral-gyrus",
    149: "Left-MPoG--postcentral-gyrus",
    150: "Right-MPrG--precentral-gyrus",
    151: "Left-MPrG--precentral-gyrus",
    152: "Right-MSFG--superior-frontal-gyrus",
    153: "Left-MSFG--superior-frontal-gyrus",
    154: "Right-MTG---middle-temporal-gyrus",
    155: "Left-MTG---middle-temporal-gyrus",
    156: "Right-OCP---occipital-pole",
    157: "Left-OCP---occipital-pole",
    160: "Right-OFuG--occipital-fusiform-gyrus",
    161: "Left-OFuG--occipital-fusiform-gyrus",
    162: "Right-OpIFG-opercular-part-of-the-IFG",
    163: "Left-OpIFG-opercular-part-of-the-IFG",
    164: "Right-OrIFG-orbital-part-of-the-IFG",
    165: "Left-OrIFG-orbital-part-of-the-IFG",
    166: "Right-PCgG--posterior-cingulate-gyrus",
    167: "Left-PCgG--posterior-cingulate-gyrus",
    168: "Right-PCu---precuneus",
    169: "Left-PCu---precuneus",
    170: "Right-PHG---parahippocampal-gyrus",
    171: "Left-PHG---parahippocampal-gyrus",
    172: "Right-PIns--posterior-insula",
    173: "Left-PIns--posterior-insula",
    174: "Right-PO----parietal-operculum",
    175: "Left-PO----parietal-operculum",
    176: "Right-PoG---postcentral-gyrus",
    177: "Left-PoG---postcentral-gyrus",
    178: "Right-POrG--posterior-orbital-gyrus",
    179: "Left-POrG--posterior-orbital-gyrus",
    180: "Right-PP----planum-polare",
    181: "Left-PP----planum-polare",
    182: "Right-PrG---precentral-gyrus",
    183: "Left-PrG---precentral-gyrus",
    184: "Right-PT----planum-temporale",
    185: "Left-PT----planum-temporale",
    186: "Right-SCA---subcallosal-area",
    187: "Left-SCA---subcallosal-area",
    190: "Right-SFG---superior-frontal-gyrus",
    191: "Left-SFG---superior-frontal-gyrus",
    192: "Right-SMC---supplementary-motor-cortex",
    193: "Left-SMC---supplementary-motor-cortex",
    194: "Right-SMG---supramarginal-gyrus",
    195: "Left-SMG---supramarginal-gyrus",
    196: "Right-SOG---superior-occipital-gyrus",
    197: "Left-SOG---superior-occipital-gyrus",
    198: "Right-SPL---superior-parietal-lobule",
    199: "Left-SPL---superior-parietal-lobule",
    200: "Right-STG---superior-temporal-gyrus",
    201: "Left-STG---superior-temporal-gyrus",
    202: "Right-TMP---temporal-pole",
    203: "Left-TMP---temporal-pole",
    204: "Right-TrIFG-triangular-part-of-the-IFG",
    205: "Left-TrIFG-triangular-part-of-the-IFG",
    206: "Right-TTG---transverse-temporal-gyrus",
    207: "Left-TTG---transverse-temporal-gyrus",
}

# Dense re-indexing: map sparse SLANT label IDs (0, 4, 11, ..., 207) to contiguous
# integers (0, 1, 2, ..., 132). This ensures uniform spacing when normalizing to [0, 1]
# and avoids the ordinal encoding problem of dividing by MAX_SLANT_LABEL.
_SLANT_LABEL_TO_DENSE = {
    label: idx for idx, label in enumerate(sorted(LABELS_SLANT.keys()))
}
NUM_SLANT_LABELS = len(LABELS_SLANT)  # 133

# Pre-build a lookup table (LUT) for fast vectorized remapping
_MAX_SLANT_RAW = max(LABELS_SLANT.keys())  # 207
_SLANT_DENSE_LUT = np.zeros(_MAX_SLANT_RAW + 1, dtype=np.float32)
for raw_label, dense_idx in _SLANT_LABEL_TO_DENSE.items():
    _SLANT_DENSE_LUT[raw_label] = dense_idx / (
        NUM_SLANT_LABELS - 1
    )  # normalize to [0, 1]


class NilearnDownsample(Transform):
    """Downsample a 3D tensor using nilearn's resample_img.

    Parameters
    ----------
    new_size : tuple[int, int, int]
        Target dimensions (D, H, W) for the downsampled image.
    interpolation : str, default="linear"
        Interpolation method: "linear" for continuous data (T1),
        "nearest" for categorical data (segmentation labels).
    """

    def __init__(self, new_size: tuple[int, int, int], interpolation: str = "linear"):
        self.new_size = new_size
        self.interpolation = interpolation

    def __call__(self, tensor_img):
        # We assume 'tensor_img' is already a nib-compatible array
        # Convert torch.Tensor to nib.Nifti1Image
        affine = np.eye(4)
        nib_img = nib.Nifti1Image(tensor_img.cpu().numpy(), affine)

        # Compute new zooms via ratio of old/new shapes
        old_shape = nib_img.shape
        zoom_factors = [old / new for old, new in zip(old_shape, self.new_size)]
        new_affine = np.diag([affine[0, 0] * zoom for zoom in zoom_factors] + [1])

        # Downsample using nilearn
        resampled_img = resample_img(
            nib_img, target_affine=new_affine, interpolation=self.interpolation
        )
        data_downsampled = torch.from_numpy(resampled_img.get_fdata()).float()
        return data_downsampled


# Maximum SLANT segmentation label value (for normalization to [0, 1])
MAX_SLANT_LABEL = 207


class DataPrepa:
    """Preprocess T1 MRI + Segmentation: skull-strip, crop, save as float16 tensors.

    Creates 2-channel tensors: channel 0 = T1 (masked), channel 1 = Segmentation (normalized).
    Expects metadata with columns: Subject, T1_path, Mask_path, Seg_path, Diagnosis.
    """

    def __init__(
        self,
        metadata: str | pd.DataFrame,
        has_seg: bool,
        preprocess_data_dir: str,
        device: torch.device,
    ):
        self.metadata = (
            metadata if isinstance(metadata, pd.DataFrame) else pd.read_csv(metadata)
        )
        self.has_seg = has_seg
        self.preprocess_data_dir = preprocess_data_dir
        self.device = device

    def _process_single_subject(self, subject_data, crop, downsample_size):
        """
        Process a single subject's MRI data (T1 + Segmentation).

        Creates a 2-channel tensor:
        - Channel 0: T1-weighted MRI (masked with brain mask)
        - Channel 1: Segmentation labels (dense re-indexed and normalized to [0, 1])

        Parameters
        ----------
        subject_data : pd.Series
            Row from metadata containing subject information (T1_path, Mask_path, Seg_path)
        crop : tuple[int, int, int]
            The size of the ROI to crop the image
        downsample_size : tuple[int, int, int] or None
            The size to downsample the image to

        Returns
        -------
        bool
            True if processing was successful, False otherwise
        """
        try:
            # Need to use CPU for joblib parallel processing
            local_device = torch.device("cpu")

            # Load T1 image, brain mask, and segmentation if needed
            t1 = torch.from_numpy(nib.load(subject_data.T1_path).get_fdata()).to(
                local_device
            )
            mask = torch.from_numpy(nib.load(subject_data.Mask_path).get_fdata()).to(
                local_device
            )
            if self.has_seg:
                seg = torch.from_numpy(nib.load(subject_data.Seg_path).get_fdata()).to(
                    local_device
                )

            # Apply mask
            t1 = t1 * mask

            if self.has_seg:
                seg = seg * mask

            # Crop both images identically
            crop_transform = CenterSpatialCrop(crop)
            t1 = crop_transform(t1.unsqueeze(0)).squeeze(0)  # (D, H, W)
            if self.has_seg:
                seg = crop_transform(seg.unsqueeze(0)).squeeze(0)  # (D, H, W)

            # Downsample if needed (different interpolation for T1 vs Seg)
            if downsample_size and downsample_size != crop:
                # T1: use linear interpolation for smooth intensity values
                downsample_t1 = NilearnDownsample(
                    downsample_size, interpolation="linear"
                )
                t1 = downsample_t1(t1)

                if self.has_seg:
                    # Segmentation: use nearest interpolation to preserve discrete labels
                    downsample_seg = NilearnDownsample(
                        downsample_size, interpolation="nearest"
                    )
                    seg = downsample_seg(seg)

            # Re-index segmentation labels to dense [0, 1] range
            # Maps sparse SLANT labels (0, 4, 11, ..., 207) to uniformly-spaced
            # values (0/132, 1/132, ..., 132/132) using a pre-built lookup table.
            if self.has_seg:
                seg_np = seg.numpy().astype(np.int32)
                seg_np = np.clip(seg_np, 0, _MAX_SLANT_RAW)
                seg = torch.from_numpy(_SLANT_DENSE_LUT[seg_np])

            # Stack into 1/2-channel tensor: (1/2, D, H, W) depending on whether segmentation is included
            if self.has_seg:
                combined = torch.stack([t1, seg], dim=0)
            else:
                combined = t1.unsqueeze(0)

            # Convert to float16 for memory efficiency
            combined = combined.type(torch.float16)

            # Save the processed image
            output_path = Path(self.preprocess_data_dir) / f"{subject_data.Subject}.pt"
            torch.save(combined, output_path)

            return True
        except Exception as e:  # noqa: BLE001 - isolate failures per subject
            print(f"Error processing subject {subject_data.Subject}: {e}")
            return False

    def preprocess_data(
        self,
        crop: tuple[int, int, int],
        downsample: tuple[int, int, int] | None = None,
        tqdm_kwargs: dict[str, str | int] | None = None,
        n_jobs: int = -1,
        batch_size: int = 1,
        verbose: int = 0,
        backend: str = "threading",
        prefer: str = "threads",
    ):
        """
        Preprocesses the MRI data in parallel and saves the results to disk.
        Parameters
        ----------
        crop : tuple[int, int, int]
            The size of the ROI to crop the image.
        downsample : tuple[int, int, int], optional
            The size to downsample the image to. If None, no downsampling is performed.
        tqdm_kwargs : dict, optional
            Keyword arguments to pass to the tqdm progress bar.
        n_jobs : int, default=-1
            Number of jobs to run in parallel. -1 means using all processors.
        batch_size : int, default=1
            Number of subjects to process per job.
        verbose : int, default=0
            The verbosity level: 0 for quiet, higher for more verbose output.
        backend : str, default='threading'
            The backend to use for parallelization. Options: 'threading', 'multiprocessing', 'loky'.
            On clusters, 'threading' is often more stable than 'multiprocessing' or 'loky'.
        prefer : str, default='threads'
            Preference when choosing backend. Options: 'processes', 'threads'.
            For MPI compatibility, 'threads' is recommended.
        """
        # Ensure crop has the correct length
        if len(crop) != 3:
            raise ValueError(f"Crop size must be a tuple of length 3, got {len(crop)}.")

        # Create output directory
        os.makedirs(self.preprocess_data_dir, exist_ok=True)

        # Check correspondence between df subjects and preprocessed files
        metadata_to_process = self._check_subjects_files_correspondence()
        if metadata_to_process is None:  # Perfect match, nothing to process
            return

        # Set default tqdm parameters if not provided
        tqdm_default_kwargs = {
            "desc": "Data preprocessing",
            "position": 0,
            "dynamic_ncols": True,
            "ncols": 80,
        }
        # Update defaults with provided kwargs if kwargs is not None
        if tqdm_kwargs:
            tqdm_default_kwargs.update(tqdm_kwargs)

        if verbose > 0:
            print(
                f"Starting parallel processing with {n_jobs} jobs using {backend} backend..."
            )

        # Try to use threading backend which has better compatibility with MPI environments
        try:
            # Process subjects in parallel
            results = Parallel(
                n_jobs=n_jobs,
                verbose=verbose,
                batch_size=batch_size,
                backend=backend,
                prefer=prefer,
            )(
                delayed(self._process_single_subject)(
                    metadata_to_process.iloc[i], crop, downsample
                )
                for i in tqdm(range(len(metadata_to_process)), **tqdm_default_kwargs)
            )

            # Report results
            success_count = sum(results)
            if verbose > 0:
                print(
                    f"Processed {success_count} out of {len(metadata_to_process)} subjects successfully."
                )

        except Exception as e:  # noqa: BLE001 - fall back to sequential processing
            print(f"Parallel processing failed with error: {e}")
            print("Falling back to sequential processing...")

            # Fall back to sequential processing if parallel fails
            results = []
            for i in tqdm(range(len(metadata_to_process)), **tqdm_default_kwargs):
                result = self._process_single_subject(
                    metadata_to_process.iloc[i], crop, downsample
                )
                results.append(result)

            success_count = sum(results)
            if verbose > 0:
                print(
                    f"Processed {success_count} out of {len(metadata_to_process)} subjects successfully."
                )

    def _check_subjects_files_correspondence(self):
        """
        Checks correspondence between subjects in dataframe and files in preprocessed directory.

        Returns:
        --------
        pd.DataFrame or None:
            Subset of metadata for subjects that need processing, or None if all subjects are processed.
        """
        preprocess_dir = Path(self.preprocess_data_dir)

        # Get subject IDs from dataframe and files
        df_subjects = set(self.metadata.Subject.values)
        file_subjects = {f.stem for f in preprocess_dir.glob("*.pt")}

        # Identify subjects to process and files to remove
        subjects_to_process = df_subjects - file_subjects
        files_to_remove = file_subjects - df_subjects

        # Remove files that are not in the dataframe
        if files_to_remove:
            print(f"Removing {len(files_to_remove)} files not in metadata.")
            for subject in files_to_remove:
                os.remove(preprocess_dir / f"{subject}.pt")

        if not subjects_to_process:
            return None

        print(
            f"Processing {len(subjects_to_process)} subjects out of {len(df_subjects)}."
        )
        return self.metadata[self.metadata.Subject.isin(subjects_to_process)]


def average_by_structure(seg: Path, mask: Path) -> np.ndarray:
    """
    Calculate the average volume of each region in the SLANT atlas.

    Parameters
    ----------
    seg : Path
        Path to the segmentation image.
    mask : Path
        Path to the mask image.

    Returns
    -------
    np.ndarray
        The average volume of each region in the SLANT atlas.
    """
    # Load the data
    seg = nib.load(seg).get_fdata().astype(np.uint8)
    mask = nib.load(mask).get_fdata().astype(np.uint8)

    # Create a view of the segmentation where the mask is non-zero
    masked_seg = seg[mask != 0]

    # Get unique labels and their counts
    labels, counts = np.unique(masked_seg, return_counts=True)

    slant_keys = np.array(list(LABELS_SLANT.keys()))
    max_slant_label = slant_keys.max()
    label_counts = np.zeros(max_slant_label + 1, dtype=counts.dtype)
    label_counts[labels] = counts
    avg_structure = label_counts[slant_keys]

    assert len(avg_structure) == len(slant_keys), (
        f"Expected {len(slant_keys)} structures, got {len(avg_structure)}"
    )

    # Normalize by total volume
    avg_structure = avg_structure / mask.sum()

    return avg_structure


def load_svm_features(
    preprocess_dir: Path, metadata: pd.DataFrame, diseases: list[str]
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load preprocessed features directly into NumPy arrays without Dataset/DataLoader overhead.

    Parameters
    ----------
    preprocess_dir : Path
        Directory containing preprocessed .pt files
    metadata : pd.DataFrame
        Metadata DataFrame with Subject and Diagnosis columns
    diseases : list[str]
        List of disease labels

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Feature matrix X and label vector Y
    """
    X_list = []
    Y_list = []

    for idx in range(len(metadata)):
        subject_id = metadata.Subject.iloc[idx]
        diagnosis = metadata.Diagnosis.iloc[idx]

        feature_path = preprocess_dir / f"{subject_id}.pt"
        features = torch.load(feature_path, map_location="cpu", weights_only=False)

        X_list.append(features.cpu().numpy())

        y_idx = diseases.index(diagnosis)
        Y_list.append(y_idx)

    X = np.vstack(X_list)
    Y = np.array(Y_list)

    return X, Y


class DataPrepaSVM:
    """Data Preparation for SVM using segmented volumes with parallel processing.

    This class preprocesses MRI segmentation data by computing average volumes
    per brain structure from SLANT atlas. Processing can be done in parallel
    using joblib for faster preprocessing.

    Parameters
    ----------
    metadata : str | pd.DataFrame
        Path to the metadata CSV file or the metadata DataFrame itself.
    preprocess_data_dir : str | Path
        Directory to save the preprocessed data.
    device : str | torch.device
        Device for processing (CPU recommended for SVM preprocessing).
    """

    def __init__(
        self,
        metadata: str | pd.DataFrame,
        preprocess_data_dir: str | Path,
        device: str | torch.device = "cpu",
    ):
        self.metadata = (
            metadata if isinstance(metadata, pd.DataFrame) else pd.read_csv(metadata)
        )
        self.device = torch.device(device) if isinstance(device, str) else device
        self.preprocess_data_dir = Path(preprocess_data_dir)

    def _process_single_subject(self, subject_data):
        """Process a single subject's segmentation data.

        Parameters
        ----------
        subject_data : pd.Series
            Row from metadata containing subject information

        Returns
        -------
        bool
            True if processing was successful, False otherwise
        """
        try:
            avg_structure = average_by_structure(
                subject_data.Seg_path,
                subject_data.Mask_path,
            )
            avg_structure = torch.from_numpy(avg_structure)
            output_path = self.preprocess_data_dir / f"{subject_data.Subject}.pt"
            torch.save(avg_structure, output_path)
            return True
        except Exception as e:  # noqa: BLE001 - isolate failures per subject
            print(f"Error processing subject {subject_data.Subject}: {e}")
            return False

    def _check_subjects_files_correspondence(self):
        """Check correspondence between subjects in dataframe and preprocessed files.

        Returns
        -------
        pd.DataFrame or None
            Subset of metadata for subjects that need processing, or None if all processed.
        """
        # Get subject IDs from dataframe and files
        df_subjects = set(self.metadata.Subject.values)
        file_subjects = {f.stem for f in self.preprocess_data_dir.glob("*.pt")}

        # Identify subjects to process and files to remove
        subjects_to_process = df_subjects - file_subjects
        files_to_remove = file_subjects - df_subjects

        # Remove files that are not in the dataframe
        if files_to_remove:
            print(f"Removing {len(files_to_remove)} files not in metadata.")
            for subject in files_to_remove:
                os.remove(self.preprocess_data_dir / f"{subject}.pt")

        if not subjects_to_process:
            return None

        print(
            f"Processing {len(subjects_to_process)} subjects out of {len(df_subjects)}."
        )
        return self.metadata[self.metadata.Subject.isin(subjects_to_process)]

    def preprocess_data(
        self,
        tqdm_kwargs: dict[str, str | int] | None = None,
        n_jobs: int = -1,
        batch_size: int = 1,
        verbose: int = 0,
    ):
        """
        Preprocesses the MRI segmentation volumes in parallel and saves results to disk.

        Parameters
        ----------
        tqdm_kwargs : dict, optional
            Keyword arguments to pass to the tqdm progress bar.
        n_jobs : int, default=-1
            Number of jobs to run in parallel. -1 means using all processors.
        batch_size : int, default=1
            Number of subjects to process per job.
        verbose : int, default=0
            The verbosity level: 0 for quiet, higher for more verbose output.
        """
        os.makedirs(self.preprocess_data_dir, exist_ok=True)

        # Check correspondence between df subjects and preprocessed files
        metadata_to_process = self._check_subjects_files_correspondence()
        if metadata_to_process is None:  # Perfect match, nothing to process
            return

        tqdm_default_kwargs = {
            "desc": "SVM preprocessing",
            "position": 0,
            "dynamic_ncols": True,
            "ncols": 80,
        }
        if tqdm_kwargs:
            tqdm_default_kwargs.update(tqdm_kwargs)

        if verbose > 0:
            print(f"Starting parallel processing with {n_jobs} jobs...")

        results = Parallel(
            n_jobs=n_jobs,
            verbose=0,
            batch_size=batch_size,
        )(
            delayed(self._process_single_subject)(metadata_to_process.iloc[i])
            for i in tqdm(range(len(metadata_to_process)), **tqdm_default_kwargs)
        )

        success_count = sum(results)
        if verbose > 0:
            print(
                f"Processed {success_count} out of {len(metadata_to_process)} subjects successfully."
            )
