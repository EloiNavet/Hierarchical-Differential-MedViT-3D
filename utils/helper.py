"""Training utilities: schedulers, parameter grouping, path validators."""

import math
import os
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch


def _trunc_normal_(
    tensor: torch.Tensor,
    mean: float = 0.0,
    std: float = 1.0,
    a: float = -2.0,
    b: float = 2.0,
) -> torch.Tensor:
    """Fill a tensor with values drawn from a truncated normal distribution.

    This function fills the input tensor with values from a truncated normal distribution defined by
    the mean and standard deviation, ensuring that values fall within the specified bounds [a, b].

    This is the *internal* implementation used by :func:`trunc_normal_`.

    Note: Copy-pasted from PyTorch official master until it's in a few official releases - RW
    Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf

    Parameters
    ----------
    tensor : torch.Tensor
        The tensor to fill with truncated normal values.
    mean : float, optional
        The mean of the normal distribution. Default is 0.
    std : float, optional
        The standard deviation of the normal distribution. Default is 1.
    a : float, optional
        The minimum cutoff value. Default is -2.
    b : float, optional
        The maximum cutoff value. Default is 2.

    Returns
    -------
    torch.Tensor
        The input tensor filled with truncated normal values.
    """

    def norm_cdf(x):
        # Inner function for calculating the cumulative distribution function (CDF)
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn(
            "mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
            "The distribution of values may be incorrect.",
            stacklevel=2,
        )

    lower = norm_cdf((a - mean) / std)
    u = norm_cdf((b - mean) / std)

    # Uniformly fill tensor with values from [lower, u], then translate to
    # [2*lower-1, 2u-1].
    tensor.uniform_(2 * lower - 1, 2 * u - 1)

    # Use inverse cdf transform for normal distribution to get truncated
    # standard normal
    tensor.erfinv_()

    # Transform to proper mean, std
    tensor.mul_(std * math.sqrt(2.0))
    tensor.add_(mean)

    # Clamp to ensure it's in the proper range
    tensor.clamp_(min=a, max=b)
    return tensor


def trunc_normal_(
    tensor: torch.Tensor,
    mean: float = 0.0,
    std: float = 1.0,
    a: float = -2.0,
    b: float = 2.0,
) -> torch.Tensor:
    r"""Public API: fills the input Tensor with values drawn from a truncated
    normal distribution. The values are effectively drawn from the
    normal distribution :math:`\mathcal{N}(\text{mean}, \text{std}^2)`
    with values outside :math:`[a, b]` redrawn until they are within
    the bounds. The method used for generating the random values works
    best when :math:`a \leq \text{mean} \leq b`.
    NOTE: this impl is similar to the PyTorch trunc_normal_, the bounds [a, b] are
    applied while sampling the normal with mean/std applied, therefore a, b args
    should be adjusted to match the range of mean, std args.

    Parameters
    ----------
    tensor : torch.Tensor
        The n-dimensional tensor to fill with truncated normal values.
    mean : float, optional
        The mean of the normal distribution. Default is 0.
    std : float, optional
        The standard deviation of the normal distribution. Default is 1.
    a : float, optional
        The minimum cutoff value. Default is -2.
    b : float, optional
        The maximum cutoff value. Default is 2.

    Returns
    -------
    torch.Tensor
        The tensor filled with truncated normal values.
    """
    # Wrapper exists so callers have a stable public API while the sampling
    # logic lives in a testable internal helper.
    with torch.no_grad():
        return _trunc_normal_(tensor, mean, std, a, b)


def cosine_scheduler(
    base_value: float,
    final_value: float,
    total_steps: int,
    warmup_steps: int = 0,
    start_warmup_value: float = 0.0,
) -> np.ndarray:
    """Creates a cosine schedule based on steps.

    This function generates a cosine annealing schedule with an optional warmup phase.

    Parameters
    ----------
    base_value : float
        The initial learning rate.
    final_value : float
        The final learning rate.
    total_steps : int
        The total number of training steps.
    warmup_steps : int
        The number of warmup steps. Default is 0.
    start_warmup_value : float
        The starting learning rate during the warmup phase. Default is 0.0.

    Returns
    -------
    np.ndarray
        The learning rate schedule.
    """
    if warmup_steps > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_steps)
    else:
        warmup_schedule = np.array([])

    remaining_steps = total_steps - warmup_steps
    if remaining_steps > 0:
        steps = np.arange(remaining_steps)
        schedule = final_value + 0.5 * (base_value - final_value) * (
            1 + np.cos(np.pi * steps / remaining_steps)
        )
        schedule = np.concatenate((warmup_schedule, schedule))
    else:
        schedule = warmup_schedule

    assert len(schedule) == total_steps, (
        f"`len(schedule)` = {len(schedule)} != {total_steps} = `total_steps`"
    )
    return schedule


def get_params_groups(model: torch.nn.Module, config) -> list[dict]:
    """Organizes model parameters into groups for optimization.

    This function separates the model parameters into two groups: those that will be regularized
    (e.g., weight decay) and those that will not (e.g., biases).

    Parameters
    ----------
    model : torch.nn.Module
        The model whose parameters are to be grouped.
    config : Any
        Configuration object containing hyperparameters, including weight decay.
    Returns
    -------
    list[dict]
        A list containing two dictionaries: one for regularized parameters and another for
        non-regularized parameters.
    """
    regularized = []
    not_regularized = []

    skip_list = {}
    if hasattr(model, "no_weight_decay"):
        skip_list = model.no_weight_decay()

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        # we do not regularize biases nor Norm parameters
        if (
            name.endswith(".bias")
            or len(param.shape) == 1
            or "lambda_" in name
            or name in skip_list
        ):
            not_regularized.append(param)
        else:
            regularized.append(param)

    return [
        {"params": regularized, "weight_decay": config.WD_BASE},
        {
            "params": not_regularized,
            "weight_decay": 0.0,
            "skip_weight_decay": True,
        },
    ]


def stringify_object(obj, level=0):
    """Convert an object into a string representation.

    This function recursively retrieves the attributes of an object and formats them into a string.
    It handles nested objects by exploring their __dict__ attributes.

    Parameters
    ----------
    obj : Any
        The object to convert to string.
    level : int
        The current recursion depth (default is 0).

    Returns
    -------
    str
        The string representation of the object.
    """
    if not isinstance(obj, tuple):
        obj = (obj,)
    t = [getattr(o, "__dict__", None) for o in obj]
    t = [1 for o in t if o is not None]
    if sum(t) == 0:
        return str(obj) if len(obj) > 1 else str(obj[0])

    dict_obj = {}
    for i, o in enumerate(obj):
        if getattr(o, "__dict__", None) and getattr(o, "__class__", None):
            key = f"{i}_{str(o.__class__).split('.')[-1][:-2]}"
            key = key.replace("'", "")
            val = {}
            dct = o.__dict__
            for k, v in dct.items():
                if k.startswith("_") or k in ["R", "dtype"]:
                    continue
                if isinstance(v, torch.Tensor):
                    val[k] = str(v)
                else:
                    val[k] = stringify_object(v, level + 1)
            dict_obj[key] = val
    return str(dict_obj)


def file_path(string):
    """Check if the given string is a valid file path.

    Parameters
    ----------
    string : str
        The file path to check.

    Returns
    -------
    str
        The valid file path.

    Raises
    ------
    FileNotFoundError
        If the file path does not exist.
    """
    if os.path.isfile(string):
        return string
    else:
        raise FileNotFoundError(string)


def dir_path(string):
    """Check if the given string is a valid directory path.

    Parameters
    ----------
    string : str
        The directory path to check.

    Returns
    -------
    str
        The valid directory path.

    Raises
    ------
    NotADirectoryError
        If the directory path does not exist.
    """
    import os

    if not os.path.isdir(string):
        try:
            os.makedirs(string, exist_ok=True)
        except Exception as e:
            raise NotADirectoryError(f"Could not create directory: {string}") from e
    return string


def get_train_val_test(
    metadata_dir: str, fold: int, kfold: int, split: tuple[int, int, int] = (7, 2, 1)
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Retrieve training, validation, and testing metadata based on a specified fold.

    This function assumes a 10-fold cross-validation setup, where:
        - {split[0]} folds are used for training,
        - {split[1]} folds are used for validation,
        - {split[2]} fold is used for testing.

    The function reads the corresponding CSV files for each fold, concatenates them,
    and returns the datasets.

    Parameters
    ----------
    metadata_dir : str
        The directory containing the fold CSV files.
    fold : int
        The index of the fold configuration (0 to 9).
    kfold: int
        The total number of folds used in cross-validation.
    split : tuple[int, int, int], optional
        A tuple indicating the number of folds for training, validation, and testing.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]
        A tuple containing:
            - metadata_train: DataFrame with training data.
            - metadata_val: DataFrame with validation data.
            - metadata_test: DataFrame with testing data.
            - metadata_all: DataFrame with all combined data (training + validation + testing).
    """
    assert len(split) == 3, (
        "The split must contain three values (train, validation, test)."
    )
    assert kfold >= 3, (
        "kfold must be at least 3 (one fold for each train, test and validation)."
    )
    assert sum(split) == kfold, f"The sum of split values must equal {kfold}."
    assert fold in range(kfold), f"Fold must be between 0 and {kfold - 1}."
    trains, vals, tests = [], [], []

    files = list(Path(metadata_dir).glob("fold_*.csv"))
    fold_files = [file for file in files if re.match(r"fold_[0-9]+\.csv", file.name)]
    assert len(fold_files) == kfold, f"Number of fold files is not {kfold}."

    # Custom split (using split[0], split[1], split[2])
    for i in range(split[0]):
        ind = (fold + i) % kfold
        trains.append(pd.read_csv(os.path.join(metadata_dir, f"fold_{ind}.csv")))
    for i in range(split[0], split[0] + split[1]):
        ind = (fold + i) % kfold
        vals.append(pd.read_csv(os.path.join(metadata_dir, f"fold_{ind}.csv")))
    for i in range(split[0] + split[1], kfold):
        ind = (fold + i) % kfold
        tests.append(pd.read_csv(os.path.join(metadata_dir, f"fold_{ind}.csv")))

    metadata_train = pd.concat(trains, ignore_index=True).reset_index(drop=True)
    metadata_val = pd.concat(vals, ignore_index=True).reset_index(drop=True)
    metadata_test = pd.concat(tests, ignore_index=True).reset_index(drop=True)
    metadata_all = (
        pd.concat([metadata_train, metadata_val, metadata_test])
        .sort_values(by="Subject")
        .reset_index(drop=True)
    )

    return metadata_train, metadata_val, metadata_test, metadata_all


def print_sensitivity(gt: np.ndarray, pred: np.ndarray, cls: int) -> float:
    """Calculate and return sensitivity for a given class.

    Parameters
    ----------
    gt : np.ndarray
        Ground truth labels.
    pred : np.ndarray
        Predicted labels.
    cls : int
        Class index to calculate sensitivity for.

    Returns
    -------
    float
        Sensitivity score (true positive rate) for the specified class.
    """
    tp = ((gt == cls) & (pred == cls)).sum()
    fn = ((gt == cls) & (pred != cls)).sum()
    sensitivity = tp / (tp + fn)
    return sensitivity


def get_model_size(model: torch.nn.Module) -> float:
    """Calculate the model size in megabytes.

    This function calculates the total memory usage of a PyTorch model by summing
    the size of all parameters and buffers.

    Parameters
    ----------
    model : torch.nn.Module
        The PyTorch model to measure.

    Returns
    -------
    float
        Model size in megabytes.
    """
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    size_all_mb = (param_size + buffer_size) / 1024**2
    return size_all_mb


def count_parameters(model: torch.nn.Module) -> int:
    """Count the number of trainable parameters in a model.

    Parameters
    ----------
    model : torch.nn.Module
        The PyTorch model to analyze.

    Returns
    -------
    int
        Number of trainable parameters in the model.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
