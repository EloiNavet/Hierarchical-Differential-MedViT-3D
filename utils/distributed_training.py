"""PyTorch Distributed Data Parallel (DDP) utilities."""

import inspect
import os
import sys

import torch
import torch.distributed as dist


def init_distributed_mode(args):
    """
    Initialize the distributed training environment.

    This function sets up the distributed training mode for PyTorch. It checks if the
    process was launched with distributed arguments and initializes the process group
    accordingly. It also handles configurations for running on a single GPU or
    multiple GPUs.

    Parameters
    ----------
    args : Namespace or wandb.Config
        Command-line arguments containing rank, world size, and other settings.
    """
    # Check if launched with torch.distributed.launch
    # launched with torch.distributed.launch
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        # Check if args is a wandb config object
        if hasattr(args, "update"):
            # WandB config object - use update with allow_val_change
            args.update({"rank": int(os.environ["RANK"])}, allow_val_change=True)
            args.update(
                {"world_size": int(os.environ["WORLD_SIZE"])}, allow_val_change=True
            )
            args.update({"gpu": int(os.environ["LOCAL_RANK"])}, allow_val_change=True)
        else:
            # Regular argparse Namespace - use direct assignment
            args.rank = int(os.environ["RANK"])
            args.world_size = int(os.environ["WORLD_SIZE"])
            args.gpu = int(os.environ["LOCAL_RANK"])

    # Check if launched without distributed (single GPU)
    # launched naively with `python main_dino.py`
    # we manually add MASTER_ADDR and MASTER_PORT to env variables
    elif torch.cuda.is_available():
        print("Will run the code on one GPU.")
        if hasattr(args, "update"):
            args.update({"rank": 0}, allow_val_change=True)
            args.update({"gpu": 0}, allow_val_change=True)
            args.update({"world_size": 1}, allow_val_change=True)
        else:
            args.rank, args.gpu, args.world_size = 0, 0, 1
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = "29500"
    else:
        print("Does not support training without GPU.")
        sys.exit(1)

    # Set the device for the current GPU BEFORE initializing the process group
    # This avoids potential NCCL hangs due to using an unexpected default device
    torch.cuda.set_device(args.gpu)

    # Initialize the distributed process group
    init_kwargs = {
        "backend": "nccl",
        "init_method": "env://",
        "world_size": args.world_size,
        "rank": args.rank,
    }

    if "device_id" in inspect.signature(dist.init_process_group).parameters:
        init_kwargs["device_id"] = torch.device("cuda", args.gpu)

    dist.init_process_group(**init_kwargs)

    print(f"| distributed init (rank {args.rank}): env://", flush=True)
    # Synchronize all processes (no device_ids arg; deprecated and can hang)
    dist.barrier()
    setup_for_distributed(args.rank == 0)  # Configure printing for master process


def setup_for_distributed(is_master):
    """
    Disable printing for non-master processes.

    This function redefines the print function to disable output unless
    the current process is the master process.

    Parameters
    ----------
    is_master : bool
        Indicates if the current process is the master process.
    """
    import builtins as __builtin__

    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop("force", False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print  # Override the built-in print function


def is_dist_avail_and_initialized():
    """
    Check if distributed training is available and initialized.

    Returns
    -------
    bool
        True if distributed training is available and initialized, False otherwise.
    """
    return not (not dist.is_available() or not dist.is_initialized())


def get_rank():
    """
    Get the rank of the current process in distributed training.

    Returns
    -------
    int
        The rank of the current process (0 if not initialized).
    """
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    """
    Check if the current process is the main process.

    Returns
    -------
    bool
        True if the current process is the main process (rank 0), False otherwise.
    """
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    """
    Save a checkpoint or model state on the master process.

    This function checks if the current process is the master and saves the
    provided arguments using torch.save.

    Parameters
    ----------
    *args : Variable length argument list
        Arguments to pass to torch.save.
    **kwargs : Arbitrary keyword arguments
        Keyword arguments to pass to torch.save.
    """
    if is_main_process():
        torch.save(*args, **kwargs)


def get_world_size():
    """
    Get the world size of the distributed training.

    Returns
    -------
    int
        The world size (number of processes).
    """
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()
