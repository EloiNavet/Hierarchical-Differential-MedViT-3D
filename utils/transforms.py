from collections.abc import Sequence

import torch
from monai.transforms import RandAffine, Resize, Transform


class ChannelSelectiveTransform(Transform):
    """
    Wrapper that applies a MONAI transform only to specified channels.

    For multimodal inputs (e.g., T1 + Segmentation), intensity-based transforms
    should only be applied to the T1 channel (channel 0) while leaving the
    segmentation channel (channel 1) intact.

    Parameters
    ----------
    transform : Transform
        The MONAI transform to apply selectively.
    channels : List[int] or int
        Channel indices to apply the transform to. Other channels are preserved.

    Example
    -------
    >>> # Apply RandBiasField only to T1 channel (channel 0)
    >>> selective_bias = ChannelSelectiveTransform(RandBiasField(prob=0.3), channels=[0])
    >>> output = selective_bias(input_tensor)  # input_tensor shape: (2, D, H, W)
    """

    def __init__(self, transform: Transform, channels: int | list[int] | Sequence[int]):
        super().__init__()
        self.transform = transform
        self.channels = [channels] if isinstance(channels, int) else list(channels)

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        """
        Apply transform selectively to specified channels.

        Parameters
        ----------
        img : torch.Tensor
            Input tensor of shape (C, D, H, W) where C is the number of channels.

        Returns
        -------
        torch.Tensor
            Output tensor with transform applied only to specified channels.
        """
        if img.shape[0] == 1:
            # Single channel: apply transform directly (backward compatible)
            return self.transform(img)

        # Multi-channel: apply selectively
        result = img.clone()
        for ch in self.channels:
            if ch < img.shape[0]:
                # Extract single channel, add channel dim for MONAI compatibility
                channel_data = img[ch : ch + 1]  # Shape: (1, D, H, W)
                transformed = self.transform(channel_data)
                result[ch] = transformed[0]

        return result


class ChannelWiseNormalize(Transform):
    """
    Normalize each channel independently using z-score normalization.

    For multimodal inputs, this applies separate normalization to each channel,
    which is essential when channels have different value ranges/distributions
    (e.g., T1 intensity vs segmentation labels normalized to [0,1]).

    Parameters
    ----------
    subtrahend : float or None
        Value to subtract. If None, uses channel mean.
    divisor : float or None
        Value to divide by. If None, uses channel std.
    nonzero : bool
        If True, compute statistics only on non-zero values.
    channel_wise : bool
        If True (default), normalize each channel independently.
        If False, normalize across all channels together.
    """

    def __init__(
        self,
        subtrahend: float | None = None,
        divisor: float | None = None,
        nonzero: bool = False,
        channel_wise: bool = True,
    ):
        super().__init__()
        self.subtrahend = subtrahend
        self.divisor = divisor
        self.nonzero = nonzero
        self.channel_wise = channel_wise

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        """
        Normalize tensor channel-wise or globally.

        Parameters
        ----------
        img : torch.Tensor
            Input tensor of shape (C, D, H, W).

        Returns
        -------
        torch.Tensor
            Normalized tensor.
        """
        if not self.channel_wise or img.shape[0] == 1:
            # Global normalization
            if self.nonzero:
                mask = img != 0
                vals = img[mask]
            else:
                vals = img.flatten()

            mean = self.subtrahend if self.subtrahend is not None else vals.mean()
            std = self.divisor if self.divisor is not None else vals.std()

            return (img - mean) / (std + 1e-8)

        # Channel-wise normalization
        result = img.clone()
        for ch in range(img.shape[0]):
            channel = img[ch]

            if self.nonzero:
                mask = channel != 0
                vals = channel[mask]
            else:
                vals = channel.flatten()

            mean = self.subtrahend if self.subtrahend is not None else vals.mean()
            std = self.divisor if self.divisor is not None else vals.std()

            result[ch] = (channel - mean) / (std + 1e-8)

        return result


class AdaptiveGaussianNoise(Transform):
    """
    Applies temporary normalization, adds Gaussian noise, then restores original scale.

    Uses PyTorch's random state for full reproducibility. All randomness (probability
    check and noise generation) is controlled by torch RNG state.
    """

    def __init__(self, prob: float = 0.1, noise_factor: float = 0.1):
        super().__init__()
        self.prob = prob
        self.noise_factor = noise_factor

    def __call__(self, img):
        if torch.rand(1).item() < self.prob:
            orig_min = torch.min(img)
            orig_max = torch.max(img)

            img_normalized = (img - orig_min) / (orig_max - orig_min + 1e-8)

            noise = torch.randn_like(img_normalized) * self.noise_factor
            img_normalized = img_normalized + noise

            img = img_normalized * (orig_max - orig_min) + orig_min

        return img


class AdaptiveRicianNoise(Transform):
    """
    Applies Rician noise while preserving the original image scale.
    Rician noise is the actual noise distribution in magnitude MR images.

    Note: Uses torch's global random state for reproducibility.
    Ensure torch is seeded for deterministic behavior.
    """

    def __init__(self, prob: float = 0.1, noise_factor: float = 0.1):
        super().__init__()
        self.prob = prob
        self.noise_factor = noise_factor

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        if torch.rand(1).item() < self.prob:
            orig_min = torch.min(img)
            orig_max = torch.max(img)

            img_normalized = (img - orig_min) / (orig_max - orig_min)

            # Generate Rician noise
            # Rician noise is sqrt((v + n1)^2 + n2^2) where v is the signal
            # and n1, n2 are independent Gaussian noise components
            sigma = self.noise_factor * torch.mean(img_normalized)
            n1 = torch.randn_like(img_normalized) * sigma
            n2 = torch.randn_like(img_normalized) * sigma

            img_noisy = torch.sqrt((img_normalized + n1) ** 2 + n2**2)
            img = img_noisy * (orig_max - orig_min) + orig_min
            img = torch.clamp(img, min=orig_min, max=orig_max)

        return img


class MultiChannelRandAffine(Transform):
    """
    Apply RandAffine with different interpolation modes per channel.

    For multimodal inputs (T1 + Segmentation), geometric transforms must use:
    - Trilinear interpolation for the T1 channel (continuous intensities)
    - Nearest-neighbor interpolation for the segmentation channel (discrete labels)

    Both channels receive the SAME geometric transformation (rotation, scale,
    translation) to maintain spatial correspondence. Uses a single RandAffine
    instance: randomize once, then apply twice with different ``mode`` args.

    Parameters
    ----------
    prob : float
        Probability of applying the transform.
    rotate_range : tuple
        Range of rotation angles in radians.
    scale_range : tuple
        Range of scaling factors.
    translate_range : tuple
        Range of translation in voxels.
    padding_mode : str
        Padding mode for out-of-bounds voxels.
    continuous_channels : list[int]
        Channels to interpolate with trilinear (default: [0]).
    nearest_channels : list[int]
        Channels to interpolate with nearest-neighbor (default: [1]).
    """

    def __init__(
        self,
        prob: float = 0.5,
        rotate_range=(0, 0),
        scale_range=(0, 0),
        translate_range=(0, 0),
        padding_mode: str = "border",
        continuous_channels: list | None = None,
        nearest_channels: list | None = None,
    ):
        super().__init__()
        self.continuous_channels = continuous_channels or [0]
        self.nearest_channels = nearest_channels or [1]
        self._affine = RandAffine(
            prob=prob,
            rotate_range=rotate_range,
            scale_range=scale_range,
            translate_range=translate_range,
            padding_mode=padding_mode,
            mode="bilinear",  # default, overridden per channel in __call__
        )

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        if img.shape[0] == 1:
            return self._affine(img)

        # Randomize geometric parameters once
        self._affine.randomize()

        if not self._affine._do_transform:
            return img

        result = img.clone()
        for ch in self.continuous_channels:
            if ch < img.shape[0]:
                result[ch : ch + 1] = self._affine(
                    img[ch : ch + 1], randomize=False, mode="bilinear"
                )
        for ch in self.nearest_channels:
            if ch < img.shape[0]:
                result[ch : ch + 1] = self._affine(
                    img[ch : ch + 1], randomize=False, mode="nearest"
                )

        return result


class MultiChannelResize(Transform):
    """
    Resize with different interpolation modes per channel.

    For multimodal inputs (T1 + Segmentation):
    - Trilinear interpolation for continuous channels (T1)
    - Nearest-neighbor interpolation for categorical channels (segmentation)

    Parameters
    ----------
    spatial_size : tuple[int, ...]
        Target spatial dimensions.
    continuous_channels : list[int]
        Channels to resize with trilinear interpolation (default: [0]).
    nearest_channels : list[int]
        Channels to resize with nearest-neighbor (default: [1]).
    """

    def __init__(
        self,
        spatial_size: tuple,
        continuous_channels: list | None = None,
        nearest_channels: list | None = None,
    ):
        super().__init__()
        self.spatial_size = spatial_size
        self.continuous_channels = continuous_channels or [0]
        self.nearest_channels = nearest_channels or [1]
        self._resize_trilinear = Resize(spatial_size, mode="trilinear")
        self._resize_nearest = Resize(spatial_size, mode="nearest")

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        if img.shape[0] == 1:
            return self._resize_trilinear(img)

        # Process each channel with appropriate interpolation
        channels = []
        for ch in range(img.shape[0]):
            if ch in self.nearest_channels:
                channels.append(self._resize_nearest(img[ch : ch + 1]))
            else:
                channels.append(self._resize_trilinear(img[ch : ch + 1]))

        return torch.cat(channels, dim=0)
