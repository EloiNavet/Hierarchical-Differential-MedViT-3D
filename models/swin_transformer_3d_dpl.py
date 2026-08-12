import torch
from torch import nn


class SwinTransformer(nn.Module):
    """3D Swin Transformer with DPL for volumetric classification.

    Parameters
    ----------
    img_size : tuple[int, int, int]
        Input image size (D, H, W).
    in_channels : int
        Number of input image channels.
    patch_size : int | tuple[int, int, int]
        Patch size for embedding.
    embed_dim : int
        Patch embedding dimension.
    depths : list[int]
        Depth of each Swin Transformer layer.
    num_heads : list[int]
        Number of attention heads in different layers.
    window_size : tuple[int, int, int]
        Window size for self-attention.
    mlp_ratio : float
        Ratio of MLP hidden dim to embedding dim.
    dropout : float
        Dropout rate.
    attention_dropout : float
        Attention dropout rate.
    stochastic_depth_prob : float
        Stochastic depth rate.
    num_classes : int
        Number of output classes.
    norm_layer : nn.Module
        Normalization layer.
    patch_norm : bool
        If True, applies normalization after patch embedding.
    qkv_bias : bool
        If True, adds learnable bias to query, key, value.
    use_checkpoint : bool
        If True, uses gradient checkpointing.
    use_diff_attn : bool
        If True, uses Differential Attention.
    use_rope : bool
        If True, uses 3D Rotary Position Embeddings.
    use_relative_pos_bias : bool
        If True, uses relative position bias.
    """

    def __init__(
        self,
        img_size: tuple = (160, 192, 160),
        in_channels: int = 1,
        patch_size: int | tuple = (4, 4, 4),
        embed_dim: int = 96,
        depths: list | None = None,
        num_heads: list | None = None,
        window_size: tuple = (8, 8, 8),
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        stochastic_depth_prob: float = 0.2,
        num_classes: int = 7,
        norm_layer: nn.Module = nn.LayerNorm,
        patch_norm: bool = True,
        qkv_bias: bool = True,
        use_checkpoint: bool = False,
        use_diff_attn: bool = False,
        use_rope: bool = False,
        use_relative_pos_bias: bool = True,
    ):
        if num_heads is None:
            num_heads = [3, 6, 12, 24]
        if depths is None:
            depths = [2, 2, 6, 2]
        super().__init__()
        raise NotImplementedError(
            "SwinTransformer 3D DPL implementation is not included in this "
            "repository. It is used as a baseline comparison only. "
            "Pre-trained weights can be loaded from checkpoints."
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class SwinTransformerT(SwinTransformer):
    """Swin Transformer Tiny variant."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class SwinTransformerS(SwinTransformer):
    """Swin Transformer Small variant."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class SwinTransformerB(SwinTransformer):
    """Swin Transformer Base variant."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class SwinTransformerL(SwinTransformer):
    """Swin Transformer Large variant."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
