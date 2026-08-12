"""
MedViT-3D with Differential Attention and 3D RoPE
Based on:
- MedViTV2: Medical Image Classification with KAN-Integrated Transformers and Dilated Neighborhood Attention (https://github.com/Omid-Nejati/MedViTV2)
- Differential Transformer (https://github.com/microsoft/unilm/tree/master/Diff-Transformer)
"""

import logging
import math
from functools import partial
from math import exp

import torch

logger = logging.getLogger(__name__)
import torch.nn.functional as F
from einops import rearrange
from natten import NeighborhoodAttention3D as NeighborhoodAttention
from timm.layers import DropPath, to_3tuple, trunc_normal_
from timm.models import register_model
from torch import nn
from torch.utils import checkpoint

from .modules.fasterkan import FasterKAN
from .modules.rms_norm import RMSNorm
from .modules.rotary_kernel import (
    apply_rotary_emb,
    apply_rotary_emb_pytorch,
    precompute_rope_3d,
)

NORM_EPS = 1e-5


def lambda_init_fn(layer_index: int) -> float:
    """Initialize lambda parameter for differential attention.

    From Differential Transformer paper:
    lambda_init = 0.8 - 0.6 * exp^{-0.3 * (l - 1)} where l ∈ [1, L]
    """
    return 0.8 - 0.6 * exp(-0.3 * layer_index)


def merge_pre_bn(module, pre_bn_1, pre_bn_2=None):
    """Merge pre BN to reduce inference runtime."""
    weight = module.weight.data
    if module.bias is None:
        zeros = torch.zeros(module.out_channels, device=weight.device).type(
            weight.type()
        )
        module.bias = nn.Parameter(zeros)
    bias = module.bias.data
    if pre_bn_2 is None:
        assert pre_bn_1.track_running_stats is True, (
            "Unsupport bn_module.track_running_stats is False"
        )
        assert pre_bn_1.affine is True, "Unsupport bn_module.affine is False"

        scale_invstd = pre_bn_1.running_var.add(pre_bn_1.eps).pow(-0.5)
        extra_weight = scale_invstd * pre_bn_1.weight
        extra_bias = (
            pre_bn_1.bias - pre_bn_1.weight * pre_bn_1.running_mean * scale_invstd
        )
    else:
        assert pre_bn_1.track_running_stats is True, (
            "Unsupport bn_module.track_running_stats is False"
        )
        assert pre_bn_1.affine is True, "Unsupport bn_module.affine is False"

        assert pre_bn_2.track_running_stats is True, (
            "Unsupport bn_module.track_running_stats is False"
        )
        assert pre_bn_2.affine is True, "Unsupport bn_module.affine is False"

        scale_invstd_1 = pre_bn_1.running_var.add(pre_bn_1.eps).pow(-0.5)
        scale_invstd_2 = pre_bn_2.running_var.add(pre_bn_2.eps).pow(-0.5)

        extra_weight = (
            scale_invstd_1 * pre_bn_1.weight * scale_invstd_2 * pre_bn_2.weight
        )
        extra_bias = (
            scale_invstd_2
            * pre_bn_2.weight
            * (
                pre_bn_1.bias
                - pre_bn_1.weight * pre_bn_1.running_mean * scale_invstd_1
                - pre_bn_2.running_mean
            )
            + pre_bn_2.bias
        )

    if isinstance(module, nn.Linear):
        extra_bias = weight @ extra_bias
        weight.mul_(extra_weight.view(1, weight.size(1)).expand_as(weight))
    elif isinstance(module, nn.Conv3d):
        assert weight.shape[2] == 1 and weight.shape[3] == 1 and weight.shape[4] == 1
        weight = weight.reshape(weight.shape[0], weight.shape[1])
        extra_bias = weight @ extra_bias
        weight.mul_(extra_weight.view(1, weight.size(1)).expand_as(weight))
        weight = weight.reshape(weight.shape[0], weight.shape[1], 1, 1, 1)
    bias.add_(extra_bias)

    module.weight.data = weight
    module.bias.data = bias


class ConvBNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, groups=1):
        super().__init__()
        kernel_size = to_3tuple(kernel_size)
        stride = to_3tuple(stride)

        padding = tuple(k // 2 for k in kernel_size)

        self.conv = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=False,
        )
        self.norm = nn.BatchNorm3d(out_channels, eps=NORM_EPS)
        self.act = nn.ReLU(inplace=False)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x


def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class PatchEmbed(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        norm_layer = partial(nn.BatchNorm3d, eps=NORM_EPS)

        s = to_3tuple(stride)

        if s[0] > 1 or s[1] > 1 or s[2] > 1:
            pool_stride = tuple(st if st > 0 else 1 for st in s)
            self.avgpool = nn.AvgPool3d(
                kernel_size=pool_stride,
                stride=pool_stride,
                ceil_mode=True,
                count_include_pad=False,
            )
            self.conv = nn.Conv3d(
                in_channels, out_channels, kernel_size=1, stride=1, bias=False
            )
            self.norm = norm_layer(out_channels)
        elif in_channels != out_channels:
            self.avgpool = nn.Identity()
            self.conv = nn.Conv3d(
                in_channels, out_channels, kernel_size=1, stride=1, bias=False
            )
            self.norm = norm_layer(out_channels)
        else:
            self.avgpool = nn.Identity()
            self.conv = nn.Identity()
            self.norm = nn.Identity()

    def forward(self, x):
        return self.norm(self.conv(self.avgpool(x)))


class MHCA(nn.Module):
    """
    Multi-Head Convolutional Attention
    """

    def __init__(self, out_channels, head_dim):
        super().__init__()
        norm_layer = partial(nn.BatchNorm3d, eps=NORM_EPS)
        self.group_conv3x3 = nn.Conv3d(
            out_channels,
            out_channels,
            kernel_size=(3, 3, 3),
            stride=1,
            padding=1,
            groups=out_channels // head_dim,
            bias=False,
        )
        self.norm = norm_layer(out_channels)
        self.act = nn.ReLU(inplace=False)
        self.projection = nn.Conv3d(
            out_channels, out_channels, kernel_size=1, bias=False
        )

    def forward(self, x):
        out = self.group_conv3x3(x)
        out = self.norm(out)
        out = self.act(out)
        out = self.projection(out)
        return out


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super().__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super().__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class ECALayer(nn.Module):
    def __init__(self, channel, gamma=2, b=1, sigmoid_type="sigmoid"):
        super().__init__()
        t = int(abs((math.log2(channel) + b) / gamma))
        k = t if t % 2 else t + 1

        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=k // 2, bias=False)
        if sigmoid_type == "sigmoid":
            self.sigmoid = nn.Sigmoid()
        elif sigmoid_type == "h_sigmoid":
            self.sigmoid = h_sigmoid()
        else:
            raise NotImplementedError(
                f"Sigmoid type {sigmoid_type} not implemented for ECALayer"
            )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2))
        y = y.transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)


class SELayer(nn.Module):
    def __init__(self, channel, reduction=4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace=False),
            nn.Linear(channel // reduction, channel),
            h_sigmoid(),
        )

    def forward(self, x):
        b, c, _, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1, 1)
        return x * y


class LocalityFeedForward(nn.Module):
    def __init__(
        self,
        in_dim=64,
        out_dim=96,
        kernel_size=3,
        stride=1,
        expand_ratio=4.0,
        act="hs+se",
        reduction=4,
        wo_dp_conv=False,
        dp_first=False,
    ):
        """
        :param in_dim: the input dimension
        :param out_dim: the output dimension. The input and output dimension should be the same.
        :param stride: stride of the depth-wise convolution.
        :param expand_ratio: expansion ratio of the hidden dimension.
        :param act: the activation function.
                    relu: ReLU
                    hs: h_swish
                    hs+se: h_swish and SE module
                    hs+eca: h_swish and ECA module
                    hs+ecah: h_swish and ECA module. Compared with eca, h_sigmoid is used.
        :param reduction: reduction rate in SE module.
        :param wo_dp_conv: without depth-wise convolution.
        :param dp_first: place depth-wise convolution as the first layer.
        """
        super().__init__()
        hidden_dim = int(in_dim * expand_ratio)

        kernel_size = to_3tuple(kernel_size)
        stride = to_3tuple(stride)

        layers = []
        # the first linear layer is replaced by 1x1x1 convolution.
        layers.extend(
            [
                nn.Conv3d(
                    in_dim, hidden_dim, kernel_size=1, stride=1, padding=0, bias=False
                ),
                nn.BatchNorm3d(hidden_dim),
                h_swish() if act.find("hs") >= 0 else nn.ReLU6(inplace=False),
            ]
        )

        # the depth-wise convolution between the two linear layers
        if not wo_dp_conv:
            dp = [
                nn.Conv3d(
                    hidden_dim,
                    hidden_dim,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=tuple(k // 2 for k in kernel_size),
                    groups=hidden_dim,
                    bias=False,
                ),
                nn.BatchNorm3d(hidden_dim),
                h_swish() if act.find("hs") >= 0 else nn.ReLU6(inplace=False),
            ]
            if dp_first:
                layers = dp + layers
            else:
                layers.extend(dp)

        if act.find("+") >= 0:
            attn_type = act.split("+")[1]
            if attn_type == "se":
                layers.append(SELayer(hidden_dim, reduction=reduction))
            elif attn_type.find("eca") >= 0:
                sigmoid_activation = "sigmoid" if attn_type == "eca" else "h_sigmoid"
                layers.append(ECALayer(hidden_dim, sigmoid_type=sigmoid_activation))
            else:
                raise NotImplementedError(f"Activation type {act} is not implemented")

        # the second linear layer is replaced by 1x1x1 convolution.
        layers.extend(
            [
                nn.Conv3d(
                    hidden_dim, out_dim, kernel_size=1, stride=1, padding=0, bias=False
                ),
                nn.BatchNorm3d(out_dim),
            ]
        )
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        x = x + self.conv(x)
        return x


class MLP(nn.Module):
    def __init__(
        self, in_features, out_features=None, mlp_ratio=None, drop=0.0, bias=True
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_dim = _make_divisible(in_features * mlp_ratio, 32)
        self.conv1 = nn.Conv3d(in_features, hidden_dim, kernel_size=1, bias=bias)
        self.act = nn.ReLU(inplace=False)
        self.conv2 = nn.Conv3d(hidden_dim, out_features, kernel_size=1, bias=bias)
        self.drop = nn.Dropout(drop)

    def merge_bn(self, pre_norm):
        merge_pre_bn(self.conv1, pre_norm)

    def forward(self, x):
        x = self.conv1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.conv2(x)
        x = self.drop(x)
        return x


class LFP(nn.Module):
    """
    Efficient Convolution Block (Local Feature Processing)
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        path_dropout=0.2,
        drop=0,
        head_dim=32,
        mlp_ratio=3,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        norm_layer = partial(nn.BatchNorm3d, eps=NORM_EPS)
        assert out_channels % head_dim == 0

        self.patch_embed = PatchEmbed(in_channels, out_channels, stride)
        self.norm1 = norm_layer(out_channels)
        self.attn = NeighborhoodAttention(
            out_channels,
            num_heads=(out_channels // head_dim),
            kernel_size=3,
            dilation=None,
            qkv_bias=True,
            qk_scale=None,
            proj_drop=0.0,
        )
        self.attention_path_dropout = DropPath(path_dropout)

        self.conv = LocalityFeedForward(
            out_channels,
            out_channels,
            kernel_size,
            1,
            mlp_ratio,
            reduction=out_channels,
        )

        self.norm2 = norm_layer(out_channels)
        self.is_bn_merged = False

    def merge_bn(self):
        if not self.is_bn_merged:
            self.mlp.merge_bn(self.norm)
            self.is_bn_merged = True

    def forward(self, x):
        x = self.patch_embed(x)
        b, c, d, h, w = x.shape
        shortcut = x
        x = self.norm1(x)
        x = self.attn(x.reshape(b, d, h, w, c))
        x = shortcut + self.attention_path_dropout(x.reshape(b, c, d, h, w))
        if not torch.onnx.is_in_onnx_export() and not self.is_bn_merged:
            out = self.norm2(x)
        else:
            out = x
        x = x + self.conv(out)
        return x


class DifferentialAttention(nn.Module):
    """
    Differential Multi-Head Self-Attention with 3D RoPE.

    Implements the differential attention mechanism from:
    "Differential Transformer", Ye et al., 2024

    Key features:
    - Two sets of Q/K projections for differential computation
    - 3D RoPE for volumetric positional encoding
    - Lambda-weighted differential attention map
    """

    def __init__(
        self,
        dim: int,
        layer_index: int,
        num_heads: int = 8,
        head_dim: int = 32,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        use_rope: bool = True,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.use_rope = use_rope
        self.scale = head_dim**-0.5

        # For differential attention, we have 2 heads per attention head
        self.num_kv_heads = num_heads
        self.n_rep = 1

        # Q, K, V projections - output dim is 2x for differential pairs
        self.q_proj = nn.Linear(dim, 2 * num_heads * head_dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, 2 * num_heads * head_dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, num_heads * 2 * head_dim, bias=qkv_bias)
        self.out_proj = nn.Linear(num_heads * 2 * head_dim, dim)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

        # Lambda parameters for differential attention
        self.lambda_init = lambda_init_fn(layer_index)
        self.lambda_q1 = nn.Parameter(
            torch.zeros(head_dim, dtype=torch.float32).normal_(mean=0, std=0.1)
        )
        self.lambda_k1 = nn.Parameter(
            torch.zeros(head_dim, dtype=torch.float32).normal_(mean=0, std=0.1)
        )
        self.lambda_q2 = nn.Parameter(
            torch.zeros(head_dim, dtype=torch.float32).normal_(mean=0, std=0.1)
        )
        self.lambda_k2 = nn.Parameter(
            torch.zeros(head_dim, dtype=torch.float32).normal_(mean=0, std=0.1)
        )

        # Sub-layer normalization
        self.subln = RMSNorm(2 * head_dim, eps=1e-5, elementwise_affine=True)

    def forward(self, x: torch.Tensor, grid_size: tuple[int, int, int]) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Input tokens of shape (B, N, C) where N = D*H*W
        grid_size : tuple[int, int, int]
            Spatial dimensions (D, H, W) of the token grid

        Returns
        -------
        torch.Tensor
            Output tokens of shape (B, N, C)
        """
        B, N, _C = x.shape

        # Project to Q, K, V
        q = self.q_proj(x).view(B, N, 2 * self.num_heads, self.head_dim)
        k = self.k_proj(x).view(B, N, 2 * self.num_heads, self.head_dim)
        v = self.v_proj(x).view(B, N, self.num_heads, 2 * self.head_dim)

        # Apply 3D RoPE if enabled
        if self.use_rope:
            with torch.amp.autocast("cuda", enabled=False):
                cos, sin = precompute_rope_3d(
                    grid_size=grid_size,
                    head_dim=self.head_dim,
                    device=q.device,
                    dtype=q.dtype,
                )

                cos = cos.to(dtype=q.dtype, device=q.device)
                sin = sin.to(dtype=q.dtype, device=q.device)

            # Apply RoPE: (B, N, 2*num_heads, head_dim)
            # Use PyTorch fallback if on CPU or if Triton fails
            if q.device.type == "cpu":
                q = apply_rotary_emb_pytorch(q, cos, sin, interleaved=True)
                k = apply_rotary_emb_pytorch(k, cos, sin, interleaved=True)
            else:
                try:
                    q = apply_rotary_emb(q, cos, sin, interleaved=True)
                    k = apply_rotary_emb(k, cos, sin, interleaved=True)
                except (ValueError, RuntimeError):
                    # Fallback to PyTorch implementation
                    q = apply_rotary_emb_pytorch(q, cos, sin, interleaved=True)
                    k = apply_rotary_emb_pytorch(k, cos, sin, interleaved=True)

        # Reshape for attention: (B, num_heads, N, head_dim)
        q = q.transpose(1, 2)  # (B, 2*num_heads, N, head_dim)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)  # (B, num_heads, N, 2*head_dim)

        # Scale queries
        q = q * self.scale

        # Compute attention scores
        attn = torch.matmul(q, k.transpose(-1, -2))  # (B, 2*num_heads, N, N)

        # Softmax
        attn = F.softmax(attn, dim=-1, dtype=torch.float32).type_as(attn)

        # Differential attention computation
        # Split attention into two parts and compute weighted difference
        lambda_1 = torch.exp(
            torch.sum(self.lambda_q1 * self.lambda_k1, dim=-1).float()
        ).type_as(q)
        lambda_2 = torch.exp(
            torch.sum(self.lambda_q2 * self.lambda_k2, dim=-1).float()
        ).type_as(q)
        lambda_full = lambda_1 - lambda_2 + self.lambda_init

        # Reshape and compute differential: (B, num_heads, 2, N, N)
        attn = attn.view(B, self.num_heads, 2, N, N)
        attn = attn[:, :, 0] - lambda_full * attn[:, :, 1]  # (B, num_heads, N, N)

        # Apply dropout
        attn = self.attn_drop(attn)

        # Apply attention to values
        out = torch.matmul(attn, v)  # (B, num_heads, N, 2*head_dim)

        # Sub-layer normalization
        out = self.subln(out)

        # Scale by (1 - lambda_init)
        out = out * (1 - self.lambda_init)

        # Reshape and project output
        out = out.transpose(1, 2).reshape(B, N, self.num_heads * 2 * self.head_dim)
        out = self.out_proj(out)
        out = self.proj_drop(out)

        return out


class GFP(nn.Module):
    """
    Global Feature Processing block with Differential Attention.

    Replaces E-MHSA with DifferentialAttention + 3D RoPE.
    Keeps MHCA and KAN layers from MedViT-V2.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        path_dropout,
        layer_index: int = 0,
        stride=1,
        mlp_ratio=2,
        head_dim=32,
        mix_block_ratio=0.75,
        attn_drop=0,
        drop=0,
        use_diff_attn=True,
        use_rope=True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.mix_block_ratio = mix_block_ratio
        self.use_diff_attn = use_diff_attn
        norm_func = partial(nn.BatchNorm3d, eps=NORM_EPS)

        self.mhsa_out_channels = _make_divisible(
            int(out_channels * mix_block_ratio), 32
        )
        self.mhca_out_channels = out_channels - self.mhsa_out_channels

        self.patch_embed = PatchEmbed(in_channels, self.mhsa_out_channels, stride)
        self.norm1 = norm_func(self.mhsa_out_channels)

        # Use Differential Attention instead of E-MHSA
        if use_diff_attn:
            num_heads = (
                self.mhsa_out_channels // head_dim // 2
            )  # Divide by 2 for diff attn
            self.diff_attn = DifferentialAttention(
                dim=self.mhsa_out_channels,
                layer_index=layer_index,
                num_heads=num_heads,
                head_dim=head_dim,
                qkv_bias=True,
                attn_drop=attn_drop,
                proj_drop=drop,
                use_rope=use_rope,
            )

        self.mhsa_path_dropout = DropPath(path_dropout * mix_block_ratio)

        self.projection = PatchEmbed(
            self.mhsa_out_channels, self.mhca_out_channels, stride=1
        )
        self.mhca = MHCA(self.mhca_out_channels, head_dim=head_dim)
        self.mhca_path_dropout = DropPath(path_dropout * (1 - mix_block_ratio))

        self.norm2 = norm_func(out_channels)
        self.mlp_path_dropout = DropPath(path_dropout)
        hidden_dim = int(out_channels * mlp_ratio)
        self.kan = FasterKAN([out_channels, hidden_dim, out_channels])

        self.is_bn_merged = False

    def merge_bn(self):
        if not self.is_bn_merged:
            self.is_bn_merged = True

    def forward(self, x):
        # Patch embed and get spatial dimensions
        x = self.patch_embed(x)
        _B, _C, D, H, W = x.shape
        grid_size = (D, H, W)

        # Apply normalization
        if not torch.onnx.is_in_onnx_export() and not self.is_bn_merged:
            out = self.norm1(x)
        else:
            out = x

        # Convert to token sequence for attention
        out = rearrange(out, "b c d h w -> b (d h w) c")

        # Apply Differential Attention with 3D RoPE
        if self.use_diff_attn:
            out = self.mhsa_path_dropout(self.diff_attn(out, grid_size))

        # Convert back to spatial
        out = rearrange(out, "b (d h w) c -> b c d h w", d=D, h=H, w=W)
        x = x + out

        # MHCA path
        out = self.projection(x)
        out = out + self.mhca_path_dropout(self.mhca(out))
        x = torch.cat([x, out], dim=1)

        # KAN feedforward
        if not torch.onnx.is_in_onnx_export() and not self.is_bn_merged:
            out = self.norm2(x)
        else:
            out = x

        b, c, d, h, w = out.shape
        x = x + self.mlp_path_dropout(
            self.kan(out.reshape(-1, out.shape[1])).reshape(b, c, d, h, w)
        )
        return x


class MedViT(nn.Module):
    """
    MedViT with Differential Attention and 3D RoPE.

    Parameters
    ----------
    in_channels : int
        Number of input channels
    stem_chs : list[int]
        Stem channel dimensions
    depths : list[int]
        Number of blocks in each stage
    dims : list[int]
        Channel dimensions for each stage
    path_dropout : float
        Stochastic depth dropout probability
    attn_drop : float
        Attention dropout
    drop : float
        Dropout rate
    num_classes : int
        Number of output classes
    strides : list[int]
        Stride for each stage
    head_dim : int
        Dimension per attention head
    mix_block_ratio : float
        Ratio for mixing MHSA and MHCA in GFP blocks
    use_checkpoint : bool
        Whether to use gradient checkpointing
    use_diff_attn : bool
        Whether to use differential attention in GFP blocks
    """

    def __init__(
        self,
        in_channels=3,
        stem_chs=None,
        depths=None,
        dims=None,
        path_dropout=0.1,
        attn_drop=0,
        drop=0,
        num_classes=3,
        strides=None,
        head_dim=32,
        mix_block_ratio=0.75,
        use_checkpoint=False,
        use_diff_attn=True,
        use_rope=True,
    ):
        if strides is None:
            strides = [1, 2, 2, 2]
        if dims is None:
            dims = [64, 128, 320, 512]
        if depths is None:
            depths = [2, 2, 6, 2]
        if stem_chs is None:
            stem_chs = [64, 32, 64]
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.use_diff_attn = use_diff_attn

        self.stage_out_channels = [
            [dims[0]] * (depths[0]),
            [dims[1]] * (depths[1] - 1) + [dims[1]],
            [dims[2], dims[2], dims[2]] * (depths[2] // 3),
            [dims[3]] * (depths[3]),
        ]

        # Architecture: LFP blocks for local features, GFP for global
        self.stage_block_types = [
            [LFP] * depths[0],
            [LFP] * (depths[1] - 1) + [GFP],
            [LFP, LFP, GFP] * (depths[2] // 3),
            [GFP] * (depths[3]),
        ]

        self.stem = nn.Sequential(
            ConvBNReLU(in_channels, stem_chs[0], kernel_size=3, stride=2),
            ConvBNReLU(stem_chs[0], stem_chs[1], kernel_size=3, stride=1),
            ConvBNReLU(stem_chs[1], stem_chs[2], kernel_size=3, stride=1),
            ConvBNReLU(stem_chs[2], stem_chs[2], kernel_size=3, stride=2),
        )
        input_channel = stem_chs[-1]
        features = []
        idx = 0
        dpr = [
            x.item() for x in torch.linspace(0, path_dropout, sum(depths))
        ]  # stochastic depth decay rule

        global_block_idx = 0  # Track global block index for lambda init

        for stage_id in range(len(depths)):
            kernel = 7 if stage_id == 0 else 3
            numrepeat = depths[stage_id]
            output_channels = self.stage_out_channels[stage_id]
            block_types = self.stage_block_types[stage_id]

            for block_id in range(numrepeat):
                if strides[stage_id] == 2 and block_id == 0:
                    stride = 2
                else:
                    stride = 1
                output_channel = output_channels[block_id]
                block_type = block_types[block_id]

                if block_type is LFP:
                    layer = LFP(
                        input_channel,
                        output_channel,
                        stride=stride,
                        kernel_size=kernel,
                        path_dropout=dpr[idx + block_id],
                        drop=drop,
                        head_dim=head_dim,
                    )
                    features.append(layer)
                elif block_type is GFP:
                    layer = GFP(
                        input_channel,
                        output_channel,
                        path_dropout=dpr[idx + block_id],
                        layer_index=global_block_idx,
                        stride=stride,
                        head_dim=head_dim,
                        mix_block_ratio=mix_block_ratio,
                        attn_drop=attn_drop,
                        drop=drop,
                        use_diff_attn=use_diff_attn,
                        use_rope=use_rope,
                    )
                    features.append(layer)
                    global_block_idx += 1

                input_channel = output_channel
            idx += numrepeat
        self.features = nn.Sequential(*features)

        self.norm = nn.BatchNorm3d(output_channel, eps=NORM_EPS)

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.proj_head = nn.Sequential(
            nn.Linear(output_channel, num_classes),
        )

        self.stage_out_idx = [sum(depths[: idx + 1]) - 1 for idx in range(len(depths))]
        self._initialize_weights()

    def merge_bn(self):
        self.eval()
        for idx, module in self.named_modules():
            if isinstance(module, (LFP, GFP)) and hasattr(module, "merge_bn"):
                module.merge_bn()

    def _initialize_weights(self):
        for n, m in self.named_modules():
            if isinstance(
                m,
                (
                    nn.BatchNorm1d,
                    nn.BatchNorm2d,
                    nn.BatchNorm3d,
                    nn.GroupNorm,
                    nn.LayerNorm,
                ),
            ):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.Linear, nn.Conv2d, nn.Conv3d)):
                trunc_normal_(m.weight, std=0.02)
                if hasattr(m, "bias") and m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.stem(x)
        for layer in self.features:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(layer, x, use_reentrant=False)
            else:
                x = layer(x)
        x = self.norm(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.proj_head(x)
        return x

    def load_checkpoint(self, checkpoint_path, device):
        if checkpoint_path is not None:
            logger.info("Loading checkpoint from %s", checkpoint_path)
            checkpoint = torch.load(
                checkpoint_path, map_location=device, weights_only=False
            )
            self.load_state_dict(checkpoint["model"], strict=True)
            logger.info("Checkpoint loaded successfully.")
        else:
            logger.info("No checkpoint path provided.")


@register_model
def MedViT_Diff_tiny(
    pretrained=False, pretrained_cfg=None, pretrained_cfg_overlay=None, **kwargs
):
    model = MedViT(
        stem_chs=[64, 32, 64],
        depths=[2, 2, 6, 1],
        dims=[64, 128, 192, 384],
        path_dropout=0.1,
        **kwargs,
    )
    return model


@register_model
def MedViT_Diff_small(
    pretrained=False, pretrained_cfg=None, pretrained_cfg_overlay=None, **kwargs
):
    model = MedViT(
        stem_chs=[64, 32, 64],
        depths=[2, 2, 6, 2],
        dims=[64, 128, 256, 512],
        path_dropout=0.1,
        **kwargs,
    )
    return model


@register_model
def MedViT_Diff_base(
    pretrained=False, pretrained_cfg=None, pretrained_cfg_overlay=None, **kwargs
):
    model = MedViT(
        stem_chs=[64, 32, 64],
        depths=[2, 2, 6, 2],
        dims=[96, 192, 384, 768],
        path_dropout=0.2,
        **kwargs,
    )
    return model


@register_model
def MedViT_Diff_large(
    pretrained=False, pretrained_cfg=None, pretrained_cfg_overlay=None, **kwargs
):
    model = MedViT(
        stem_chs=[64, 32, 64],
        depths=[2, 2, 6, 2],
        dims=[96, 256, 512, 1024],
        path_dropout=0.2,
        **kwargs,
    )
    return model
