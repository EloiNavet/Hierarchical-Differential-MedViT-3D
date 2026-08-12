import torch
import triton
import triton.language as tl


# @triton.autotune(
#     configs=[
#         triton.Config({"BLOCK_M": 2}),
#         triton.Config({"BLOCK_M": 4}),
#         triton.Config({"BLOCK_M": 8}),
#         triton.Config({"BLOCK_M": 16}),
#     ],
#     key=["CACHE_KEY_SEQLEN", "BLOCK_K", "INTERLEAVED"],
# )
@triton.jit
def rotary_kernel(
    OUT,  # Pointers to matrices
    X,
    COS,
    SIN,
    CU_SEQLENS,
    SEQLEN_OFFSETS,  # this could be int or a pointer
    # Matrix dimensions
    seqlen,
    nheads,
    rotary_dim,
    seqlen_ro,
    CACHE_KEY_SEQLEN,
    # strides
    stride_out_batch,
    stride_out_seqlen,
    stride_out_nheads,
    stride_out_headdim,
    stride_x_batch,
    stride_x_seqlen,
    stride_x_nheads,
    stride_x_headdim,
    # Meta-parameters
    BLOCK_K: tl.constexpr,
    IS_SEQLEN_OFFSETS_TENSOR: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    INTERLEAVED: tl.constexpr,
    CONJUGATE: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_batch = tl.program_id(axis=1)
    pid_head = tl.program_id(axis=2)
    rotary_dim_half = rotary_dim // 2

    if not IS_VARLEN:
        X = X + pid_batch * stride_x_batch + pid_head * stride_x_nheads
        OUT = OUT + pid_batch * stride_out_batch + pid_head * stride_out_nheads
    else:
        start_idx = tl.load(CU_SEQLENS + pid_batch)
        seqlen = tl.load(CU_SEQLENS + pid_batch + 1) - start_idx
        X = X + start_idx * stride_x_seqlen + pid_head * stride_x_nheads
        OUT = OUT + start_idx * stride_out_seqlen + pid_head * stride_out_nheads

    if pid_m * BLOCK_M >= seqlen:
        return
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    if not IS_SEQLEN_OFFSETS_TENSOR:
        rm_cs = rm + SEQLEN_OFFSETS
    else:
        rm_cs = rm + tl.load(SEQLEN_OFFSETS + pid_batch)
    rk = tl.arange(0, BLOCK_K)
    rk_half = tl.arange(0, BLOCK_K // 2)

    if not INTERLEAVED:
        # Load the 1st and 2nd halves of X, do calculation, then store to 1st and 2nd halves of OUT
        X = X + (rm[:, None] * stride_x_seqlen + rk_half[None, :] * stride_x_headdim)
        COS = COS + (rm_cs[:, None] * rotary_dim_half + rk_half[None, :])
        SIN = SIN + (rm_cs[:, None] * rotary_dim_half + rk_half[None, :])
        cos = tl.load(
            COS,
            mask=(rm_cs[:, None] < seqlen_ro) & (rk_half[None, :] < rotary_dim_half),
            other=1.0,
        ).to(tl.float32)
        sin = tl.load(
            SIN,
            mask=(rm_cs[:, None] < seqlen_ro) & (rk_half[None, :] < rotary_dim_half),
            other=0.0,
        ).to(tl.float32)
        x0 = tl.load(
            X,
            mask=(rm[:, None] < seqlen) & (rk_half[None, :] < rotary_dim_half),
            other=0.0,
        ).to(tl.float32)
        x1 = tl.load(
            X + rotary_dim_half * stride_x_headdim,
            mask=(rm[:, None] < seqlen) & (rk_half[None, :] < rotary_dim_half),
            other=0.0,
        ).to(tl.float32)
        if CONJUGATE:
            sin = -sin
        o0 = x0 * cos - x1 * sin
        o1 = x0 * sin + x1 * cos
        # write back result
        OUT = OUT + (
            rm[:, None] * stride_out_seqlen + rk_half[None, :] * stride_out_headdim
        )
        tl.store(
            OUT, o0, mask=(rm[:, None] < seqlen) & (rk_half[None, :] < rotary_dim_half)
        )
        tl.store(
            OUT + rotary_dim_half * stride_out_headdim,
            o1,
            mask=(rm[:, None] < seqlen) & (rk_half[None, :] < rotary_dim_half),
        )
    else:
        # We don't want to load X[0, 2, 4, ...] and X[1, 3, 5, ...] separately since both are slow.
        # Instead, we load x0 = X[0, 1, 2, 3, ...] and x1 = X[1, 0, 3, 2, ...].
        # Loading x0 will be fast but x1 will be slow.
        # Then we load cos = COS[0, 0, 1, 1, ...] and sin = SIN[0, 0, 1, 1, ...].
        # Then we do the calculation and use tl.where to pick put the right outputs for the even
        # and for the odd indices.
        rk_swap = rk + ((rk + 1) % 2) * 2 - 1  # 1, 0, 3, 2, 5, 4, ...
        rk_repeat = tl.arange(0, BLOCK_K) // 2
        X0 = X + (rm[:, None] * stride_x_seqlen + rk[None, :] * stride_x_headdim)
        X1 = X + (rm[:, None] * stride_x_seqlen + rk_swap[None, :] * stride_x_headdim)
        COS = COS + (rm_cs[:, None] * rotary_dim_half + rk_repeat[None, :])
        SIN = SIN + (rm_cs[:, None] * rotary_dim_half + rk_repeat[None, :])
        cos = tl.load(
            COS,
            mask=(rm_cs[:, None] < seqlen_ro) & (rk_repeat[None, :] < rotary_dim_half),
            other=1.0,
        ).to(tl.float32)
        sin = tl.load(
            SIN,
            mask=(rm_cs[:, None] < seqlen_ro) & (rk_repeat[None, :] < rotary_dim_half),
            other=0.0,
        ).to(tl.float32)
        x0 = tl.load(
            X0, mask=(rm[:, None] < seqlen) & (rk[None, :] < rotary_dim), other=0.0
        ).to(tl.float32)
        x1 = tl.load(
            X1, mask=(rm[:, None] < seqlen) & (rk_swap[None, :] < rotary_dim), other=0.0
        ).to(tl.float32)
        if CONJUGATE:
            sin = -sin
        x0_cos = x0 * cos
        x1_sin = x1 * sin
        out = tl.where(rk[None, :] % 2 == 0, x0_cos - x1_sin, x0_cos + x1_sin)
        OUT = OUT + (rm[:, None] * stride_out_seqlen + rk[None, :] * stride_out_headdim)
        tl.store(OUT, out, mask=(rm[:, None] < seqlen) & (rk[None, :] < rotary_dim))


def apply_rotary(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    seqlen_offsets: int | torch.Tensor = 0,
    cu_seqlens: torch.Tensor | None = None,
    max_seqlen: int | None = None,
    interleaved=False,
    inplace=False,
    conjugate=False,
) -> torch.Tensor:
    """
    Arguments:
        x: (batch, seqlen, nheads, headdim) if cu_seqlens is None
            else (total_seqlen, nheads, headdim).
        cos: (seqlen_ro, rotary_dim / 2)
        sin: (seqlen_ro, rotary_dim / 2)
        seqlen_offsets: integer or integer tensor of size (batch,)
        cu_seqlens: (batch + 1,) or None
        max_seqlen: int
    Returns:
        y: (batch, seqlen, nheads, headdim)
    """
    is_varlen = cu_seqlens is not None
    if not is_varlen:
        batch, seqlen, nheads, headdim = x.shape
    else:
        assert max_seqlen is not None, (
            "If cu_seqlens is passed in, then max_seqlen must be passed"
        )
        _total_seqlen, nheads, headdim = x.shape
        batch_p_1 = cu_seqlens.shape[0]
        batch = batch_p_1 - 1
        seqlen = max_seqlen
    seqlen_ro, rotary_dim = cos.shape
    assert sin.shape == cos.shape
    rotary_dim *= 2
    assert rotary_dim <= headdim, "rotary_dim must be <= headdim"
    assert headdim <= 256, "Only support headdim <= 256"
    assert seqlen_ro >= seqlen, "seqlen_ro must be >= seqlen"

    assert cos.dtype == sin.dtype, (
        f"cos and sin must have the same dtype, got {cos.dtype} and {sin.dtype}"
    )
    assert x.dtype == cos.dtype, (
        f"Input and cos/sin must have the same dtype, got {x.dtype} and {cos.dtype}"
    )

    cos, sin = cos.contiguous(), sin.contiguous()
    if isinstance(seqlen_offsets, torch.Tensor):
        assert seqlen_offsets.shape == (batch,)
        assert seqlen_offsets.dtype in [torch.int32, torch.int64]
        seqlen_offsets = seqlen_offsets.contiguous()
    else:
        assert seqlen_offsets + seqlen <= seqlen_ro

    output = torch.empty_like(x) if not inplace else x
    if rotary_dim < headdim and not inplace:
        output[..., rotary_dim:].copy_(x[..., rotary_dim:])

    BLOCK_K = (
        32
        if rotary_dim <= 32
        else (64 if rotary_dim <= 64 else (128 if rotary_dim <= 128 else 256))
    )
    grid = lambda META: (triton.cdiv(seqlen, META["BLOCK_M"]), batch, nheads)
    BLOCK_M = 4 if interleaved else (8 if rotary_dim <= 64 else 4)

    # Need this, otherwise Triton tries to launch from cuda:0 and we get
    # ValueError: Pointer argument (at 0) cannot be accessed from Triton (cpu tensor?)
    with torch.cuda.device(x.device.index):
        rotary_kernel[grid](
            output,  # data ptrs
            x,
            cos,
            sin,
            cu_seqlens,
            seqlen_offsets,
            seqlen,  # shapes
            nheads,
            rotary_dim,
            seqlen_ro,
            seqlen // 128,  # key for triton cache (limit number of compilations)
            (
                output.stride(0) if not is_varlen else 0
            ),  # batch_strides if not varlen else 0
            output.stride(-3),  # seqlen_stride or total_seqlen_stride
            output.stride(-2),  # nheads_stride
            output.stride(-1),  # headdim_stride
            x.stride(0) if not is_varlen else 0,  # batch_strides if not varlen else 0
            x.stride(-3),  # seqlen stride or total_seqlen_stride
            x.stride(-2),  # nheads stride
            x.stride(-1),  # headdim stride
            BLOCK_K,
            isinstance(seqlen_offsets, torch.Tensor),
            is_varlen,
            interleaved,
            conjugate,
            BLOCK_M,
        )
    return output


class ApplyRotaryEmb(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x,
        cos,
        sin,
        interleaved=False,
        inplace=False,
        seqlen_offsets: int | torch.Tensor = 0,
        cu_seqlens: torch.Tensor | None = None,
        max_seqlen: int | None = None,
    ):
        out = apply_rotary(
            x,
            cos,
            sin,
            seqlen_offsets=seqlen_offsets,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            interleaved=interleaved,
            inplace=inplace,
        )
        if isinstance(seqlen_offsets, int):
            # Can't save int with save_for_backward
            ctx.save_for_backward(cos, sin, cu_seqlens)
            ctx.seqlen_offsets = seqlen_offsets
        else:
            ctx.save_for_backward(cos, sin, cu_seqlens, seqlen_offsets)
            ctx.seqlen_offsets = None
        ctx.interleaved = interleaved
        ctx.inplace = inplace
        ctx.max_seqlen = max_seqlen
        return out if not inplace else x

    @staticmethod
    def backward(ctx, do):
        seqlen_offsets = ctx.seqlen_offsets
        if seqlen_offsets is None:
            cos, sin, cu_seqlens, seqlen_offsets = ctx.saved_tensors
        else:
            cos, sin, cu_seqlens = ctx.saved_tensors
        # TD [2023-09-02]: For some reason Triton (2.0.0.post1) errors with
        # "[CUDA]: invalid device context", and cloning makes it work. Idk why. Triton 2.1.0 works.
        if not ctx.interleaved and not ctx.inplace:
            do = do.clone()
        dx = apply_rotary(
            do,
            cos,
            sin,
            seqlen_offsets=seqlen_offsets,
            cu_seqlens=cu_seqlens,
            max_seqlen=ctx.max_seqlen,
            interleaved=ctx.interleaved,
            inplace=ctx.inplace,
            conjugate=True,
        )
        return dx, None, None, None, None, None, None, None


def apply_rotary_emb(
    x,
    cos,
    sin,
    interleaved=False,
    inplace=False,
    seqlen_offsets: int | torch.Tensor = 0,
    cu_seqlens: torch.Tensor | None = None,
    max_seqlen: int | None = None,
):
    """
    Arguments:
        x: (batch_size, seqlen, nheads, headdim) if cu_seqlens is None
            else (total_seqlen, nheads, headdim)
        cos, sin: (seqlen_rotary, rotary_dim / 2)
        interleaved: if True, rotate pairs of even and odd dimensions (GPT-J style) instead
            of 1st half and 2nd half (GPT-NeoX style).
        inplace: if True, apply rotary embedding in-place.
        seqlen_offsets: (batch_size,) or int. Each sequence in x is shifted by this amount.
            Most commonly used in inference when we have KV cache.
        cu_seqlens: (batch + 1,) or None
        max_seqlen: int
    Return:
        out: (batch_size, seqlen, nheads, headdim) if cu_seqlens is None
            else (total_seqlen, nheads, headdim)
    rotary_dim must be <= headdim
    Apply rotary embedding to the first rotary_dim of x.
    """
    return ApplyRotaryEmb.apply(
        x, cos, sin, interleaved, inplace, seqlen_offsets, cu_seqlens, max_seqlen
    )


def precompute_rope_cos_sin(
    seq_len: int,
    head_dim: int,
    theta: float = 10000.0,
    device: torch.device = None,
    dtype: torch.dtype = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Precompute cosine and sine tensors for Rotary Position Embedding (RoPE).

    Parameters
    ----------
    seq_len : int
        The sequence length.
    head_dim : int
        The head dimension.
    theta : float, optional
        The theta parameter for RoPE. Default is 10000.0.
    device : torch.device, optional
        The device to place the tensors on.
    dtype : torch.dtype, optional
        The dtype for the tensors. If None, defaults to torch.float32.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        A tuple of (cos, sin) tensors with shape (seq_len, head_dim // 2).
    """
    if dtype is None:
        dtype = torch.float32

    # Create position indices
    position = torch.arange(seq_len, dtype=dtype, device=device)

    # Create frequency indices
    freqs = 1.0 / (
        theta ** (torch.arange(0, head_dim, 2, dtype=dtype, device=device) / head_dim)
    )

    # Compute angles: position * freqs
    angles = position[:, None] * freqs[None, :]  # (seq_len, head_dim // 2)

    # Compute cos and sin
    cos = torch.cos(angles)
    sin = torch.sin(angles)

    return cos, sin


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Repeat key and value tensors for multi-query attention."""
    bs, n_kv_heads, slen, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, None, :, :]
        .expand(bs, n_kv_heads, n_rep, slen, head_dim)
        .reshape(bs, n_kv_heads * n_rep, slen, head_dim)
    )


def apply_rotary_emb_pytorch(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    interleaved: bool = True,
) -> torch.Tensor:
    """
    PyTorch fallback implementation of RoPE (when Triton not available or CPU tensors).

    Parameters
    ----------
    x : torch.Tensor
        Input tensor of shape (B, N, num_heads, head_dim)
    cos : torch.Tensor
        Cosine tensor of shape (N, head_dim // 2)
    sin : torch.Tensor
        Sine tensor of shape (N, head_dim // 2)
    interleaved : bool
        Whether to use interleaved rotation (GPT-J style)

    Returns
    -------
    torch.Tensor
        Rotated tensor of same shape as input
    """
    B, N, num_heads, _head_dim = x.shape
    rotary_dim = cos.shape[-1] * 2

    # Extract the part to rotate
    x_rot = x[..., :rotary_dim]
    x_pass = x[..., rotary_dim:]

    # Expand cos/sin to match batch and head dimensions
    cos = cos[None, :, None, :]  # (1, N, 1, head_dim//2)
    sin = sin[None, :, None, :]  # (1, N, 1, head_dim//2)

    if interleaved:
        # GPT-J style: rotate pairs [0,1], [2,3], [4,5], ...
        x_rot = x_rot.reshape(B, N, num_heads, -1, 2)  # (B, N, H, rotary_dim//2, 2)
        x1 = x_rot[..., 0]  # (B, N, H, rotary_dim//2)
        x2 = x_rot[..., 1]  # (B, N, H, rotary_dim//2)

        # Apply rotation
        rotated_1 = x1 * cos - x2 * sin
        rotated_2 = x1 * sin + x2 * cos

        # Stack back
        x_rot_out = torch.stack([rotated_1, rotated_2], dim=-1).reshape(
            B, N, num_heads, rotary_dim
        )
    else:
        # GPT-NeoX style: rotate first half and second half
        x1 = x_rot[..., : rotary_dim // 2]
        x2 = x_rot[..., rotary_dim // 2 :]

        # Apply rotation
        x_rot_out = torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)

    # Concatenate rotated and pass-through parts
    return torch.cat([x_rot_out, x_pass], dim=-1)


def precompute_rope_3d(
    grid_size: tuple[int, int, int],
    head_dim: int,
    theta: float = 10000.0,
    device: torch.device = None,
    dtype: torch.dtype = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Precompute 3D RoPE cos/sin tensors for volumetric data.

    Uses a 3D-aware approach by computing a combined positional index from 3D coordinates,
    then applying standard RoPE. This works for any head_dim divisible by 2.

    Parameters
    ----------
    grid_size : tuple[int, int, int]
        (D, H, W) dimensions of the 3D volume
    head_dim : int
        Head dimension (must be divisible by 2)
    theta : float
        Base frequency for RoPE
    device : torch.device
        Device to place tensors on
    dtype : torch.dtype
        Data type for tensors

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        cos and sin tensors of shape (N, head_dim // 2) where N = D*H*W
    """
    if dtype is None:
        dtype = torch.float32

    D, H, W = grid_size
    N = D * H * W

    # Create 3D coordinate grid
    coords_d = torch.arange(D, dtype=dtype, device=device)
    coords_h = torch.arange(H, dtype=dtype, device=device)
    coords_w = torch.arange(W, dtype=dtype, device=device)

    # Create meshgrid
    mesh_args = torch.meshgrid.__kwdefaults__
    if mesh_args is not None:
        grid_d, grid_h, grid_w = torch.meshgrid(
            coords_d, coords_h, coords_w, indexing="ij"
        )
    else:
        grid_d, grid_h, grid_w = torch.meshgrid(coords_d, coords_h, coords_w)

    # Flatten to get positions (N,)
    # Create a combined 3D position index that respects spatial structure
    # Using a space-filling curve-like approach: position = d * (H*W) + h * W + w
    positions = (
        grid_d.flatten() * (H * W) + grid_h.flatten() * W + grid_w.flatten()
    ).float()

    # Normalize positions to reasonable range (helps with numerical stability)
    positions = (
        positions / max(N - 1, 1) * N * 0.1
    )  # Scale factor 0.1 keeps values reasonable

    # Create frequency bands
    freqs = 1.0 / (
        theta
        ** (torch.arange(0, head_dim, 2, dtype=dtype, device=device).float() / head_dim)
    )

    # Compute angles: (N, 1) * (1, head_dim//2) = (N, head_dim//2)
    angles = positions[:, None] * freqs[None, :]

    # Compute cos and sin
    cos = torch.cos(angles)
    sin = torch.sin(angles)

    return cos, sin
