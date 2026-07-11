"""Triton FlashAttention implementation for the mini-train-sys backend.

This module follows the same layering as the local RMSNorm/RoPE/SwiGLU
operators:

1. JIT kernels implement the tiled attention forward and backward.
2. Python launchers pass MiniTrain's native `(batch, heads, seq, head_dim)`
   tensors and their strides to the kernels.
3. `MiniTrainFlashAttentionFunction` bridges the launchers into PyTorch autograd.
4. `flash_attention()` is the function consumed by `TritonOpsBackend`.

The kernel is intentionally scoped to the model path used in this repo: dense
attention, equal Q/K/V head counts, fp16/bf16/fp32 CUDA tensors, and
head_dim <= 128. Unsupported cases fall back to PyTorch SDPA in the backend
facade.
"""

from __future__ import annotations

import math
import os

import torch

from minitrain.kernels.triton.cache import configure_triton_cache


configure_triton_cache()

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - exercised by environments without Triton.
    triton = None
    tl = None


_SUPPORTED_DTYPES = {torch.float16, torch.bfloat16, torch.float32}
_MAX_HEAD_DIM = 128
_MIN_HEAD_DIM_BLOCK = 16
_MAX_DROPOUT_COUNTER = 2**32

# Kernel tile and autotune policy. Keep these together so the register/occupancy
# trade-offs are visible without having to inspect individual kernels.
_FWD_BLOCK_M_VALUES = (64, 128)
_FWD_BLOCK_N_VALUES = (32, 64, 128)
_FWD_NUM_WARPS = (4, 8)
_FWD_NUM_STAGES = (2, 3, 4)
_FWD_LSE_BLOCK_M = 128

# Dropout keeps the no-dropout score tile plus RNG offsets and a keep predicate
# live at once. These limits retain 64x64 and 128x64 candidates, but exclude
# the high-pressure 128x128 tile and low-warp large tiles.
_DROPOUT_FWD_MAX_SCORE_TILE = 64 * 128
_DROPOUT_FWD_MAX_STAGES = 3
_DROPOUT_FWD_LARGE_TILE = 64 * 64
_DROPOUT_FWD_LARGE_TILE_WARPS = 8

_BWD_BLOCK_M = 128
_BWD_BLOCK_N = 128
_BWD_NUM_WARPS = (4, 8)
_BWD_NUM_STAGES = (1, 2, 3, 4)
# dK and dV each retain a 128x128 FP32 accumulator. For dropout, keep the
# higher-warp, lower-stage candidates that limit register pressure while still
# allowing enough pipelining to hide memory latency.
_DROPOUT_BWD_NUM_WARPS = (8,)
_DROPOUT_BWD_NUM_STAGES = (1, 2, 3)
_BWD_DKDV_BLOCK_M_STEP = 32
_BWD_DQ_BLOCK_N_STEP = 32
_BWD_CAUSAL_DIAGONAL_DIVISOR = 2


def _round_up_to_block(x: int, block: int = _FWD_LSE_BLOCK_M) -> int:
    return math.ceil(x / block) * block


def _head_dim_block(head_dim: int) -> int:
    if triton is None:
        raise RuntimeError("Triton is not installed. Install mini-train-sys[triton].")
    return max(triton.next_power_of_2(head_dim), _MIN_HEAD_DIM_BLOCK)


def _cache_key_dim(x: int, bucket: int = 32) -> int:
    """Bucket dynamic dimensions to limit autotune/JIT cache explosion."""

    return max(1, math.ceil(x / bucket))


def is_flash_attention_supported(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    dropout_p: float,
) -> bool:
    """Return whether tensors should use the local Triton FlashAttention path."""

    if triton is None:
        return False
    if not (0.0 <= dropout_p < 1.0):
        return False
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        return False
    if not (q.is_cuda and k.is_cuda and v.is_cuda):
        return False
    if q.device != k.device or q.device != v.device:
        return False
    if q.dtype not in _SUPPORTED_DTYPES or k.dtype != q.dtype or v.dtype != q.dtype:
        return False
    if q.shape[0] != k.shape[0] or q.shape[0] != v.shape[0]:
        return False
    if q.shape[1] != k.shape[1] or q.shape[1] != v.shape[1]:
        return False
    if q.shape[-1] != k.shape[-1] or q.shape[-1] != v.shape[-1]:
        return False
    if q.shape[-1] > _MAX_HEAD_DIM:
        return False
    if dropout_p != 0.0 and q.shape[-2] * k.shape[-2] > _MAX_DROPOUT_COUNTER:
        return False
    if q.stride(-1) != 1 or k.stride(-1) != 1 or v.stride(-1) != 1:
        return False
    return True


if triton is not None:

    def _is_cuda_target() -> bool:
        return torch.cuda.is_available() and torch.version.hip is None

    def _is_hopper_target() -> bool:
        return _is_cuda_target() and torch.cuda.get_device_capability()[0] == 9

    def _keep_fwd_config(conf) -> bool:
        """Mirror the upstream guard against poor Hopper small-tile configs."""

        block_m = conf.kwargs["BLOCK_M"]
        block_n = conf.kwargs["BLOCK_N"]
        return not (_is_hopper_target() and block_m * block_n < 128 * 128 and conf.num_warps == 8)

    def _prune_fwd_configs(configs, named_args, **kwargs):
        """Drop clearly invalid configs before Triton spends time benchmarking them.

        Causal attention is most efficient when a query tile is at least as tall
        as the key tile, because the diagonal/on-band work stays compact. For
        non-causal attention the full K/V range is scanned, so rectangular
        variants are fair game.
        """

        named_args = named_args if isinstance(named_args, dict) else {}
        is_causal = kwargs.get("IS_CAUSAL", named_args.get("IS_CAUSAL", False))
        is_dropout = kwargs.get("IS_DROPOUT", named_args.get("IS_DROPOUT", False))
        head_dim = kwargs.get("BLOCK_HEADDIM", named_args.get("BLOCK_HEADDIM", _MAX_HEAD_DIM))
        pruned = []
        for conf in configs:
            block_m = conf.kwargs["BLOCK_M"]
            block_n = conf.kwargs["BLOCK_N"]
            if is_causal and block_m < block_n:
                continue
            # Very small head dimensions tend to suffer more from oversized
            # key tiles than they gain from fewer loop iterations.
            if head_dim <= 32 and block_n > 64:
                continue
            if is_dropout:
                score_tile = block_m * block_n
                if score_tile > _DROPOUT_FWD_MAX_SCORE_TILE:
                    continue
                if conf.num_stages > _DROPOUT_FWD_MAX_STAGES:
                    continue
                if (
                    head_dim == _MAX_HEAD_DIM
                    and (block_m == _MAX_HEAD_DIM or score_tile > _DROPOUT_FWD_LARGE_TILE)
                    and conf.num_warps < _DROPOUT_FWD_LARGE_TILE_WARPS
                ):
                    continue
            pruned.append(conf)
        return pruned

    def _prune_bwd_configs(configs, named_args, **kwargs):
        """Keep no-dropout tuning unchanged; trim only spill-prone dropout configs."""

        named_args = named_args if isinstance(named_args, dict) else {}
        is_dropout = kwargs.get("IS_DROPOUT", named_args.get("IS_DROPOUT", False))
        if not is_dropout:
            return configs
        return [
            conf
            for conf in configs
            if conf.num_warps in _DROPOUT_BWD_NUM_WARPS
            and conf.num_stages in _DROPOUT_BWD_NUM_STAGES
        ]

    _FWD_CONFIGS = [
        triton.Config({"BLOCK_M": block_m, "BLOCK_N": block_n}, num_stages=num_stages, num_warps=num_warps)
        for block_m in _FWD_BLOCK_M_VALUES
        for block_n in _FWD_BLOCK_N_VALUES
        for num_stages in _FWD_NUM_STAGES
        for num_warps in _FWD_NUM_WARPS
    ]
    _FWD_CONFIGS = [conf for conf in _FWD_CONFIGS if _keep_fwd_config(conf)]

    _BWD_CONFIGS = [
        triton.Config(
            {"BLOCK_M": _BWD_BLOCK_M, "BLOCK_N": _BWD_BLOCK_N},
            num_warps=num_warps,
            num_stages=num_stages,
        )
        for num_warps in _BWD_NUM_WARPS
        for num_stages in _BWD_NUM_STAGES
    ]

    @triton.autotune(
        configs=_FWD_CONFIGS,
        key=[
            "CACHE_KEY_BATCH_HEADS",
            "CACHE_KEY_SEQLEN_Q",
            "CACHE_KEY_SEQLEN_K",
            "IS_CAUSAL",
            "IS_DROPOUT",
            "BLOCK_HEADDIM",
        ],
        prune_configs_by={"early_config_prune": _prune_fwd_configs},
    )
    @triton.heuristics(
        {
            "EVEN_M": lambda args: args["seqlen_q"] % args["BLOCK_M"] == 0,
            "EVEN_N": lambda args: args["seqlen_k"] % args["BLOCK_N"] == 0,
            "EVEN_HEADDIM": lambda args: args["headdim"] == args["BLOCK_HEADDIM"],
        }
    )
    @triton.jit
    def _flash_fwd_kernel(
        Q,
        K,
        V,
        Out,
        LSE,
        DropoutSeed,
        # TMP,
        softmax_scale,
        dropout_p,
        dropout_scale,
        stride_qb,
        stride_qh,
        stride_qm,
        stride_kb,
        stride_kh,
        stride_kn,
        stride_vb,
        stride_vh,
        stride_vn,
        stride_ob,
        stride_oh,
        stride_om,
        nheads,
        CACHE_KEY_BATCH_HEADS,
        seqlen_q,
        seqlen_k,
        seqlen_q_rounded,
        headdim,
        CACHE_KEY_SEQLEN_Q,
        CACHE_KEY_SEQLEN_K,
        IS_CAUSAL: tl.constexpr,
        IS_DROPOUT: tl.constexpr,
        BLOCK_HEADDIM: tl.constexpr,
        EVEN_M: tl.constexpr,
        EVEN_N: tl.constexpr,
        EVEN_HEADDIM: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        start_m = tl.program_id(0)
        off_hb = tl.program_id(1)
        off_b = off_hb // nheads
        off_h = off_hb % nheads

        offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = tl.arange(0, BLOCK_N)
        offs_d = tl.arange(0, BLOCK_HEADDIM)

        q_ptrs = Q + off_b * stride_qb + off_h * stride_qh + (
            offs_m[:, None] * stride_qm + offs_d[None, :]
        )
        k_ptrs = K + off_b * stride_kb + off_h * stride_kh + (
            offs_n[:, None] * stride_kn + offs_d[None, :]
        )
        v_ptrs = V + off_b * stride_vb + off_h * stride_vh + (
            offs_n[:, None] * stride_vn + offs_d[None, :]
        )
        # tmp_ptrs = TMP + off_hb * seqlen_q_rounded + offs_m

        m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
        l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
        acc_o = tl.zeros([BLOCK_M, BLOCK_HEADDIM], dtype=tl.float32)
        qk_scale = softmax_scale * 1.4426950408889634

        if IS_DROPOUT:
            # tl.rand uses a 32-bit counter. Split the logical RNG stream into
            # a per-(batch, head) seed and a per-head QK counter, so counters
            # stay unique for sequence lengths up to 65536.
            dropout_seed = tl.load(DropoutSeed) + off_hb.to(tl.int32)

        if EVEN_M & EVEN_N:
            if EVEN_HEADDIM:
                q = tl.load(q_ptrs)
            else:
                q = tl.load(q_ptrs, mask=offs_d[None, :] < headdim, other=0.0)
        else:
            if EVEN_HEADDIM:
                q = tl.load(q_ptrs, mask=offs_m[:, None] < seqlen_q, other=0.0)
            else:
                q = tl.load(q_ptrs, mask=(offs_m[:, None] < seqlen_q) & (offs_d[None, :] < headdim), other=0.0)

        end_n = seqlen_k if not IS_CAUSAL else tl.minimum((start_m + 1) * BLOCK_M, seqlen_k)
        for start_n in range(0, end_n, BLOCK_N):
            start_n = tl.multiple_of(start_n, BLOCK_N)
            if EVEN_N & EVEN_M:
                if EVEN_HEADDIM:
                    k = tl.load(k_ptrs + start_n * stride_kn)
                else:
                    k = tl.load(
                        k_ptrs + start_n * stride_kn,
                        mask=offs_d[None, :] < headdim,
                        other=0.0,
                    )
            else:
                if EVEN_HEADDIM:
                    k = tl.load(
                        k_ptrs + start_n * stride_kn,
                        mask=(start_n + offs_n)[:, None] < seqlen_k,
                        other=0.0,
                    )
                else:
                    k = tl.load(
                        k_ptrs + start_n * stride_kn,
                        mask=((start_n + offs_n)[:, None] < seqlen_k)
                        & (offs_d[None, :] < headdim),
                        other=0.0,
                    )

            qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
            # qk += tl.dot(q, k, trans_b=True)
            qk += tl.dot(q, tl.trans(k))
            if not EVEN_N:
                qk += tl.where((start_n + offs_n)[None, :] < seqlen_k, 0, float("-inf"))
            if IS_CAUSAL:
                qk += tl.where(offs_m[:, None] >= (start_n + offs_n)[None, :], 0, float("-inf"))

            qk = qk * qk_scale
            m_ij = tl.maximum(tl.max(qk, 1), m_i)
            p = tl.math.exp2(qk - m_ij[:, None])
            l_ij = tl.sum(p, 1)

            acc_o_scale = tl.math.exp2(m_i - m_ij)
            # Old Triton examples materialized this scale through TMP as a
            # compiler workaround. We keep it in registers to avoid the extra
            # global store/load.
            # tl.store(tmp_ptrs, acc_o_scale)
            # acc_o_scale = tl.load(tmp_ptrs)
            acc_o = acc_o * acc_o_scale[:, None]

            if EVEN_N & EVEN_M:
                if EVEN_HEADDIM:
                    v_block = tl.load(v_ptrs + start_n * stride_vn)
                else:
                    v_block = tl.load(
                        v_ptrs + start_n * stride_vn,
                        mask=offs_d[None, :] < headdim,
                        other=0.0,
                    )
            else:
                if EVEN_HEADDIM:
                    v_block = tl.load(
                        v_ptrs + start_n * stride_vn,
                        mask=(start_n + offs_n)[:, None] < seqlen_k,
                        other=0.0,
                    )
                else:
                    v_block = tl.load(
                        v_ptrs + start_n * stride_vn,
                        mask=((start_n + offs_n)[:, None] < seqlen_k)
                        & (offs_d[None, :] < headdim),
                        other=0.0,
                    )
            if IS_DROPOUT:
                rng_offsets = (
                    offs_m[:, None].to(tl.int32) * seqlen_k
                    + (start_n + offs_n)[None, :].to(tl.int32)
                )
                keep = tl.rand(dropout_seed, rng_offsets) > dropout_p
                p_for_v = tl.where(keep, p * dropout_scale, 0.0)
                acc_o += tl.dot(p_for_v.to(v_block.dtype), v_block)
            else:
                # Keep the no-dropout specialization identical to the original
                # FlashAttention accumulation path.
                acc_o += tl.dot(p.to(v_block.dtype), v_block)

            l_i = l_i * acc_o_scale + l_ij
            m_i = m_ij

        lse_i = m_i + tl.math.log2(l_i)
        o_scale = 1.0 / l_i
        # tl.store(tmp_ptrs, o_scale)
        # o_scale = tl.load(tmp_ptrs)
        acc_o = acc_o * o_scale[:, None]

        offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
        tl.store(LSE + off_hb * seqlen_q_rounded + offs_m, lse_i)

        out_ptrs = Out + off_b * stride_ob + off_h * stride_oh + (
            offs_m[:, None] * stride_om + offs_d[None, :]
        )
        if EVEN_M:
            if EVEN_HEADDIM:
                tl.store(out_ptrs, acc_o)
            else:
                tl.store(out_ptrs, acc_o, mask=offs_d[None, :] < headdim)
        else:
            if EVEN_HEADDIM:
                tl.store(out_ptrs, acc_o, mask=offs_m[:, None] < seqlen_q)
            else:
                tl.store(out_ptrs, acc_o, mask=(offs_m[:, None] < seqlen_q) & (offs_d[None, :] < headdim))

    @triton.jit
    def _flash_bwd_preprocess_kernel(
        Out,
        DO,
        Delta,
        stride_ob,
        stride_oh,
        stride_om,
        stride_dob,
        stride_doh,
        stride_dom,
        nheads,
        seqlen_q,
        seqlen_q_rounded,
        headdim,
        BLOCK_M: tl.constexpr,
        BLOCK_HEADDIM: tl.constexpr,
    ):
        start_m = tl.program_id(0)
        off_hb = tl.program_id(1)
        off_b = off_hb // nheads
        off_h = off_hb % nheads
        offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_d = tl.arange(0, BLOCK_HEADDIM)

        o = tl.load(Out + off_b * stride_ob + off_h * stride_oh + offs_m[:, None] * stride_om + offs_d[None, :], mask=(offs_m[:, None] < seqlen_q) & (offs_d[None, :] < headdim), other=0.0).to(tl.float32)
        do = tl.load(DO + off_b * stride_dob + off_h * stride_doh + offs_m[:, None] * stride_dom + offs_d[None, :], mask=(offs_m[:, None] < seqlen_q) & (offs_d[None, :] < headdim), other=0.0).to(tl.float32)
        delta = tl.sum(o * do, axis=1)
        tl.store(Delta + off_hb * seqlen_q_rounded + offs_m, delta)

    @triton.jit
    def _flash_bwd_store_dk_dv(
        dk_ptrs,
        dv_ptrs,
        dk,
        dv,
        offs_n,
        offs_d,
        seqlen_k,
        headdim,
        EVEN_M: tl.constexpr,
        EVEN_N: tl.constexpr,
        EVEN_HEADDIM: tl.constexpr,
    ):
        if EVEN_N & EVEN_M:
            if EVEN_HEADDIM:
                tl.store(dv_ptrs, dv)
                tl.store(dk_ptrs, dk)
            else:
                tl.store(dv_ptrs, dv, mask=offs_d[None, :] < headdim)
                tl.store(dk_ptrs, dk, mask=offs_d[None, :] < headdim)
        else:
            if EVEN_HEADDIM:
                tl.store(dv_ptrs, dv, mask=offs_n[:, None] < seqlen_k)
                tl.store(dk_ptrs, dk, mask=offs_n[:, None] < seqlen_k)
            else:
                tl.store(
                    dv_ptrs,
                    dv,
                    mask=(offs_n[:, None] < seqlen_k) & (offs_d[None, :] < headdim),
                )
                tl.store(
                    dk_ptrs,
                    dk,
                    mask=(offs_n[:, None] < seqlen_k) & (offs_d[None, :] < headdim),
                )

    @triton.jit
    def _flash_bwd_dkdv_tiled(
        dk,
        dv,
        Q,
        k,
        v,
        DO,
        LSE,
        Delta,
        DropoutSeed,
        qk_scale,
        softmax_scale,
        dropout_p,
        dropout_scale,
        stride_qm,
        stride_dom,
        seqlen_q,
        seqlen_k,
        headdim,
        start_n,
        start_m,
        end_m,
        off_hb,
        MASK: tl.constexpr,
        IS_DROPOUT: tl.constexpr,
        BLOCK_M_STEP: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_HEADDIM: tl.constexpr,
    ):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        offs_d = tl.arange(0, BLOCK_HEADDIM)
        if IS_DROPOUT:
            dropout_seed = tl.load(DropoutSeed) + off_hb.to(tl.int32)

        for curr_m in tl.range(start_m, end_m, BLOCK_M_STEP):
            offs_m = curr_m + tl.arange(0, BLOCK_M_STEP)
            qT_ptrs = Q + offs_m[None, :] * stride_qm + offs_d[:, None]
            do_ptrs = DO + offs_m[:, None] * stride_dom + offs_d[None, :]
            qT = tl.load(
                qT_ptrs,
                mask=(offs_m[None, :] < seqlen_q) & (offs_d[:, None] < headdim),
                other=0.0,
            )
            m = tl.load(LSE + offs_m, mask=offs_m < seqlen_q, other=float("inf"))

            qkT = tl.dot(k, qT) * qk_scale
            pT = tl.math.exp2(qkT - m[None, :])
            pT = tl.where(offs_n[:, None] < seqlen_k, pT, 0.0)
            if MASK:
                pT = tl.where(offs_m[None, :] >= offs_n[:, None], pT, 0.0)
            if IS_DROPOUT:
                rng_offsets = (
                    offs_m[None, :].to(tl.int32) * seqlen_k
                    + offs_n[:, None].to(tl.int32)
                )
                keep = tl.rand(dropout_seed, rng_offsets) > dropout_p
                pT_dropout = tl.where(keep, pT * dropout_scale, 0.0)

            do = tl.load(do_ptrs, mask=(offs_m[:, None] < seqlen_q) & (offs_d[None, :] < headdim), other=0.0,)
            delta = tl.load(Delta + offs_m, mask=offs_m < seqlen_q, other=0.0)
            dpT = tl.dot(v, tl.trans(do)).to(tl.float32)
            # qk_scale is only for recomputing p in log2 space. Gradients are
            # with respect to the original QK scores, so use softmax_scale here.
            if IS_DROPOUT:
                dv += tl.dot(pT_dropout.to(do.dtype), do)
                dsT = (
                    (dpT * pT_dropout - pT * delta[None, :]) * softmax_scale
                ).to(qT.dtype)
            else:
                dv += tl.dot(pT.to(do.dtype), do)
                dsT = (pT * (dpT - delta[None, :]) * softmax_scale).to(qT.dtype)
            dk += tl.dot(dsT, tl.trans(qT))

        return dk, dv

    @triton.jit
    def _flash_bwd_dq_tiled(
        dq,
        q,
        K,
        V,
        do,
        m,
        delta,
        DropoutSeed,
        qk_scale,
        softmax_scale,
        dropout_p,
        dropout_scale,
        stride_kn,
        stride_vn,
        seqlen_k,
        headdim,
        start_m,
        start_n,
        end_n,
        off_hb,
        MASK: tl.constexpr,
        IS_DROPOUT: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N_STEP: tl.constexpr,
        BLOCK_HEADDIM: tl.constexpr,
    ):
        offs_m = start_m + tl.arange(0, BLOCK_M)
        offs_d = tl.arange(0, BLOCK_HEADDIM)
        if IS_DROPOUT:
            dropout_seed = tl.load(DropoutSeed) + off_hb.to(tl.int32)

        for curr_n in tl.range(start_n, end_n, BLOCK_N_STEP):
            offs_n = curr_n + tl.arange(0, BLOCK_N_STEP)
            kT_ptrs = K + offs_n[None, :] * stride_kn + offs_d[:, None]
            vT_ptrs = V + offs_n[None, :] * stride_vn + offs_d[:, None]
            kT = tl.load(
                kT_ptrs,
                mask=(offs_n[None, :] < seqlen_k) & (offs_d[:, None] < headdim),
                other=0.0,
            )
            vT = tl.load(
                vT_ptrs,
                mask=(offs_n[None, :] < seqlen_k) & (offs_d[:, None] < headdim),
                other=0.0,
            )

            qk = tl.dot(q, kT) * qk_scale
            p = tl.math.exp2(qk - m)
            p = tl.where(offs_n[None, :] < seqlen_k, p, 0.0)
            if MASK:
                p = tl.where(offs_m[:, None] >= offs_n[None, :], p, 0.0)
            if IS_DROPOUT:
                rng_offsets = (
                    offs_m[:, None].to(tl.int32) * seqlen_k
                    + offs_n[None, :].to(tl.int32)
                )
                keep = tl.rand(dropout_seed, rng_offsets) > dropout_p
                p_dropout = tl.where(keep, p * dropout_scale, 0.0)

            dp = tl.dot(do, vT).to(tl.float32)
            # K is not pre-scaled in this implementation. Keep the true
            # softmax_scale in dS and do not apply an LN2 correction later.
            if IS_DROPOUT:
                ds = ((dp * p_dropout - p * delta[:, None]) * softmax_scale).to(q.dtype)
            else:
                ds = (p * (dp - delta[:, None]) * softmax_scale).to(q.dtype)
            dq += tl.dot(ds, tl.trans(kT))

        return dq

    @triton.autotune(
        configs=_BWD_CONFIGS,
        key=[
            "CACHE_KEY_BATCH_HEADS",
            "CACHE_KEY_SEQLEN_Q",
            "CACHE_KEY_SEQLEN_K",
            "IS_CAUSAL",
            "IS_DROPOUT",
            "BLOCK_HEADDIM",
        ],
        prune_configs_by={"early_config_prune": _prune_bwd_configs},
    )
    @triton.heuristics(
        {
            "EVEN_M": lambda args: args["seqlen_q"] % args["BLOCK_M"] == 0,
            "EVEN_N": lambda args: args["seqlen_k"] % args["BLOCK_N"] == 0,
            "EVEN_HEADDIM": lambda args: args["headdim"] == args["BLOCK_HEADDIM"],
        }
    )
    @triton.jit
    def _flash_bwd_kernel(
        Q,
        K,
        V,
        DO,
        DQ,
        DK,
        DV,
        LSE,
        Delta,
        DropoutSeed,
        softmax_scale,
        dropout_p,
        dropout_scale,
        stride_qb,
        stride_qh,
        stride_qm,
        stride_kb,
        stride_kh,
        stride_kn,
        stride_vb,
        stride_vh,
        stride_vn,
        stride_dob,
        stride_doh,
        stride_dom,
        stride_dqb,
        stride_dqh,
        stride_dqm,
        stride_dkb,
        stride_dkh,
        stride_dkn,
        stride_dvb,
        stride_dvh,
        stride_dvn,
        nheads,
        CACHE_KEY_BATCH_HEADS,
        seqlen_q,
        seqlen_k,
        seqlen_q_rounded,
        headdim,
        CACHE_KEY_SEQLEN_Q,
        CACHE_KEY_SEQLEN_K,
        IS_CAUSAL: tl.constexpr,
        IS_DROPOUT: tl.constexpr,
        BLOCK_HEADDIM: tl.constexpr,
        BWD_DKDV_BLOCK_M_STEP: tl.constexpr,
        BWD_DQ_BLOCK_N_STEP: tl.constexpr,
        BWD_CAUSAL_DIAGONAL_DIVISOR: tl.constexpr,
        EVEN_M: tl.constexpr,
        EVEN_N: tl.constexpr,
        EVEN_HEADDIM: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        off_hb = tl.program_id(1)
        off_b = off_hb // nheads
        off_h = off_hb % nheads

        Q += off_b * stride_qb + off_h * stride_qh
        K += off_b * stride_kb + off_h * stride_kh
        V += off_b * stride_vb + off_h * stride_vh
        DO += off_b * stride_dob + off_h * stride_doh
        DQ += off_b * stride_dqb + off_h * stride_dqh
        DK += off_b * stride_dkb + off_h * stride_dkh
        DV += off_b * stride_dvb + off_h * stride_dvh
        Delta += off_hb * seqlen_q_rounded
        LSE += off_hb * seqlen_q_rounded

        # The active tiled layout writes one DK/DV column block and one DQ row
        # block per program, so DQ needs no global-memory atomics.
        pid = tl.program_id(0)
        qk_scale = softmax_scale * 1.4426950408889634
        offs_d = tl.arange(0, BLOCK_HEADDIM)

        MASK_BLOCK_M_DKDV: tl.constexpr = BWD_DKDV_BLOCK_M_STEP // BWD_CAUSAL_DIAGONAL_DIVISOR
        MASK_BLOCK_N_DQ: tl.constexpr = BWD_DQ_BLOCK_N_STEP // BWD_CAUSAL_DIAGONAL_DIVISOR

        # Compute dK and dV for one K/V column block.
        start_n = pid * BLOCK_N
        offs_n = start_n + tl.arange(0, BLOCK_N)
        k = tl.load(
            K + offs_n[:, None] * stride_kn + offs_d[None, :],
            mask=(offs_n[:, None] < seqlen_k) & (offs_d[None, :] < headdim),
            other=0.0,
        )
        v = tl.load(
            V + offs_n[:, None] * stride_vn + offs_d[None, :],
            mask=(offs_n[:, None] < seqlen_k) & (offs_d[None, :] < headdim),
            other=0.0,
        )
        dk = tl.zeros([BLOCK_N, BLOCK_HEADDIM], dtype=tl.float32)
        dv = tl.zeros([BLOCK_N, BLOCK_HEADDIM], dtype=tl.float32)

        start_m = 0
        if IS_CAUSAL:
            diag_hi = tl.minimum(start_n + BLOCK_N, seqlen_q)
            diag_end = start_n + tl.cdiv(tl.maximum(0, diag_hi - start_n), MASK_BLOCK_M_DKDV) * MASK_BLOCK_M_DKDV
            dk, dv = _flash_bwd_dkdv_tiled(
                dk,
                dv,
                Q,
                k,
                v,
                DO,
                LSE,
                Delta,
                DropoutSeed,
                qk_scale,
                softmax_scale,
                dropout_p,
                dropout_scale,
                stride_qm,
                stride_dom,
                seqlen_q,
                seqlen_k,
                headdim,
                start_n,
                start_n,
                diag_end,
                off_hb,
                MASK=True,
                IS_DROPOUT=IS_DROPOUT,
                BLOCK_M_STEP=MASK_BLOCK_M_DKDV,
                BLOCK_N=BLOCK_N,
                BLOCK_HEADDIM=BLOCK_HEADDIM,
            )
            start_m = diag_end

        tail_end = (
            start_m
            + tl.cdiv(tl.maximum(0, seqlen_q - start_m), BWD_DKDV_BLOCK_M_STEP)
            * BWD_DKDV_BLOCK_M_STEP
        )
        dk, dv = _flash_bwd_dkdv_tiled(
            dk,
            dv,
            Q,
            k,
            v,
            DO,
            LSE,
            Delta,
            DropoutSeed,
            qk_scale,
            softmax_scale,
            dropout_p,
            dropout_scale,
            stride_qm,
            stride_dom,
            seqlen_q,
            seqlen_k,
            headdim,
            start_n,
            start_m,
            tail_end,
            off_hb,
            MASK=False,
            IS_DROPOUT=IS_DROPOUT,
            BLOCK_M_STEP=BWD_DKDV_BLOCK_M_STEP,
            BLOCK_N=BLOCK_N,
            BLOCK_HEADDIM=BLOCK_HEADDIM,
        )

        dv_ptrs = DV + offs_n[:, None] * stride_dvn + offs_d[None, :]
        dk_ptrs = DK + offs_n[:, None] * stride_dkn + offs_d[None, :]
        _flash_bwd_store_dk_dv(
            dk_ptrs,
            dv_ptrs,
            dk,
            dv,
            offs_n,
            offs_d,
            seqlen_k,
            headdim,
            EVEN_M=EVEN_M,
            EVEN_N=EVEN_N,
            EVEN_HEADDIM=EVEN_HEADDIM,
        )

        # Compute dQ for one Q row block.
        start_m = pid * BLOCK_M
        offs_m = start_m + tl.arange(0, BLOCK_M)
        q = tl.load(Q + offs_m[:, None] * stride_qm + offs_d[None, :], mask=(offs_m[:, None] < seqlen_q) & (offs_d[None, :] < headdim), other=0.0)
        do = tl.load(DO + offs_m[:, None] * stride_dom + offs_d[None, :], mask=(offs_m[:, None] < seqlen_q) & (offs_d[None, :] < headdim), other=0.0)
        m = tl.load(LSE + offs_m, mask=offs_m < seqlen_q, other=float("inf"))[:, None]
        delta = tl.load(Delta + offs_m, mask=offs_m < seqlen_q, other=0.0)
        dq = tl.zeros([BLOCK_M, BLOCK_HEADDIM], dtype=tl.float32)

        if IS_CAUSAL:
            full_end = tl.cdiv(tl.minimum(start_m, seqlen_k), BWD_DQ_BLOCK_N_STEP) * BWD_DQ_BLOCK_N_STEP
            dq = _flash_bwd_dq_tiled(
                dq,
                q,
                K,
                V,
                do,
                m,
                delta,
                DropoutSeed,
                qk_scale,
                softmax_scale,
                dropout_p,
                dropout_scale,
                stride_kn,
                stride_vn,
                seqlen_k,
                headdim,
                start_m,
                0,
                full_end,
                off_hb,
                MASK=False,
                IS_DROPOUT=IS_DROPOUT,
                BLOCK_M=BLOCK_M,
                BLOCK_N_STEP=BWD_DQ_BLOCK_N_STEP,
                BLOCK_HEADDIM=BLOCK_HEADDIM,
            )
            diag_hi = tl.minimum(start_m + BLOCK_M, seqlen_k)
            diag_end = start_m + tl.cdiv(tl.maximum(0, diag_hi - start_m), MASK_BLOCK_N_DQ) * MASK_BLOCK_N_DQ
            dq = _flash_bwd_dq_tiled(
                dq,
                q,
                K,
                V,
                do,
                m,
                delta,
                DropoutSeed,
                qk_scale,
                softmax_scale,
                dropout_p,
                dropout_scale,
                stride_kn,
                stride_vn,
                seqlen_k,
                headdim,
                start_m,
                start_m,
                diag_end,
                off_hb,
                MASK=True,
                IS_DROPOUT=IS_DROPOUT,
                BLOCK_M=BLOCK_M,
                BLOCK_N_STEP=MASK_BLOCK_N_DQ,
                BLOCK_HEADDIM=BLOCK_HEADDIM,
            )
        else:
            end_n = tl.cdiv(seqlen_k, BWD_DQ_BLOCK_N_STEP) * BWD_DQ_BLOCK_N_STEP
            dq = _flash_bwd_dq_tiled(
                dq,
                q,
                K,
                V,
                do,
                m,
                delta,
                DropoutSeed,
                qk_scale,
                softmax_scale,
                dropout_p,
                dropout_scale,
                stride_kn,
                stride_vn,
                seqlen_k,
                headdim,
                start_m,
                0,
                end_n,
                off_hb,
                MASK=False,
                IS_DROPOUT=IS_DROPOUT,
                BLOCK_M=BLOCK_M,
                BLOCK_N_STEP=BWD_DQ_BLOCK_N_STEP,
                BLOCK_HEADDIM=BLOCK_HEADDIM,
            )

        dq_ptrs = DQ + offs_m[:, None] * stride_dqm + offs_d[None, :]
        tl.store(dq_ptrs, dq, mask=(offs_m[:, None] < seqlen_q) & (offs_d[None, :] < headdim))

    @triton.jit
    def _flash_dropout_mask_kernel(
        Mask,
        DropoutSeed,
        dropout_p,
        seqlen_q,
        seqlen_k,
        n_elements,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        valid = offsets < n_elements

        offs_k = offsets % seqlen_k
        tmp = offsets // seqlen_k
        offs_q = tmp % seqlen_q
        off_hb = tmp // seqlen_q
        rng_offsets = offs_q.to(tl.int32) * seqlen_k + offs_k.to(tl.int32)

        seed = tl.load(DropoutSeed) + off_hb.to(tl.int32)
        keep = tl.rand(seed, rng_offsets) > dropout_p
        tl.store(Mask + offsets, keep, mask=valid)


def flash_attention_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    is_causal: bool,
    dropout_p: float,
    dropout_seed: torch.Tensor | None = None,
    softmax_scale: float | None = None,
):
    """Launch the local Triton FlashAttention forward kernel.

    Inputs use MiniTrain's native `(batch, heads, seq, head_dim)` layout and
    must be contiguous in the last dimension.
    """

    if triton is None:
        raise RuntimeError("Triton is not installed. Install mini-train-sys[triton].")
    batch, nheads, seqlen_q, head_dim = q.shape
    _, k_heads, seqlen_k, k_head_dim = k.shape
    if k.shape != (batch, nheads, seqlen_k, head_dim):
        raise ValueError(f"k shape {tuple(k.shape)} is incompatible with q shape {tuple(q.shape)}.")
    if v.shape != (batch, nheads, seqlen_k, head_dim):
        raise ValueError(f"v shape {tuple(v.shape)} is incompatible with q shape {tuple(q.shape)}.")
    if k_heads != nheads or k_head_dim != head_dim:
        raise ValueError("FlashAttention requires matching Q/K/V head counts and head dimensions.")
    if head_dim > _MAX_HEAD_DIM:
        raise ValueError(f"Triton FlashAttention supports head_dim <= {_MAX_HEAD_DIM}.")
    if q.dtype not in _SUPPORTED_DTYPES or k.dtype != q.dtype or v.dtype != q.dtype:
        raise TypeError("Triton FlashAttention supports matching fp16/bf16/fp32 Q/K/V tensors only.")
    if not (q.is_cuda and k.is_cuda and v.is_cuda):
        raise RuntimeError("Triton FlashAttention requires CUDA tensors.")
    if not (0.0 <= dropout_p < 1.0):
        raise ValueError("Triton FlashAttention requires 0.0 <= dropout_p < 1.0.")
    if dropout_p != 0.0 and seqlen_q * seqlen_k > _MAX_DROPOUT_COUNTER:
        raise ValueError("Triton FlashAttention dropout requires seqlen_q * seqlen_k <= 2**32.")

    softmax_scale = softmax_scale or 1.0 / math.sqrt(head_dim)
    dropout_scale = 1.0 / (1.0 - dropout_p) if dropout_p != 0.0 else 1.0
    seqlen_q_rounded = _round_up_to_block(seqlen_q)
    block_headdim = _head_dim_block(head_dim)
    if dropout_p != 0.0:
        if dropout_seed is None:
            dropout_seed = torch.empty((), device=q.device, dtype=torch.int32)
            dropout_seed.random_(0, 2**31 - 1)
        elif dropout_seed.device != q.device or dropout_seed.dtype != torch.int32 or dropout_seed.numel() != 1:
            raise ValueError("dropout_seed must be a one-element CUDA int32 tensor on the same device as q.")
    else:
        dropout_seed = None

    out = torch.empty_like(q)
    lse = torch.empty((batch, nheads, seqlen_q_rounded), device=q.device, dtype=torch.float32)
    # TMP used to materialize forward rescale factors for a Triton compiler
    # workaround. The kernel now keeps those factors in registers.
    # tmp = torch.empty_like(lse)
    grid = lambda META: (triton.cdiv(seqlen_q, META["BLOCK_M"]), batch * nheads)
    _flash_fwd_kernel[grid](
        q,
        k,
        v,
        out,
        lse,
        dropout_seed if dropout_seed is not None else q,
        # tmp,
        softmax_scale,
        dropout_p,
        dropout_scale,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        nheads,
        batch * nheads,
        seqlen_q,
        seqlen_k,
        seqlen_q_rounded,
        head_dim,
        _cache_key_dim(seqlen_q),
        _cache_key_dim(seqlen_k),
        is_causal,
        dropout_p != 0.0,
        block_headdim,
    )
    return out, lse, dropout_seed, softmax_scale


def flash_attention_backward(
    do: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    lse: torch.Tensor,
    dropout_seed: torch.Tensor | None,
    *,
    is_causal: bool,
    dropout_p: float,
    softmax_scale: float,
):
    """Launch the local Triton FlashAttention backward kernels."""

    if do.stride(-1) != 1:
        do = do.contiguous()
    batch, nheads, seqlen_q, head_dim = q.shape
    _, _, seqlen_k, _ = k.shape
    seqlen_q_rounded = _round_up_to_block(seqlen_q)
    if lse.shape != (batch, nheads, seqlen_q_rounded):
        raise ValueError(f"Unexpected LSE shape {tuple(lse.shape)}.")
    if not (0.0 <= dropout_p < 1.0):
        raise ValueError("Triton FlashAttention requires 0.0 <= dropout_p < 1.0.")
    if dropout_p != 0.0 and seqlen_q * seqlen_k > _MAX_DROPOUT_COUNTER:
        raise ValueError("Triton FlashAttention dropout requires seqlen_q * seqlen_k <= 2**32.")
    if dropout_p != 0.0 and (
        dropout_seed is None
        or dropout_seed.device != q.device
        or dropout_seed.dtype != torch.int32
        or dropout_seed.numel() != 1
    ):
        raise ValueError("dropout_seed must be a one-element CUDA int32 tensor on the same device as q.")

    block_headdim = _head_dim_block(head_dim)
    dropout_scale = 1.0 / (1.0 - dropout_p) if dropout_p != 0.0 else 1.0
    dq = torch.empty_like(q)
    dk = torch.empty_like(k)
    dv = torch.empty_like(v)
    delta = torch.empty_like(lse)

    preprocess_grid = lambda META: (triton.cdiv(seqlen_q, META["BLOCK_M"]), batch * nheads)
    _flash_bwd_preprocess_kernel[preprocess_grid](
        out,
        do,
        delta,
        out.stride(0),
        out.stride(1),
        out.stride(2),
        do.stride(0),
        do.stride(1),
        do.stride(2),
        nheads,
        seqlen_q,
        seqlen_q_rounded,
        head_dim,
        BLOCK_M=_FWD_LSE_BLOCK_M,
        BLOCK_HEADDIM=block_headdim,
    )

    backward_grid = lambda META: (
        max(triton.cdiv(seqlen_k, META["BLOCK_N"]), triton.cdiv(seqlen_q, META["BLOCK_M"])),
        batch * nheads,
    )
    _flash_bwd_kernel[backward_grid](
        q,
        k,
        v,
        do,
        dq,
        dk,
        dv,
        lse,
        delta,
        dropout_seed if dropout_seed is not None else q,
        softmax_scale,
        dropout_p,
        dropout_scale,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        do.stride(0),
        do.stride(1),
        do.stride(2),
        dq.stride(0),
        dq.stride(1),
        dq.stride(2),
        dk.stride(0),
        dk.stride(1),
        dk.stride(2),
        dv.stride(0),
        dv.stride(1),
        dv.stride(2),
        nheads,
        batch * nheads,
        seqlen_q,
        seqlen_k,
        seqlen_q_rounded,
        head_dim,
        _cache_key_dim(seqlen_q),
        _cache_key_dim(seqlen_k),
        is_causal,
        dropout_p != 0.0,
        block_headdim,
        BWD_DKDV_BLOCK_M_STEP=_BWD_DKDV_BLOCK_M_STEP,
        BWD_DQ_BLOCK_N_STEP=_BWD_DQ_BLOCK_N_STEP,
        BWD_CAUSAL_DIAGONAL_DIVISOR=_BWD_CAUSAL_DIAGONAL_DIVISOR,
    )
    return dq, dk, dv


def flash_attention_dropout_mask(
    batch: int,
    nheads: int,
    seqlen_q: int,
    seqlen_k: int,
    *,
    device: torch.device | str,
    dropout_p: float,
    dropout_seed: torch.Tensor,
) -> torch.Tensor:
    """Materialize the exact stateless dropout mask used by the Triton kernels.

    This is intended for small correctness checks. Production forward/backward
    regenerate the mask online and never store it.
    """

    if triton is None:
        raise RuntimeError("Triton is not installed. Install mini-train-sys[triton].")
    if not (0.0 <= dropout_p < 1.0):
        raise ValueError("Triton FlashAttention requires 0.0 <= dropout_p < 1.0.")
    if seqlen_q * seqlen_k > _MAX_DROPOUT_COUNTER:
        raise ValueError("Triton FlashAttention dropout requires seqlen_q * seqlen_k <= 2**32.")
    if not dropout_seed.is_cuda or dropout_seed.dtype != torch.int32 or dropout_seed.numel() != 1:
        raise ValueError("Triton FlashAttention dropout mask requires a one-element CUDA int32 seed tensor.")

    mask = torch.empty((batch, nheads, seqlen_q, seqlen_k), device=device, dtype=torch.bool)
    n_elements = mask.numel()
    grid = (triton.cdiv(n_elements, 1024),)
    _flash_dropout_mask_kernel[grid](
        mask,
        dropout_seed,
        dropout_p,
        seqlen_q,
        seqlen_k,
        n_elements,
        BLOCK_SIZE=1024,
    )
    return mask


def flash_attention_autotune_kernels() -> dict[str, object]:
    """Return FlashAttention autotuners for notebook benchmark reporting."""

    if triton is None:
        return {}
    return {
        "flash_attention_forward": _flash_fwd_kernel,
        "flash_attention_backward": _flash_bwd_kernel,
    }


class MiniTrainFlashAttentionFunction(torch.autograd.Function):
    """Autograd bridge around the local Triton FlashAttention launchers."""

    @staticmethod
    def forward(ctx, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, is_causal: bool, dropout_p: float):
        q, k, v = [x if x.stride(-1) == 1 else x.contiguous() for x in (q, k, v)]
        out, lse, dropout_seed, softmax_scale = flash_attention_forward(
            q,
            k,
            v,
            is_causal=is_causal,
            dropout_p=dropout_p,
        )
        ctx.save_for_backward(q, k, v, out, lse)
        ctx.dropout_seed = dropout_seed
        ctx.is_causal = is_causal
        ctx.dropout_p = dropout_p
        ctx.softmax_scale = softmax_scale
        return out

    @staticmethod
    def backward(ctx, do: torch.Tensor):
        q, k, v, out, lse = ctx.saved_tensors
        # Triton launches may mutate tensor version counters during autotune/JIT
        # setup. Running under inference_mode matches upstream's autograd bridge.
        with torch.inference_mode():
            dq, dk, dv = flash_attention_backward(
                do,
                q,
                k,
                v,
                out,
                lse,
                ctx.dropout_seed,
                is_causal=ctx.is_causal,
                dropout_p=ctx.dropout_p,
                softmax_scale=ctx.softmax_scale,
            )
        return dq, dk, dv, None, None


def flash_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    is_causal: bool,
    dropout_p: float,
) -> torch.Tensor:
    """Compute attention through the local Triton FlashAttention implementation.

    Args:
        q, k, v: `(batch, heads, seq, head_dim)` tensors.
        is_causal: Whether to apply a causal mask.
        dropout_p: Attention dropout probability. Nonzero dropout uses a
            stateless Triton RNG mask that is regenerated in backward.
    """

    if not is_flash_attention_supported(q, k, v, dropout_p=dropout_p):
        raise RuntimeError(
            "Local Triton FlashAttention requires CUDA fp16/bf16/fp32 Q/K/V tensors, "
            "matching batch/head/head_dim shapes, head_dim <= 128, contiguous last "
            "dimension, and 0.0 <= dropout_p < 1.0."
        )
    return MiniTrainFlashAttentionFunction.apply(q, k, v, is_causal, float(dropout_p))
