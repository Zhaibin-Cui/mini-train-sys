# SPDX-License-Identifier: BSD-2-Clause
# Adapted from Liger-Kernel's fused MoE implementation.

import functools

import torch
import triton

from minitrain.kernels.triton.fused_moe_kernels import _fused_down_proj_kernel
from minitrain.kernels.triton.fused_moe_kernels import _fused_up_proj_swiglu_kernel
from minitrain.kernels.triton.fused_moe_kernels import _moe_bwd_down_proj_kernel
from minitrain.kernels.triton.fused_moe_kernels import _moe_bwd_dW1_kernel
from minitrain.kernels.triton.fused_moe_kernels import _moe_bwd_dW2_kernel
from minitrain.kernels.triton.fused_moe_kernels import _moe_bwd_dX_expanded_kernel
from minitrain.kernels.triton.fused_moe_kernels import _moe_router_histogram_kernel
from minitrain.kernels.triton.fused_moe_kernels import _moe_router_prefix_sum_kernel
from minitrain.kernels.triton.fused_moe_kernels import _moe_router_scatter_kernel
from minitrain.kernels.triton.fused_moe_kernels import _token_gather_weighted_sum_kernel


def ensure_contiguous(fn):
    @functools.wraps(fn)
    def wrapper(ctx, *args, **kwargs):
        args = [arg.contiguous() if isinstance(arg, torch.Tensor) else arg for arg in args]
        kwargs = {
            key: value.contiguous() if isinstance(value, torch.Tensor) else value
            for key, value in kwargs.items()
        }
        return fn(ctx, *args, **kwargs)

    return wrapper


BLOCK_M_TOKEN = 64


def is_fused_moe_supported(
    x: torch.Tensor,
    gate_up_proj: torch.Tensor,
    down_proj: torch.Tensor,
    top_k_index: torch.Tensor,
    top_k_weights: torch.Tensor,
) -> bool:
    tensors = (x, gate_up_proj, down_proj, top_k_index, top_k_weights)
    if not all(tensor.is_cuda for tensor in tensors):
        return False
    if len({tensor.device for tensor in tensors}) != 1:
        return False
    if x.ndim != 2 or gate_up_proj.ndim != 3 or down_proj.ndim != 3:
        return False
    if top_k_index.ndim != 2 or top_k_weights.shape != top_k_index.shape:
        return False
    tokens, hidden = x.shape
    experts, twice_intermediate, weight_hidden = gate_up_proj.shape
    intermediate = twice_intermediate // 2
    supported_dtypes = (torch.float16, torch.bfloat16, torch.float32)
    return (
        tokens > 0
        and experts > 0
        and experts < 0xFFFF
        and twice_intermediate % 2 == 0
        and weight_hidden == hidden
        and down_proj.shape == (experts, hidden, intermediate)
        and top_k_index.shape[0] == tokens
        and 0 < top_k_index.shape[1] <= min(experts, 32768)
        and top_k_index.dtype == torch.int32
        and top_k_weights.dtype == x.dtype
        and x.dtype in supported_dtypes
        and gate_up_proj.dtype in supported_dtypes
        and down_proj.dtype in supported_dtypes
    )


def compute_routing_metadata(
    topk_indices: torch.Tensor,
    E: int,
    block_m_token: int = BLOCK_M_TOKEN,
):
    T, K = topk_indices.shape
    TK = T * K
    device = topk_indices.device
    E_POW2 = triton.next_power_of_2(E)
    K_POW2 = triton.next_power_of_2(K)
    TOKENS_PER_BLOCK = max(1, 1024 // K_POW2)
    n_tiles = triton.cdiv(T, TOKENS_PER_BLOCK)

    tile_expert_counts = torch.empty(E, n_tiles, dtype=torch.int32, device=device)
    _moe_router_histogram_kernel[(n_tiles,)](
        topk_indices,
        tile_expert_counts,
        T,
        E=E,
        n_tiles=n_tiles,
        TOKENS_PER_TILE=TOKENS_PER_BLOCK,
        K_POW2=K_POW2,
        K=K,
        E_POW2=E_POW2,
    )

    expert_token_count = tile_expert_counts.sum(dim=1, dtype=torch.int32)  # (E,)

    expert_start_idx = torch.empty(E + 1, dtype=torch.int32, device=device)
    expert_tile_offset = torch.empty(E + 1, dtype=torch.int32, device=device)
    _moe_router_prefix_sum_kernel[(E + 2,)](
        expert_token_count,
        expert_start_idx,
        expert_tile_offset,
        E=E,
        partial_sum_ptr=tile_expert_counts,
        n_tiles=n_tiles,
        TK=TK,
        BLOCK_M=128,
        BLOCK_N=E_POW2,
        BLOCK_M_TOKEN=block_m_token,
    )

    tile_count = expert_tile_offset[-1:]
    max_m_tiles = min(TK, triton.cdiv(TK, block_m_token) + E - 1)
    tile_row_start = torch.empty(max_m_tiles, dtype=torch.int32, device=device)
    tile_expert = torch.empty(max_m_tiles, dtype=torch.int32, device=device)

    s_scatter_idx = torch.empty(TK, dtype=torch.int32, device=device)
    s_reverse_scatter_idx = torch.empty(TK, dtype=torch.int32, device=device)
    x_gather_idx = torch.empty(TK, dtype=torch.int32, device=device)

    if TK > 0:
        _moe_router_scatter_kernel[(n_tiles,)](
            s_scatter_idx,
            s_reverse_scatter_idx,
            x_gather_idx,
            tile_row_start,
            tile_expert,
            topk_indices,
            T,
            tile_expert_counts,  # non-contiguous (E, n_tiles) view
            n_tiles,
            expert_start_idx[:E],  # E entries (without TK sentinel)
            expert_tile_offset[:E],  # E entries of cumulative tile counts
            K_POW2=K_POW2,
            K=K,
            TOKENS_PER_BLOCK=TOKENS_PER_BLOCK,
            BLOCK_M_TOKEN=block_m_token,
        )

    return (
        expert_token_count,
        expert_start_idx,
        x_gather_idx,
        s_scatter_idx,
        s_reverse_scatter_idx,
        tile_row_start,
        tile_expert,
        tile_count,
    )


def _token_aggregation(Y, topk_weights_flat, s_reverse_scatter_idx, T, K, H):
    out = torch.empty(T, H, dtype=Y.dtype, device=Y.device)
    _token_gather_weighted_sum_kernel[(T,)](
        Y,
        topk_weights_flat,
        s_reverse_scatter_idx,
        out,
        H_dim=H,
        K_dim=K,
        stride_Y_TK=Y.stride(0),
        stride_Y_H=Y.stride(1),
        stride_out_T=out.stride(0),
        stride_out_H=out.stride(1),
        w_is_None=False,
    )
    return out


class _FusedMoEFunction(torch.autograd.Function):
    @staticmethod
    @ensure_contiguous
    def forward(ctx, x, gate_up_proj, down_proj, top_k_index, top_k_weights):
        T, K = top_k_index.shape
        E = gate_up_proj.shape[0]
        H = x.shape[1]
        intermediate_dim = gate_up_proj.shape[1] // 2
        TK = T * K

        with torch.no_grad():
            (
                _,
                expert_start_idx,
                x_gather_idx,
                s_scatter_idx,
                s_reverse_scatter_idx,
                tile_row_start,
                tile_expert,
                tile_count,
            ) = compute_routing_metadata(top_k_index, E)

        max_m_tiles = tile_row_start.shape[0]

        pre_act = torch.empty(TK, 2 * intermediate_dim, dtype=x.dtype, device=x.device)
        post_act = torch.empty(TK, intermediate_dim, dtype=x.dtype, device=x.device)

        if max_m_tiles > 0:
            _fused_up_proj_swiglu_kernel[
                lambda meta: (max_m_tiles, triton.cdiv(intermediate_dim, meta["BLOCK_N"]))
            ](
                x,
                gate_up_proj,
                x_gather_idx,
                expert_start_idx,
                tile_row_start,
                tile_expert,
                tile_count,
                pre_act,
                post_act,
                H_dim=H,
                I_dim=intermediate_dim,
                stride_x_T=x.stride(0),
                stride_x_H=x.stride(1),
                stride_w_E=gate_up_proj.stride(0),
                stride_w_N=gate_up_proj.stride(1),
                stride_w_K=gate_up_proj.stride(2),
                stride_pre_TK=pre_act.stride(0),
                stride_pre_N=pre_act.stride(1),
                stride_post_TK=post_act.stride(0),
                stride_post_N=post_act.stride(1),
                BLOCK_M=BLOCK_M_TOKEN,
            )

        Y = torch.empty(TK, H, dtype=x.dtype, device=x.device)

        if max_m_tiles > 0:
            _fused_down_proj_kernel[lambda meta: (max_m_tiles, triton.cdiv(H, meta["BLOCK_N"]))](
                post_act,
                down_proj,
                expert_start_idx,
                tile_row_start,
                tile_expert,
                tile_count,
                Y,
                H_dim=H,
                I_dim=intermediate_dim,
                stride_post_TK=post_act.stride(0),
                stride_post_I=post_act.stride(1),
                stride_w_E=down_proj.stride(0),
                stride_w_H=down_proj.stride(1),
                stride_w_I=down_proj.stride(2),
                stride_Y_TK=Y.stride(0),
                stride_Y_H=Y.stride(1),
                BLOCK_M=BLOCK_M_TOKEN,
            )

        topk_weights_flat = top_k_weights.flatten().contiguous()
        out = _token_aggregation(Y, topk_weights_flat, s_reverse_scatter_idx, T, K, H)

        ctx.save_for_backward(
            x,
            gate_up_proj,
            down_proj,
            pre_act,
            topk_weights_flat,
            expert_start_idx,
            x_gather_idx,
            s_scatter_idx,
            s_reverse_scatter_idx,
            tile_row_start,
            tile_expert,
            tile_count,
        )
        ctx.T = T
        ctx.K = K
        ctx.E = E
        ctx.H = H
        ctx.intermediate_dim = intermediate_dim
        ctx.TK = TK
        ctx.max_m_tiles = max_m_tiles
        ctx.mark_non_differentiable(top_k_index)
        ctx.set_materialize_grads(False)

        return out

    @staticmethod
    @ensure_contiguous
    def backward(ctx, dO):
        if dO is None:
            return None, None, None, None, None

        (
            x,
            gate_up_proj,
            down_proj,
            pre_act,
            topk_weights_flat,
            expert_start_idx,
            x_gather_idx,
            s_scatter_idx,
            s_reverse_scatter_idx,
            tile_row_start,
            tile_expert,
            tile_count,
        ) = ctx.saved_tensors

        T = ctx.T
        K = ctx.K
        E = ctx.E
        H = ctx.H
        intermediate_dim = ctx.intermediate_dim
        TK = ctx.TK
        max_m_tiles = ctx.max_m_tiles

        d_pre_act = torch.empty(TK, 2 * intermediate_dim, dtype=dO.dtype, device=dO.device)
        weighted_act = torch.empty(TK, intermediate_dim, dtype=dO.dtype, device=dO.device)
        # Triton/CUDA do not provide a portable bf16 atomic add. Routing-weight
        # gradients accumulate across N tiles in fp32 and are cast only at the
        # autograd boundary.
        dS = torch.zeros(TK, dtype=torch.float32, device=dO.device)

        if max_m_tiles > 0:
            _moe_bwd_down_proj_kernel[
                lambda meta: (max_m_tiles, triton.cdiv(intermediate_dim, meta["BLOCK_N"]))
            ](
                dO,
                x_gather_idx,
                s_scatter_idx,
                topk_weights_flat,
                down_proj,
                pre_act,
                expert_start_idx,
                tile_row_start,
                tile_expert,
                tile_count,
                d_pre_act,
                weighted_act,
                dS,
                H_dim=H,
                I_dim=intermediate_dim,
                stride_dO_T=dO.stride(0),
                stride_dO_H=dO.stride(1),
                stride_w_E=down_proj.stride(0),
                stride_w_H=down_proj.stride(1),
                stride_w_I=down_proj.stride(2),
                stride_pre_TK=pre_act.stride(0),
                stride_pre_N=pre_act.stride(1),
                stride_d_pre_TK=d_pre_act.stride(0),
                stride_d_pre_N=d_pre_act.stride(1),
                stride_wact_TK=weighted_act.stride(0),
                stride_wact_I=weighted_act.stride(1),
                BLOCK_M=BLOCK_M_TOKEN,
            )

        ddown_proj = torch.zeros_like(down_proj)
        _moe_bwd_dW2_kernel[
            lambda meta: (
                E * triton.cdiv(intermediate_dim, meta["BLOCK_M"]),
                triton.cdiv(H, meta["BLOCK_N"]),
            )
        ](
            weighted_act,
            dO,
            x_gather_idx,
            expert_start_idx,
            ddown_proj,
            H_dim=H,
            I_dim=intermediate_dim,
            stride_wact_TK=weighted_act.stride(0),
            stride_wact_I=weighted_act.stride(1),
            stride_dout_T=dO.stride(0),
            stride_dout_H=dO.stride(1),
            stride_dW2_E=ddown_proj.stride(0),
            stride_dW2_H=ddown_proj.stride(1),
            stride_dW2_I=ddown_proj.stride(2),
        )

        dx_expanded = torch.empty(TK, H, dtype=dO.dtype, device=dO.device)

        if max_m_tiles > 0:
            _moe_bwd_dX_expanded_kernel[
                lambda meta: (max_m_tiles, triton.cdiv(H, meta["BLOCK_N"]))
            ](
                d_pre_act,
                gate_up_proj,
                expert_start_idx,
                tile_row_start,
                tile_expert,
                tile_count,
                dx_expanded,
                H_dim=H,
                I_dim=intermediate_dim,
                stride_d_pre_TK=d_pre_act.stride(0),
                stride_d_pre_N=d_pre_act.stride(1),
                stride_w_E=gate_up_proj.stride(0),
                stride_w_N=gate_up_proj.stride(1),
                stride_w_K=gate_up_proj.stride(2),
                stride_dxe_TK=dx_expanded.stride(0),
                stride_dxe_H=dx_expanded.stride(1),
                BLOCK_M=BLOCK_M_TOKEN,
            )

        dx = torch.zeros(T, H, dtype=dO.dtype, device=dO.device)
        if TK > 0:
            _token_gather_weighted_sum_kernel[(T,)](
                dx_expanded,
                dS,  # dummy w_ptr --never loaded when w_is_None=True
                s_reverse_scatter_idx,
                dx,
                H_dim=H,
                K_dim=K,
                stride_Y_TK=dx_expanded.stride(0),
                stride_Y_H=dx_expanded.stride(1),
                stride_out_T=dx.stride(0),
                stride_out_H=dx.stride(1),
                w_is_None=True,
            )

        dgate_up_proj = torch.zeros_like(gate_up_proj)
        _moe_bwd_dW1_kernel[
            lambda meta: (
                E * triton.cdiv(H, meta["BLOCK_M"]),
                triton.cdiv(2 * intermediate_dim, meta["BLOCK_N"]),
            )
        ](
            x,
            d_pre_act,
            x_gather_idx,
            expert_start_idx,
            dgate_up_proj,
            H_dim=H,
            I_dim=intermediate_dim,
            stride_x_T=x.stride(0),
            stride_x_H=x.stride(1),
            stride_d_pre_TK=d_pre_act.stride(0),
            stride_d_pre_N=d_pre_act.stride(1),
            stride_dW1_E=dgate_up_proj.stride(0),
            stride_dW1_N=dgate_up_proj.stride(1),
            stride_dW1_H=dgate_up_proj.stride(2),
        )

        return (
            dx,
            dgate_up_proj,
            ddown_proj,
            None,
            dS.to(topk_weights_flat.dtype).view(T, K),
        )


def fused_moe(x, gate_up_proj, down_proj, top_k_index, top_k_weights):
    if not is_fused_moe_supported(x, gate_up_proj, down_proj, top_k_index, top_k_weights):
        raise RuntimeError("Unsupported tensor contract for Triton fused MoE")
    return _FusedMoEFunction.apply(x, gate_up_proj, down_proj, top_k_index, top_k_weights)
