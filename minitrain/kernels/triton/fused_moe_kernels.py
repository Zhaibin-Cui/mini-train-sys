# SPDX-License-Identifier: BSD-2-Clause
import os

import triton
import triton.language as tl

_AUTOTUNE_DISABLED = os.environ.get("MINITRAIN_MOE_AUTOTUNE", "0").lower() in (
    "0",
    "false",
    "no",
)

@triton.jit
def _keyed_add(x, y):
    # Segmented addition on packed (key, count) values.
    # Example: [(e0, 1), (e0, 1), (e1, 1)] scans to
    #          [(e0, 1), (e0, 2), (e1, 1)].
    key_mask: tl.constexpr = 0xFFFF0000
    kx = x & key_mask
    ky = y & key_mask
    z = tl.where(kx == ky, x + y - kx, y)
    return z


@triton.jit
def _moe_router_histogram_kernel(
    topk_indices_ptr,  # (T, K) int32
    partial_sum_ptr,  # (E, n_tiles) int32 --output; partial_sum[e, tile] = count
    T,
    E: tl.constexpr,
    n_tiles,
    TOKENS_PER_TILE: tl.constexpr,
    K_POW2: tl.constexpr,
    K: tl.constexpr,
    E_POW2: tl.constexpr,
):
    # Example with TOKENS_PER_TILE=2 and topk experts
    #   [[0, 2], [1, 2], [0, 1], [2, 2]]:
    # each column below is one token tile, so partial_sum[e, tile] becomes
    #   expert 0: [1, 1], expert 1: [1, 1], expert 2: [2, 2].
    tile_id = tl.program_id(0)

    e_offs = tl.arange(0, E_POW2)
    tl.store(
        partial_sum_ptr + e_offs * n_tiles + tile_id,
        tl.zeros([E_POW2], tl.int32),
        mask=e_offs < E,
    )

    tok_offs = tile_id * TOKENS_PER_TILE + tl.arange(0, TOKENS_PER_TILE)
    k_offs = tl.arange(0, K_POW2)
    tok_mask = tok_offs < T
    load_mask = tok_mask[:, None] & (k_offs[None, :] < K)
    safe_k = tl.minimum(k_offs, K - 1)  # clamp for out-of-bounds k slots
    expert_ids = tl.load(
        topk_indices_ptr + tok_offs[:, None] * K + safe_k[None, :],
        mask=load_mask,
        other=-1,
    )

    flat_experts = tl.reshape(expert_ids, [TOKENS_PER_TILE * K_POW2])
    flat_mask = tl.reshape(load_mask, [TOKENS_PER_TILE * K_POW2])
    safe_experts = tl.where(flat_mask, flat_experts, 0)  # redirect masked lanes to expert 0

    tl.atomic_add(
        partial_sum_ptr + safe_experts * n_tiles + tile_id,
        tl.full([TOKENS_PER_TILE * K_POW2], 1, dtype=tl.int32),
        mask=flat_mask,
    )


@triton.jit
def _moe_router_prefix_sum_kernel(
    expert_freq_ptr,  # (E,) int32 --total tokens assigned to each expert
    expert_freq_offs_ptr,  # (E+1,) int32 --output: exclusive cumsum of expert_frequency
    expert_tile_offset_ptr,  # (E+1,) int32 --output: exclusive cumsum of ceil(freq/BLOCK_M_TOKEN)
    E: tl.constexpr,
    partial_sum_ptr,  # (E, n_tiles) int32 --in-place: raw tile counts ->tile prefix sums
    n_tiles,
    TK,  # T * K, written as sentinel into expert_freq_offs[E]
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_M_TOKEN: tl.constexpr,
):
    # Continuing the histogram example, raw per-tile counts
    #   [[1, 1], [1, 1], [2, 2]]
    # become exclusive per-expert tile prefixes
    #   [[0, 1], [0, 1], [0, 2]].
    # With expert_freq=[2, 2, 4] and BLOCK_M_TOKEN=2, this also produces
    #   expert_freq_offs=[0, 2, 4, 8] and expert_tile_offset=[0, 1, 2, 4].
    pid = tl.program_id(0)
    if pid < E:
        expert_partial_sum_ptr = partial_sum_ptr + pid * n_tiles
        curr_sum = 0
        for start in range(0, n_tiles, BLOCK_M):
            offs = start + tl.arange(0, BLOCK_M)
            tile_counts = tl.load(expert_partial_sum_ptr + offs, mask=offs < n_tiles, other=0)
            excl_cumsum = tl.cumsum(tile_counts, 0) - tile_counts + curr_sum
            curr_sum += tl.sum(tile_counts, 0)
            tl.store(expert_partial_sum_ptr + offs, excl_cumsum, mask=offs < n_tiles)
    elif pid == E:
        curr_freq_sum = 0
        curr_tile_sum = 0
        for start in tl.static_range(0, E, BLOCK_N):
            offs = start + tl.arange(0, BLOCK_N)
            expert_freq = tl.load(expert_freq_ptr + offs, mask=offs < E, other=0)

            excl_freq = tl.cumsum(expert_freq, 0) - expert_freq + curr_freq_sum
            curr_freq_sum += tl.sum(expert_freq, 0)
            tl.store(expert_freq_offs_ptr + offs, excl_freq, mask=offs < E)

            expert_m_tiles = (expert_freq + BLOCK_M_TOKEN - 1) // BLOCK_M_TOKEN
            excl_tile = tl.cumsum(expert_m_tiles, 0) - expert_m_tiles + curr_tile_sum
            curr_tile_sum += tl.sum(expert_m_tiles, 0)
            tl.store(expert_tile_offset_ptr + offs, excl_tile, mask=offs < E)

        tl.store(expert_tile_offset_ptr + E, curr_tile_sum)
    elif pid == E + 1:
        tl.store(expert_freq_offs_ptr + E, TK)


@triton.jit
def _moe_router_scatter_kernel(
    s_scatter_idx_ptr,  # (TK,) int32 --output: sorted_pos ->flat (t,k) index
    s_reverse_scatter_idx_ptr,  # (TK,) int32 --output: flat (t,k) ->sorted_pos
    x_gather_idx_ptr,  # (TK,) int32 --output: sorted_pos ->token index t
    tile_row_start_ptr,  # (num_m_tiles,) int32 --output: absolute row_start per M-tile
    tile_expert_ptr,  # (num_m_tiles,) int32 --output: expert index per M-tile
    topk_indices_ptr,  # (T, K) int32
    T,
    partial_sum_ptr,  # (E, n_tiles) int32 --tile prefix sums from K2 (read-only here)
    n_tiles,
    expert_offs_ptr,  # (E,) int32 --expert_start_idx[0:E] from K2
    expert_tile_offset_ptr,  # (E,) int32 --expert_tile_offset[0:E] from K2
    K_POW2: tl.constexpr,
    K: tl.constexpr,
    TOKENS_PER_BLOCK: tl.constexpr,
    BLOCK_M_TOKEN: tl.constexpr,
):
    # Running example: T=4, K=2, E=3, TOKENS_PER_BLOCK=2, BLOCK_M_TOKEN=2.
    # Two programs process tokens [0,1] and [2,3], respectively:
    #   token 0 -> [2, 0]       token 2 -> [0, 1]
    #   token 1 -> [1, 2]       token 3 -> [2, 1]
    # Inputs already produced by the prefix-sum kernel are:
    #   partial_sum[e, pid] = [[0,1], [0,1], [0,2]]
    #   expert_offs         = [0,2,5]
    #   expert_tile_offset  = [0,1,3].
    BLOCK_SIZE: tl.constexpr = TOKENS_PER_BLOCK * K_POW2
    IS_POW2_K: tl.constexpr = K == K_POW2
    tl.static_assert(BLOCK_SIZE <= 32768)

    # Step 1 -- load one program's routes. In the example K_POW2=K=2:
    #   pid 0: offs_local=[0,1,2,3], expert=[2,0,1,2]
    #   pid 1: offs_local=[0,1,2,3], expert=[0,1,2,1].
    pid_m = tl.program_id(0)
    offs_local = tl.arange(0, BLOCK_SIZE)
    offs_global = pid_m * BLOCK_SIZE + offs_local
    mask = offs_global < T * K_POW2

    if IS_POW2_K:
        expert = tl.load(topk_indices_ptr + offs_global, mask=mask, other=-1).to(tl.uint32)
    else:
        token_i_local = offs_local // K_POW2
        k_slot = offs_local % K_POW2
        token_i_global = pid_m * TOKENS_PER_BLOCK + token_i_local
        load_mask = mask & (k_slot < K)
        safe_k = tl.minimum(k_slot, K - 1)
        expert = tl.load(
            topk_indices_ptr + token_i_global * K + safe_k,
            mask=load_mask,
            other=-1,
        ).to(tl.uint32)

    # Step 2 -- pack (expert, offs_local) and sort by expert:
    #   pid 0: [(2,0),(0,1),(1,2),(2,3)] ->[(0,1),(1,2),(2,0),(2,3)]
    #   pid 1: [(0,0),(1,1),(2,2),(1,3)] ->[(0,0),(1,1),(1,3),(2,2)].
    kv_pairs = tl.sort(((expert << 16) | offs_local).to(tl.uint32), 0)
    expert = kv_pairs >> 16
    mask = expert != 0xFFFF  # mask out padding entries introduced by K_POW2 rounding
    expert_i32 = expert.to(tl.int32)

    # Step 3 -- count each route's rank inside its local expert run:
    #   pid 0 sorted experts [0,1,2,2] ->rank [0,0,0,1]
    #   pid 1 sorted experts [0,1,1,2] ->rank [0,0,1,0].
    scan_input = (kv_pairs & 0xFFFF0000) | 0x00000001
    inclusive_run_lengths = tl.associative_scan(scan_input, 0, _keyed_add)
    within_expert_rank = ((inclusive_run_lengths - 1) & 0xFFFF).to(tl.int32)

    # Step 4 -- add routes of the same expert from earlier programs:
    #   reference : partial_sum[e, pid] = [[0,1], [0,1], [0,2]]
    #   pid 0: earlier=[0,0,0,0] + rank=[0,0,0,1] ->within=[0,0,0,1]
    #   pid 1: earlier=[1,1,1,2] + rank=[0,0,1,0] ->within=[1,1,2,2].
    within_expert = tl.load(partial_sum_ptr + pid_m + expert_i32 * n_tiles, mask=mask, other=0) + within_expert_rank
    expert_start = tl.load(expert_offs_ptr + expert_i32, mask=mask, other=0)
    # Step 5 -- add expert_offs to obtain the globally unique sorted position:
    #   pid 0: starts=[0,2,5,5] + within=[0,0,0,1] ->s_reverse=[0,2,5,6]
    #   pid 1: starts=[0,2,2,5] + within=[1,1,2,2] ->s_reverse=[1,3,4,7].
    s_reverse = expert_start + within_expert

    # Step 6 -- start one GEMM tile at within_expert=0,2,4,...:
    #   pid 0 writes (row, expert)=(0,0),(2,1),(5,2)
    #   pid 1 writes (row, expert)=(4,1),(7,2)
    #   pid for n_pid-th tile of each expert
    # giving tile_row_start=[0,2,4,5,7], tile_expert=[0,1,1,2,2].
    is_tile_start = (within_expert % BLOCK_M_TOKEN) == 0
    t_within = within_expert // BLOCK_M_TOKEN
    tile_base = tl.load(
        expert_tile_offset_ptr + expert_i32,
        mask=mask & is_tile_start,
        other=0,
    ).to(tl.int32)
    flat_tile_idx = tile_base + t_within
    tl.store(tile_row_start_ptr + flat_tile_idx, s_reverse.to(tl.int32), mask=mask & is_tile_start)
    tl.store(tile_expert_ptr + flat_tile_idx, expert.to(tl.int32), mask=mask & is_tile_start)

    # Step 7 -- recover entry_idx and write the final mappings. Combining pids:
    #   sorted_pos:   0 1 | 2 3 4 | 5 6 7
    #   expert:       0 0 | 1 1 1 | 2 2 2
    #   entry_idx:    1 4 | 2 5 7 | 0 3 6 = s_scatter_idx
    #   token:        0 2 | 1 2 3 | 0 1 3 = x_gather_idx
    #   entry_idx ->sorted_pos gives s_reverse_scatter_idx=[5,0,2,6,1,3,7,4].
    if IS_POW2_K:
        presort_offs = (kv_pairs & 0xFFFF).to(tl.int32)
        entry_idx = pid_m * BLOCK_SIZE + presort_offs  # flat (t, k) index in [0, TK)
        tl.store(s_reverse_scatter_idx_ptr + entry_idx, s_reverse, mask=mask)
        tl.store(s_scatter_idx_ptr + s_reverse, entry_idx, mask=mask)
        tl.store(x_gather_idx_ptr + s_reverse, entry_idx // K_POW2, mask=mask)
    else:
        presort_offs = (kv_pairs & 0xFFFF).to(tl.int32)
        token_i_global_s = pid_m * TOKENS_PER_BLOCK + presort_offs // K_POW2
        entry_idx = token_i_global_s * K + presort_offs % K_POW2
        tl.store(s_reverse_scatter_idx_ptr + entry_idx, s_reverse, mask=mask)
        tl.store(s_scatter_idx_ptr + s_reverse, entry_idx, mask=mask)
        tl.store(x_gather_idx_ptr + s_reverse, token_i_global_s, mask=mask)

def _get_gemm_autotune_configs():
    if _AUTOTUNE_DISABLED:
        return [triton.Config({"BLOCK_N": 128, "BLOCK_K": 64}, num_warps=8, num_stages=2)]
    configs = []
    for bn in [64, 128]:
        for bk in [32, 64]:
            for nw in [4, 8]:
                for ns in [2, 3, 4, 5]:
                    configs.append(
                        triton.Config(
                            {"BLOCK_N": bn, "BLOCK_K": bk},
                            num_warps=nw,
                            num_stages=ns,
                        )
                    )
    return configs

def _get_dW_autotune_configs():
    if _AUTOTUNE_DISABLED:
        return [triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=8, num_stages=1)]
    return [
        triton.Config({"BLOCK_M": bm, "BLOCK_N": bn, "BLOCK_K": bk}, num_warps=nw, num_stages=1)
        for bm in [64, 128]
        for bn in [64, 128]
        for bk in [16, 32]
        for nw in [4, 8]
    ]

@triton.autotune(
    configs=_get_gemm_autotune_configs(),
    key=["H_dim", "I_dim"],
)
@triton.jit
def _fused_up_proj_swiglu_kernel(
    x_ptr,  # (T, H)
    gate_up_proj_ptr,  # (E, 2*I, H)
    x_gather_idx_ptr,  # (TK,) int32
    expert_start_ptr,  # (E+1,) int32
    tile_row_start_ptr,  # (num_m_tiles,) int32 --row_start per M-tile
    tile_expert_ptr,  # (num_m_tiles,) int32 --expert index per M-tile
    tile_count_ptr,
    pre_act_ptr,  # (TK, 2*I)  pre-SwiGLU activations [saved for backward]
    post_act_ptr,  # (TK, I)    post-SwiGLU activations
    H_dim: tl.constexpr,
    I_dim: tl.constexpr,
    stride_x_T,
    stride_x_H: tl.constexpr,
    stride_w_E,
    stride_w_N,
    stride_w_K: tl.constexpr,
    stride_pre_TK,
    stride_pre_N: tl.constexpr,
    stride_post_TK,
    stride_post_N: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # Example for sorted row r routed to expert e with original token t:
    #   gate[r] = x[t] @ W_gate[e]^T
    #   up[r]   = x[t] @ W_up[e]^T
    #   post_act[r] = silu(gate[r]) * up[r].
    # pre_act stores [gate, up] for backward; rows stay grouped by expert.
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    active_tile = pid_m < tl.load(tile_count_ptr)
    row_start = tl.load(tile_row_start_ptr + pid_m, mask=active_tile, other=0)
    expert_idx = tl.load(
        tile_expert_ptr + pid_m, mask=active_tile, other=0
    ).to(tl.int64)
    n_start = pid_n * BLOCK_N
    expert_end = tl.load(expert_start_ptr + expert_idx + 1)

    m_offs = tl.arange(0, BLOCK_M)
    n_offs = tl.arange(0, BLOCK_N)
    k_offs = tl.arange(0, BLOCK_K)

    row_offs = (row_start + m_offs).to(tl.int64)
    row_mask = active_tile & (row_offs < expert_end)

    acc_gate = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc_up = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    n_idx = n_start + n_offs
    n_mask = n_idx < I_dim
    token_idx = tl.load(x_gather_idx_ptr + row_offs, mask=row_mask, other=0).to(tl.int64)
    for k in tl.range(0, H_dim, BLOCK_K):
        k_idx = k + k_offs
        k_mask = k_idx < H_dim

        x_ptrs = x_ptr + token_idx[:, None] * stride_x_T + k_idx[None, :] * stride_x_H
        x_tile = tl.load(
            x_ptrs,
            mask=row_mask[:, None] & k_mask[None, :],
            other=0.0,
            eviction_policy="evict_first",  # token rows not reused; free L2 for weights
        )

        w_mask = n_mask[:, None] & k_mask[None, :]
        w_gate_ptrs = (
            gate_up_proj_ptr + expert_idx * stride_w_E + n_idx[:, None] * stride_w_N + k_idx[None, :] * stride_w_K
        )
        w_gate = tl.load(
            w_gate_ptrs,
            mask=w_mask,
            other=0.0,
        ).to(x_ptr.dtype.element_ty)
        acc_gate = tl.dot(x_tile, tl.trans(w_gate), acc=acc_gate)

        w_up_ptrs = w_gate_ptrs + I_dim * stride_w_N
        w_up = tl.load(
            w_up_ptrs,
            mask=w_mask,
            other=0.0,
        ).to(x_ptr.dtype.element_ty)

        acc_up = tl.dot(x_tile, tl.trans(w_up), acc=acc_up)

    out_mask = row_mask[:, None] & n_mask[None, :]

    pre_gate_ptrs = pre_act_ptr + row_offs[:, None] * stride_pre_TK + n_idx[None, :] * stride_pre_N
    pre_up_ptrs = pre_gate_ptrs + I_dim * stride_pre_N
    tl.store(pre_gate_ptrs, acc_gate.to(pre_act_ptr.dtype.element_ty), mask=out_mask)
    tl.store(pre_up_ptrs, acc_up.to(pre_act_ptr.dtype.element_ty), mask=out_mask)

    sig_gate = tl.sigmoid(acc_gate)
    silu_gate = acc_gate * sig_gate
    a_out = silu_gate * acc_up

    post_ptrs = post_act_ptr + row_offs[:, None] * stride_post_TK + n_idx[None, :] * stride_post_N
    tl.store(post_ptrs, a_out.to(post_act_ptr.dtype.element_ty), mask=out_mask)

@triton.autotune(
    configs=_get_gemm_autotune_configs(),
    key=["H_dim", "I_dim"],
)
@triton.jit
def _fused_down_proj_kernel(
    post_act_ptr,  # (TK, I)
    down_proj_ptr,  # (E, H, I)
    expert_start_ptr,  # (E+1,) int32
    tile_row_start_ptr,  # (num_m_tiles,) int32
    tile_expert_ptr,  # (num_m_tiles,) int32
    tile_count_ptr,
    Y_ptr,  # (TK, H)
    H_dim: tl.constexpr,
    I_dim: tl.constexpr,
    stride_post_TK,
    stride_post_I: tl.constexpr,
    stride_w_E,
    stride_w_H,
    stride_w_I: tl.constexpr,
    stride_Y_TK,
    stride_Y_H: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # Example for a sorted row r belonging to expert e:
    #   Y[r] = post_act[r] @ W_down[e]^T, with shapes (I) @ (I, H) -> (H).
    # If expert e owns sorted rows [4, 7), only those three rows use W_down[e].
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    active_tile = pid_m < tl.load(tile_count_ptr)
    row_start = tl.load(tile_row_start_ptr + pid_m, mask=active_tile, other=0)
    expert_idx = tl.load(
        tile_expert_ptr + pid_m, mask=active_tile, other=0
    ).to(tl.int64)
    n_start = pid_n * BLOCK_N
    expert_end = tl.load(expert_start_ptr + expert_idx + 1)

    m_offs = tl.arange(0, BLOCK_M)
    n_offs = tl.arange(0, BLOCK_N)
    k_offs = tl.arange(0, BLOCK_K)

    row_offs = (row_start + m_offs).to(tl.int64)
    row_mask = active_tile & (row_offs < expert_end)
    n_idx = n_start + n_offs
    n_mask = n_idx < H_dim

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in tl.range(0, I_dim, BLOCK_K):
        k_idx = k + k_offs
        k_mask = k_idx < I_dim

        a_ptrs = post_act_ptr + row_offs[:, None] * stride_post_TK + k_idx[None, :] * stride_post_I
        a_tile = tl.load(a_ptrs, mask=row_mask[:, None] & k_mask[None, :], other=0.0)

        w_ptrs = down_proj_ptr + expert_idx * stride_w_E + n_idx[:, None] * stride_w_H + k_idx[None, :] * stride_w_I
        w_tile = tl.load(
            w_ptrs,
            mask=n_mask[:, None] & k_mask[None, :],
            other=0.0,
        ).to(post_act_ptr.dtype.element_ty)

        acc = tl.dot(a_tile, tl.trans(w_tile), acc=acc)

    Y_ptrs = Y_ptr + row_offs[:, None] * stride_Y_TK + n_idx[None, :] * stride_Y_H
    tl.store(Y_ptrs, acc.to(Y_ptr.dtype.element_ty), mask=row_mask[:, None] & n_mask[None, :])


def _get_token_gather_autotune_configs():
    if _AUTOTUNE_DISABLED:
        return [triton.Config({"BLOCK_H": 128, "BLOCK_K": 4}, num_warps=4, num_stages=4)]
    configs = []
    for bh in [64, 128, 256, 512]:
        for bk in [1, 2, 4, 8, 16]:
            for nw in [4, 8]:
                if bk * bh <= 32768:
                    configs.append(triton.Config({"BLOCK_H": bh, "BLOCK_K": bk}, num_warps=nw, num_stages=4))
    return configs

@triton.autotune(
    configs=_get_token_gather_autotune_configs(),
    key=["H_dim", "K_dim", "w_is_None"],
)
@triton.jit
def _token_gather_weighted_sum_kernel(
    Y_ptr,  # (TK, H)
    w_ptr,  # (TK,) routing weights, or None when w_is_None=True
    s_rev_ptr,  # (TK,) int32 s_reverse_scatter_idx: flat(t,k) ->sorted position
    out_ptr,  # (T, H)
    H_dim: tl.constexpr,
    K_dim: tl.constexpr,
    stride_Y_TK,
    stride_Y_H: tl.constexpr,
    stride_out_T,
    stride_out_H: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_K: tl.constexpr,
    w_is_None: tl.constexpr,  # True ->unweighted gather-sum (used for dx backward)
):
    # Example for K=2: if token t's flat routes map through s_rev to rows [5, 1],
    #   out[t] = w[t,0] * Y[5] + w[t,1] * Y[1].
    # With w_is_None=True, the same kernel computes out[t] = Y[5] + Y[1].
    t = tl.program_id(0).to(tl.int64)

    for h_tile in tl.static_range(triton.cdiv(H_dim, BLOCK_H)):
        h_idx = (h_tile * BLOCK_H + tl.arange(0, BLOCK_H)).to(tl.uint32)
        h_mask = h_idx < H_dim
        acc = tl.zeros([BLOCK_H], dtype=tl.float32)

        for k_tile in tl.range(triton.cdiv(K_dim, BLOCK_K)):
            k_offs = (k_tile * BLOCK_K + tl.arange(0, BLOCK_K)).to(tl.uint32)
            k_mask = k_offs < K_dim

            flat_idx = t * K_dim + k_offs
            perm_idx = tl.load(s_rev_ptr + flat_idx, mask=k_mask, other=0).to(tl.int64)

            y_ptrs = Y_ptr + perm_idx[:, None] * stride_Y_TK + h_idx[None, :] * stride_Y_H
            y_vals = tl.load(y_ptrs, mask=k_mask[:, None] & h_mask[None, :], other=0.0).to(tl.float32)

            if w_is_None:
                acc += tl.sum(y_vals, axis=0)
            else:
                w_vals = tl.load(w_ptr + flat_idx, mask=k_mask, other=0.0).to(tl.float32)
                acc += tl.sum(y_vals * w_vals[:, None], axis=0)

        out_ptrs = out_ptr + t * stride_out_T + h_idx * stride_out_H
        tl.store(out_ptrs, acc.to(out_ptr.dtype.element_ty), mask=h_mask)

@triton.autotune(
    configs=_get_gemm_autotune_configs(),
    key=["H_dim", "I_dim"],
    reset_to_zero=["dS_ptr"],  # autotune runs multiple configs; atomic_add accumulates, so reset between runs
)
@triton.jit
def _moe_bwd_down_proj_kernel(
    dO_ptr,  # (T, H)   --/, upstream gradient
    x_gather_idx_ptr,  # (TK,)    --_x: sorted_pos ->original token index
    s_scatter_idx_ptr,  # (TK,)    --_s: sorted_pos ->flat (t,k) index
    topk_weights_ptr,  # (TK,)    --s_k: routing weights in flat (t,k) order
    down_proj_ptr,  # (E, H, I) --W2
    pre_act_ptr,  # (TK, 2I) --z = [gate, up] saved from forward
    expert_start_ptr,  # (E+1,)   int32
    tile_row_start_ptr,  # (num_m_tiles,) int32
    tile_expert_ptr,  # (num_m_tiles,) int32
    tile_count_ptr,
    d_pre_act_ptr,  # (TK, 2I) --output: / = [dgate, dup]
    weighted_act_ptr,  # (TK, I)  --output: s_k * y1 (for dW2 kernel)
    dS_ptr,  # (TK,)    --output: /_k, indexed by flat (t,k)
    H_dim: tl.constexpr,
    I_dim: tl.constexpr,
    stride_dO_T,
    stride_dO_H: tl.constexpr,
    stride_w_E,
    stride_w_H,
    stride_w_I: tl.constexpr,
    stride_pre_TK,
    stride_pre_N: tl.constexpr,
    stride_d_pre_TK,
    stride_d_pre_N: tl.constexpr,
    stride_wact_TK,
    stride_wact_I: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # Example for sorted route r=(token t ->expert e) with routing weight s:
    #   dy1      = s * (dO[t] @ W2[e])
    #   dS[t,k]  = dot(dO[t], W2[e] @ y1[r])
    #   weighted_act[r] = s * y1[r].
    # dy1 is then differentiated through y1=silu(gate)*up into [dgate, dup].
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    active_tile = pid_m < tl.load(tile_count_ptr)
    row_start = tl.load(tile_row_start_ptr + pid_m, mask=active_tile, other=0)
    expert_idx = tl.load(
        tile_expert_ptr + pid_m, mask=active_tile, other=0
    ).to(tl.int64)
    n_start = pid_n * BLOCK_N
    expert_end = tl.load(expert_start_ptr + expert_idx + 1)

    m_offs = tl.arange(0, BLOCK_M)
    n_offs = tl.arange(0, BLOCK_N)
    k_offs = tl.arange(0, BLOCK_K)

    row_offs = (row_start + m_offs).to(tl.int64)
    row_mask = active_tile & (row_offs < expert_end)
    n_idx = n_start + n_offs
    n_mask = n_idx < I_dim
    out_mask = row_mask[:, None] & n_mask[None, :]

    token_idx = tl.load(x_gather_idx_ptr + row_offs, mask=row_mask, other=0).to(tl.int64)
    flat_tk_idx = tl.load(s_scatter_idx_ptr + row_offs, mask=row_mask, other=0)
    weights = tl.load(topk_weights_ptr + flat_tk_idx, mask=row_mask, other=0.0).to(tl.float32)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in tl.range(0, H_dim, BLOCK_K):
        k_idx = k + k_offs
        k_mask = k_idx < H_dim

        dO_ptrs = dO_ptr + token_idx[:, None] * stride_dO_T + k_idx[None, :] * stride_dO_H
        dO_tile = tl.load(dO_ptrs, mask=row_mask[:, None] & k_mask[None, :], other=0.0)

        w_ptrs = down_proj_ptr + expert_idx * stride_w_E + k_idx[:, None] * stride_w_H + n_idx[None, :] * stride_w_I
        w_tile = tl.load(w_ptrs, mask=k_mask[:, None] & n_mask[None, :], other=0.0).to(dO_ptr.dtype.element_ty)
        acc = tl.dot(dO_tile, w_tile, acc=acc)

    gate_ptrs = pre_act_ptr + row_offs[:, None] * stride_pre_TK + n_idx[None, :] * stride_pre_N
    up_ptrs = gate_ptrs + I_dim * stride_pre_N
    gate = tl.load(gate_ptrs, mask=out_mask, other=0.0).to(tl.float32)
    up = tl.load(up_ptrs, mask=out_mask, other=0.0).to(tl.float32)
    sig_gate = tl.sigmoid(gate)
    silu_gate = gate * sig_gate
    y1 = silu_gate * up  # (BLOCK_M, BLOCK_N)

    wact_ptrs = weighted_act_ptr + row_offs[:, None] * stride_wact_TK + n_idx[None, :] * stride_wact_I
    tl.store(wact_ptrs, (weights[:, None] * y1).to(weighted_act_ptr.dtype.element_ty), mask=out_mask)

    dS_partial = tl.sum(acc * y1, axis=1)
    tl.atomic_add(dS_ptr + flat_tk_idx, dS_partial, mask=row_mask)

    acc = acc * weights[:, None]

    dgate = acc * (silu_gate * (1.0 - sig_gate) + sig_gate) * up
    dup = acc * silu_gate
    dgate_ptrs = d_pre_act_ptr + row_offs[:, None] * stride_d_pre_TK + n_idx[None, :] * stride_d_pre_N
    dup_ptrs = dgate_ptrs + I_dim * stride_d_pre_N
    tl.store(dgate_ptrs, dgate.to(d_pre_act_ptr.dtype.element_ty), mask=out_mask)
    tl.store(dup_ptrs, dup.to(d_pre_act_ptr.dtype.element_ty), mask=out_mask)

@triton.autotune(
    configs=_get_dW_autotune_configs(),
    key=["H_dim", "I_dim"],
    reset_to_zero=["dW2_ptr"],
)
@triton.jit
def _moe_bwd_dW2_kernel(
    weighted_act_ptr,  # (TK, I) --s_k * y1 from backward down-proj kernel
    dout_ptr,  # (T, H)  --upstream gradient (gathered by x_gather_idx)
    x_gather_idx_ptr,  # (TK,)   --sorted_pos ->original token index
    expert_start_ptr,  # (E+1,)  int32
    dW2_ptr,  # (E, H, I) --output
    H_dim: tl.constexpr,
    I_dim: tl.constexpr,
    stride_wact_TK,
    stride_wact_I: tl.constexpr,
    stride_dout_T,
    stride_dout_H: tl.constexpr,
    stride_dW2_E,
    stride_dW2_H,
    stride_dW2_I: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # Example: if expert e owns sorted rows r in [start_e, end_e), then
    #   dW2[e,h,i] = sum_r dO[token_of(r),h] * (s_r * y1[r,i]).
    # The K loop reduces over all routes assigned to this expert.
    pid0 = tl.program_id(0)
    pid1 = tl.program_id(1)

    N_M_TILES: tl.constexpr = (I_dim + BLOCK_M - 1) // BLOCK_M
    expert_idx = (pid0 // N_M_TILES).to(tl.int64)
    m_tile = pid0 % N_M_TILES

    expert_start = tl.load(expert_start_ptr + expert_idx)
    expert_end = tl.load(expert_start_ptr + expert_idx + 1)
    M_e = expert_end - expert_start
    if M_e == 0:
        return

    m_start = m_tile * BLOCK_M
    n_start = pid1 * BLOCK_N

    m_offs = tl.arange(0, BLOCK_M)
    n_offs = tl.arange(0, BLOCK_N)
    k_offs = tl.arange(0, BLOCK_K)

    i_idx = m_start + m_offs
    h_idx = n_start + n_offs
    i_mask = i_idx < I_dim
    h_mask = h_idx < H_dim

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in tl.range(0, M_e, BLOCK_K):
        k_idx = k + k_offs
        k_mask = k_idx < M_e
        row_offs = (expert_start + k_idx).to(tl.int64)

        wact_ptrs = weighted_act_ptr + row_offs[None, :] * stride_wact_TK + i_idx[:, None] * stride_wact_I
        wact_tile = tl.load(wact_ptrs, mask=k_mask[None, :] & i_mask[:, None], other=0.0)

        token_idx = tl.load(x_gather_idx_ptr + row_offs, mask=k_mask, other=0).to(tl.int64)
        dout_ptrs = dout_ptr + token_idx[:, None] * stride_dout_T + h_idx[None, :] * stride_dout_H
        dout_tile = tl.load(dout_ptrs, mask=k_mask[:, None] & h_mask[None, :], other=0.0)

        acc = tl.dot(wact_tile, dout_tile, acc=acc)

    dW2_ptrs = dW2_ptr + expert_idx * stride_dW2_E + h_idx[None, :] * stride_dW2_H + i_idx[:, None] * stride_dW2_I
    tl.store(dW2_ptrs, acc.to(dW2_ptr.dtype.element_ty), mask=i_mask[:, None] & h_mask[None, :])

@triton.autotune(
    configs=_get_gemm_autotune_configs(),
    key=["H_dim", "I_dim"],
)
@triton.jit
def _moe_bwd_dX_expanded_kernel(
    d_pre_act_ptr,  # (TK, 2*I)
    gate_up_proj_ptr,  # (E, 2*I, H) --W1
    expert_start_ptr,  # (E+1,) int32
    tile_row_start_ptr,  # (num_m_tiles,) int32
    tile_expert_ptr,  # (num_m_tiles,) int32
    tile_count_ptr,
    dx_expanded_ptr,  # (TK, H) --output: clean write, indexed by sorted_pos
    H_dim: tl.constexpr,
    I_dim: tl.constexpr,
    stride_d_pre_TK,
    stride_d_pre_N: tl.constexpr,
    stride_w_E,
    stride_w_N,
    stride_w_K: tl.constexpr,
    stride_dxe_TK,
    stride_dxe_H: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # Example for sorted route r assigned to expert e:
    #   dx_expanded[r] = dgate[r] @ W_gate[e] + dup[r] @ W_up[e].
    # A later unweighted gather-sum combines the K route gradients per token.
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    active_tile = pid_m < tl.load(tile_count_ptr)
    row_start = tl.load(tile_row_start_ptr + pid_m, mask=active_tile, other=0)
    expert_idx = tl.load(
        tile_expert_ptr + pid_m, mask=active_tile, other=0
    ).to(tl.int64)
    n_start = pid_n * BLOCK_N
    expert_end = tl.load(expert_start_ptr + expert_idx + 1)

    m_offs = tl.arange(0, BLOCK_M)
    n_offs = tl.arange(0, BLOCK_N)
    k_offs = tl.arange(0, BLOCK_K)

    row_offs = (row_start + m_offs).to(tl.int64)
    row_mask = active_tile & (row_offs < expert_end)
    h_idx = n_start + n_offs
    h_mask = h_idx < H_dim

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in tl.range(0, I_dim, BLOCK_K):
        k_idx = k + k_offs
        k_mask = k_idx < I_dim

        d_gate_ptrs = d_pre_act_ptr + row_offs[:, None] * stride_d_pre_TK + k_idx[None, :] * stride_d_pre_N
        d_gate = tl.load(d_gate_ptrs, mask=row_mask[:, None] & k_mask[None, :], other=0.0)

        w_gate_ptrs = (
            gate_up_proj_ptr + expert_idx * stride_w_E + k_idx[:, None] * stride_w_N + h_idx[None, :] * stride_w_K
        )
        w_gate = tl.load(w_gate_ptrs, mask=k_mask[:, None] & h_mask[None, :], other=0.0).to(
            d_pre_act_ptr.dtype.element_ty
        )
        acc = tl.dot(d_gate, w_gate, acc=acc)

        d_up_ptrs = d_pre_act_ptr + row_offs[:, None] * stride_d_pre_TK + (I_dim + k_idx)[None, :] * stride_d_pre_N
        d_up = tl.load(d_up_ptrs, mask=row_mask[:, None] & k_mask[None, :], other=0.0)

        w_up_ptrs = (
            gate_up_proj_ptr
            + expert_idx * stride_w_E
            + (I_dim + k_idx)[:, None] * stride_w_N
            + h_idx[None, :] * stride_w_K
        )
        w_up = tl.load(w_up_ptrs, mask=k_mask[:, None] & h_mask[None, :], other=0.0).to(
            d_pre_act_ptr.dtype.element_ty
        )

        acc = tl.dot(d_up, w_up, acc=acc)

    dxe_ptrs = dx_expanded_ptr + row_offs[:, None] * stride_dxe_TK + h_idx[None, :] * stride_dxe_H
    tl.store(dxe_ptrs, acc.to(dx_expanded_ptr.dtype.element_ty), mask=row_mask[:, None] & h_mask[None, :])

@triton.autotune(
    configs=_get_dW_autotune_configs(),
    key=["H_dim", "I_dim"],
    reset_to_zero=["dW1_ptr"],
)
@triton.jit
def _moe_bwd_dW1_kernel(
    x_ptr,  # (T, H)
    d_pre_act_ptr,  # (TK, 2*I)
    x_gather_idx_ptr,  # (TK,) int32
    expert_start_ptr,  # (E+1,) int32
    dW1_ptr,  # (E, 2*I, H) --output
    H_dim: tl.constexpr,
    I_dim: tl.constexpr,
    stride_x_T,
    stride_x_H: tl.constexpr,
    stride_d_pre_TK,
    stride_d_pre_N: tl.constexpr,
    stride_dW1_E,
    stride_dW1_N,
    stride_dW1_H: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # Example: for every sorted route r owned by expert e, with token t_r,
    #   dW1[e,n,h] = sum_r d_pre_act[r,n] * x[t_r,h].
    # n spans both gate and up halves, so dW1 has shape (E, 2*I, H).
    pid0 = tl.program_id(0)
    pid1 = tl.program_id(1)

    N_M_TILES: tl.constexpr = (H_dim + BLOCK_M - 1) // BLOCK_M
    expert_idx = (pid0 // N_M_TILES).to(tl.int64)
    m_tile = pid0 % N_M_TILES

    expert_start = tl.load(expert_start_ptr + expert_idx)
    expert_end = tl.load(expert_start_ptr + expert_idx + 1)
    M_e = expert_end - expert_start
    if M_e == 0:
        return

    m_start = m_tile * BLOCK_M
    n_start = pid1 * BLOCK_N

    m_offs = tl.arange(0, BLOCK_M)
    n_offs = tl.arange(0, BLOCK_N)
    k_offs = tl.arange(0, BLOCK_K)

    h_idx = m_start + m_offs
    n_idx = n_start + n_offs
    h_mask = h_idx < H_dim
    n_mask = n_idx < 2 * I_dim

    acc = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float32)

    for k in tl.range(0, M_e, BLOCK_K):
        k_idx = k + k_offs
        k_mask = k_idx < M_e
        row_offs = (expert_start + k_idx).to(tl.int64)

        token_idx = tl.load(x_gather_idx_ptr + row_offs, mask=k_mask, other=0).to(tl.int64)
        x_ptrs = x_ptr + token_idx[:, None] * stride_x_T + h_idx[None, :] * stride_x_H
        x_tile = tl.load(x_ptrs, mask=k_mask[:, None] & h_mask[None, :], other=0.0)

        d_pre_ptrs = d_pre_act_ptr + row_offs[:, None] * stride_d_pre_TK + n_idx[None, :] * stride_d_pre_N
        d_pre_tile = tl.load(d_pre_ptrs, mask=k_mask[:, None] & n_mask[None, :], other=0.0)

        acc = tl.dot(tl.trans(d_pre_tile), x_tile, acc=acc)

    dW1_ptrs = dW1_ptr + expert_idx * stride_dW1_E + n_idx[:, None] * stride_dW1_N + h_idx[None, :] * stride_dW1_H
    tl.store(dW1_ptrs, acc.to(dW1_ptr.dtype.element_ty), mask=n_mask[:, None] & h_mask[None, :])
