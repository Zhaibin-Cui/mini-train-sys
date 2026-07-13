/******************************************************************************
 * Copyright (c) 2024, Tri Dao.
 * Copyright (c) 2026, mini-train-sys contributors.
 *
 * This file is a deliberately small adaptation of FlashAttention 2.8.4's
 * csrc/flash_attn/flash_api.cpp. The CUDA kernels and launch templates remain
 * upstream code under csrc/third_party; this layer only maps MiniTrain's
 * (batch, heads, sequence, head_dim) tensors to upstream parameter structs.
 ******************************************************************************/

#include <ATen/cuda/CUDAGeneratorImpl.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>

#include <cmath>
#include <cstdint>
#include <mutex>
#include <optional>
#include <vector>

#include <cutlass/numeric_types.h>

#include "flash.h"
#include "hardware_info.h"

#define CHECK_TENSOR_CUDA(x) TORCH_CHECK((x).is_cuda(), #x " must be a CUDA tensor")
#define CHECK_LAST_CONTIGUOUS(x) \
    TORCH_CHECK((x).stride(-1) == 1, #x " must have contiguous head_dim")

namespace minitrain_flash {

namespace {

// Round dimensions exactly as upstream does. The selected head-dimension
// specialization handles values below its bucket with masked loads/stores.
int round_up(int value, int multiple) {
    return (value + multiple - 1) / multiple * multiple;
}

// Validate the deliberately narrow MiniTrain contract at the C++ boundary.
// Python performs the same check before dispatch; this protects direct pybind
// calls and turns unsupported configurations into useful errors.
void check_qkv(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v) {
    CHECK_TENSOR_CUDA(q);
    CHECK_TENSOR_CUDA(k);
    CHECK_TENSOR_CUDA(v);
    CHECK_LAST_CONTIGUOUS(q);
    CHECK_LAST_CONTIGUOUS(k);
    CHECK_LAST_CONTIGUOUS(v);

    TORCH_CHECK(q.dim() == 4, "q must have shape (batch, heads, sequence, head_dim)");
    TORCH_CHECK(k.sizes() == q.sizes(), "k must have the same shape as q");
    TORCH_CHECK(v.sizes() == q.sizes(), "v must have the same shape as q");
    TORCH_CHECK(k.device() == q.device(), "q and k must be on the same CUDA device");
    TORCH_CHECK(v.device() == q.device(), "q and v must be on the same CUDA device");
    TORCH_CHECK(k.scalar_type() == q.scalar_type(), "q and k must have the same dtype");
    TORCH_CHECK(v.scalar_type() == q.scalar_type(), "q and v must have the same dtype");
    TORCH_CHECK(
        q.scalar_type() == at::kHalf || q.scalar_type() == at::kBFloat16,
        "upstream CUDA FlashAttention supports fp16 and bf16 inputs");
    TORCH_CHECK(q.size(0) > 0 && q.size(1) > 0 && q.size(2) > 0, "B, H, and S must be positive");
    TORCH_CHECK(q.size(3) > 0 && q.size(3) <= 256, "head_dim must be in [1, 256]");
    TORCH_CHECK(q.size(3) % 8 == 0, "head_dim must be a multiple of 8");

    const auto [major, minor] = get_compute_capability(q.get_device());
    TORCH_CHECK(major >= 8, "upstream CUDA FlashAttention requires sm80 or newer");
}

// Guard an upstream hardware limitation that is easy to miss in the template
// launcher.  For head dimensions above 192, backward dispatches to the D=256
// family.  Its dropout specialization requires at least 144 KiB of opt-in
// shared memory per block.  On sm86/sm89 the upstream low-shared-memory branch
// deliberately launches only the no-dropout specialization; silently entering
// it with dropout would leave dq/dk/dv uninitialized.
void check_dropout_backward_support(
    const at::Tensor& q,
    float dropout_p) {
    if (dropout_p == 0.0f || q.size(3) <= 192) {
        return;
    }

    int max_smem_per_block = 0;
    C10_CUDA_CHECK(cudaDeviceGetAttribute(
        &max_smem_per_block,
        cudaDevAttrMaxSharedMemoryPerBlockOptin,
        q.get_device()));
    TORCH_CHECK(
        max_smem_per_block >= 144 * 1024,
        "head_dim > 192 dropout backward requires at least 144 KiB opt-in "
        "shared memory per block; use the Triton/PyTorch fallback on this GPU");
}

// Fill the common forward parameter block. Keeping this close to upstream is
// important: kernel headers consume strides in elements and interpret rows as
// sequence positions, regardless of the physical Python dimension order.
void set_forward_params(
    Flash_fwd_params& params,
    const at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& v,
    at::Tensor& out,
    at::Tensor& lse,
    at::Tensor& rng_state,
    float dropout_p,
    float softmax_scale,
    bool is_causal) {
    params = {};

    const int batch = static_cast<int>(q.size(0));
    const int heads = static_cast<int>(q.size(1));
    const int seqlen = static_cast<int>(q.size(2));
    const int head_dim = static_cast<int>(q.size(3));

    // MiniTrain layout is B,H,S,D. Upstream names the S stride "row" and the H
    // stride "head"; arbitrary outer strides are supported by the kernels.
    params.q_ptr = q.data_ptr();
    params.k_ptr = k.data_ptr();
    params.v_ptr = v.data_ptr();
    params.o_ptr = out.data_ptr();
    params.q_batch_stride = q.stride(0);
    params.k_batch_stride = k.stride(0);
    params.v_batch_stride = v.stride(0);
    params.o_batch_stride = out.stride(0);
    params.q_head_stride = q.stride(1);
    params.k_head_stride = k.stride(1);
    params.v_head_stride = v.stride(1);
    params.o_head_stride = out.stride(1);
    params.q_row_stride = q.stride(2);
    params.k_row_stride = k.stride(2);
    params.v_row_stride = v.stride(2);
    params.o_row_stride = out.stride(2);

    // This reduced build supports equal Q/K/V heads and fixed-length dense
    // batches only. All optional pointer fields stay null after zero-init.
    params.b = batch;
    params.h = heads;
    params.h_k = heads;
    params.h_h_k_ratio = 1;
    params.seqlen_q = seqlen;
    params.seqlen_k = seqlen;
    params.seqlen_q_rounded = round_up(seqlen, 128);
    params.seqlen_k_rounded = round_up(seqlen, 128);
    params.d = head_dim;
    params.d_rounded = round_up(head_dim, head_dim <= 128 ? 32 : 64);
    params.softmax_lse_ptr = lse.data_ptr();

    // Softmax uses log2 exponentiation internally, as in upstream. No softcap,
    // local attention, ALiBi, return-softmax, or split-KV state is configured.
    params.scale_softmax = softmax_scale;
    params.scale_softmax_log2 = softmax_scale * static_cast<float>(M_LOG2E);
    params.softcap = 0.0f;
    params.is_causal = is_causal;
    params.window_size_left = -1;
    params.window_size_right = is_causal ? 0 : -1;
    params.is_seqlens_k_cumulative = true;
    params.is_bf16 = q.scalar_type() == at::kBFloat16;
    params.rng_state = rng_state.numel() == 2
        ? reinterpret_cast<uint64_t*>(rng_state.data_ptr())
        : nullptr;

    // Upstream stores the keep probability in params.p_dropout. The compile-
    // time no-dropout specialization removes these values and all RNG work.
    const float keep_p = 1.0f - dropout_p;
    params.p_dropout = keep_p;
    params.p_dropout_in_uint8_t = static_cast<uint8_t>(std::floor(keep_p * 255.0f));
    params.rp_dropout = 1.0f / keep_p;
    params.scale_softmax_rp_dropout = params.rp_dropout * softmax_scale;
}

// Dispatch only to instantiations included by build.py. A local workstation can
// compile one bucket while a server/wheel build enables the full matrix.
template <typename scalar_t>
void dispatch_forward_dtype(Flash_fwd_params& params, cudaStream_t stream) {
    const bool causal = params.is_causal;
#ifdef MINITRAIN_FLASH_HDIM_32
    if (params.d <= 32) {
        causal ? run_mha_fwd_<scalar_t, 32, true>(params, stream)
               : run_mha_fwd_<scalar_t, 32, false>(params, stream);
        return;
    }
#endif
#ifdef MINITRAIN_FLASH_HDIM_64
    if (params.d <= 64) {
        causal ? run_mha_fwd_<scalar_t, 64, true>(params, stream)
               : run_mha_fwd_<scalar_t, 64, false>(params, stream);
        return;
    }
#endif
#ifdef MINITRAIN_FLASH_HDIM_96
    if (params.d <= 96) {
        causal ? run_mha_fwd_<scalar_t, 96, true>(params, stream)
               : run_mha_fwd_<scalar_t, 96, false>(params, stream);
        return;
    }
#endif
#ifdef MINITRAIN_FLASH_HDIM_128
    if (params.d <= 128) {
        causal ? run_mha_fwd_<scalar_t, 128, true>(params, stream)
               : run_mha_fwd_<scalar_t, 128, false>(params, stream);
        return;
    }
#endif
#ifdef MINITRAIN_FLASH_HDIM_192
    if (params.d <= 192) {
        causal ? run_mha_fwd_<scalar_t, 192, true>(params, stream)
               : run_mha_fwd_<scalar_t, 192, false>(params, stream);
        return;
    }
#endif
#ifdef MINITRAIN_FLASH_HDIM_256
    if (params.d <= 256) {
        causal ? run_mha_fwd_<scalar_t, 256, true>(params, stream)
               : run_mha_fwd_<scalar_t, 256, false>(params, stream);
        return;
    }
#endif
    TORCH_CHECK(false, "head_dim ", params.d, " is not included in this cuda_ext build");
}

// Backward mirrors forward dispatch so a profile can never accidentally call
// an unlinked template specialization.
template <typename scalar_t>
void dispatch_backward_dtype(Flash_bwd_params& params, cudaStream_t stream) {
    const bool causal = params.is_causal;
#ifdef MINITRAIN_FLASH_HDIM_32
    if (params.d <= 32) {
        causal ? run_mha_bwd_<scalar_t, 32, true>(params, stream)
               : run_mha_bwd_<scalar_t, 32, false>(params, stream);
        return;
    }
#endif
#ifdef MINITRAIN_FLASH_HDIM_64
    if (params.d <= 64) {
        causal ? run_mha_bwd_<scalar_t, 64, true>(params, stream)
               : run_mha_bwd_<scalar_t, 64, false>(params, stream);
        return;
    }
#endif
#ifdef MINITRAIN_FLASH_HDIM_96
    if (params.d <= 96) {
        causal ? run_mha_bwd_<scalar_t, 96, true>(params, stream)
               : run_mha_bwd_<scalar_t, 96, false>(params, stream);
        return;
    }
#endif
#ifdef MINITRAIN_FLASH_HDIM_128
    if (params.d <= 128) {
        causal ? run_mha_bwd_<scalar_t, 128, true>(params, stream)
               : run_mha_bwd_<scalar_t, 128, false>(params, stream);
        return;
    }
#endif
#ifdef MINITRAIN_FLASH_HDIM_192
    if (params.d <= 192) {
        causal ? run_mha_bwd_<scalar_t, 192, true>(params, stream)
               : run_mha_bwd_<scalar_t, 192, false>(params, stream);
        return;
    }
#endif
#ifdef MINITRAIN_FLASH_HDIM_256
    if (params.d <= 256) {
        causal ? run_mha_bwd_<scalar_t, 256, true>(params, stream)
               : run_mha_bwd_<scalar_t, 256, false>(params, stream);
        return;
    }
#endif
    TORCH_CHECK(false, "head_dim ", params.d, " is not included in this cuda_ext build");
}

void dispatch_forward(Flash_fwd_params& params, cudaStream_t stream) {
    if (params.is_bf16) {
#ifdef MINITRAIN_FLASH_ENABLE_BF16
        dispatch_forward_dtype<cutlass::bfloat16_t>(params, stream);
        return;
#endif
    } else {
#ifdef MINITRAIN_FLASH_ENABLE_FP16
        dispatch_forward_dtype<cutlass::half_t>(params, stream);
        return;
#endif
    }
    TORCH_CHECK(false, params.is_bf16 ? "bf16" : "fp16", " is not included in this cuda_ext build");
}

void dispatch_backward(Flash_bwd_params& params, cudaStream_t stream) {
    if (params.is_bf16) {
#ifdef MINITRAIN_FLASH_ENABLE_BF16
        dispatch_backward_dtype<cutlass::bfloat16_t>(params, stream);
        return;
#endif
    } else {
#ifdef MINITRAIN_FLASH_ENABLE_FP16
        dispatch_backward_dtype<cutlass::half_t>(params, stream);
        return;
#endif
    }
    TORCH_CHECK(false, params.is_bf16 ? "bf16" : "fp16", " is not included in this cuda_ext build");
}

}  // namespace

// Forward returns all state needed by a custom autograd function. LSE is the
// only numerical intermediate retained; attention scores and dropout masks are
// never materialized.
std::vector<at::Tensor> flash_attn_fwd(
    const at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& v,
    bool is_causal,
    double dropout_p_value,
    bool return_softmax) {
    check_qkv(q, k, v);
    TORCH_CHECK(dropout_p_value >= 0.0 && dropout_p_value < 1.0, "dropout_p must be in [0, 1)");
    at::cuda::CUDAGuard device_guard(q.device());

    const float dropout_p = static_cast<float>(dropout_p_value);
    TORCH_CHECK(
        dropout_p >= 0.0f && dropout_p < 1.0f,
        "dropout_p must remain in [0, 1) after float32 conversion");
    const float softmax_scale = 1.0f / std::sqrt(static_cast<float>(q.size(3)));
    auto out = at::empty_like(q);
    auto lse = at::empty({q.size(0), q.size(1), q.size(2)}, q.options().dtype(at::kFloat));
    auto rng_state = dropout_p > 0.0f
        ? at::empty({2}, q.options().dtype(at::kLong))
        : at::empty({0}, q.options().dtype(at::kLong));
    TORCH_CHECK(!return_softmax || dropout_p > 0.0f, "debug softmax is only available with dropout");
    const int seqlen_rounded = round_up(static_cast<int>(q.size(2)), 128);
    auto softmax = return_softmax
        ? at::empty(
              {q.size(0), q.size(1), seqlen_rounded, seqlen_rounded},
              q.options())
        : at::empty({0}, q.options());

    Flash_fwd_params params;
    set_forward_params(params, q, k, v, out, lse, rng_state, dropout_p, softmax_scale, is_causal);
    // Upstream stores debug probabilities with the dropout decision encoded in
    // the sign bit. Production calls leave this null and compile-time dispatch
    // selects Return_softmax=false.
    params.p_ptr = return_softmax ? softmax.data_ptr() : nullptr;

    // Reserve one Philox subsequence per batch/head as upstream does. Only the
    // dropout specialization reads this state and writes seed/offset to tensor.
    if (dropout_p > 0.0f) {
        auto gen = at::get_generator_or_default<at::CUDAGeneratorImpl>(
            std::nullopt,
            at::cuda::detail::getDefaultCUDAGenerator());
        std::lock_guard<std::mutex> lock(gen->mutex_);
        params.philox_args = gen->philox_cuda_state(params.b * params.h * 32);
    }

    dispatch_forward(params, at::cuda::getCurrentCUDAStream().stream());
    return {out, lse, rng_state, softmax};
}

// Backward reuses forward LSE and, for dropout, the exact Philox seed/offset.
// The upstream kernel recomputes softmax probabilities and the mask tile by
// tile, preserving FlashAttention's linear-memory property.
std::vector<at::Tensor> flash_attn_bwd(
    const at::Tensor& dout_input,
    const at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& v,
    const at::Tensor& out,
    const at::Tensor& lse,
    const at::Tensor& rng_state,
    bool is_causal,
    double dropout_p_value) {
    check_qkv(q, k, v);
    CHECK_TENSOR_CUDA(dout_input);
    TORCH_CHECK(dout_input.sizes() == q.sizes(), "dout must have the same shape as q");
    TORCH_CHECK(dout_input.device() == q.device(), "dout must be on the same CUDA device as q");
    TORCH_CHECK(dout_input.scalar_type() == q.scalar_type(), "dout must have the same dtype as q");

    // Saved tensors normally come from this extension's forward. Validate them
    // anyway so a direct pybind caller cannot pass foreign-device pointers or
    // layouts that the upstream kernels would interpret incorrectly.
    CHECK_TENSOR_CUDA(out);
    CHECK_LAST_CONTIGUOUS(out);
    TORCH_CHECK(out.device() == q.device(), "saved out must be on the same CUDA device as q");
    TORCH_CHECK(out.sizes() == q.sizes() && out.scalar_type() == q.scalar_type(), "invalid saved out tensor");
    CHECK_TENSOR_CUDA(lse);
    TORCH_CHECK(lse.device() == q.device(), "saved LSE must be on the same CUDA device as q");
    TORCH_CHECK(lse.scalar_type() == at::kFloat && lse.is_contiguous(), "saved LSE must be contiguous float32");
    TORCH_CHECK(lse.sizes() == at::IntArrayRef({q.size(0), q.size(1), q.size(2)}), "invalid saved LSE tensor");
    TORCH_CHECK(dropout_p_value >= 0.0 && dropout_p_value < 1.0, "dropout_p must be in [0, 1)");
    const float dropout_p = static_cast<float>(dropout_p_value);
    TORCH_CHECK(
        dropout_p >= 0.0f && dropout_p < 1.0f,
        "dropout_p must remain in [0, 1) after float32 conversion");
    if (dropout_p > 0.0f) {
        CHECK_TENSOR_CUDA(rng_state);
        TORCH_CHECK(rng_state.device() == q.device(), "saved RNG state must be on the same CUDA device as q");
        TORCH_CHECK(
            rng_state.scalar_type() == at::kLong && rng_state.is_contiguous() && rng_state.numel() == 2,
            "dropout backward requires a contiguous int64 seed/offset tensor");
    } else {
        TORCH_CHECK(rng_state.numel() == 0, "no-dropout backward expects an empty RNG state");
    }
    at::cuda::CUDAGuard device_guard(q.device());

    check_dropout_backward_support(q, dropout_p);

    // Autograd may provide an expanded gradient with stride(-1) == 0, for
    // example after out.sum(). The kernels require a contiguous head dimension,
    // so materialize only that exceptional gradient instead of rejecting it.
    const at::Tensor dout = dout_input.stride(-1) == 1 ? dout_input : dout_input.contiguous();
    auto dq = at::empty_like(q);
    auto dk = at::empty_like(k);
    auto dv = at::empty_like(v);

    const int seqlen_rounded = round_up(static_cast<int>(q.size(2)), 128);
    const int head_dim_rounded = round_up(
        static_cast<int>(q.size(3)), q.size(3) <= 128 ? 32 : 64);
    auto float_options = q.options().dtype(at::kFloat);
    auto softmax_d = at::empty({q.size(0), q.size(1), seqlen_rounded}, float_options);
    auto dq_accum = at::empty(
        {q.size(0), seqlen_rounded, q.size(1), head_dim_rounded}, float_options);

    Flash_bwd_params params;
    at::Tensor mutable_lse = lse;
    at::Tensor mutable_rng_state = rng_state;
    at::Tensor mutable_out = out;
    const float softmax_scale = 1.0f / std::sqrt(static_cast<float>(q.size(3)));
    set_forward_params(
        params, q, k, v, mutable_out, mutable_lse, mutable_rng_state,
        dropout_p, softmax_scale, is_causal);

    // Populate gradient pointers and strides in MiniTrain's B,H,S,D layout.
    params.do_ptr = dout.data_ptr();
    params.dq_ptr = dq.data_ptr();
    params.dk_ptr = dk.data_ptr();
    params.dv_ptr = dv.data_ptr();
    params.do_batch_stride = dout.stride(0);
    params.dq_batch_stride = dq.stride(0);
    params.dk_batch_stride = dk.stride(0);
    params.dv_batch_stride = dv.stride(0);
    params.do_head_stride = dout.stride(1);
    params.dq_head_stride = dq.stride(1);
    params.dk_head_stride = dk.stride(1);
    params.dv_head_stride = dv.stride(1);
    params.do_row_stride = dout.stride(2);
    params.dq_row_stride = dq.stride(2);
    params.dk_row_stride = dk.stride(2);
    params.dv_row_stride = dv.stride(2);
    params.dq_accum_ptr = dq_accum.data_ptr();
    params.dsoftmax_sum = softmax_d.data_ptr();
    params.deterministic = false;
    params.dq_accum_split_stride = 0;

    dispatch_backward(params, at::cuda::getCurrentCUDAStream().stream());
    return {dq, dk, dv};
}

}  // namespace minitrain_flash

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.doc() = "MiniTrain dense CUDA FlashAttention (FlashAttention 2.8.4 kernels)";
    module.def(
        "flash_attn_fwd",
        &minitrain_flash::flash_attn_fwd,
        "Dense FlashAttention forward",
        pybind11::arg("q"),
        pybind11::arg("k"),
        pybind11::arg("v"),
        pybind11::arg("is_causal"),
        pybind11::arg("dropout_p"),
        pybind11::arg("return_softmax") = false);
    module.def("flash_attn_bwd", &minitrain_flash::flash_attn_bwd, "Dense FlashAttention backward");
}
