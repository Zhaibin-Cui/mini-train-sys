// Copyright (c) 2024, Tri Dao.
// Adapted for mini-train-sys from FlashAttention 2.8.4 generate_kernels.py.
// This file is generated. Edit cuda_ext/generate_kernels.py instead.

#include "namespace_config.h"
#include "flash_fwd_launch_template.h"

namespace FLASH_NAMESPACE {

// Bind one public dispatch symbol to one fully compile-time-specialized tree.
template <>
void run_mha_fwd_<cutlass::bfloat16_t, 192, true>(
    Flash_fwd_params& params,
    cudaStream_t stream) {
    run_mha_fwd_hdim192<cutlass::bfloat16_t, true>(params, stream);
}

}  // namespace FLASH_NAMESPACE
