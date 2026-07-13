// Copyright (c) 2024, Tri Dao.
// Adapted for mini-train-sys from FlashAttention 2.8.4 generate_kernels.py.
// This file is generated. Edit cuda_ext/generate_kernels.py instead.

#include "namespace_config.h"
#include "flash_bwd_launch_template.h"

namespace FLASH_NAMESPACE {

// Bind one public dispatch symbol to one fully compile-time-specialized tree.
template <>
void run_mha_bwd_<cutlass::bfloat16_t, 192, false>(
    Flash_bwd_params& params,
    cudaStream_t stream) {
    run_mha_bwd_hdim192<cutlass::bfloat16_t, false>(params, stream);
}

}  // namespace FLASH_NAMESPACE
