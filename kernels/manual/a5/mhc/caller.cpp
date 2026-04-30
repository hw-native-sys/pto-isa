/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
CANN Open Software License Agreement Version 2.0

Caller wrappers for MHC kernels. Each wrapper exports a C function that
launches the corresponding __global__ kernel with the <<<>>> syntax.
*/
#include "expand_to_mhc_fwd.cpp"
#include <cstdint>

extern "C" void call_expand_fwd(uint32_t blockDim, void *stream,
    uint8_t *x, uint8_t *out, int32_t tokens, int32_t hidden) {
    tilekernels_mhc_expand_to_mhc_fwd_m4<<<blockDim, nullptr, stream>>>(
        (bfloat16_t *)x, (bfloat16_t *)out, tokens, hidden);
}
