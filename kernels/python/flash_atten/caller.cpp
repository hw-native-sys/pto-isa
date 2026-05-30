/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 *
 * Host shim for the pto-dsl flash-attention kernel. compile.sh injects
 * KERNEL_CPP with the generated or patched kernel source that defines
 * `call_both`. The single exported symbol `call_kernel` is what run.py calls
 * via ctypes.
 */

#ifndef KERNEL_CPP
#error "KERNEL_CPP must be defined at compile time (see compile.sh)."
#endif

#include <cstdint>

extern "C" int rtGetC2cCtrlAddr(uint64_t *ctrlAddr, uint32_t *ctrlLen);

#include KERNEL_CPP

extern "C" void call_kernel(uint32_t blockDim, void *stream, uint8_t *gmSlotBuffer, uint8_t *q, uint8_t *k, uint8_t *v,
                            uint8_t *o, int64_t s0, int64_t s1)
{
    void *fftsAddr = nullptr;
    uint32_t fftsLen = 0;
    (void)rtGetC2cCtrlAddr(reinterpret_cast<uint64_t *>(&fftsAddr), &fftsLen);
    (void)fftsLen;

    call_both<<<blockDim, nullptr, stream>>>((__gm__ uint64_t *)fftsAddr, (__gm__ float *)gmSlotBuffer,
                                             (__gm__ half *)gmSlotBuffer, (__gm__ half *)q, (__gm__ half *)k,
                                             (__gm__ half *)v, (__gm__ float *)o, s0, s1);
}
