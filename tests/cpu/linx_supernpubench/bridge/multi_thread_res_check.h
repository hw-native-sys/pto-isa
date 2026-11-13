/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#pragma once
#ifdef __CPU_SIM

#include <cstdint>
#include <atomic>

// Hosted RES_CHECK runs the same main() on four PEs.  File I/O must remain on
// PE0, while the input/output buffers live in shared static storage.  This
// small SPMD barrier publishes PE0's input writes and prevents PE0 from
// exporting a partially-computed result before PE1..PE3 have finished.
struct MultiThreadResCheckSync {
    std::atomic<std::uint32_t> input_ready;
    std::atomic<std::uint32_t> done[4];
};

inline void res_check_publish_inputs(MultiThreadResCheckSync& sync, std::uint32_t tid)
{
    if (tid == 0) {
        __asm__ volatile("" : : : "memory");
        sync.input_ready = 1;
    } else {
        while (sync.input_ready == 0) {
        }
        __asm__ volatile("" : : : "memory");
    }
}

inline void res_check_wait_for_all(MultiThreadResCheckSync& sync, std::uint32_t tid)
{
    __asm__ volatile("" : : : "memory");
    sync.done[tid] = 1;
    if (tid == 0) {
        for (std::uint32_t pe = 1; pe < 4; ++pe) {
            while (sync.done[pe] == 0) {
            }
        }
        __asm__ volatile("" : : : "memory");
    }
}

#else
#include_next "multi_thread_res_check.h"
#endif
