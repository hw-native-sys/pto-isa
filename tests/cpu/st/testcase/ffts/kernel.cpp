/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#include <cstdint>
#include <pto/pto-inst.hpp>

using namespace pto;

extern "C" {
__attribute__((visibility("default"))) void registerTestHooks(void* lane, void* storage)
{
    cpu_sim::register_hooks(lane, storage);
}

__attribute__((visibility("default"))) void signalEvent(uint16_t message)
{
    __builtin_cce_ffts_cross_core_sync(PIPE_FIX, message);
}

__attribute__((visibility("default"))) void waitEvent(int event) { __builtin_cce_wait_flag_dev(event); }

__attribute__((visibility("default"))) void setBase(uint64_t address) { set_ffts_base_addr(address); }

__attribute__((visibility("default"))) int pipeline(int* data, int* results, int rounds)
{
    constexpr int SLOTS = 3;
    int errors = 0;
#if defined(__DAV_CUBE__)
    for (int i = 0; i < rounds + SLOTS; ++i) {
        const int slot = i % SLOTS;
        if (i >= SLOTS) {
            wait_flag_dev(slot + SLOTS);
            for (int lane = 0; lane < 2; ++lane) {
                errors += results[lane * SLOTS + slot] != (i - SLOTS + 1) * (lane + 2);
            }
        }
        if (i < rounds) {
            data[slot] = i + 1;
            ffts_cross_core_sync(PIPE_FIX, getFFTSMsg(FFTS_MODE_VAL, slot));
        }
    }
#else
    const auto lane = get_subblockid();
    for (int i = 0; i < rounds; ++i) {
        const int slot = i % SLOTS;
        wait_flag_dev(slot);
        errors += data[slot] != i + 1;
        results[lane * SLOTS + slot] = data[slot] * (lane + 2);
        ffts_cross_core_sync(PIPE_MTE3, getFFTSMsg(FFTS_MODE_VAL, slot + SLOTS));
    }
#endif
    return errors;
}
}
