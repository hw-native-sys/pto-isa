/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#include <pto/cpu/linx/runtime.hpp>
#include <atomic>
#include <cstdio>
#include <array>
#ifndef PES
#define PES 4
#endif
extern "C" int supernpubench_main();
int main()
{
    std::atomic<int> status{0};
    std::vector<std::byte> low(1 << 20), high(16 << 20);
    std::array<pto::linx_cpu::MemoryRegion, 2> regions{{{0, low}, {0x4000800000ULL, high}}};
    try {
        pto::linx_cpu::runCore(
            [&] {
                int r = supernpubench_main();
                if (r)
                    status.store(r);
            },
            {PES, regions});
    } catch (const std::exception& e) {
        std::fprintf(stderr, "CPU_EXCEPTION: %s\n", e.what());
        return 90;
    }
    return status.load();
}
