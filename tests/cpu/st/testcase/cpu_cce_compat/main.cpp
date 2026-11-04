/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#include <pto/pto-inst.hpp>
#include <gtest/gtest.h>
#include <cstdio>
using namespace pto;
TEST(CpuCceCompat, ResetPreservesExplicitStoreModes) {
    using V = Tile<TileType::Vec, float, 1, 16>;
    using G = GlobalTensor<float, Shape<1, 1, 1, 1, 16>, Stride<16, 16, 16, 16, 1>>;
    float output[16] = {};
    V input;
    TASSIGN(input, 0);
    for (int i = 0; i < 16; ++i) input.data()[i] = i + 1;
    G global(output);
    set_atomic_none();
    TSTORE(global, input);
    for (int i = 0; i < 16; ++i) EXPECT_EQ(output[i], i + 1);
    set_atomic_none();
    TSTORE<V, G, AtomicType::AtomicAdd>(global, input);
    for (int i = 0; i < 16; ++i) EXPECT_EQ(output[i], 2 * (i + 1));
    set_atomic_none();
    TSTORE(global, input);
    for (int i = 0; i < 16; ++i) EXPECT_EQ(output[i], i + 1);
    static_assert(SINGLE_CACHE_LINE == cache_line_t::SINGLE_CACHE_LINE);
    dcci(output, SINGLE_CACHE_LINE);
    dcci(output, cache_line_t::SINGLE_CACHE_LINE, dcci_dst_t::CACHELINE_OUT);
    pipe_barrier(PIPE_ALL);
    set_flag(PIPE_MTE2, PIPE_V, 0);
    wait_flag(PIPE_MTE2, PIPE_V, 0);
    puts("PASS: default store, explicit atomic add, reset, cache aliases and existing flags");
}
