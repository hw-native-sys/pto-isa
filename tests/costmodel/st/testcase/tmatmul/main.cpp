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
#include <pto/common/constants.hpp>
#include <gtest/gtest.h>

#include "cost_check.hpp"

using namespace pto;

namespace {

template <typename T>
constexpr T CeilAlign(T num_1, T num_2)
{
    return num_2 == 0 ? 0 : (num_1 + num_2 - 1) / num_2 * num_2;
}

template <
    typename outType, typename AType, typename BType, int validM, int validK, int validN, float profiling,
    float accuracy>
void runTMatmul()
{
    constexpr int blockAlign = (sizeof(AType) == 1) ? 32 : 16;
    constexpr int M = CeilAlign<int>(validM, blockAlign);
    constexpr int N = CeilAlign<int>(validN, blockAlign);
    constexpr int K = CeilAlign<int>(validK, blockAlign);

    using LeftTile = TileLeft<AType, M, K, validM, validK>;
    using RightTile = TileRight<BType, K, N, validK, validN>;
    using AccTile = TileAcc<outType, M, N, validM, validN>;

    LeftTile aTile;
    RightTile bTile;
    AccTile cTile;
    TASSIGN(aTile, 0x0);
    TASSIGN(bTile, 0x10000);
    TASSIGN(cTile, 0x20000);

    TMATMUL(cTile, aTile, bTile);

    EXPECT_CYCLE_NEAR(profiling, accuracy);
}

template <
    typename outType, typename AType, typename BType, int M, int K, int N, int numRepeats, float profiling,
    float accuracy>
void runTMatmulSplitK()
{
    using LeftTile = TileLeft<AType, M, K, M, K>;
    using RightTile = TileRight<BType, K, N, K, N>;
    using AccTile = TileAcc<outType, M, N, M, N>;

    LeftTile aTile;
    RightTile bTile;
    AccTile cTile;
    TASSIGN(aTile, 0x0);
    TASSIGN(bTile, 0x10000);
    TASSIGN(cTile, 0x20000);

    for (uint32_t i = 0; i < numRepeats; i++) {
        if (i == 0) {
            TMATMUL(cTile, aTile, bTile);
        } else {
            TMATMUL_ACC(cTile, cTile, aTile, bTile);
        }
    }

    EXPECT_CYCLE_NEAR(profiling, accuracy);
}

} // namespace

TEST(TMatmul, half_40x50x60) { runTMatmul<float, half, half, 40, 50, 60, 54.0f, 1.0f>(); }

TEST(TMatmul, int8_6x7x8) { runTMatmul<int32_t, int8_t, int8_t, 6, 7, 8, 7.0f, 1.0f>(); }

TEST(TMatmul, split_k_half_128x128x64_reps5) { runTMatmulSplitK<float, half, half, 128, 128, 64, 5, 262.0f, 1.0f>(); }

TEST(TMatmul, float_120x110x50) { runTMatmul<float, float, float, 120, 110, 50, 902.0f, 1.0f>(); }
