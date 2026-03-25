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
#include <cmath>

using namespace pto;

// TCVT cycle formula (no PIPE_V between rows):
//   elemPerRepeat = REPEAT_BYTE / max(sizeof(DstT), sizeof(SrcT))
//   numRepeatPerLine = validCol / elemPerRepeat
//   Stats: [vconv(rpt)] * validRow  (back-to-back, no barrier)
//   Total: startup(13) + validRow * numRepeatPerLine * per_repeat(1)
//
// FP32->FP16: repeatWidth=4, elemPerRepeat=64
//   4x64 tile: numRepeatPerLine=1, total = 13 + 4*1 = 17
// FP16->FP32: repeatWidth=4, elemPerRepeat=64 (same)
//   4x64 tile: total = 17

template <typename DstT, typename SrcT, int kTRows_, int kTCols_, float profiling, float accuracy>
AICORE void runTCvt()
{
    using DstTile = Tile<TileType::Vec, DstT, kTRows_, kTCols_, BLayout::RowMajor, -1, -1>;
    using SrcTile = Tile<TileType::Vec, SrcT, kTRows_, kTCols_, BLayout::RowMajor, -1, -1>;

    DstTile dstTile(kTRows_, kTCols_);
    SrcTile srcTile(kTRows_, kTCols_);

    TASSIGN(srcTile, 0x0);
    TASSIGN(dstTile, 0x8000);

    TCVT(dstTile, srcTile, RoundMode::CAST_NONE);

    float costResult = dstTile.GetCycle();
    float precision = 1 - fabs(profiling - costResult) / profiling;
    bool ret = precision >= accuracy;
    EXPECT_TRUE(ret);
}

// FP32->FP16
template <int kTRows_, int kTCols_, float profiling, float accuracy>
void LaunchTCvtF32ToF16(void *stream)
{
    runTCvt<half, float, kTRows_, kTCols_, profiling, accuracy>();
}

// FP16->FP32
template <int kTRows_, int kTCols_, float profiling, float accuracy>
void LaunchTCvtF16ToF32(void *stream)
{
    runTCvt<float, half, kTRows_, kTCols_, profiling, accuracy>();
}

// Instantiate used templates
// FP32->FP16: startup=13 + 4*1 = 17
template void LaunchTCvtF32ToF16<4, 64, 17.0f, 0.0f>(void *stream);
// FP16->FP32: startup=13 + 4*1 = 17
template void LaunchTCvtF16ToF32<4, 64, 17.0f, 0.0f>(void *stream);
