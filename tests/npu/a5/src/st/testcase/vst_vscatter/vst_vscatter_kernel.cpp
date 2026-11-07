/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

// Avoid pulling broken URMA batch decls from pto_comm_inst.hpp (tree-local issue).
#ifndef PTO_COMM_NOT_SUPPORTED
#define PTO_COMM_NOT_SUPPORTED
#endif

#include <pto/pto-inst.hpp>
#include <acl/acl.h>

using namespace pto;

// Minimal intrinsic-level probe (no TSCATTER):
//   VF1: vsts zeros into a float UB tile (srcTile)
//   VF2: vscatter meaningful payload into the same tile by index
// The two steps live in separate __VEC_SCOPE__ blocks (see issue #409).
//
// Full GM shapes that exceed UB (payload+src+idx) are processed in row tiles.
template <int kTileRows, int kCols>
PTO_INTERNAL void runVstVscatterTile(__gm__ float* out, __gm__ float* payload, __gm__ uint32_t* indexes, int gmRows)
{
    using TileShape = Shape<1, 1, 1, kTileRows, kCols>;
    using TileStride = pto::Stride<1, 1, 1, kCols, 1>;
    using GlobalData = GlobalTensor<float, TileShape, TileStride>;
    using GlobalIdx = GlobalTensor<uint32_t, TileShape, TileStride>;

    using TileData = Tile<TileType::Vec, float, kTileRows, kCols>;
    using TileIdx = Tile<TileType::Vec, uint32_t, kTileRows, kCols>;

    constexpr uint32_t tileNumel = TileData::Numel;
    constexpr uint16_t nElemPerVL = CCE_VL / sizeof(float);
    constexpr uint16_t zeroRepeats = (tileNumel + nElemPerVL - 1) / nElemPerVL;
    constexpr uint16_t scatterBatch = CCE_VL / sizeof(uint32_t);
    constexpr uint16_t scatterColRepeats = (kCols + scatterBatch - 1) / scatterBatch;
    constexpr auto distValue =
        std::integral_constant<::DistVST, static_cast<::DistVST>(GetDistVst<float, pto::DistVST::DIST_NORM>())>();

    TileData payloadTile;
    TileData srcTile; // zeroed by vsts, then filled by vscatter
    TileIdx idxTile;

    TASSIGN(payloadTile, 0x0);
    TASSIGN(srcTile, tileNumel * sizeof(float));
    TASSIGN(idxTile, 2 * tileNumel * sizeof(float));

    __ubuf__ float* srcPtr = (__ubuf__ float*)srcTile.data();
    __ubuf__ float* payloadPtr = (__ubuf__ float*)payloadTile.data();
    __ubuf__ uint32_t* idxPtr = (__ubuf__ uint32_t*)idxTile.data();

    for (int rowBase = 0; rowBase < gmRows; rowBase += kTileRows) {
        GlobalData payloadGlobal(payload + rowBase * kCols);
        GlobalData dstGlobal(out + rowBase * kCols);
        GlobalIdx idxGlobal(indexes + rowBase * kCols);

        TLOAD(payloadTile, payloadGlobal);
        TLOAD(idxTile, idxGlobal);
#ifndef __PTO_AUTO__
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
#endif

        // VF1: clear srcTile with vsts
        __VEC_SCOPE__
        {
            RegTensor<float> zeros;
            vbr(zeros, 0.0f);
            uint32_t remaining = tileNumel;
            for (uint16_t i = 0; i < zeroRepeats; ++i) {
                MaskReg preg = CreatePredicate<float>(remaining);
                vsts(zeros, srcPtr, i * nElemPerVL, distValue, preg);
            }
            mem_bar(VST_VST);
        }

        // VF2: scatter payload -> srcTile by index (no TSCATTER)
        __VEC_SCOPE__
        {
            RegTensor<float> vPayload;
            RegTensor<uint32_t> vIdx;
            for (uint16_t row = 0; row < static_cast<uint16_t>(kTileRows); ++row) {
                uint32_t sReg = kCols;
                for (uint16_t colRepeat = 0; colRepeat < scatterColRepeats; ++colRepeat) {
                    MaskReg pReg = CreatePredicate<uint32_t>(sReg);
                    uint32_t offset = row * kCols + colRepeat * scatterBatch;
                    vlds(vPayload, payloadPtr, offset, NORM);
                    vlds(vIdx, idxPtr, offset, NORM);
                    vscatter(vPayload, srcPtr, vIdx, pReg);
                }
            }
        }

#ifndef __PTO_AUTO__
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
#endif
        TSTORE(dstGlobal, srcTile);
#ifndef __PTO_AUTO__
        set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
        wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
#endif
    }
}

extern "C" __global__ AICORE void launchVstVscatterCase1(
    __gm__ float* out, __gm__ float* payload, __gm__ uint32_t* indexes)
{
    // 32x64 fits entirely in one UB tile group.
    runVstVscatterTile<32, 64>(out, payload, indexes, 32);
}

extern "C" __global__ AICORE void launchVstVscatterCase2(
    __gm__ float* out, __gm__ float* payload, __gm__ uint32_t* indexes)
{
    // 64x1024 GM: one float tile alone is 256KB, so tile as 16x1024 (x4).
    runVstVscatterTile<16, 1024>(out, payload, indexes, 64);
}

template <uint32_t caseId>
void launchVstVscatterTestCase(void* out, void* payload, void* indexes, void* stream)
{
    switch (caseId) {
        case 1:
            launchVstVscatterCase1<<<1, nullptr, stream>>>((float*)out, (float*)payload, (uint32_t*)indexes);
            break;
        case 2:
            launchVstVscatterCase2<<<1, nullptr, stream>>>((float*)out, (float*)payload, (uint32_t*)indexes);
            break;
        default:
            break;
    }
}

template void launchVstVscatterTestCase<1>(void* out, void* payload, void* indexes, void* stream);
template void launchVstVscatterTestCase<2>(void* out, void* payload, void* indexes, void* stream);
