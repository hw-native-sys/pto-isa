/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#include <pto/pto-inst.hpp>
#include <pto/common/constants.hpp>
#include <acl/acl.h>

using namespace std;
using namespace pto;

// =============================================================================
// Pure index mode kernel (3-arg TCOLARGMAX)
// =============================================================================
template <typename T, int srcRow, int srcValidRow, int dstRow, int col, int validCol>
PTO_INTERNAL void runTColCMax(__gm__ uint32_t __out__* out, __gm__ T __in__* src, bool isBinary)
{
    using DynDim2Shape = Shape<1, 1, 1, -1, -1>;
    using DynDim2Stride = pto::Stride<1, 1, -1, -1, 1>;
    using GlobalData = GlobalTensor<T, DynDim2Shape, DynDim2Stride>;
    using GlobalDataDst = GlobalTensor<uint32_t, DynDim2Shape, DynDim2Stride>;
    GlobalData srcGlobal(src, DynDim2Shape(srcValidRow, validCol), DynDim2Stride(srcRow, col));
    GlobalDataDst dstGlobal(out, DynDim2Shape(dstRow, validCol), DynDim2Stride(dstRow, col));

    using SrcTileData = Tile<TileType::Vec, T, srcRow, col, BLayout::RowMajor, -1, -1>;
    using DstTileData = Tile<TileType::Vec, uint32_t, dstRow, col, BLayout::RowMajor, -1, -1>;
    using TmpTile = Tile<TileType::Vec, T, 1, 32, BLayout::RowMajor, -1, -1>;
    SrcTileData srcTile(srcValidRow, validCol);
    DstTileData dstTile(dstRow, validCol);
    TmpTile tmpTile(1, 32);
    TASSIGN(srcTile, 0x0);
    TASSIGN(dstTile, srcRow * col * sizeof(T));
    TASSIGN(tmpTile, srcRow * col * sizeof(T) + col * sizeof(uint32_t));

    TLOAD(srcTile, srcGlobal);

    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    TCOLARGMAX(dstTile, srcTile, tmpTile);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    TSTORE(dstGlobal, dstTile);
    out = dstGlobal.data();
}

// =============================================================================
// Value + index mode kernel (4-arg TCOLARGMAX)
// =============================================================================
template <typename TVal, typename TIdx, int srcRow, int srcValidRow, int dstRow, int col, int validCol>
PTO_INTERNAL void runTColIdxValMax(__gm__ TVal __out__* outVal, __gm__ TIdx __out__* outIdx, __gm__ TVal __in__* src)
{
    using DynDim2Shape = Shape<1, 1, 1, -1, -1>;
    using DynDim2Stride = pto::Stride<1, 1, -1, -1, 1>;

    using GlobalDataSrc = GlobalTensor<TVal, DynDim2Shape, DynDim2Stride>;
    using GlobalDataDstVal = GlobalTensor<TVal, DynDim2Shape, DynDim2Stride>;
    using GlobalDataDstIdx = GlobalTensor<TIdx, DynDim2Shape, DynDim2Stride>;

    GlobalDataSrc srcGlobal(src, DynDim2Shape(srcValidRow, validCol), DynDim2Stride(srcRow, col));
    GlobalDataDstVal dstValGlobal(outVal, DynDim2Shape(dstRow, validCol), DynDim2Stride(dstRow, col));
    GlobalDataDstIdx dstIdxGlobal(outIdx, DynDim2Shape(dstRow, validCol), DynDim2Stride(dstRow, col));

    using SrcTileData = Tile<TileType::Vec, TVal, srcRow, col, BLayout::RowMajor, -1, -1>;
    using DstValTileData = Tile<TileType::Vec, TVal, dstRow, col, BLayout::RowMajor, -1, -1>;
    using DstIdxTileData = Tile<TileType::Vec, TIdx, dstRow, col, BLayout::RowMajor, -1, -1>;
    using TmpTile = Tile<TileType::Vec, TVal, 1, 32, BLayout::RowMajor, -1, -1>;

    SrcTileData srcTile(srcValidRow, validCol);
    DstValTileData valTile(dstRow, validCol);
    DstIdxTileData idxTile(dstRow, validCol);
    TmpTile tmpTile(1, 32);

    TASSIGN(srcTile, 0x0);
    TASSIGN(valTile, srcRow * col * sizeof(TVal));
    TASSIGN(idxTile, (srcRow + 1) * col * sizeof(TVal));
    TASSIGN(tmpTile, (srcRow + 2) * col * sizeof(TVal));

    TLOAD(srcTile, srcGlobal);

    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    TCOLARGMAX(valTile, idxTile, srcTile, tmpTile);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    TSTORE(dstIdxGlobal, idxTile);
    TSTORE(dstValGlobal, valTile);

    outVal = dstValGlobal.data();
    outIdx = dstIdxGlobal.data();
}

// =============================================================================
// Pure index extern "C" entry points -- float32
// =============================================================================
extern "C" __global__ AICORE void launchTCOLCMAXCase01(__gm__ uint32_t* out, __gm__ float* src)
{
    runTColCMax<float, 1, 1, 1, 256, 255>(out, src, false);
}
extern "C" __global__ AICORE void launchTCOLCMAXCase02(__gm__ uint32_t* out, __gm__ float* src)
{
    runTColCMax<float, 16, 16, 1, 128, 127>(out, src, false);
}
extern "C" __global__ AICORE void launchTCOLCMAXCase03(__gm__ uint32_t* out, __gm__ float* src)
{
    runTColCMax<float, 16, 15, 1, 256, 255>(out, src, false);
}

// =============================================================================
// Pure index extern "C" entry points -- float16
// =============================================================================
extern "C" __global__ AICORE void launchTCOLCMAXCase11(__gm__ uint32_t* out, __gm__ half* src)
{
    runTColCMax<half, 1, 1, 1, 256, 255>(out, src, false);
}
extern "C" __global__ AICORE void launchTCOLCMAXCase12(__gm__ uint32_t* out, __gm__ half* src)
{
    runTColCMax<half, 16, 16, 1, 128, 127>(out, src, false);
}
extern "C" __global__ AICORE void launchTCOLCMAXCase13(__gm__ uint32_t* out, __gm__ half* src)
{
    runTColCMax<half, 16, 15, 1, 256, 255>(out, src, false);
}

// =============================================================================
// Pure index extern "C" entry points -- int8
// =============================================================================
extern "C" __global__ AICORE void launchTCOLCMAXCase21(__gm__ uint32_t* out, __gm__ int8_t* src)
{
    runTColCMax<int8_t, 1, 1, 1, 256, 255>(out, src, false);
}
extern "C" __global__ AICORE void launchTCOLCMAXCase22(__gm__ uint32_t* out, __gm__ int8_t* src)
{
    runTColCMax<int8_t, 16, 16, 1, 128, 127>(out, src, false);
}
extern "C" __global__ AICORE void launchTCOLCMAXCase23(__gm__ uint32_t* out, __gm__ int8_t* src)
{
    runTColCMax<int8_t, 16, 15, 1, 256, 255>(out, src, false);
}

// =============================================================================
// Pure index extern "C" entry points -- uint8
// =============================================================================
extern "C" __global__ AICORE void launchTCOLCMAXCase31(__gm__ uint32_t* out, __gm__ uint8_t* src)
{
    runTColCMax<uint8_t, 1, 1, 1, 256, 255>(out, src, false);
}
extern "C" __global__ AICORE void launchTCOLCMAXCase32(__gm__ uint32_t* out, __gm__ uint8_t* src)
{
    runTColCMax<uint8_t, 16, 16, 1, 128, 127>(out, src, false);
}
extern "C" __global__ AICORE void launchTCOLCMAXCase33(__gm__ uint32_t* out, __gm__ uint8_t* src)
{
    runTColCMax<uint8_t, 16, 15, 1, 256, 255>(out, src, false);
}

// =============================================================================
// Pure index extern "C" entry points -- int16
// =============================================================================
extern "C" __global__ AICORE void launchTCOLCMAXCase41(__gm__ uint32_t* out, __gm__ int16_t* src)
{
    runTColCMax<int16_t, 1, 1, 1, 256, 255>(out, src, false);
}
extern "C" __global__ AICORE void launchTCOLCMAXCase42(__gm__ uint32_t* out, __gm__ int16_t* src)
{
    runTColCMax<int16_t, 16, 16, 1, 128, 127>(out, src, false);
}
extern "C" __global__ AICORE void launchTCOLCMAXCase43(__gm__ uint32_t* out, __gm__ int16_t* src)
{
    runTColCMax<int16_t, 16, 15, 1, 256, 255>(out, src, false);
}

// =============================================================================
// Pure index extern "C" entry points -- uint16
// =============================================================================
extern "C" __global__ AICORE void launchTCOLCMAXCase51(__gm__ uint32_t* out, __gm__ uint16_t* src)
{
    runTColCMax<uint16_t, 1, 1, 1, 256, 255>(out, src, false);
}
extern "C" __global__ AICORE void launchTCOLCMAXCase52(__gm__ uint32_t* out, __gm__ uint16_t* src)
{
    runTColCMax<uint16_t, 16, 16, 1, 128, 127>(out, src, false);
}
extern "C" __global__ AICORE void launchTCOLCMAXCase53(__gm__ uint32_t* out, __gm__ uint16_t* src)
{
    runTColCMax<uint16_t, 16, 15, 1, 256, 255>(out, src, false);
}

// =============================================================================
// Pure index extern "C" entry points -- int32
// =============================================================================
extern "C" __global__ AICORE void launchTCOLCMAXCase61(__gm__ uint32_t* out, __gm__ int32_t* src)
{
    runTColCMax<int32_t, 1, 1, 1, 256, 255>(out, src, false);
}
extern "C" __global__ AICORE void launchTCOLCMAXCase62(__gm__ uint32_t* out, __gm__ int32_t* src)
{
    runTColCMax<int32_t, 16, 16, 1, 128, 127>(out, src, false);
}
extern "C" __global__ AICORE void launchTCOLCMAXCase63(__gm__ uint32_t* out, __gm__ int32_t* src)
{
    runTColCMax<int32_t, 16, 15, 1, 256, 255>(out, src, false);
}

// =============================================================================
// Pure index extern "C" entry points -- uint32
// =============================================================================
extern "C" __global__ AICORE void launchTCOLCMAXCase71(__gm__ uint32_t* out, __gm__ uint32_t* src)
{
    runTColCMax<uint32_t, 1, 1, 1, 256, 255>(out, src, false);
}
extern "C" __global__ AICORE void launchTCOLCMAXCase72(__gm__ uint32_t* out, __gm__ uint32_t* src)
{
    runTColCMax<uint32_t, 16, 16, 1, 128, 127>(out, src, false);
}
extern "C" __global__ AICORE void launchTCOLCMAXCase73(__gm__ uint32_t* out, __gm__ uint32_t* src)
{
    runTColCMax<uint32_t, 16, 15, 1, 256, 255>(out, src, false);
}

// =============================================================================
// Pure index extern "C" entry points -- small dim edge cases
// =============================================================================
extern "C" __global__ AICORE void launchTCOLCMAXCase81(__gm__ uint32_t* out, __gm__ half* src)
{
    runTColCMax<half, 16, 16, 1, 32, 32>(out, src, false);
}
extern "C" __global__ AICORE void launchTCOLCMAXCase82(__gm__ uint32_t* out, __gm__ uint16_t* src)
{
    runTColCMax<uint16_t, 16, 16, 1, 32, 32>(out, src, false);
}
extern "C" __global__ AICORE void launchTCOLCMAXCase83(__gm__ uint32_t* out, __gm__ uint32_t* src)
{
    runTColCMax<uint32_t, 16, 16, 1, 32, 31>(out, src, false);
}
extern "C" __global__ AICORE void launchTCOLCMAXCase84(__gm__ uint32_t* out, __gm__ float* src)
{
    runTColCMax<float, 16, 16, 1, 32, 31>(out, src, false);
}
extern "C" __global__ AICORE void launchTCOLCMAXCase85(__gm__ uint32_t* out, __gm__ int8_t* src)
{
    runTColCMax<int8_t, 16, 16, 1, 32, 31>(out, src, false);
}
extern "C" __global__ AICORE void launchTCOLCMAXCase86(__gm__ uint32_t* out, __gm__ uint8_t* src)
{
    runTColCMax<uint8_t, 16, 16, 1, 32, 31>(out, src, false);
}
extern "C" __global__ AICORE void launchTCOLCMAXCase87(__gm__ uint32_t* out, __gm__ int16_t* src)
{
    runTColCMax<int16_t, 16, 16, 1, 32, 31>(out, src, false);
}
extern "C" __global__ AICORE void launchTCOLCMAXCase88(__gm__ uint32_t* out, __gm__ int32_t* src)
{
    runTColCMax<int32_t, 16, 16, 1, 32, 31>(out, src, false);
}
extern "C" __global__ AICORE void launchTCOLCMAXCase91(__gm__ uint32_t* out, __gm__ uint16_t* src)
{
    runTColCMax<uint16_t, 16, 16, 1, 128, 120>(out, src, false);
}
extern "C" __global__ AICORE void launchTCOLCMAXCase92(__gm__ uint32_t* out, __gm__ half* src)
{
    runTColCMax<half, 16, 16, 1, 96, 88>(out, src, false);
}
extern "C" __global__ AICORE void launchTCOLCMAXCase93(__gm__ uint32_t* out, __gm__ uint16_t* src)
{
    runTColCMax<uint16_t, 4, 4, 1, 48, 34>(out, src, false);
}

// =============================================================================
// Value + index extern "C" entry points -- float32 + int32_t index
// =============================================================================
extern "C" __global__ AICORE void launchTCOLIDXVALMAXCase001(
    __gm__ float* outVal, __gm__ int32_t* outIdx, __gm__ float* src)
{
    runTColIdxValMax<float, int32_t, 1, 1, 1, 256, 255>(outVal, outIdx, src);
}
extern "C" __global__ AICORE void launchTCOLIDXVALMAXCase002(
    __gm__ float* outVal, __gm__ int32_t* outIdx, __gm__ float* src)
{
    runTColIdxValMax<float, int32_t, 16, 16, 1, 128, 127>(outVal, outIdx, src);
}
extern "C" __global__ AICORE void launchTCOLIDXVALMAXCase003(
    __gm__ float* outVal, __gm__ int32_t* outIdx, __gm__ float* src)
{
    runTColIdxValMax<float, int32_t, 16, 15, 1, 256, 255>(outVal, outIdx, src);
}

// =============================================================================
// Value + index extern "C" entry points -- float16 + int16_t index
// =============================================================================
extern "C" __global__ AICORE void launchTCOLIDXVALMAXCase011(
    __gm__ half* outVal, __gm__ int16_t* outIdx, __gm__ half* src)
{
    runTColIdxValMax<half, int16_t, 1, 1, 1, 256, 255>(outVal, outIdx, src);
}
extern "C" __global__ AICORE void launchTCOLIDXVALMAXCase012(
    __gm__ half* outVal, __gm__ int16_t* outIdx, __gm__ half* src)
{
    runTColIdxValMax<half, int16_t, 16, 16, 1, 128, 127>(outVal, outIdx, src);
}
extern "C" __global__ AICORE void launchTCOLIDXVALMAXCase013(
    __gm__ half* outVal, __gm__ int16_t* outIdx, __gm__ half* src)
{
    runTColIdxValMax<half, int16_t, 16, 15, 1, 256, 255>(outVal, outIdx, src);
}

// =============================================================================
// Value + index extern "C" entry points -- int16 + int16_t index
// =============================================================================
extern "C" __global__ AICORE void launchTCOLIDXVALMAXCase041(
    __gm__ int16_t* outVal, __gm__ int16_t* outIdx, __gm__ int16_t* src)
{
    runTColIdxValMax<int16_t, int16_t, 1, 1, 1, 256, 255>(outVal, outIdx, src);
}
extern "C" __global__ AICORE void launchTCOLIDXVALMAXCase042(
    __gm__ int16_t* outVal, __gm__ int16_t* outIdx, __gm__ int16_t* src)
{
    runTColIdxValMax<int16_t, int16_t, 16, 16, 1, 128, 127>(outVal, outIdx, src);
}
extern "C" __global__ AICORE void launchTCOLIDXVALMAXCase043(
    __gm__ int16_t* outVal, __gm__ int16_t* outIdx, __gm__ int16_t* src)
{
    runTColIdxValMax<int16_t, int16_t, 16, 15, 1, 256, 255>(outVal, outIdx, src);
}

// =============================================================================
// Value + index extern "C" entry points -- uint16 + int16_t index
// =============================================================================
extern "C" __global__ AICORE void launchTCOLIDXVALMAXCase051(
    __gm__ uint16_t* outVal, __gm__ int16_t* outIdx, __gm__ uint16_t* src)
{
    runTColIdxValMax<uint16_t, int16_t, 1, 1, 1, 256, 255>(outVal, outIdx, src);
}
extern "C" __global__ AICORE void launchTCOLIDXVALMAXCase052(
    __gm__ uint16_t* outVal, __gm__ int16_t* outIdx, __gm__ uint16_t* src)
{
    runTColIdxValMax<uint16_t, int16_t, 16, 16, 1, 128, 127>(outVal, outIdx, src);
}
extern "C" __global__ AICORE void launchTCOLIDXVALMAXCase053(
    __gm__ uint16_t* outVal, __gm__ int16_t* outIdx, __gm__ uint16_t* src)
{
    runTColIdxValMax<uint16_t, int16_t, 16, 15, 1, 256, 255>(outVal, outIdx, src);
}

// =============================================================================
// Value + index extern "C" entry points -- int32 + int32_t index
// =============================================================================
extern "C" __global__ AICORE void launchTCOLIDXVALMAXCase061(
    __gm__ int32_t* outVal, __gm__ int32_t* outIdx, __gm__ int32_t* src)
{
    runTColIdxValMax<int32_t, int32_t, 1, 1, 1, 256, 255>(outVal, outIdx, src);
}
extern "C" __global__ AICORE void launchTCOLIDXVALMAXCase062(
    __gm__ int32_t* outVal, __gm__ int32_t* outIdx, __gm__ int32_t* src)
{
    runTColIdxValMax<int32_t, int32_t, 16, 16, 1, 128, 127>(outVal, outIdx, src);
}
extern "C" __global__ AICORE void launchTCOLIDXVALMAXCase063(
    __gm__ int32_t* outVal, __gm__ int32_t* outIdx, __gm__ int32_t* src)
{
    runTColIdxValMax<int32_t, int32_t, 16, 15, 1, 256, 255>(outVal, outIdx, src);
}

// =============================================================================
// Value + index extern "C" entry points -- uint32 + int32_t index
// =============================================================================
extern "C" __global__ AICORE void launchTCOLIDXVALMAXCase071(
    __gm__ uint32_t* outVal, __gm__ int32_t* outIdx, __gm__ uint32_t* src)
{
    runTColIdxValMax<uint32_t, int32_t, 1, 1, 1, 256, 255>(outVal, outIdx, src);
}
extern "C" __global__ AICORE void launchTCOLIDXVALMAXCase072(
    __gm__ uint32_t* outVal, __gm__ int32_t* outIdx, __gm__ uint32_t* src)
{
    runTColIdxValMax<uint32_t, int32_t, 16, 16, 1, 128, 127>(outVal, outIdx, src);
}
extern "C" __global__ AICORE void launchTCOLIDXVALMAXCase073(
    __gm__ uint32_t* outVal, __gm__ int32_t* outIdx, __gm__ uint32_t* src)
{
    runTColIdxValMax<uint32_t, int32_t, 16, 15, 1, 256, 255>(outVal, outIdx, src);
}

// =============================================================================
// Value + index extern "C" entry points -- small dim edge cases
// =============================================================================
extern "C" __global__ AICORE void launchTCOLIDXVALMAXCase081(
    __gm__ half* outVal, __gm__ int16_t* outIdx, __gm__ half* src)
{
    runTColIdxValMax<half, int16_t, 16, 16, 1, 32, 32>(outVal, outIdx, src);
}
extern "C" __global__ AICORE void launchTCOLIDXVALMAXCase082(
    __gm__ uint16_t* outVal, __gm__ int16_t* outIdx, __gm__ uint16_t* src)
{
    runTColIdxValMax<uint16_t, int16_t, 16, 16, 1, 32, 32>(outVal, outIdx, src);
}
extern "C" __global__ AICORE void launchTCOLIDXVALMAXCase083(
    __gm__ uint32_t* outVal, __gm__ int32_t* outIdx, __gm__ uint32_t* src)
{
    runTColIdxValMax<uint32_t, int32_t, 16, 16, 1, 32, 31>(outVal, outIdx, src);
}
extern "C" __global__ AICORE void launchTCOLIDXVALMAXCase084(
    __gm__ float* outVal, __gm__ int32_t* outIdx, __gm__ float* src)
{
    runTColIdxValMax<float, int32_t, 16, 16, 1, 32, 31>(outVal, outIdx, src);
}
extern "C" __global__ AICORE void launchTCOLIDXVALMAXCase085(
    __gm__ int16_t* outVal, __gm__ int16_t* outIdx, __gm__ int16_t* src)
{
    runTColIdxValMax<int16_t, int16_t, 16, 16, 1, 32, 31>(outVal, outIdx, src);
}
extern "C" __global__ AICORE void launchTCOLIDXVALMAXCase086(
    __gm__ int32_t* outVal, __gm__ int32_t* outIdx, __gm__ int32_t* src)
{
    runTColIdxValMax<int32_t, int32_t, 16, 16, 1, 32, 31>(outVal, outIdx, src);
}
extern "C" __global__ AICORE void launchTCOLIDXVALMAXCase091(
    __gm__ uint16_t* outVal, __gm__ int16_t* outIdx, __gm__ uint16_t* src)
{
    runTColIdxValMax<uint16_t, int16_t, 16, 16, 1, 128, 120>(outVal, outIdx, src);
}
extern "C" __global__ AICORE void launchTCOLIDXVALMAXCase092(
    __gm__ half* outVal, __gm__ int16_t* outIdx, __gm__ half* src)
{
    runTColIdxValMax<half, int16_t, 16, 16, 1, 96, 88>(outVal, outIdx, src);
}
extern "C" __global__ AICORE void launchTCOLIDXVALMAXCase093(
    __gm__ uint16_t* outVal, __gm__ int16_t* outIdx, __gm__ uint16_t* src)
{
    runTColIdxValMax<uint16_t, int16_t, 4, 4, 1, 48, 34>(outVal, outIdx, src);
}

// =============================================================================
// Pure index dispatcher
// =============================================================================
template <uint32_t caseId>
void launchTCOLCMAXTestCase(void* out, void* src, aclrtStream stream)
{
    switch (caseId) {
        case 1: {
            launchTCOLCMAXCase01<<<1, nullptr, stream>>>((uint32_t*)out, (float*)src);
            break;
        }
        case 2: {
            launchTCOLCMAXCase02<<<1, nullptr, stream>>>((uint32_t*)out, (float*)src);
            break;
        }
        case 3: {
            launchTCOLCMAXCase03<<<1, nullptr, stream>>>((uint32_t*)out, (float*)src);
            break;
        }
        case 11: {
            launchTCOLCMAXCase11<<<1, nullptr, stream>>>((uint32_t*)out, (half*)src);
            break;
        }
        case 12: {
            launchTCOLCMAXCase12<<<1, nullptr, stream>>>((uint32_t*)out, (half*)src);
            break;
        }
        case 13: {
            launchTCOLCMAXCase13<<<1, nullptr, stream>>>((uint32_t*)out, (half*)src);
            break;
        }
        case 21: {
            launchTCOLCMAXCase21<<<1, nullptr, stream>>>((uint32_t*)out, (int8_t*)src);
            break;
        }
        case 22: {
            launchTCOLCMAXCase22<<<1, nullptr, stream>>>((uint32_t*)out, (int8_t*)src);
            break;
        }
        case 23: {
            launchTCOLCMAXCase23<<<1, nullptr, stream>>>((uint32_t*)out, (int8_t*)src);
            break;
        }
        case 31: {
            launchTCOLCMAXCase31<<<1, nullptr, stream>>>((uint32_t*)out, (uint8_t*)src);
            break;
        }
        case 32: {
            launchTCOLCMAXCase32<<<1, nullptr, stream>>>((uint32_t*)out, (uint8_t*)src);
            break;
        }
        case 33: {
            launchTCOLCMAXCase33<<<1, nullptr, stream>>>((uint32_t*)out, (uint8_t*)src);
            break;
        }
        case 41: {
            launchTCOLCMAXCase41<<<1, nullptr, stream>>>((uint32_t*)out, (int16_t*)src);
            break;
        }
        case 42: {
            launchTCOLCMAXCase42<<<1, nullptr, stream>>>((uint32_t*)out, (int16_t*)src);
            break;
        }
        case 43: {
            launchTCOLCMAXCase43<<<1, nullptr, stream>>>((uint32_t*)out, (int16_t*)src);
            break;
        }
        case 51: {
            launchTCOLCMAXCase51<<<1, nullptr, stream>>>((uint32_t*)out, (uint16_t*)src);
            break;
        }
        case 52: {
            launchTCOLCMAXCase52<<<1, nullptr, stream>>>((uint32_t*)out, (uint16_t*)src);
            break;
        }
        case 53: {
            launchTCOLCMAXCase53<<<1, nullptr, stream>>>((uint32_t*)out, (uint16_t*)src);
            break;
        }
        case 61: {
            launchTCOLCMAXCase61<<<1, nullptr, stream>>>((uint32_t*)out, (int32_t*)src);
            break;
        }
        case 62: {
            launchTCOLCMAXCase62<<<1, nullptr, stream>>>((uint32_t*)out, (int32_t*)src);
            break;
        }
        case 63: {
            launchTCOLCMAXCase63<<<1, nullptr, stream>>>((uint32_t*)out, (int32_t*)src);
            break;
        }
        case 71: {
            launchTCOLCMAXCase71<<<1, nullptr, stream>>>((uint32_t*)out, (uint32_t*)src);
            break;
        }
        case 72: {
            launchTCOLCMAXCase72<<<1, nullptr, stream>>>((uint32_t*)out, (uint32_t*)src);
            break;
        }
        case 73: {
            launchTCOLCMAXCase73<<<1, nullptr, stream>>>((uint32_t*)out, (uint32_t*)src);
            break;
        }
        case 81: {
            launchTCOLCMAXCase81<<<1, nullptr, stream>>>((uint32_t*)out, (half*)src);
            break;
        }
        case 82: {
            launchTCOLCMAXCase82<<<1, nullptr, stream>>>((uint32_t*)out, (uint16_t*)src);
            break;
        }
        case 83: {
            launchTCOLCMAXCase83<<<1, nullptr, stream>>>((uint32_t*)out, (uint32_t*)src);
            break;
        }
        case 84: {
            launchTCOLCMAXCase84<<<1, nullptr, stream>>>((uint32_t*)out, (float*)src);
            break;
        }
        case 85: {
            launchTCOLCMAXCase85<<<1, nullptr, stream>>>((uint32_t*)out, (int8_t*)src);
            break;
        }
        case 86: {
            launchTCOLCMAXCase86<<<1, nullptr, stream>>>((uint32_t*)out, (uint8_t*)src);
            break;
        }
        case 87: {
            launchTCOLCMAXCase87<<<1, nullptr, stream>>>((uint32_t*)out, (int16_t*)src);
            break;
        }
        case 88: {
            launchTCOLCMAXCase88<<<1, nullptr, stream>>>((uint32_t*)out, (int32_t*)src);
            break;
        }
        case 91: {
            launchTCOLCMAXCase91<<<1, nullptr, stream>>>((uint32_t*)out, (uint16_t*)src);
            break;
        }
        case 92: {
            launchTCOLCMAXCase92<<<1, nullptr, stream>>>((uint32_t*)out, (half*)src);
            break;
        }
        case 93: {
            launchTCOLCMAXCase93<<<1, nullptr, stream>>>((uint32_t*)out, (uint16_t*)src);
            break;
        }
        default: {
        }
    }
}

// =============================================================================
// Value + index dispatcher
// =============================================================================
template <uint32_t caseId>
void launchTCOLIDXVALMAXCase(void* outVal, void* outIdx, void* src, aclrtStream stream)
{
    switch (caseId) {
        case 1: {
            launchTCOLIDXVALMAXCase001<<<1, nullptr, stream>>>((float*)outVal, (int32_t*)outIdx, (float*)src);
            break;
        }
        case 2: {
            launchTCOLIDXVALMAXCase002<<<1, nullptr, stream>>>((float*)outVal, (int32_t*)outIdx, (float*)src);
            break;
        }
        case 3: {
            launchTCOLIDXVALMAXCase003<<<1, nullptr, stream>>>((float*)outVal, (int32_t*)outIdx, (float*)src);
            break;
        }
        case 11: {
            launchTCOLIDXVALMAXCase011<<<1, nullptr, stream>>>((half*)outVal, (int16_t*)outIdx, (half*)src);
            break;
        }
        case 12: {
            launchTCOLIDXVALMAXCase012<<<1, nullptr, stream>>>((half*)outVal, (int16_t*)outIdx, (half*)src);
            break;
        }
        case 13: {
            launchTCOLIDXVALMAXCase013<<<1, nullptr, stream>>>((half*)outVal, (int16_t*)outIdx, (half*)src);
            break;
        }
        case 41: {
            launchTCOLIDXVALMAXCase041<<<1, nullptr, stream>>>((int16_t*)outVal, (int16_t*)outIdx, (int16_t*)src);
            break;
        }
        case 42: {
            launchTCOLIDXVALMAXCase042<<<1, nullptr, stream>>>((int16_t*)outVal, (int16_t*)outIdx, (int16_t*)src);
            break;
        }
        case 43: {
            launchTCOLIDXVALMAXCase043<<<1, nullptr, stream>>>((int16_t*)outVal, (int16_t*)outIdx, (int16_t*)src);
            break;
        }
        case 51: {
            launchTCOLIDXVALMAXCase051<<<1, nullptr, stream>>>((uint16_t*)outVal, (int16_t*)outIdx, (uint16_t*)src);
            break;
        }
        case 52: {
            launchTCOLIDXVALMAXCase052<<<1, nullptr, stream>>>((uint16_t*)outVal, (int16_t*)outIdx, (uint16_t*)src);
            break;
        }
        case 53: {
            launchTCOLIDXVALMAXCase053<<<1, nullptr, stream>>>((uint16_t*)outVal, (int16_t*)outIdx, (uint16_t*)src);
            break;
        }
        case 61: {
            launchTCOLIDXVALMAXCase061<<<1, nullptr, stream>>>((int32_t*)outVal, (int32_t*)outIdx, (int32_t*)src);
            break;
        }
        case 62: {
            launchTCOLIDXVALMAXCase062<<<1, nullptr, stream>>>((int32_t*)outVal, (int32_t*)outIdx, (int32_t*)src);
            break;
        }
        case 63: {
            launchTCOLIDXVALMAXCase063<<<1, nullptr, stream>>>((int32_t*)outVal, (int32_t*)outIdx, (int32_t*)src);
            break;
        }
        case 71: {
            launchTCOLIDXVALMAXCase071<<<1, nullptr, stream>>>((uint32_t*)outVal, (int32_t*)outIdx, (uint32_t*)src);
            break;
        }
        case 72: {
            launchTCOLIDXVALMAXCase072<<<1, nullptr, stream>>>((uint32_t*)outVal, (int32_t*)outIdx, (uint32_t*)src);
            break;
        }
        case 73: {
            launchTCOLIDXVALMAXCase073<<<1, nullptr, stream>>>((uint32_t*)outVal, (int32_t*)outIdx, (uint32_t*)src);
            break;
        }
        case 81: {
            launchTCOLIDXVALMAXCase081<<<1, nullptr, stream>>>((half*)outVal, (int16_t*)outIdx, (half*)src);
            break;
        }
        case 82: {
            launchTCOLIDXVALMAXCase082<<<1, nullptr, stream>>>((uint16_t*)outVal, (int16_t*)outIdx, (uint16_t*)src);
            break;
        }
        case 83: {
            launchTCOLIDXVALMAXCase083<<<1, nullptr, stream>>>((uint32_t*)outVal, (int32_t*)outIdx, (uint32_t*)src);
            break;
        }
        case 84: {
            launchTCOLIDXVALMAXCase084<<<1, nullptr, stream>>>((float*)outVal, (int32_t*)outIdx, (float*)src);
            break;
        }
        case 85: {
            launchTCOLIDXVALMAXCase085<<<1, nullptr, stream>>>((int16_t*)outVal, (int16_t*)outIdx, (int16_t*)src);
            break;
        }
        case 86: {
            launchTCOLIDXVALMAXCase086<<<1, nullptr, stream>>>((int32_t*)outVal, (int32_t*)outIdx, (int32_t*)src);
            break;
        }
        case 91: {
            launchTCOLIDXVALMAXCase091<<<1, nullptr, stream>>>((uint16_t*)outVal, (int16_t*)outIdx, (uint16_t*)src);
            break;
        }
        case 92: {
            launchTCOLIDXVALMAXCase092<<<1, nullptr, stream>>>((half*)outVal, (int16_t*)outIdx, (half*)src);
            break;
        }
        case 93: {
            launchTCOLIDXVALMAXCase093<<<1, nullptr, stream>>>((uint16_t*)outVal, (int16_t*)outIdx, (uint16_t*)src);
            break;
        }
        default: {
        }
    }
}

// =============================================================================
// Pure index template instantiations
// =============================================================================
template void launchTCOLCMAXTestCase<1>(void* out, void* src, aclrtStream stream);
template void launchTCOLCMAXTestCase<2>(void* out, void* src, aclrtStream stream);
template void launchTCOLCMAXTestCase<3>(void* out, void* src, aclrtStream stream);
template void launchTCOLCMAXTestCase<11>(void* out, void* src, aclrtStream stream);
template void launchTCOLCMAXTestCase<12>(void* out, void* src, aclrtStream stream);
template void launchTCOLCMAXTestCase<13>(void* out, void* src, aclrtStream stream);
template void launchTCOLCMAXTestCase<21>(void* out, void* src, aclrtStream stream);
template void launchTCOLCMAXTestCase<22>(void* out, void* src, aclrtStream stream);
template void launchTCOLCMAXTestCase<23>(void* out, void* src, aclrtStream stream);
template void launchTCOLCMAXTestCase<31>(void* out, void* src, aclrtStream stream);
template void launchTCOLCMAXTestCase<32>(void* out, void* src, aclrtStream stream);
template void launchTCOLCMAXTestCase<33>(void* out, void* src, aclrtStream stream);
template void launchTCOLCMAXTestCase<41>(void* out, void* src, aclrtStream stream);
template void launchTCOLCMAXTestCase<42>(void* out, void* src, aclrtStream stream);
template void launchTCOLCMAXTestCase<43>(void* out, void* src, aclrtStream stream);
template void launchTCOLCMAXTestCase<51>(void* out, void* src, aclrtStream stream);
template void launchTCOLCMAXTestCase<52>(void* out, void* src, aclrtStream stream);
template void launchTCOLCMAXTestCase<53>(void* out, void* src, aclrtStream stream);
template void launchTCOLCMAXTestCase<61>(void* out, void* src, aclrtStream stream);
template void launchTCOLCMAXTestCase<62>(void* out, void* src, aclrtStream stream);
template void launchTCOLCMAXTestCase<63>(void* out, void* src, aclrtStream stream);
template void launchTCOLCMAXTestCase<71>(void* out, void* src, aclrtStream stream);
template void launchTCOLCMAXTestCase<72>(void* out, void* src, aclrtStream stream);
template void launchTCOLCMAXTestCase<73>(void* out, void* src, aclrtStream stream);
template void launchTCOLCMAXTestCase<81>(void* out, void* src, aclrtStream stream);
template void launchTCOLCMAXTestCase<82>(void* out, void* src, aclrtStream stream);
template void launchTCOLCMAXTestCase<83>(void* out, void* src, aclrtStream stream);
template void launchTCOLCMAXTestCase<84>(void* out, void* src, aclrtStream stream);
template void launchTCOLCMAXTestCase<85>(void* out, void* src, aclrtStream stream);
template void launchTCOLCMAXTestCase<86>(void* out, void* src, aclrtStream stream);
template void launchTCOLCMAXTestCase<87>(void* out, void* src, aclrtStream stream);
template void launchTCOLCMAXTestCase<88>(void* out, void* src, aclrtStream stream);
template void launchTCOLCMAXTestCase<91>(void* out, void* src, aclrtStream stream);
template void launchTCOLCMAXTestCase<92>(void* out, void* src, aclrtStream stream);
template void launchTCOLCMAXTestCase<93>(void* out, void* src, aclrtStream stream);

// =============================================================================
// Value + index template instantiations
// =============================================================================
template void launchTCOLIDXVALMAXCase<1>(void* outVal, void* outIdx, void* src, aclrtStream stream);
template void launchTCOLIDXVALMAXCase<2>(void* outVal, void* outIdx, void* src, aclrtStream stream);
template void launchTCOLIDXVALMAXCase<3>(void* outVal, void* outIdx, void* src, aclrtStream stream);
template void launchTCOLIDXVALMAXCase<11>(void* outVal, void* outIdx, void* src, aclrtStream stream);
template void launchTCOLIDXVALMAXCase<12>(void* outVal, void* outIdx, void* src, aclrtStream stream);
template void launchTCOLIDXVALMAXCase<13>(void* outVal, void* outIdx, void* src, aclrtStream stream);
template void launchTCOLIDXVALMAXCase<41>(void* outVal, void* outIdx, void* src, aclrtStream stream);
template void launchTCOLIDXVALMAXCase<42>(void* outVal, void* outIdx, void* src, aclrtStream stream);
template void launchTCOLIDXVALMAXCase<43>(void* outVal, void* outIdx, void* src, aclrtStream stream);
template void launchTCOLIDXVALMAXCase<51>(void* outVal, void* outIdx, void* src, aclrtStream stream);
template void launchTCOLIDXVALMAXCase<52>(void* outVal, void* outIdx, void* src, aclrtStream stream);
template void launchTCOLIDXVALMAXCase<53>(void* outVal, void* outIdx, void* src, aclrtStream stream);
template void launchTCOLIDXVALMAXCase<61>(void* outVal, void* outIdx, void* src, aclrtStream stream);
template void launchTCOLIDXVALMAXCase<62>(void* outVal, void* outIdx, void* src, aclrtStream stream);
template void launchTCOLIDXVALMAXCase<63>(void* outVal, void* outIdx, void* src, aclrtStream stream);
template void launchTCOLIDXVALMAXCase<71>(void* outVal, void* outIdx, void* src, aclrtStream stream);
template void launchTCOLIDXVALMAXCase<72>(void* outVal, void* outIdx, void* src, aclrtStream stream);
template void launchTCOLIDXVALMAXCase<73>(void* outVal, void* outIdx, void* src, aclrtStream stream);
template void launchTCOLIDXVALMAXCase<81>(void* outVal, void* outIdx, void* src, aclrtStream stream);
template void launchTCOLIDXVALMAXCase<82>(void* outVal, void* outIdx, void* src, aclrtStream stream);
template void launchTCOLIDXVALMAXCase<83>(void* outVal, void* outIdx, void* src, aclrtStream stream);
template void launchTCOLIDXVALMAXCase<84>(void* outVal, void* outIdx, void* src, aclrtStream stream);
template void launchTCOLIDXVALMAXCase<85>(void* outVal, void* outIdx, void* src, aclrtStream stream);
template void launchTCOLIDXVALMAXCase<86>(void* outVal, void* outIdx, void* src, aclrtStream stream);
template void launchTCOLIDXVALMAXCase<91>(void* outVal, void* outIdx, void* src, aclrtStream stream);
template void launchTCOLIDXVALMAXCase<92>(void* outVal, void* outIdx, void* src, aclrtStream stream);
template void launchTCOLIDXVALMAXCase<93>(void* outVal, void* outIdx, void* src, aclrtStream stream);
