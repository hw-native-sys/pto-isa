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
#include "acl/acl.h"

using namespace pto;
namespace TRowExpandMulTest {

template <typename T, uint32_t dstRow, uint32_t dstCol, uint32_t src1Row, uint32_t src1Col, bool src0eqdst>
__global__ AICORE void runROWEXPANDMUL(__gm__ T __out__* out, __gm__ T __in__* src0, __gm__ T __in__* src1)
{
    using DynShapeDim5 = Shape<1, 1, 1, src1Row, src1Col>;
    using DynStridDim5 = pto::Stride<1, 1, 1, src1Col, 1>;
    using GlobalData = GlobalTensor<T, DynShapeDim5, DynStridDim5, Layout::DN>;
    using TileData = Tile<TileType::Vec, T, src1Row, 1, BLayout::ColMajor, -1, -1>;

    using DstDynShapeDim5 = Shape<1, 1, 1, dstRow, dstCol>;
    using DstDynStridDim5 = pto::Stride<1, 1, 1, dstCol, 1>;
    using DstGlobalData = GlobalTensor<T, DstDynShapeDim5, DstDynStridDim5>;
    using DstTileData = Tile<TileType::Vec, T, dstRow, dstCol, BLayout::RowMajor, -1, -1>;

    DstTileData src0Tile(dstRow, dstCol);
    TileData src1Tile(src1Row, 1);
    DstTileData dstTile(dstRow, dstCol);
    size_t size = dstRow * dstCol * sizeof(T);
    TASSIGN(src0Tile, 0x0);
    TASSIGN(src1Tile, size);
    TASSIGN(dstTile, 0x0);

    int offset = 0;
    DstGlobalData src0Global(src0 + offset);
    GlobalData src1Global(src1 + offset);
    DstGlobalData dstGlobal(out + offset);

    TLOAD(src0Tile, src0Global);
    TLOAD(src1Tile, src1Global);
#ifndef __PTO_AUTO__
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
#endif
    if constexpr (src0eqdst) {
        TROWEXPANDMUL(dstTile, src0Tile, src1Tile);
    } else {
        TROWEXPANDMUL(dstTile, src1Tile, src0Tile);
    }
#ifndef __PTO_AUTO__
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
#endif
    TSTORE(dstGlobal, dstTile);
    out = dstGlobal.data();
}

template <typename T, uint32_t dstRow, uint32_t dstCol, uint32_t src1Row, uint32_t src1Col, bool src0eqdst>
__global__ AICORE void runROWEXPANDMUL2(__gm__ T __out__* out, __gm__ T __in__* src0, __gm__ T __in__* src1)
{
    using DynShapeDim5 = Shape<1, 1, 1, src1Row, src1Col>;
    using DynStridDim5 = pto::Stride<1, 1, 1, src1Col, 1>;
    using GlobalData = GlobalTensor<T, DynShapeDim5, DynStridDim5>;
    using TileData = Tile<TileType::Vec, T, src1Row, src1Col, BLayout::RowMajor, -1, -1>;

    using DstDynShapeDim5 = Shape<1, 1, 1, dstRow, dstCol>;
    using DstDynStridDim5 = pto::Stride<1, 1, 1, dstCol, 1>;
    using DstGlobalData = GlobalTensor<T, DstDynShapeDim5, DstDynStridDim5>;
    using DstTileData = Tile<TileType::Vec, T, dstRow, dstCol, BLayout::RowMajor, -1, -1>;

    DstTileData src0Tile(dstRow, dstCol);
    TileData src1Tile(src1Row, src1Col);
    DstTileData dstTile(dstRow, dstCol);
    size_t size = dstRow * dstCol * sizeof(T);
    TASSIGN(src0Tile, 0x0);
    TASSIGN(src1Tile, size);
    TASSIGN(dstTile, 0x0);

    int offset = 0;
    DstGlobalData src0Global(src0 + offset);
    GlobalData src1Global(src1 + offset);
    DstGlobalData dstGlobal(out + offset);

    TLOAD(src0Tile, src0Global);
    TLOAD(src1Tile, src1Global);
#ifndef __PTO_AUTO__
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
#endif
    if constexpr (src0eqdst) {
        TROWEXPANDMUL(dstTile, src0Tile, src1Tile);
    } else {
        TROWEXPANDMUL(dstTile, src1Tile, src0Tile);
    }
#ifndef __PTO_AUTO__
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
#endif
    TSTORE(dstGlobal, dstTile);
    out = dstGlobal.data();
}

template <typename T, uint32_t dstRow, uint32_t dstCol, uint32_t src1Row, uint32_t src1Col, bool src0eqdst>
void launchTRowExpandMul(T* out, T* src0, T* src1, void* stream)
{
    if constexpr (std::is_same_v<T, aclFloat16>) {
        runROWEXPANDMUL<half, dstRow, dstCol, src1Row, src1Col, src0eqdst>
            <<<1, nullptr, stream>>>((half*)out, (half*)src0, (half*)src1);
    } else {
        runROWEXPANDMUL<T, dstRow, dstCol, src1Row, src1Col, src0eqdst><<<1, nullptr, stream>>>(out, src0, src1);
    }
}

template <typename T, uint32_t dstRow, uint32_t dstCol, uint32_t src1Row, uint32_t src1Col, bool src0eqdst>
void launchTRowExpandMul2(T* out, T* src0, T* src1, void* stream)
{
    if constexpr (std::is_same_v<T, aclFloat16>) {
        runROWEXPANDMUL2<half, dstRow, dstCol, src1Row, src1Col, src0eqdst>
            <<<1, nullptr, stream>>>((half*)out, (half*)src0, (half*)src1);
    } else {
        runROWEXPANDMUL2<T, dstRow, dstCol, src1Row, src1Col, src0eqdst><<<1, nullptr, stream>>>(out, src0, src1);
    }
}

template void launchTRowExpandMul<float, 16, 32, 16, 1, true>(float* out, float* src0, float* src1, void* stream);
template void launchTRowExpandMul<float, 56, 128, 56, 1, true>(float* out, float* src0, float* src1, void* stream);
template void launchTRowExpandMul<aclFloat16, 48, 64, 48, 1, true>(
    aclFloat16* out, aclFloat16* src0, aclFloat16* src1, void* stream);
template void launchTRowExpandMul<aclFloat16, 16, 128, 16, 1, true>(
    aclFloat16* out, aclFloat16* src0, aclFloat16* src1, void* stream);
template void launchTRowExpandMul2<float, 24, 64, 24, 8, true>(float* out, float* src0, float* src1, void* stream);
template void launchTRowExpandMul<aclFloat16, 32, 64, 32, 1, false>(
    aclFloat16* out, aclFloat16* src0, aclFloat16* src1, void* stream);
template void launchTRowExpandMul2<float, 20, 64, 20, 8, false>(float* out, float* src0, float* src1, void* stream);
template void launchTRowExpandMul<int32_t, 16, 32, 16, 1, true>(
    int32_t* out, int32_t* src0, int32_t* src1, void* stream);
template void launchTRowExpandMul<int16_t, 16, 64, 16, 1, true>(
    int16_t* out, int16_t* src0, int16_t* src1, void* stream);
template void launchTRowExpandMul<uint32_t, 16, 32, 16, 1, true>(
    uint32_t* out, uint32_t* src0, uint32_t* src1, void* stream);
template void launchTRowExpandMul<uint16_t, 16, 64, 16, 1, true>(
    uint16_t* out, uint16_t* src0, uint16_t* src1, void* stream);
template void launchTRowExpandMul<int64_t, 16, 32, 16, 1, true>(
    int64_t* out, int64_t* src0, int64_t* src1, void* stream);
template void launchTRowExpandMul2<int64_t, 24, 64, 24, 4, true>(
    int64_t* out, int64_t* src0, int64_t* src1, void* stream);
template void launchTRowExpandMul<uint64_t, 16, 32, 16, 1, true>(
    uint64_t* out, uint64_t* src0, uint64_t* src1, void* stream);

// issue #828: bf16 was never exercised on A5 (vmul bf16 is native; regression coverage)
template <uint32_t dstRow, uint32_t dstCol, uint32_t src1Row, uint32_t src1Col, bool src0eqdst>
void launchTRowExpandMulBf16(uint16_t* out, uint16_t* src0, uint16_t* src1, void* stream)
{
    runROWEXPANDMUL<bfloat16_t, dstRow, dstCol, src1Row, src1Col, src0eqdst>
        <<<1, nullptr, stream>>>((bfloat16_t*)out, (bfloat16_t*)src0, (bfloat16_t*)src1);
}

template void launchTRowExpandMulBf16<16, 32, 16, 1, true>(uint16_t* out, uint16_t* src0, uint16_t* src1, void* stream);
// Packed short-row cases (issue #406): widths 8/16/32 with 2-8 rows per vector, plus the model
// shapes (1024x16 FP4 groups, 512x32 FP8 groups) and a half variant.
template void launchTRowExpandMul<float, 8, 8, 8, 1, true>(float* out, float* src0, float* src1, void* stream);
template void launchTRowExpandMul<float, 16, 16, 16, 1, true>(float* out, float* src0, float* src1, void* stream);
template void launchTRowExpandMul<float, 1024, 16, 1024, 1, true>(float* out, float* src0, float* src1, void* stream);
template void launchTRowExpandMul<float, 512, 32, 512, 1, true>(float* out, float* src0, float* src1, void* stream);
template void launchTRowExpandMul<aclFloat16, 32, 16, 32, 1, true>(
    aclFloat16* out, aclFloat16* src0, aclFloat16* src1, void* stream);

// Row-tail case for the packed short-row path: dst tiles carry a static ColValid so
// TileData::ValidCol == TileData::Cols (isContiguous), while the runtime valid row count (21 of
// physical 24) leaves a trailing partial group (21 % 4 == 1); physical 24 rows keep the
// scalar-carrier tile 32B-aligned.
template <typename T, uint32_t physRow, uint32_t physCol, uint32_t validRow>
__global__ AICORE void runROWEXPANDMUL_VALID(__gm__ T __out__* out, __gm__ T __in__* src0, __gm__ T __in__* src1)
{
    using DynShapeDim5 = Shape<1, 1, 1, validRow, 1>;
    using DynStridDim5 = pto::Stride<1, 1, 1, 1, 1>;
    using GlobalData = GlobalTensor<T, DynShapeDim5, DynStridDim5, Layout::DN>;
    using TileData = Tile<TileType::Vec, T, physRow, 1, BLayout::ColMajor, -1, -1>;

    using DstDynShapeDim5 = Shape<1, 1, 1, validRow, physCol>;
    using DstDynStridDim5 = pto::Stride<1, 1, 1, physCol, 1>;
    using DstGlobalData = GlobalTensor<T, DstDynShapeDim5, DstDynStridDim5>;
    using DstTileData = Tile<TileType::Vec, T, physRow, physCol, BLayout::RowMajor, -1, physCol>;

    DstTileData src0Tile(validRow);
    TileData src1Tile(validRow, 1);
    DstTileData dstTile(validRow);
    size_t size = physRow * physCol * sizeof(T);
    TASSIGN(src0Tile, 0x0);
    TASSIGN(src1Tile, size);
    TASSIGN(dstTile, 0x0);

    DstGlobalData src0Global(src0);
    GlobalData src1Global(src1);
    DstGlobalData dstGlobal(out);

    TLOAD(src0Tile, src0Global);
    TLOAD(src1Tile, src1Global);
#ifndef __PTO_AUTO__
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
#endif
    TROWEXPANDMUL(dstTile, src0Tile, src1Tile);
#ifndef __PTO_AUTO__
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
#endif
    TSTORE(dstGlobal, dstTile);
    out = dstGlobal.data();
}

template <typename T, uint32_t physRow, uint32_t physCol, uint32_t validRow>
void launchTRowExpandMulValid(T* out, T* src0, T* src1, void* stream)
{
    runROWEXPANDMUL_VALID<T, physRow, physCol, validRow><<<1, nullptr, stream>>>(out, src0, src1);
}

template void launchTRowExpandMulValid<float, 24, 16, 21>(float* out, float* src0, float* src1, void* stream);

// Dynamic-valid-extent cases for the runtime packed-path guard: tiles keep the DYNAMIC (-1) valid
// dimensions with runtime valid extents, as PyPTO generates. Full runtime columns enter
// TRowExpandBinOps_1D_PackedShortRows through the guard (including row tails: 1021 % 4 == 1,
// 509 % 2 == 1 on the C1A model shapes); runtime partial columns keep the strided 2D path.
template <typename T, uint32_t physRow, uint32_t physCol, uint32_t validRow, uint32_t validCol>
__global__ AICORE void runROWEXPANDMUL_DYNVALID(__gm__ T __out__* out, __gm__ T __in__* src0, __gm__ T __in__* src1)
{
    using DynShapeDim5 = Shape<1, 1, 1, validRow, 1>;
    using DynStridDim5 = pto::Stride<1, 1, 1, 1, 1>;
    using GlobalData = GlobalTensor<T, DynShapeDim5, DynStridDim5, Layout::DN>;
    using TileData = Tile<TileType::Vec, T, physRow, 1, BLayout::ColMajor, -1, -1>;

    using DstDynShapeDim5 = Shape<1, 1, 1, validRow, validCol>;
    using DstDynStridDim5 = pto::Stride<1, 1, 1, physCol, 1>;
    using DstGlobalData = GlobalTensor<T, DstDynShapeDim5, DstDynStridDim5>;
    using DstTileData = Tile<TileType::Vec, T, physRow, physCol, BLayout::RowMajor, -1, -1>;

    DstTileData src0Tile(validRow, validCol);
    TileData src1Tile(validRow, 1);
    DstTileData dstTile(validRow, validCol);
    size_t size = physRow * physCol * sizeof(T);
    TASSIGN(src0Tile, 0x0);
    TASSIGN(src1Tile, size);
    TASSIGN(dstTile, 0x0);

    DstGlobalData src0Global(src0);
    GlobalData src1Global(src1);
    DstGlobalData dstGlobal(out);

    TLOAD(src0Tile, src0Global);
    TLOAD(src1Tile, src1Global);
#ifndef __PTO_AUTO__
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
#endif
    TROWEXPANDMUL(dstTile, src0Tile, src1Tile);
#ifndef __PTO_AUTO__
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
#endif
    TSTORE(dstGlobal, dstTile);
    out = dstGlobal.data();
}

template <typename T, uint32_t physRow, uint32_t physCol, uint32_t validRow, uint32_t validCol>
void launchTRowExpandMulDynValid(T* out, T* src0, T* src1, void* stream)
{
    runROWEXPANDMUL_DYNVALID<T, physRow, physCol, validRow, validCol><<<1, nullptr, stream>>>(out, src0, src1);
}

template void launchTRowExpandMulDynValid<float, 1024, 16, 1021, 16>(
    float* out, float* src0, float* src1, void* stream);
template void launchTRowExpandMulDynValid<float, 512, 32, 509, 32>(float* out, float* src0, float* src1, void* stream);
template void launchTRowExpandMulDynValid<float, 24, 32, 21, 30>(float* out, float* src0, float* src1, void* stream);

// Static-valid-extent cases for the packed short-row path: tiles omit the DYNAMIC (-1) valid
// dimensions, so TileData::ValidCol == TileData::Cols makes isContiguous true and the dispatch
// enters TRowExpandBinOps_1D_PackedShortRows. Covers groupRows = 2 (16x32), 4 (16x16) and
// 8 (8x8; 16x16 half with elementsPerRepeat 128) with per-row nonuniform scales.
template <typename T, uint32_t dstRow, uint32_t dstCol>
__global__ AICORE void runROWEXPANDMUL_STATIC(__gm__ T __out__* out, __gm__ T __in__* src0, __gm__ T __in__* src1)
{
    using DynShapeDim5 = Shape<1, 1, 1, dstRow, 1>;
    using DynStridDim5 = pto::Stride<1, 1, 1, 1, 1>;
    using GlobalData = GlobalTensor<T, DynShapeDim5, DynStridDim5, Layout::DN>;
    using TileData = Tile<TileType::Vec, T, dstRow, 1, BLayout::ColMajor>;

    using DstDynShapeDim5 = Shape<1, 1, 1, dstRow, dstCol>;
    using DstDynStridDim5 = pto::Stride<1, 1, 1, dstCol, 1>;
    using DstGlobalData = GlobalTensor<T, DstDynShapeDim5, DstDynStridDim5>;
    using DstTileData = Tile<TileType::Vec, T, dstRow, dstCol, BLayout::RowMajor>;

    DstTileData src0Tile;
    TileData src1Tile;
    DstTileData dstTile;
    size_t size = dstRow * dstCol * sizeof(T);
    TASSIGN(src0Tile, 0x0);
    TASSIGN(src1Tile, size);
    TASSIGN(dstTile, 0x0);

    DstGlobalData src0Global(src0);
    GlobalData src1Global(src1);
    DstGlobalData dstGlobal(out);

    TLOAD(src0Tile, src0Global);
    TLOAD(src1Tile, src1Global);
#ifndef __PTO_AUTO__
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
#endif
    TROWEXPANDMUL(dstTile, src0Tile, src1Tile);
#ifndef __PTO_AUTO__
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
#endif
    TSTORE(dstGlobal, dstTile);
    out = dstGlobal.data();
}

template <typename T, uint32_t dstRow, uint32_t dstCol>
void launchTRowExpandMulStatic(T* out, T* src0, T* src1, void* stream)
{
    if constexpr (std::is_same_v<T, aclFloat16>) {
        runROWEXPANDMUL_STATIC<half, dstRow, dstCol>
            <<<1, nullptr, stream>>>((half*)(out), (half*)(src0), (half*)(src1));
    } else {
        runROWEXPANDMUL_STATIC<T, dstRow, dstCol><<<1, nullptr, stream>>>(out, src0, src1);
    }
}

template void launchTRowExpandMulStatic<float, 16, 32>(float* out, float* src0, float* src1, void* stream);
template void launchTRowExpandMulStatic<float, 16, 16>(float* out, float* src0, float* src1, void* stream);
template void launchTRowExpandMulStatic<float, 8, 8>(float* out, float* src0, float* src1, void* stream);
template void launchTRowExpandMulStatic<aclFloat16, 16, 16>(
    aclFloat16* out, aclFloat16* src0, aclFloat16* src1, void* stream);
} // namespace TRowExpandMulTest
