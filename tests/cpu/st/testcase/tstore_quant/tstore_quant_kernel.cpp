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
#include <iostream>

using namespace std;
using namespace pto;

template <typename Dst, typename SrcT, int gShape0, int gShape1, int gShape2, int gShape3, int gShape4,
          int gWholeShape0, int gWholeShape1, int gWholeShape2, int gWholeShape3, int gWholeShape4, bool is_v_quant,
          bool saturate_inf, bool apply_relu>
AICORE inline void RunTStoreRowMajorQuant(__gm__ Dst __out__ *out, __gm__ SrcT __in__ *src,
                                          __gm__ uint64_t __in__ *fbQuant)
{
    constexpr int gStride[5] = {gWholeShape1 * gWholeShape2 * gWholeShape3 * gWholeShape4,
                                gWholeShape2 * gWholeShape3 * gWholeShape4, gWholeShape3 * gWholeShape4, gWholeShape4,
                                1};
    constexpr int blockSize = 32 / sizeof(SrcT);
    constexpr int validRow = gShape0 * gShape1 * gShape2 * gShape3;
    constexpr int validCol = gShape4;
    constexpr int Rows = gShape0 * gShape1 * gShape2 * gShape3;
    constexpr int Cols = (gShape4 + blockSize - 1) / blockSize * blockSize;

    using DynShapeDim5 = Shape<gShape0, gShape1, gShape2, gShape3, gShape4>;
    using DynStridDim5 = pto::Stride<gStride[0], gStride[1], gStride[2], gStride[3], gStride[4]>;
    using GlobalDataDst = GlobalTensor<Dst, DynShapeDim5, DynStridDim5>;
    using GlobalDataSrc = GlobalTensor<SrcT, DynShapeDim5, DynStridDim5>;

    using TileData = Tile<TileType::Vec, SrcT, Rows, Cols, BLayout::RowMajor, -1, -1>;
    constexpr ReluPreMode reluPreMode = apply_relu ? ReluPreMode::NormalRelu : ReluPreMode::NoRelu;

    TileData srcTile(validRow, validCol);

    TASSIGN(srcTile, 0x0);

    GlobalDataDst dstGlobal(out);
    GlobalDataSrc srcGlobal(src);

    TLOAD(srcTile, srcGlobal);
    set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);

    if constexpr (is_v_quant) {
        using GlobalDataFp = GlobalTensor<uint64_t, pto::Shape<1, 1, 1, 1, validCol>,
                                          pto::Stride<1 * validCol, validCol, validCol, validCol, 1>>;
        GlobalDataFp fpGlobal(fbQuant);

        using QuantTile = Tile<TileType::Vec, uint64_t, 1, Cols, BLayout::RowMajor, 1, validCol>;
        QuantTile qTile;
        TASSIGN(qTile, 0x400 + Rows * Cols * sizeof(SrcT));

        TLOAD(qTile, fpGlobal);
        TSTORE_FP<TileData, GlobalDataDst, QuantTile, AtomicType::AtomicNone, reluPreMode>(dstGlobal, srcTile, qTile);
    } else {
        TSTORE<TileData, GlobalDataDst, AtomicType::AtomicNone, reluPreMode>(dstGlobal, srcTile, fbQuant[0]);
    }

    out = dstGlobal.data();
}

template <typename Dst, typename SrcT, int gShape0, int gShape1, int gShape2, int gShape3, int gShape4,
          int gWholeShape0, int gWholeShape1, int gWholeShape2, int gWholeShape3, int gWholeShape4, bool is_v_quant,
          bool saturate_inf, bool apply_relu>
AICORE inline void RunTStoreColMajorQuant(__gm__ Dst __out__ *out, __gm__ SrcT __in__ *src,
                                          __gm__ uint64_t __in__ *fbQuant)
{
    constexpr int gStride[5] = {gWholeShape1 * gWholeShape2 * gWholeShape3 * gWholeShape4,
                                gWholeShape2 * gWholeShape3 * gWholeShape4, gWholeShape3 * gWholeShape4, 1,
                                gWholeShape3};

    constexpr int blockSize = 32 / sizeof(SrcT);
    constexpr int Rows = (gShape3 + blockSize - 1) / blockSize * blockSize;
    constexpr int Cols = gShape0 * gShape1 * gShape2 * gShape4;
    constexpr int validRow = gShape3;
    constexpr int validCol = gShape0 * gShape1 * gShape2 * gShape4;

    using DynShapeDim5 = Shape<gShape0, gShape1, gShape2, gShape3, gShape4>;
    using DynStridDim5 = pto::Stride<gStride[0], gStride[1], gStride[2], gStride[3], gStride[4]>;
    using GlobalDataDst = GlobalTensor<Dst, DynShapeDim5, DynStridDim5>;
    using GlobalDataSrc = GlobalTensor<SrcT, DynShapeDim5, DynStridDim5>;
    using TileData = Tile<TileType::Vec, SrcT, Rows, Cols, BLayout::ColMajor, -1, -1>;

    constexpr ReluPreMode reluPreMode = apply_relu ? ReluPreMode::NormalRelu : ReluPreMode::NoRelu;

    TileData srcTile(validRow, validCol);

    TASSIGN(srcTile, 0x0);

    GlobalDataSrc srcGlobal(src);
    GlobalDataDst dstGlobal(out);

    TLOAD(srcTile, srcGlobal);
    set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);

    if constexpr (is_v_quant) {
        using GlobalDataFp = GlobalTensor<uint64_t, pto::Shape<1, 1, 1, 1, validRow>,
                                          pto::Stride<1 * validRow, validRow, validRow, validRow, 1>>;
        GlobalDataFp fpGlobal(fbQuant);

        using QuantTile = Tile<TileType::Vec, uint64_t, 1, Rows, BLayout::RowMajor, 1, validRow>;
        QuantTile qTile;
        TASSIGN(qTile, 0x400 + Rows * Cols * sizeof(SrcT));

        TLOAD(qTile, fpGlobal);
        TSTORE_FP<TileData, GlobalDataDst, QuantTile, AtomicType::AtomicNone, reluPreMode>(dstGlobal, srcTile, qTile);
    } else {
        TSTORE<TileData, GlobalDataDst, AtomicType::AtomicNone, reluPreMode>(dstGlobal, srcTile, fbQuant[0]);
    }

    out = dstGlobal.data();
}

template <typename Dst, typename SrcT, int gShape0, int gShape1, int gShape2, int gShape3, int gShape4,
          int gWholeShape0, int gWholeShape1, int gWholeShape2, int gWholeShape3, int gWholeShape4, bool is_v_quant,
          bool saturate_inf, bool apply_relu>
AICORE inline void RunTStoreNZQuant(__gm__ Dst __out__ *out, __gm__ SrcT __in__ *src, __gm__ uint64_t __in__ *fbQuant)
{
    constexpr int gStride[5] = {gWholeShape1 * gWholeShape2 * gWholeShape3 * gWholeShape4,
                                gWholeShape2 * gWholeShape3 * gWholeShape4, gWholeShape3 * gWholeShape4, gWholeShape4,
                                1};

    constexpr int Rows = gShape2 * gShape3;
    constexpr int Cols = gShape0 * gShape1 * gShape4;

    using DynShapeDim5 = pto::Shape<gShape0, gShape1, gShape2, gShape3, gShape4>;
    using DynStridDim5 = pto::Stride<gStride[0], gStride[1], gStride[2], gStride[3], gStride[4]>;
    using GlobalDataDst = GlobalTensor<Dst, DynShapeDim5, DynStridDim5>;
    using GlobalDataSrc = GlobalTensor<SrcT, DynShapeDim5, DynStridDim5>;
    using TileData = Tile<TileType::Vec, SrcT, Rows, Cols, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512>;

    constexpr ReluPreMode reluPreMode = apply_relu ? ReluPreMode::NormalRelu : ReluPreMode::NoRelu;

    int validRow = gShape2 * gShape3;
    constexpr int validCol = gShape0 * gShape1 * gShape4;
    TileData srcTile(validRow, validCol);

    TASSIGN(srcTile, 0x0);

    GlobalDataSrc srcGlobal(src);
    GlobalDataDst dstGlobal(out);

    TLOAD(srcTile, srcGlobal);
    set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);

    if constexpr (is_v_quant) {
        using GlobalDataFp = GlobalTensor<uint64_t, pto::Shape<1, 1, 1, 1, validCol>,
                                          pto::Stride<1 * validCol, validCol, validCol, validCol, 1>>;
        GlobalDataFp fpGlobal(fbQuant);

        using QuantTile = Tile<TileType::Vec, uint64_t, 1, Cols, BLayout::RowMajor, 1, validCol>;
        QuantTile qTile;
        TASSIGN(qTile, 0x400 + Rows * Cols * sizeof(SrcT));

        TLOAD(qTile, fpGlobal);
        TSTORE_FP<TileData, GlobalDataDst, QuantTile, AtomicType::AtomicNone, reluPreMode>(dstGlobal, srcTile, qTile);
    } else {
        TSTORE<TileData, GlobalDataDst, AtomicType::AtomicNone, reluPreMode>(dstGlobal, srcTile, fbQuant[0]);
    }

    out = dstGlobal.data();
}

template <typename DstT, typename SrcT, pto::Layout format, int gShape0, int gShape1, int gShape2, int gShape3,
          int gShape4, int gWholeShape0, int gWholeShape1, int gWholeShape2, int gWholeShape3, int gWholeShape4,
          bool is_v_quant, bool saturate_inf, bool apply_relu>
__global__ AICORE void TStoreKernelQuant(__gm__ DstT *out, __gm__ SrcT *src, __gm__ uint64_t *fbQuant)
{
    if constexpr (format == pto::Layout::ND) {
        RunTStoreRowMajorQuant<DstT, SrcT, gShape0, gShape1, gShape2, gShape3, gShape4, gWholeShape0, gWholeShape1,
                               gWholeShape2, gWholeShape3, gWholeShape4, is_v_quant, saturate_inf, apply_relu>(out, src,
                                                                                                               fbQuant);
    } else if constexpr (format == pto::Layout::DN) {
        RunTStoreColMajorQuant<DstT, SrcT, gShape0, gShape1, gShape2, gShape3, gShape4, gWholeShape0, gWholeShape1,
                               gWholeShape2, gWholeShape3, gWholeShape4, is_v_quant, saturate_inf, apply_relu>(out, src,
                                                                                                               fbQuant);
    } else if constexpr (format == pto::Layout::NZ) {
        RunTStoreNZQuant<DstT, SrcT, gShape0, gShape1, gShape2, gShape3, gShape4, gWholeShape0, gWholeShape1,
                         gWholeShape2, gWholeShape3, gWholeShape4, is_v_quant, saturate_inf, apply_relu>(out, src,
                                                                                                         fbQuant);
    }
}

template <int format, typename DstT, typename SrcT, int gShape0, int gShape1, int gShape2, int gShape3, int gShape4,
          int gWholeShape0, int gWholeShape1, int gWholeShape2, int gWholeShape3, int gWholeShape4, bool is_v_quant,
          bool saturate_inf, bool apply_relu>
void LaunchTStoreQuant(DstT *out, SrcT *src, uint64_t *fbQuant, void *stream)
{
    using ST = std::conditional_t<std::is_same_v<SrcT, aclFloat16>, half, SrcT>;
    using DT = std::conditional_t<std::is_same_v<DstT, aclFloat16>, half, DstT>;
    if constexpr (format == 0) {
        TStoreKernelQuant<DstT, SrcT, pto::Layout::ND, gShape0, gShape1, gShape2, gShape3, gShape4, gWholeShape0,
                          gWholeShape1, gWholeShape2, gWholeShape3, gWholeShape4, is_v_quant, saturate_inf, apply_relu>(
            static_cast<DT *>(out), static_cast<ST *>(src), fbQuant);
    } else if constexpr (format == 1) {
        TStoreKernelQuant<DstT, SrcT, pto::Layout::DN, gShape0, gShape1, gShape2, gShape3, gShape4, gWholeShape0,
                          gWholeShape1, gWholeShape2, gWholeShape3, gWholeShape4, is_v_quant, saturate_inf, apply_relu>(
            static_cast<DT *>(out), static_cast<ST *>(src), fbQuant);
    } else if constexpr (format == 2) {
        TStoreKernelQuant<DstT, SrcT, pto::Layout::NZ, gShape0, gShape1, gShape2, gShape3, gShape4, gWholeShape0,
                          gWholeShape1, gWholeShape2, gWholeShape3, gWholeShape4, is_v_quant, saturate_inf, apply_relu>(
            static_cast<DT *>(out), static_cast<ST *>(src), fbQuant);
    }
}

template void LaunchTStoreQuant<0, int8_t, float, 1, 1, 1, 2, 128, 1, 1, 1, 2, 128, true, true, true>(int8_t *out,
                                                                                                      float *src,
                                                                                                      uint64_t *fbQuant,
                                                                                                      void *stream);
template void LaunchTStoreQuant<0, int16_t, int32_t, 1, 2, 1, 23, 121, 3, 2, 2, 35, 125, true, true, false>(
    int16_t *out, int32_t *src, uint64_t *fbQuant, void *stream);
template void LaunchTStoreQuant<0, int8_t, int32_t, 2, 2, 3, 23, 47, 3, 3, 4, 32, 50, true, false, true>(
    int8_t *out, int32_t *src, uint64_t *fbQuant, void *stream);
template void LaunchTStoreQuant<1, aclFloat16, float, 1, 1, 1, 4, 21, 1, 1, 1, 8, 32, false, true, true>(
    aclFloat16 *out, float *src, uint64_t *fbQuant, void *stream);
template void LaunchTStoreQuant<1, aclFloat16, float, 3, 1, 1, 1, 124, 5, 1, 1, 2, 128, true, false, false>(
    aclFloat16 *out, float *src, uint64_t *fbQuant, void *stream);
template void LaunchTStoreQuant<1, int8_t, int32_t, 2, 1, 2, 32, 32, 3, 4, 3, 64, 35, false, true, false>(
    int8_t *out, int32_t *src, uint64_t *fbQuant, void *stream);
template void LaunchTStoreQuant<1, aclFloat16, float, 1, 1, 1, 16, 8, 1, 1, 2, 16, 8, false, false, true>(
    aclFloat16 *out, float *src, uint64_t *fbQuant, void *stream);
template void LaunchTStoreQuant<1, int16_t, int32_t, 2, 2, 2, 16, 16, 5, 3, 3, 16, 16, false, false, false>(
    int16_t *out, int32_t *src, uint64_t *fbQuant, void *stream);
template void LaunchTStoreQuant<1, int8_t, int32_t, 1, 2, 1, 16, 32, 2, 4, 2, 16, 32, true, true, true>(
    int8_t *out, int32_t *src, uint64_t *fbQuant, void *stream);
