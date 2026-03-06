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
#include <pto/common/pto_tile.hpp>
#include <pto/common/constants.hpp>

using namespace pto;

constexpr uint16_t BLOCK_CUBE_M_N = 16;

template <int atomicType, typename accDataType, typename dstDataType, typename srcDataType, int gShape0, int gShape1,
          int gShape2, int gShape3, int gShape4, int gWholeShape0, int gWholeShape1, int gWholeShape2, int gWholeShape3,
          int gWholeShape4, int validM, int validN, int validK, int reluMode = 0, int format = 0>
__global__ AICORE void TStoreAcc2gmNz2nd(__gm__ dstDataType *out, __gm__ srcDataType *src0, __gm__ srcDataType *src1)
{
    constexpr int gStride[5] = {gWholeShape1 * gWholeShape2 * gWholeShape3 * gWholeShape4,
                                gWholeShape2 * gWholeShape3 * gWholeShape4, gWholeShape3 * gWholeShape4, gWholeShape4,
                                1};
    constexpr int M = (validM + BLOCK_CUBE_M_N - 1) / BLOCK_CUBE_M_N * BLOCK_CUBE_M_N;
    constexpr int N = (validN + BLOCK_CUBE_M_N - 1) / BLOCK_CUBE_M_N * BLOCK_CUBE_M_N;
    constexpr int K = (validK + BLOCK_CUBE_M_N - 1) / BLOCK_CUBE_M_N * BLOCK_CUBE_M_N;
    constexpr int Rows = M;
    constexpr int Cols = N;

    using GlobalDataSrc0 =
        GlobalTensor<srcDataType, pto::Shape<1, 1, 1, validM, validK>,
                     pto::Stride<1 * validM * validK, 1 * validM * validK, validM * validK, validK, 1>>;
    using GlobalDataSrc1 =
        GlobalTensor<srcDataType, pto::Shape<1, 1, 1, validK, validN>,
                     pto::Stride<1 * validK * validN, 1 * validK * validN, validK * validN, validN, 1>>;

    using DynShapeDim5 = pto::Shape<gShape0, gShape1, gShape2, gShape3, gShape4>;
    using DynStridDim5 = pto::Stride<gStride[0], gStride[1], gStride[2], gStride[3], gStride[4]>;
    using GlobalDataOut = std::conditional_t<
        (format == 0), GlobalTensor<dstDataType, DynShapeDim5, DynStridDim5>,
        std::conditional_t<
            (format == 1), GlobalTensor<dstDataType, DynShapeDim5, DynStridDim5, Layout::NHWC>,
            std::conditional_t<
                (format == 2), GlobalTensor<dstDataType, DynShapeDim5, DynStridDim5, Layout::NCHW>,
                std::conditional_t<(format == 3), GlobalTensor<dstDataType, DynShapeDim5, DynStridDim5, Layout::NCDHW>,
                                   GlobalTensor<dstDataType, DynShapeDim5, DynStridDim5>>>>>;

    GlobalDataSrc0 src0Global(src0);
    GlobalDataSrc1 src1Global(src1);
    GlobalDataOut dstGlobal(out);

    using TileMatAData =
        Tile<TileType::Mat, srcDataType, M, K, BLayout::ColMajor, validM, validK, SLayout::RowMajor, 512>;
    using TileMatBData =
        Tile<TileType::Mat, srcDataType, K, N, BLayout::ColMajor, validK, validN, SLayout::RowMajor, 512>;
    using LeftTile = TileLeft<srcDataType, M, K, validM, validK>;
    using RightTile = TileRight<srcDataType, K, N, validK, validN>;
    using AccTile = TileAcc<accDataType, Rows, Cols, -1, -1>;

    uint32_t aMatSize = M * K * sizeof(srcDataType);

    TileMatAData aMatTile;
    TileMatBData bMatTile;
    TASSIGN(aMatTile, 0x0);
    TASSIGN(bMatTile, aMatSize);

    LeftTile aTile;
    RightTile bTile;
    AccTile cTile(validM, validN);
    TASSIGN(aTile, 0x0);
    TASSIGN(bTile, 0x0);
    TASSIGN(cTile, 0x0);

    TLOAD(aMatTile, src0Global);
    TLOAD(bMatTile, src1Global);

    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    TMOV(aTile, aMatTile);
    TMOV(bTile, bMatTile);
    set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
    wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
    TMATMUL(cTile, aTile, bTile);
    set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    constexpr AtomicType atomicTypeEnum = atomicType == 1 ? AtomicType::AtomicAdd : AtomicType::AtomicNone;
    if constexpr (reluMode == 0) {
        TSTORE<AccTile, GlobalDataOut, atomicTypeEnum>(dstGlobal, cTile);
    } else if constexpr (reluMode == 1) {
        constexpr ReluPreMode reluPreMode = ReluPreMode::NormalRelu;
        TSTORE<AccTile, GlobalDataOut, atomicTypeEnum, reluPreMode>(dstGlobal, cTile);
    }
    set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
    wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
    out = dstGlobal.data();
}

template <int atomicType, typename accDataType, typename dstDataType, typename srcDataType, int gShape0, int gShape1,
          int gShape2, int gShape3, int gShape4, int gWholeShape0, int gWholeShape1, int gWholeShape2, int gWholeShape3,
          int gWholeShape4, int validM, int validN, int validK, int reluMode = 0>
__global__ AICORE void TStoreAcc2gmNz2nz(__gm__ dstDataType *out, __gm__ srcDataType *src0, __gm__ srcDataType *src1)
{
    constexpr int gStride[5] = {gWholeShape1 * gWholeShape2 * gWholeShape3 * gWholeShape4,
                                gWholeShape2 * gWholeShape3 * gWholeShape4, gWholeShape3 * gWholeShape4, gWholeShape4,
                                1};
    constexpr int M = (validM + BLOCK_CUBE_M_N - 1) / BLOCK_CUBE_M_N * BLOCK_CUBE_M_N;
    constexpr int N = (validN + BLOCK_CUBE_M_N - 1) / BLOCK_CUBE_M_N * BLOCK_CUBE_M_N;
    constexpr int K = (validK + BLOCK_CUBE_M_N - 1) / BLOCK_CUBE_M_N * BLOCK_CUBE_M_N;
    constexpr int Rows = gShape2 * gShape3;
    constexpr int Cols = gShape0 * gShape1 * gShape4;
    int validRow = gShape2 * gShape3;
    int validCol = gShape0 * gShape1 * gShape4;
    using DynShapeDim5 = pto::Shape<gShape0, gShape1, gShape2, gShape3, gShape4>;
    using DynStridDim5 = pto::Stride<gStride[0], gStride[1], gStride[2], gStride[3], gStride[4]>;
    using GlobalDataSrc0 =
        GlobalTensor<srcDataType, pto::Shape<1, 1, 1, validM, validK>,
                     pto::Stride<1 * validM * validK, 1 * validM * validK, validM * validK, validK, 1>>;
    using GlobalDataSrc1 =
        GlobalTensor<srcDataType, pto::Shape<1, 1, 1, validK, validN>,
                     pto::Stride<1 * validK * validN, 1 * validK * validN, validK * validN, validN, 1>>;
    using GlobalDataOut = GlobalTensor<dstDataType, DynShapeDim5, DynStridDim5, Layout::NZ>;

    GlobalDataSrc0 src0Global(src0);
    GlobalDataSrc1 src1Global(src1);
    GlobalDataOut dstGlobal(out);

    using TileMatAData =
        Tile<TileType::Mat, srcDataType, M, K, BLayout::ColMajor, validM, validK, SLayout::RowMajor, 512>;
    using TileMatBData =
        Tile<TileType::Mat, srcDataType, K, N, BLayout::ColMajor, validK, validN, SLayout::RowMajor, 512>;
    using LeftTile = TileLeft<srcDataType, M, K, validM, validK>;
    using RightTile = TileRight<srcDataType, K, N, validK, validN>;
    using AccTile = TileAcc<accDataType, Rows, Cols, -1, -1>;

    uint32_t aMatSize = M * K * sizeof(srcDataType);

    TileMatAData aMatTile;
    TileMatBData bMatTile;
    TASSIGN(aMatTile, 0x0);
    TASSIGN(bMatTile, aMatSize);

    LeftTile aTile;
    RightTile bTile;
    AccTile cTile(validRow, validCol);
    TASSIGN(aTile, 0x0);
    TASSIGN(bTile, 0x0);
    TASSIGN(cTile, 0x0);

    TLOAD(aMatTile, src0Global);
    TLOAD(bMatTile, src1Global);

    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    TMOV(aTile, aMatTile);
    TMOV(bTile, bMatTile);
    set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
    wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
    TMATMUL(cTile, aTile, bTile);
    set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    constexpr AtomicType atomicTypeEnum = atomicType == 1 ? AtomicType::AtomicAdd : AtomicType::AtomicNone;
    if constexpr (reluMode == 0) {
        TSTORE<AccTile, GlobalDataOut, atomicTypeEnum>(dstGlobal, cTile);
    } else if constexpr (reluMode == 1) {
        constexpr ReluPreMode reluPreMode = ReluPreMode::NormalRelu;
        TSTORE<AccTile, GlobalDataOut, atomicTypeEnum, reluPreMode>(dstGlobal, cTile);
    }
    set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
    wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
    out = dstGlobal.data();
}

template <int atomicType, typename accDataType, typename dstDataType, typename srcDataType, int gShape0, int gShape1,
          int gShape2, int gShape3, int gShape4, int gWholeShape0, int gWholeShape1, int gWholeShape2, int gWholeShape3,
          int gWholeShape4, int validM, int validN, int validK, int reluMode = 0, int format = 0>
__global__ AICORE void TStoreAcc2gmScalarNz2nd(__gm__ dstDataType *out, __gm__ srcDataType *src0,
                                               __gm__ srcDataType *src1, float scalarQuant)
{
    constexpr int gStride[5] = {gWholeShape1 * gWholeShape2 * gWholeShape3 * gWholeShape4,
                                gWholeShape2 * gWholeShape3 * gWholeShape4, gWholeShape3 * gWholeShape4, gWholeShape4,
                                1};
    constexpr int M = (validM + BLOCK_CUBE_M_N - 1) / BLOCK_CUBE_M_N * BLOCK_CUBE_M_N;
    constexpr int N = (validN + BLOCK_CUBE_M_N - 1) / BLOCK_CUBE_M_N * BLOCK_CUBE_M_N;
    constexpr int K = (validK + BLOCK_CUBE_M_N - 1) / BLOCK_CUBE_M_N * BLOCK_CUBE_M_N;
    constexpr int Rows = M;
    constexpr int Cols = N;

    using GlobalDataSrc0 =
        GlobalTensor<srcDataType, pto::Shape<1, 1, 1, validM, validK>,
                     pto::Stride<1 * validM * validK, 1 * validM * validK, validM * validK, validK, 1>>;
    using GlobalDataSrc1 =
        GlobalTensor<srcDataType, pto::Shape<1, 1, 1, validK, validN>,
                     pto::Stride<1 * validK * validN, 1 * validK * validN, validK * validN, validN, 1>>;

    using DynShapeDim5 = pto::Shape<gShape0, gShape1, gShape2, gShape3, gShape4>;
    using DynStridDim5 = pto::Stride<gStride[0], gStride[1], gStride[2], gStride[3], gStride[4]>;
    using GlobalDataOut = std::conditional_t<
        (format == 0), GlobalTensor<dstDataType, DynShapeDim5, DynStridDim5>,
        std::conditional_t<
            (format == 1), GlobalTensor<dstDataType, DynShapeDim5, DynStridDim5, Layout::NHWC>,
            std::conditional_t<
                (format == 2), GlobalTensor<dstDataType, DynShapeDim5, DynStridDim5, Layout::NCHW>,
                std::conditional_t<(format == 3), GlobalTensor<dstDataType, DynShapeDim5, DynStridDim5, Layout::NCDHW>,
                                   GlobalTensor<dstDataType, DynShapeDim5, DynStridDim5>>>>>;
    GlobalDataSrc0 src0Global(src0);
    GlobalDataSrc1 src1Global(src1);
    GlobalDataOut dstGlobal(out);

    using TileMatAData =
        Tile<TileType::Mat, srcDataType, M, K, BLayout::ColMajor, validM, validK, SLayout::RowMajor, 512>;
    using TileMatBData =
        Tile<TileType::Mat, srcDataType, K, N, BLayout::ColMajor, validK, validN, SLayout::RowMajor, 512>;
    using LeftTile = TileLeft<srcDataType, M, K, validM, validK>;
    using RightTile = TileRight<srcDataType, K, N, validK, validN>;
    using AccTile = TileAcc<accDataType, Rows, Cols, -1, -1>;

    uint32_t aMatSize = M * K * sizeof(srcDataType);

    TileMatAData aMatTile;
    TileMatBData bMatTile;
    TASSIGN(aMatTile, 0x0);
    TASSIGN(bMatTile, aMatSize);

    LeftTile aTile;
    RightTile bTile;
    AccTile cTile(validM, validN);
    TASSIGN(aTile, 0x0);
    TASSIGN(bTile, 0x0);
    TASSIGN(cTile, 0x0);

    TLOAD(aMatTile, src0Global);
    TLOAD(bMatTile, src1Global);

    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    TMOV(aTile, aMatTile);
    TMOV(bTile, bMatTile);
    set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
    wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
    TMATMUL(cTile, aTile, bTile);
    set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    uint64_t preQuantScalar = static_cast<uint64_t>(*reinterpret_cast<int32_t *>(&scalarQuant));
    if (sizeof(dstDataType) == 1) {
        constexpr bool sign = (std::is_same_v<dstDataType, int8_t>) ? true : false;
        preQuantScalar = (preQuantScalar & ~(static_cast<uint64_t>(1) << 46)) | (static_cast<uint64_t>(sign) << 46);
    }
    constexpr AtomicType atomicTypeEnum = atomicType == 1 ? AtomicType::AtomicAdd : AtomicType::AtomicNone;

    if constexpr (reluMode == 0) {
        TSTORE<AccTile, GlobalDataOut, atomicTypeEnum>(dstGlobal, cTile, preQuantScalar);
    } else if constexpr (reluMode == 1) {
        constexpr ReluPreMode reluPreMode = ReluPreMode::NormalRelu;
        TSTORE<AccTile, GlobalDataOut, atomicTypeEnum, reluPreMode>(dstGlobal, cTile, preQuantScalar);
    }
    set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
    wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
    out = dstGlobal.data();
}

template <int atomicType, typename accDataType, typename dstDataType, typename srcDataType, int gShape0, int gShape1,
          int gShape2, int gShape3, int gShape4, int gWholeShape0, int gWholeShape1, int gWholeShape2, int gWholeShape3,
          int gWholeShape4, int validM, int validN, int validK, int reluMode = 0>
__global__ AICORE void TStoreAcc2gmScalarNz2nz(__gm__ dstDataType *out, __gm__ srcDataType *src0,
                                               __gm__ srcDataType *src1, float scalarQuant)
{
    constexpr int gStride[5] = {gWholeShape1 * gWholeShape2 * gWholeShape3 * gWholeShape4,
                                gWholeShape2 * gWholeShape3 * gWholeShape4, gWholeShape3 * gWholeShape4, gWholeShape4,
                                1};
    constexpr int M = (validM + BLOCK_CUBE_M_N - 1) / BLOCK_CUBE_M_N * BLOCK_CUBE_M_N;
    constexpr int N = (validN + BLOCK_CUBE_M_N - 1) / BLOCK_CUBE_M_N * BLOCK_CUBE_M_N;
    constexpr int K = (validK + BLOCK_CUBE_M_N - 1) / BLOCK_CUBE_M_N * BLOCK_CUBE_M_N;
    constexpr int Rows = gShape2 * gShape3;
    constexpr int Cols = gShape0 * gShape1 * gShape4;
    int validRow = Rows;
    int validCol = Cols;
    using DynShapeDim5 = pto::Shape<gShape0, gShape1, gShape2, gShape3, gShape4>;
    using DynStridDim5 = pto::Stride<gStride[0], gStride[1], gStride[2], gStride[3], gStride[4]>;
    using GlobalDataSrc0 =
        GlobalTensor<srcDataType, pto::Shape<1, 1, 1, validM, validK>,
                     pto::Stride<1 * validM * validK, 1 * validM * validK, validM * validK, validK, 1>>;
    using GlobalDataSrc1 =
        GlobalTensor<srcDataType, pto::Shape<1, 1, 1, validK, validN>,
                     pto::Stride<1 * validK * validN, 1 * validK * validN, validK * validN, validN, 1>>;
    using GlobalDataOut = GlobalTensor<dstDataType, DynShapeDim5, DynStridDim5, Layout::NZ>;

    GlobalDataSrc0 src0Global(src0);
    GlobalDataSrc1 src1Global(src1);
    GlobalDataOut dstGlobal(out);

    using TileMatAData =
        Tile<TileType::Mat, srcDataType, M, K, BLayout::ColMajor, validM, validK, SLayout::RowMajor, 512>;
    using TileMatBData =
        Tile<TileType::Mat, srcDataType, K, N, BLayout::ColMajor, validK, validN, SLayout::RowMajor, 512>;
    using LeftTile = TileLeft<srcDataType, M, K, validM, validK>;
    using RightTile = TileRight<srcDataType, K, N, validK, validN>;
    using AccTile = TileAcc<accDataType, Rows, Cols, -1, -1>;

    uint32_t aMatSize = M * K * sizeof(srcDataType);

    TileMatAData aMatTile;
    TileMatBData bMatTile;
    TASSIGN(aMatTile, 0x0);
    TASSIGN(bMatTile, aMatSize);

    LeftTile aTile;
    RightTile bTile;
    AccTile cTile(validRow, validCol);
    TASSIGN(aTile, 0x0);
    TASSIGN(bTile, 0x0);
    TASSIGN(cTile, 0x0);

    TLOAD(aMatTile, src0Global);
    TLOAD(bMatTile, src1Global);

    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    TMOV(aTile, aMatTile);
    TMOV(bTile, bMatTile);
    set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
    wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
    TMATMUL(cTile, aTile, bTile);
    set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    uint64_t preQuantScalar = static_cast<uint64_t>(*reinterpret_cast<int32_t *>(&scalarQuant));
    if (sizeof(dstDataType) == 1) {
        constexpr bool sign = (std::is_same_v<dstDataType, int8_t>) ? true : false;
        preQuantScalar = (preQuantScalar & ~(static_cast<uint64_t>(1) << 46)) | (static_cast<uint64_t>(sign) << 46);
    }
    constexpr AtomicType atomicTypeEnum = atomicType == 1 ? AtomicType::AtomicAdd : AtomicType::AtomicNone;
    if constexpr (reluMode == 0) {
        TSTORE<AccTile, GlobalDataOut, atomicTypeEnum>(dstGlobal, cTile, preQuantScalar);
    } else if constexpr (reluMode == 1) {
        constexpr ReluPreMode reluPreMode = ReluPreMode::NormalRelu;
        TSTORE<AccTile, GlobalDataOut, atomicTypeEnum, reluPreMode>(dstGlobal, cTile, preQuantScalar);
    }
    set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
    wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
    out = dstGlobal.data();
}

template <int atomicType, typename accDataType, typename dstDataType, typename srcDataType, int gShape0, int gShape1,
          int gShape2, int gShape3, int gShape4, int gWholeShape0, int gWholeShape1, int gWholeShape2, int gWholeShape3,
          int gWholeShape4, int validM, int validN, int validK, int reluMode = 0, int format = 0>
__global__ AICORE void TStoreAcc2gmVectorNz2nd(__gm__ dstDataType *out, __gm__ srcDataType *src0,
                                               __gm__ srcDataType *src1, __gm__ uint64_t *quantTensor)
{
    constexpr int gStride[5] = {gWholeShape1 * gWholeShape2 * gWholeShape3 * gWholeShape4,
                                gWholeShape2 * gWholeShape3 * gWholeShape4, gWholeShape3 * gWholeShape4, gWholeShape4,
                                1};
    constexpr int M = (validM + BLOCK_CUBE_M_N - 1) / BLOCK_CUBE_M_N * BLOCK_CUBE_M_N;
    constexpr int N = (validN + BLOCK_CUBE_M_N - 1) / BLOCK_CUBE_M_N * BLOCK_CUBE_M_N;
    constexpr int K = (validK + BLOCK_CUBE_M_N - 1) / BLOCK_CUBE_M_N * BLOCK_CUBE_M_N;
    constexpr int Rows = M;
    constexpr int Cols = N;
    constexpr int alignScalingN = ((validN * sizeof(uint64_t) + 127) / 128) * 128 / sizeof(uint64_t);

    using GlobalDataSrc0 =
        GlobalTensor<srcDataType, pto::Shape<1, 1, 1, validM, validK>,
                     pto::Stride<1 * validM * validK, 1 * validM * validK, validM * validK, validK, 1>>;
    using GlobalDataSrc1 =
        GlobalTensor<srcDataType, pto::Shape<1, 1, 1, validK, validN>,
                     pto::Stride<1 * validK * validN, 1 * validK * validN, validK * validN, validN, 1>>;
    using GlobalDataSrc2 = GlobalTensor<uint64_t, pto::Shape<1, 1, 1, 1, alignScalingN>,
                                        pto::Stride<alignScalingN, alignScalingN, alignScalingN, alignScalingN, 1>>;

    using DynShapeDim5 = pto::Shape<gShape0, gShape1, gShape2, gShape3, gShape4>;
    using DynStridDim5 = pto::Stride<gStride[0], gStride[1], gStride[2], gStride[3], gStride[4]>;
    using GlobalDataOut = std::conditional_t<
        (format == 0), GlobalTensor<dstDataType, DynShapeDim5, DynStridDim5>,
        std::conditional_t<
            (format == 1), GlobalTensor<dstDataType, DynShapeDim5, DynStridDim5, Layout::NHWC>,
            std::conditional_t<
                (format == 2), GlobalTensor<dstDataType, DynShapeDim5, DynStridDim5, Layout::NCHW>,
                std::conditional_t<(format == 3), GlobalTensor<dstDataType, DynShapeDim5, DynStridDim5, Layout::NCDHW>,
                                   GlobalTensor<dstDataType, DynShapeDim5, DynStridDim5>>>>>;
    GlobalDataSrc0 src0Global(src0);
    GlobalDataSrc1 src1Global(src1);
    GlobalDataSrc2 src2Global(quantTensor);
    GlobalDataOut dstGlobal(out);

    using TileMatAData =
        Tile<TileType::Mat, srcDataType, M, K, BLayout::ColMajor, validM, validK, SLayout::RowMajor, 512>;
    using TileMatBData =
        Tile<TileType::Mat, srcDataType, K, N, BLayout::ColMajor, validK, validN, SLayout::RowMajor, 512>;
    using TileMatScalingData =
        Tile<TileType::Mat, uint64_t, 1, alignScalingN, BLayout::RowMajor, 1, -1, SLayout::NoneBox>;
    using LeftTile = TileLeft<srcDataType, M, K, validM, validK>;
    using RightTile = TileRight<srcDataType, K, N, validK, validN>;
    using AccTile = TileAcc<accDataType, Rows, Cols, -1, -1>;
    using ScalingTile = Tile<TileType::Scaling, uint64_t, 1, alignScalingN, BLayout::RowMajor, 1, -1, SLayout::NoneBox>;

    uint32_t aMatSize = M * K * sizeof(srcDataType);
    uint32_t bMatSize = K * N * sizeof(srcDataType);
    TileMatAData aMatTile;
    TileMatBData bMatTile;
    TileMatScalingData scalingMatTile(validN);
    TASSIGN(aMatTile, 0x0);
    TASSIGN(bMatTile, aMatSize);
    TASSIGN(scalingMatTile, aMatSize + bMatSize);

    LeftTile aTile;
    RightTile bTile;
    AccTile cTile(validM, validN);
    ScalingTile scalingTile(validN);
    TASSIGN(aTile, 0x0);
    TASSIGN(bTile, 0x0);
    TASSIGN(cTile, 0x0);
    TASSIGN(scalingTile, 0x0);

    TLOAD(aMatTile, src0Global);
    TLOAD(bMatTile, src1Global);
    TLOAD(scalingMatTile, src2Global);

    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    TMOV(aTile, aMatTile);
    TMOV(bTile, bMatTile);
    set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
    wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
    TMATMUL(cTile, aTile, bTile);
    set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    TMOV(scalingTile, scalingMatTile);
    constexpr AtomicType atomicTypeEnum = atomicType == 1 ? AtomicType::AtomicAdd : AtomicType::AtomicNone;
    if constexpr (reluMode == 0) {
        TSTORE_FP<AccTile, GlobalDataOut, ScalingTile, atomicTypeEnum>(dstGlobal, cTile, scalingTile);
    } else if constexpr (reluMode == 1) {
        constexpr ReluPreMode reluPreMode = ReluPreMode::NormalRelu;
        TSTORE_FP<AccTile, GlobalDataOut, ScalingTile, atomicTypeEnum, reluPreMode>(dstGlobal, cTile, scalingTile);
    }
    set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
    wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
    out = dstGlobal.data();
}

template <int atomicType, typename accDataType, typename dstDataType, typename srcDataType, int gShape0, int gShape1,
          int gShape2, int gShape3, int gShape4, int gWholeShape0, int gWholeShape1, int gWholeShape2, int gWholeShape3,
          int gWholeShape4, int validM, int validN, int validK, int reluMode = 0>
__global__ AICORE void TStoreAcc2gmVectorNz2nz(__gm__ dstDataType *out, __gm__ srcDataType *src0,
                                               __gm__ srcDataType *src1, __gm__ uint64_t *quantTensor)
{
    constexpr int gStride[5] = {gWholeShape1 * gWholeShape2 * gWholeShape3 * gWholeShape4,
                                gWholeShape2 * gWholeShape3 * gWholeShape4, gWholeShape3 * gWholeShape4, gWholeShape4,
                                1};
    constexpr int M = (validM + BLOCK_CUBE_M_N - 1) / BLOCK_CUBE_M_N * BLOCK_CUBE_M_N;
    constexpr int N = (validN + BLOCK_CUBE_M_N - 1) / BLOCK_CUBE_M_N * BLOCK_CUBE_M_N;
    constexpr int K = (validK + BLOCK_CUBE_M_N - 1) / BLOCK_CUBE_M_N * BLOCK_CUBE_M_N;
    constexpr int Rows = gShape2 * gShape3;
    constexpr int Cols = gShape0 * gShape1 * gShape4;
    int validRow = Rows;
    int validCol = Cols;
    constexpr int alignScalingN = ((validN * sizeof(uint64_t) + 127) / 128) * 128 / sizeof(uint64_t);
    using DynShapeDim5 = pto::Shape<gShape0, gShape1, gShape2, gShape3, gShape4>;
    using DynStridDim5 = pto::Stride<gStride[0], gStride[1], gStride[2], gStride[3], gStride[4]>;
    using GlobalDataSrc0 =
        GlobalTensor<srcDataType, pto::Shape<1, 1, 1, validM, validK>,
                     pto::Stride<1 * validM * validK, 1 * validM * validK, validM * validK, validK, 1>>;
    using GlobalDataSrc1 =
        GlobalTensor<srcDataType, pto::Shape<1, 1, 1, validK, validN>,
                     pto::Stride<1 * validK * validN, 1 * validK * validN, validK * validN, validN, 1>>;
    using GlobalDataSrc2 = GlobalTensor<uint64_t, pto::Shape<1, 1, 1, 1, alignScalingN>,
                                        pto::Stride<alignScalingN, alignScalingN, alignScalingN, alignScalingN, 1>>;
    using GlobalDataOut = GlobalTensor<dstDataType, DynShapeDim5, DynStridDim5, Layout::NZ>;

    GlobalDataSrc0 src0Global(src0);
    GlobalDataSrc1 src1Global(src1);
    GlobalDataSrc2 src2Global(quantTensor);
    GlobalDataOut dstGlobal(out);

    using TileMatAData =
        Tile<TileType::Mat, srcDataType, M, K, BLayout::ColMajor, validM, validK, SLayout::RowMajor, 512>;
    using TileMatBData =
        Tile<TileType::Mat, srcDataType, K, N, BLayout::ColMajor, validK, validN, SLayout::RowMajor, 512>;
    using TileMatScalingData =
        Tile<TileType::Mat, uint64_t, 1, alignScalingN, BLayout::RowMajor, 1, -1, SLayout::NoneBox>;
    using LeftTile = TileLeft<srcDataType, M, K, validM, validK>;
    using RightTile = TileRight<srcDataType, K, N, validK, validN>;
    using AccTile = TileAcc<accDataType, Rows, Cols, -1, -1>;
    using ScalingTile = Tile<TileType::Scaling, uint64_t, 1, alignScalingN, BLayout::RowMajor, 1, -1, SLayout::NoneBox>;

    uint32_t aMatSize = M * K * sizeof(srcDataType);
    uint32_t bMatSize = K * N * sizeof(srcDataType);
    TileMatAData aMatTile;
    TileMatBData bMatTile;
    TileMatScalingData scalingMatTile(validN);
    TASSIGN(aMatTile, 0x0);
    TASSIGN(bMatTile, aMatSize);
    TASSIGN(scalingMatTile, aMatSize + bMatSize);

    LeftTile aTile;
    RightTile bTile;
    AccTile cTile(validRow, validCol);
    ScalingTile scalingTile(validN);
    TASSIGN(aTile, 0x0);
    TASSIGN(bTile, 0x0);
    TASSIGN(cTile, 0x0);
    TASSIGN(scalingTile, 0x0);

    TLOAD(aMatTile, src0Global);
    TLOAD(bMatTile, src1Global);
    TLOAD(scalingMatTile, src2Global);

    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    TMOV(aTile, aMatTile);
    TMOV(bTile, bMatTile);
    set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
    wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
    TMATMUL(cTile, aTile, bTile);
    set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    TMOV(scalingTile, scalingMatTile);
    constexpr AtomicType atomicTypeEnum = atomicType == 1 ? AtomicType::AtomicAdd : AtomicType::AtomicNone;
    if constexpr (reluMode == 0) {
        TSTORE_FP<AccTile, GlobalDataOut, ScalingTile, atomicTypeEnum>(dstGlobal, cTile, scalingTile);
    } else if constexpr (reluMode == 1) {
        constexpr ReluPreMode reluPreMode = ReluPreMode::NormalRelu;
        TSTORE_FP<AccTile, GlobalDataOut, ScalingTile, atomicTypeEnum, reluPreMode>(dstGlobal, cTile, scalingTile);
    }
    set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
    wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
    out = dstGlobal.data();
}

template <int tilingKey>
void LaunchTStoreAcc2gmNz2nd(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream)
{
    if constexpr (tilingKey == 1) {
        TStoreAcc2gmNz2nd<1, float, float, float, 1, 1, 1, 128, 128, 1, 1, 1, 128, 128, 128, 128, 16>
            <<<1, nullptr, stream>>>(reinterpret_cast<float *>(out), reinterpret_cast<float *>(src0),
                                     reinterpret_cast<float *>(src1));
    } else if constexpr (tilingKey == 2) {
        TStoreAcc2gmNz2nd<0, float, float, float, 1, 1, 1, 31, 32, 1, 1, 1, 31, 32, 31, 32, 15><<<1, nullptr, stream>>>(
            reinterpret_cast<float *>(out), reinterpret_cast<float *>(src0), reinterpret_cast<float *>(src1));
    } else if constexpr (tilingKey == 3) {
        TStoreAcc2gmNz2nd<0, float, float, half, 1, 1, 1, 65, 128, 1, 1, 1, 65, 128, 65, 128, 96>
            <<<1, nullptr, stream>>>(reinterpret_cast<float *>(out), reinterpret_cast<half *>(src0),
                                     reinterpret_cast<half *>(src1));
    } else if constexpr (tilingKey == 4) {
        TStoreAcc2gmNz2nd<0, float, half, half, 1, 1, 1, 73, 64, 1, 1, 1, 73, 64, 73, 64, 32><<<1, nullptr, stream>>>(
            reinterpret_cast<half *>(out), reinterpret_cast<half *>(src0), reinterpret_cast<half *>(src1));
    } else if constexpr (tilingKey == 5) {
        TStoreAcc2gmNz2nd<1, float, float, bfloat16_t, 1, 1, 1, 13, 32, 1, 1, 1, 13, 32, 13, 32, 25>
            <<<1, nullptr, stream>>>(reinterpret_cast<float *>(out), reinterpret_cast<bfloat16_t *>(src0),
                                     reinterpret_cast<bfloat16_t *>(src1));
    } else if constexpr (tilingKey == 6) {
        TStoreAcc2gmNz2nd<0, float, bfloat16_t, bfloat16_t, 1, 1, 1, 100, 222, 1, 1, 1, 100, 222, 100, 222, 60>
            <<<1, nullptr, stream>>>(reinterpret_cast<bfloat16_t *>(out), reinterpret_cast<bfloat16_t *>(src0),
                                     reinterpret_cast<bfloat16_t *>(src1));
    } else if constexpr (tilingKey == 7) {
        TStoreAcc2gmNz2nd<1, int32_t, int32_t, int8_t, 1, 1, 1, 44, 128, 1, 1, 1, 44, 128, 44, 128, 27>
            <<<1, nullptr, stream>>>(reinterpret_cast<int32_t *>(out), reinterpret_cast<int8_t *>(src0),
                                     reinterpret_cast<int8_t *>(src1));
    } else if constexpr (tilingKey == 21) {
        TStoreAcc2gmNz2nd<0, float, float, float, 1, 1, 1, 117, 97, 1, 1, 1, 117, 97, 117, 97, 71, 1>
            <<<1, nullptr, stream>>>(reinterpret_cast<float *>(out), reinterpret_cast<float *>(src0),
                                     reinterpret_cast<float *>(src1));
    } else if constexpr (tilingKey == 31) {
        TStoreAcc2gmNz2nd<0, float, float, float, 1, 1, 16, 8, 128, 1, 1, 16, 8, 128, 128, 128, 16, 0, 1>
            <<<1, nullptr, stream>>>(reinterpret_cast<float *>(out), reinterpret_cast<float *>(src0),
                                     reinterpret_cast<float *>(src1));
    } else if constexpr (tilingKey == 32) {
        TStoreAcc2gmNz2nd<0, int32_t, int32_t, int8_t, 1, 4, 8, 16, 63, 1, 4, 8, 16, 63, 512, 63, 31, 0, 1>
            <<<1, nullptr, stream>>>(reinterpret_cast<int32_t *>(out), reinterpret_cast<int8_t *>(src0),
                                     reinterpret_cast<int8_t *>(src1));
    } else if constexpr (tilingKey == 33) {
        TStoreAcc2gmNz2nd<0, float, bfloat16_t, float, 1, 1, 32, 32, 32, 1, 1, 32, 32, 32, 1024, 32, 8, 0, 1>
            <<<1, nullptr, stream>>>(reinterpret_cast<bfloat16_t *>(out), reinterpret_cast<float *>(src0),
                                     reinterpret_cast<float *>(src1));
    } else if constexpr (tilingKey == 34) {
        TStoreAcc2gmNz2nd<0, float, float, bfloat16_t, 1, 1, 2, 63, 43, 1, 1, 2, 63, 43, 126, 43, 64, 1, 1>
            <<<1, nullptr, stream>>>(reinterpret_cast<float *>(out), reinterpret_cast<bfloat16_t *>(src0),
                                     reinterpret_cast<bfloat16_t *>(src1));
    } else if constexpr (tilingKey == 41) {
        TStoreAcc2gmNz2nd<0, float, float, float, 1, 1, 128, 16, 8, 1, 1, 128, 16, 8, 128, 128, 16, 0, 2>
            <<<1, nullptr, stream>>>(reinterpret_cast<float *>(out), reinterpret_cast<float *>(src0),
                                     reinterpret_cast<float *>(src1));
    } else if constexpr (tilingKey == 42) {
        TStoreAcc2gmNz2nd<0, int32_t, int32_t, int8_t, 1, 1, 63, 32, 16, 1, 1, 63, 32, 16, 512, 63, 31, 0, 2>
            <<<1, nullptr, stream>>>(reinterpret_cast<int32_t *>(out), reinterpret_cast<int8_t *>(src0),
                                     reinterpret_cast<int8_t *>(src1));
    } else if constexpr (tilingKey == 43) {
        TStoreAcc2gmNz2nd<0, float, bfloat16_t, float, 1, 1, 32, 32, 32, 1, 1, 32, 32, 32, 1024, 32, 8, 0, 2>
            <<<1, nullptr, stream>>>(reinterpret_cast<bfloat16_t *>(out), reinterpret_cast<float *>(src0),
                                     reinterpret_cast<float *>(src1));
    } else if constexpr (tilingKey == 44) {
        TStoreAcc2gmNz2nd<0, float, float, bfloat16_t, 1, 1, 43, 2, 63, 1, 1, 43, 2, 63, 126, 43, 64, 1, 2>
            <<<1, nullptr, stream>>>(reinterpret_cast<float *>(out), reinterpret_cast<bfloat16_t *>(src0),
                                     reinterpret_cast<bfloat16_t *>(src1));
    } else if constexpr (tilingKey == 51) {
        TStoreAcc2gmNz2nd<0, float, float, float, 1, 128, 1, 16, 8, 1, 128, 2, 16, 8, 128, 128, 16, 0, 3>
            <<<1, nullptr, stream>>>(reinterpret_cast<float *>(out), reinterpret_cast<float *>(src0),
                                     reinterpret_cast<float *>(src1));
    } else if constexpr (tilingKey == 52) {
        TStoreAcc2gmNz2nd<0, int32_t, int32_t, int8_t, 1, 63, 1, 32, 16, 1, 63, 3, 32, 16, 512, 63, 31, 0, 3>
            <<<1, nullptr, stream>>>(reinterpret_cast<int32_t *>(out), reinterpret_cast<int8_t *>(src0),
                                     reinterpret_cast<int8_t *>(src1));
    } else if constexpr (tilingKey == 53) {
        TStoreAcc2gmNz2nd<0, float, bfloat16_t, float, 1, 32, 1, 32, 32, 1, 32, 4, 32, 32, 1024, 32, 8, 0, 3>
            <<<1, nullptr, stream>>>(reinterpret_cast<bfloat16_t *>(out), reinterpret_cast<float *>(src0),
                                     reinterpret_cast<float *>(src1));
    } else if constexpr (tilingKey == 54) {
        TStoreAcc2gmNz2nd<0, float, float, bfloat16_t, 1, 43, 1, 2, 63, 1, 43, 2, 2, 63, 126, 43, 64, 1, 3>
            <<<1, nullptr, stream>>>(reinterpret_cast<float *>(out), reinterpret_cast<bfloat16_t *>(src0),
                                     reinterpret_cast<bfloat16_t *>(src1));
    }
}

template <int tilingKey>
void LaunchTStoreAcc2gmNz2nz(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream)
{
    if constexpr (tilingKey == 1) {
        TStoreAcc2gmNz2nz<1, float, float, float, 2, 2, 2, 16, 16, 2, 2, 2, 16, 16, 32, 64, 25><<<1, nullptr, stream>>>(
            reinterpret_cast<float *>(out), reinterpret_cast<float *>(src0), reinterpret_cast<float *>(src1));
    } else if constexpr (tilingKey == 2) {
        TStoreAcc2gmNz2nz<0, float, float, float, 1, 2, 3, 16, 16, 1, 2, 3, 16, 16, 48, 32, 45><<<1, nullptr, stream>>>(
            reinterpret_cast<float *>(out), reinterpret_cast<float *>(src0), reinterpret_cast<float *>(src1));
    } else if constexpr (tilingKey == 3) {
        TStoreAcc2gmNz2nz<0, float, float, half, 2, 2, 2, 16, 16, 2, 2, 2, 16, 16, 32, 64, 24><<<1, nullptr, stream>>>(
            reinterpret_cast<float *>(out), reinterpret_cast<half *>(src0), reinterpret_cast<half *>(src1));
    } else if constexpr (tilingKey == 4) {
        TStoreAcc2gmNz2nz<0, float, half, half, 2, 3, 6, 16, 16, 2, 3, 6, 16, 16, 96, 96, 23><<<1, nullptr, stream>>>(
            reinterpret_cast<half *>(out), reinterpret_cast<half *>(src0), reinterpret_cast<half *>(src1));
    } else if constexpr (tilingKey == 5) {
        TStoreAcc2gmNz2nz<1, float, float, bfloat16_t, 2, 3, 3, 16, 16, 2, 3, 3, 16, 16, 48, 96, 22>
            <<<1, nullptr, stream>>>(reinterpret_cast<float *>(out), reinterpret_cast<bfloat16_t *>(src0),
                                     reinterpret_cast<bfloat16_t *>(src1));
    } else if constexpr (tilingKey == 6) {
        TStoreAcc2gmNz2nz<0, float, bfloat16_t, bfloat16_t, 4, 4, 3, 16, 16, 4, 4, 3, 16, 16, 48, 256, 32>
            <<<1, nullptr, stream>>>(reinterpret_cast<bfloat16_t *>(out), reinterpret_cast<bfloat16_t *>(src0),
                                     reinterpret_cast<bfloat16_t *>(src1));
    } else if constexpr (tilingKey == 7) {
        TStoreAcc2gmNz2nz<1, int32_t, int32_t, int8_t, 2, 3, 4, 16, 16, 2, 3, 4, 16, 16, 64, 96, 30>
            <<<1, nullptr, stream>>>(reinterpret_cast<int32_t *>(out), reinterpret_cast<int8_t *>(src0),
                                     reinterpret_cast<int8_t *>(src1));
    } else if constexpr (tilingKey == 8) {
        TStoreAcc2gmNz2nz<0, float, float, float, 3, 8, 4, 16, 8, 3, 8, 4, 16, 8, 64, 192, 43><<<1, nullptr, stream>>>(
            reinterpret_cast<float *>(out), reinterpret_cast<float *>(src0), reinterpret_cast<float *>(src1));
    } else if constexpr (tilingKey == 21) {
        TStoreAcc2gmNz2nz<0, float, float, half, 5, 1, 10, 16, 16, 5, 1, 10, 16, 16, 160, 80, 51, 1>
            <<<1, nullptr, stream>>>(reinterpret_cast<float *>(out), reinterpret_cast<half *>(src0),
                                     reinterpret_cast<half *>(src1));
    }
}

template <int tilingKey>
void LaunchTStoreAcc2gmScalarNz2nd(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream, float scalarQuant)
{
    if constexpr (tilingKey == 1) {
        TStoreAcc2gmScalarNz2nd<0, int32_t, half, int8_t, 1, 1, 1, 64, 64, 1, 1, 1, 64, 64, 64, 64, 64>
            <<<1, nullptr, stream>>>(reinterpret_cast<half *>(out), reinterpret_cast<int8_t *>(src0),
                                     reinterpret_cast<int8_t *>(src1), scalarQuant);
    } else if constexpr (tilingKey == 2) {
        TStoreAcc2gmScalarNz2nd<0, int32_t, int8_t, int8_t, 1, 1, 1, 31, 32, 1, 1, 1, 31, 32, 31, 32, 26>
            <<<1, nullptr, stream>>>(reinterpret_cast<int8_t *>(out), reinterpret_cast<int8_t *>(src0),
                                     reinterpret_cast<int8_t *>(src1), scalarQuant);
    } else if constexpr (tilingKey == 3) {
        TStoreAcc2gmScalarNz2nd<0, int32_t, uint8_t, int8_t, 1, 1, 1, 16, 32, 1, 1, 1, 16, 32, 16, 32, 17>
            <<<1, nullptr, stream>>>(reinterpret_cast<uint8_t *>(out), reinterpret_cast<int8_t *>(src0),
                                     reinterpret_cast<int8_t *>(src1), scalarQuant);
    } else if constexpr (tilingKey == 4) {
        TStoreAcc2gmScalarNz2nd<0, int32_t, bfloat16_t, int8_t, 1, 1, 1, 17, 32, 1, 1, 1, 17, 32, 17, 32, 31>
            <<<1, nullptr, stream>>>(reinterpret_cast<bfloat16_t *>(out), reinterpret_cast<int8_t *>(src0),
                                     reinterpret_cast<int8_t *>(src1), scalarQuant);
    } else if constexpr (tilingKey == 5) {
        TStoreAcc2gmScalarNz2nd<1, float, int8_t, half, 1, 1, 1, 25, 35, 1, 1, 1, 25, 35, 25, 35, 32>
            <<<1, nullptr, stream>>>(reinterpret_cast<int8_t *>(out), reinterpret_cast<half *>(src0),
                                     reinterpret_cast<half *>(src1), scalarQuant);
    } else if constexpr (tilingKey == 6) {
        TStoreAcc2gmScalarNz2nd<0, float, uint8_t, float, 1, 1, 1, 16, 20, 1, 1, 1, 16, 20, 16, 20, 25>
            <<<1, nullptr, stream>>>(reinterpret_cast<uint8_t *>(out), reinterpret_cast<float *>(src0),
                                     reinterpret_cast<float *>(src1), scalarQuant);
    } else if constexpr (tilingKey == 7) {
        TStoreAcc2gmScalarNz2nd<0, float, half, half, 1, 1, 1, 49, 65, 1, 1, 1, 49, 65, 49, 65, 37>
            <<<1, nullptr, stream>>>(reinterpret_cast<half *>(out), reinterpret_cast<half *>(src0),
                                     reinterpret_cast<half *>(src1), scalarQuant);
    } else if constexpr (tilingKey == 8) {
        TStoreAcc2gmScalarNz2nd<0, float, bfloat16_t, half, 1, 1, 1, 160, 79, 1, 1, 1, 160, 79, 160, 79, 51>
            <<<1, nullptr, stream>>>(reinterpret_cast<bfloat16_t *>(out), reinterpret_cast<half *>(src0),
                                     reinterpret_cast<half *>(src1), scalarQuant);
    } else if constexpr (tilingKey == 9) {
        TStoreAcc2gmScalarNz2nd<0, float, hifloat8_t, half, 1, 1, 1, 17, 57, 1, 1, 1, 17, 57, 17, 57, 33>
            <<<1, nullptr, stream>>>(reinterpret_cast<hifloat8_t *>(out), reinterpret_cast<half *>(src0),
                                     reinterpret_cast<half *>(src1), scalarQuant);
    } else if constexpr (tilingKey == 10) {
        TStoreAcc2gmScalarNz2nd<0, float, float8_e4m3_t, float8_e4m3_t, 1, 1, 1, 32, 32, 1, 1, 1, 32, 32, 32, 32, 31>
            <<<1, nullptr, stream>>>(reinterpret_cast<float8_e4m3_t *>(out), reinterpret_cast<float8_e4m3_t *>(src0),
                                     reinterpret_cast<float8_e4m3_t *>(src1), scalarQuant);
    } else if constexpr (tilingKey == 21) {
        TStoreAcc2gmScalarNz2nd<0, float, int8_t, half, 1, 1, 1, 77, 34, 1, 1, 1, 77, 34, 77, 34, 81, 1>
            <<<1, nullptr, stream>>>(reinterpret_cast<int8_t *>(out), reinterpret_cast<half *>(src0),
                                     reinterpret_cast<half *>(src1), scalarQuant);
    } else if constexpr (tilingKey == 31) {
        TStoreAcc2gmScalarNz2nd<0, float, int8_t, hifloat8_t, 1, 8, 16, 5, 64, 1, 8, 16, 5, 64, 640, 64, 96, 1, 1>
            <<<1, nullptr, stream>>>(reinterpret_cast<int8_t *>(out), reinterpret_cast<hifloat8_t *>(src0),
                                     reinterpret_cast<hifloat8_t *>(src1), scalarQuant);
    } else if constexpr (tilingKey == 41) {
        TStoreAcc2gmScalarNz2nd<0, float, int8_t, hifloat8_t, 1, 1, 64, 16, 40, 1, 1, 64, 16, 40, 640, 64, 96, 1, 2>
            <<<1, nullptr, stream>>>(reinterpret_cast<int8_t *>(out), reinterpret_cast<hifloat8_t *>(src0),
                                     reinterpret_cast<hifloat8_t *>(src1), scalarQuant);
    } else if constexpr (tilingKey == 51) {
        TStoreAcc2gmScalarNz2nd<0, float, int8_t, hifloat8_t, 1, 64, 1, 16, 40, 1, 64, 4, 16, 40, 640, 64, 96, 1, 3>
            <<<1, nullptr, stream>>>(reinterpret_cast<int8_t *>(out), reinterpret_cast<hifloat8_t *>(src0),
                                     reinterpret_cast<hifloat8_t *>(src1), scalarQuant);
    }
}

template <int tilingKey>
void LaunchTStoreAcc2gmScalarNz2nz(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream, float scalarQuant)
{
    if constexpr (tilingKey == 1) {
        TStoreAcc2gmScalarNz2nz<0, int32_t, half, int8_t, 1, 2, 4, 16, 16, 1, 2, 4, 16, 16, 64, 32, 64>
            <<<1, nullptr, stream>>>(reinterpret_cast<half *>(out), reinterpret_cast<int8_t *>(src0),
                                     reinterpret_cast<int8_t *>(src1), scalarQuant);
    } else if constexpr (tilingKey == 2) {
        TStoreAcc2gmScalarNz2nz<0, int32_t, int8_t, int8_t, 1, 1, 2, 16, 32, 1, 1, 2, 16, 32, 32, 32, 32>
            <<<1, nullptr, stream>>>(reinterpret_cast<int8_t *>(out), reinterpret_cast<int8_t *>(src0),
                                     reinterpret_cast<int8_t *>(src1), scalarQuant);
    } else if constexpr (tilingKey == 3) {
        TStoreAcc2gmScalarNz2nz<0, int32_t, uint8_t, int8_t, 2, 1, 10, 16, 32, 2, 1, 10, 16, 32, 160, 64, 17>
            <<<1, nullptr, stream>>>(reinterpret_cast<uint8_t *>(out), reinterpret_cast<int8_t *>(src0),
                                     reinterpret_cast<int8_t *>(src1), scalarQuant);
    } else if constexpr (tilingKey == 4) {
        TStoreAcc2gmScalarNz2nz<0, int32_t, bfloat16_t, int8_t, 1, 6, 1, 16, 16, 1, 6, 1, 16, 16, 16, 96, 29>
            <<<1, nullptr, stream>>>(reinterpret_cast<bfloat16_t *>(out), reinterpret_cast<int8_t *>(src0),
                                     reinterpret_cast<int8_t *>(src1), scalarQuant);
    } else if constexpr (tilingKey == 5) {
        TStoreAcc2gmScalarNz2nz<0, float, int8_t, float, 1, 2, 1, 16, 32, 1, 2, 1, 16, 32, 16, 64, 15>
            <<<1, nullptr, stream>>>(reinterpret_cast<int8_t *>(out), reinterpret_cast<float *>(src0),
                                     reinterpret_cast<float *>(src1), scalarQuant);
    } else if constexpr (tilingKey == 6) {
        TStoreAcc2gmScalarNz2nz<0, float, uint8_t, bfloat16_t, 1, 2, 2, 16, 32, 1, 2, 2, 16, 32, 32, 64, 16>
            <<<1, nullptr, stream>>>(reinterpret_cast<uint8_t *>(out), reinterpret_cast<bfloat16_t *>(src0),
                                     reinterpret_cast<bfloat16_t *>(src1), scalarQuant);
    } else if constexpr (tilingKey == 7) {
        TStoreAcc2gmScalarNz2nz<0, float, half, half, 2, 4, 8, 16, 16, 2, 4, 8, 16, 16, 128, 128, 37>
            <<<1, nullptr, stream>>>(reinterpret_cast<half *>(out), reinterpret_cast<half *>(src0),
                                     reinterpret_cast<half *>(src1), scalarQuant);
    } else if constexpr (tilingKey == 8) {
        TStoreAcc2gmScalarNz2nz<0, float, bfloat16_t, half, 2, 1, 4, 16, 16, 2, 1, 4, 16, 16, 64, 32, 31>
            <<<1, nullptr, stream>>>(reinterpret_cast<bfloat16_t *>(out), reinterpret_cast<half *>(src0),
                                     reinterpret_cast<half *>(src1), scalarQuant);
    } else if constexpr (tilingKey == 9) {
        TStoreAcc2gmScalarNz2nz<0, float, hifloat8_t, half, 2, 1, 5, 16, 32, 2, 1, 5, 16, 32, 80, 64, 10>
            <<<1, nullptr, stream>>>(reinterpret_cast<hifloat8_t *>(out), reinterpret_cast<half *>(src0),
                                     reinterpret_cast<half *>(src1), scalarQuant);
    } else if constexpr (tilingKey == 21) {
        TStoreAcc2gmScalarNz2nz<0, int32_t, int8_t, int8_t, 1, 1, 6, 16, 32, 1, 1, 6, 16, 32, 96, 32, 159, 1>
            <<<1, nullptr, stream>>>(reinterpret_cast<int8_t *>(out), reinterpret_cast<int8_t *>(src0),
                                     reinterpret_cast<int8_t *>(src1), scalarQuant);
    }
}

template <int tilingKey>
void LaunchTStoreAcc2gmVectorNz2nd(uint8_t *out, uint8_t *src0, uint8_t *src1, uint8_t *quantTensor, void *stream)
{
    if constexpr (tilingKey == 1) {
        TStoreAcc2gmVectorNz2nd<0, int32_t, half, int8_t, 1, 1, 1, 55, 88, 1, 1, 1, 55, 88, 55, 88, 32>
            <<<1, nullptr, stream>>>(reinterpret_cast<half *>(out), reinterpret_cast<int8_t *>(src0),
                                     reinterpret_cast<int8_t *>(src1), reinterpret_cast<uint64_t *>(quantTensor));
    } else if constexpr (tilingKey == 2) {
        TStoreAcc2gmVectorNz2nd<0, int32_t, int8_t, int8_t, 1, 1, 1, 34, 85, 1, 1, 1, 34, 85, 34, 85, 19>
            <<<1, nullptr, stream>>>(reinterpret_cast<int8_t *>(out), reinterpret_cast<int8_t *>(src0),
                                     reinterpret_cast<int8_t *>(src1), reinterpret_cast<uint64_t *>(quantTensor));
    } else if constexpr (tilingKey == 3) {
        TStoreAcc2gmVectorNz2nd<0, int32_t, uint8_t, int8_t, 1, 1, 1, 31, 32, 1, 1, 1, 31, 32, 31, 32, 29>
            <<<1, nullptr, stream>>>(reinterpret_cast<uint8_t *>(out), reinterpret_cast<int8_t *>(src0),
                                     reinterpret_cast<int8_t *>(src1), reinterpret_cast<uint64_t *>(quantTensor));
    } else if constexpr (tilingKey == 4) {
        TStoreAcc2gmVectorNz2nd<0, int32_t, bfloat16_t, int8_t, 1, 1, 1, 45, 81, 1, 1, 1, 45, 81, 45, 81, 26>
            <<<1, nullptr, stream>>>(reinterpret_cast<bfloat16_t *>(out), reinterpret_cast<int8_t *>(src0),
                                     reinterpret_cast<int8_t *>(src1), reinterpret_cast<uint64_t *>(quantTensor));
    } else if constexpr (tilingKey == 5) {
        TStoreAcc2gmVectorNz2nd<0, float, half, half, 1, 1, 1, 15, 15, 1, 1, 1, 15, 15, 15, 15, 31>
            <<<1, nullptr, stream>>>(reinterpret_cast<half *>(out), reinterpret_cast<half *>(src0),
                                     reinterpret_cast<half *>(src1), reinterpret_cast<uint64_t *>(quantTensor));
    } else if constexpr (tilingKey == 6) {
        TStoreAcc2gmVectorNz2nd<0, float, bfloat16_t, half, 1, 1, 1, 31, 95, 1, 1, 1, 31, 95, 31, 95, 17>
            <<<1, nullptr, stream>>>(reinterpret_cast<bfloat16_t *>(out), reinterpret_cast<half *>(src0),
                                     reinterpret_cast<half *>(src1), reinterpret_cast<uint64_t *>(quantTensor));
    } else if constexpr (tilingKey == 7) {
        TStoreAcc2gmVectorNz2nd<0, float, int8_t, half, 1, 1, 1, 33, 65, 1, 1, 1, 33, 65, 33, 65, 25>
            <<<1, nullptr, stream>>>(reinterpret_cast<int8_t *>(out), reinterpret_cast<half *>(src0),
                                     reinterpret_cast<half *>(src1), reinterpret_cast<uint64_t *>(quantTensor));
    } else if constexpr (tilingKey == 8) {
        TStoreAcc2gmVectorNz2nd<0, float, uint8_t, half, 1, 1, 1, 19, 32, 1, 1, 1, 19, 32, 19, 32, 23>
            <<<1, nullptr, stream>>>(reinterpret_cast<uint8_t *>(out), reinterpret_cast<half *>(src0),
                                     reinterpret_cast<half *>(src1), reinterpret_cast<uint64_t *>(quantTensor));
    } else if constexpr (tilingKey == 9) {
        TStoreAcc2gmVectorNz2nd<0, float, hifloat8_t, half, 1, 1, 1, 99, 100, 1, 1, 1, 99, 100, 99, 100, 15>
            <<<1, nullptr, stream>>>(reinterpret_cast<hifloat8_t *>(out), reinterpret_cast<half *>(src0),
                                     reinterpret_cast<half *>(src1), reinterpret_cast<uint64_t *>(quantTensor));
    } else if constexpr (tilingKey == 21) {
        TStoreAcc2gmVectorNz2nd<0, float, int8_t, half, 1, 1, 1, 85, 77, 1, 1, 1, 85, 77, 85, 77, 66, 1>
            <<<1, nullptr, stream>>>(reinterpret_cast<int8_t *>(out), reinterpret_cast<half *>(src0),
                                     reinterpret_cast<half *>(src1), reinterpret_cast<uint64_t *>(quantTensor));
    } else if constexpr (tilingKey == 31) {
        TStoreAcc2gmVectorNz2nd<0, float, half, float8_e4m3_t, 1, 2, 8, 22, 64, 1, 2, 8, 22, 64, 352, 64, 32, 0, 1>
            <<<1, nullptr, stream>>>(reinterpret_cast<half *>(out), reinterpret_cast<float8_e4m3_t *>(src0),
                                     reinterpret_cast<float8_e4m3_t *>(src1),
                                     reinterpret_cast<uint64_t *>(quantTensor));
    } else if constexpr (tilingKey == 32) {
        TStoreAcc2gmVectorNz2nd<0, float, float, half, 1, 1, 64, 4, 128, 1, 1, 64, 4, 128, 256, 128, 32, 1, 1>
            <<<1, nullptr, stream>>>(reinterpret_cast<float *>(out), reinterpret_cast<half *>(src0),
                                     reinterpret_cast<half *>(src1), reinterpret_cast<uint64_t *>(quantTensor));
    } else if constexpr (tilingKey == 41) {
        TStoreAcc2gmVectorNz2nd<0, float, half, float8_e4m3_t, 1, 1, 64, 8, 44, 1, 1, 64, 8, 44, 352, 64, 32, 0, 2>
            <<<1, nullptr, stream>>>(reinterpret_cast<half *>(out), reinterpret_cast<float8_e4m3_t *>(src0),
                                     reinterpret_cast<float8_e4m3_t *>(src1),
                                     reinterpret_cast<uint64_t *>(quantTensor));
    } else if constexpr (tilingKey == 42) {
        TStoreAcc2gmVectorNz2nd<0, float, float, half, 1, 1, 128, 64, 4, 1, 1, 128, 64, 4, 256, 128, 32, 1, 2>
            <<<1, nullptr, stream>>>(reinterpret_cast<float *>(out), reinterpret_cast<half *>(src0),
                                     reinterpret_cast<half *>(src1), reinterpret_cast<uint64_t *>(quantTensor));
    } else if constexpr (tilingKey == 51) {
        TStoreAcc2gmVectorNz2nd<0, float, half, float8_e4m3_t, 1, 64, 1, 8, 44, 1, 64, 3, 8, 44, 352, 64, 32, 0, 3>
            <<<1, nullptr, stream>>>(reinterpret_cast<half *>(out), reinterpret_cast<float8_e4m3_t *>(src0),
                                     reinterpret_cast<float8_e4m3_t *>(src1),
                                     reinterpret_cast<uint64_t *>(quantTensor));
    } else if constexpr (tilingKey == 52) {
        TStoreAcc2gmVectorNz2nd<0, float, float, half, 1, 128, 1, 64, 4, 1, 128, 4, 64, 4, 256, 128, 32, 1, 3>
            <<<1, nullptr, stream>>>(reinterpret_cast<float *>(out), reinterpret_cast<half *>(src0),
                                     reinterpret_cast<half *>(src1), reinterpret_cast<uint64_t *>(quantTensor));
    }
}

template <int tilingKey>
void LaunchTStoreAcc2gmVectorNz2nz(uint8_t *out, uint8_t *src0, uint8_t *src1, uint8_t *quantTensor, void *stream)
{
    if constexpr (tilingKey == 1) {
        TStoreAcc2gmVectorNz2nz<0, int32_t, half, int8_t, 4, 2, 16, 16, 16, 4, 2, 16, 16, 16, 256, 128, 63>
            <<<1, nullptr, stream>>>(reinterpret_cast<half *>(out), reinterpret_cast<int8_t *>(src0),
                                     reinterpret_cast<int8_t *>(src1), reinterpret_cast<uint64_t *>(quantTensor));
    } else if constexpr (tilingKey == 2) {
        TStoreAcc2gmVectorNz2nz<0, int32_t, int8_t, int8_t, 1, 1, 2, 16, 32, 1, 1, 2, 16, 32, 32, 32, 31>
            <<<1, nullptr, stream>>>(reinterpret_cast<int8_t *>(out), reinterpret_cast<int8_t *>(src0),
                                     reinterpret_cast<int8_t *>(src1), reinterpret_cast<uint64_t *>(quantTensor));
    } else if constexpr (tilingKey == 3) {
        TStoreAcc2gmVectorNz2nz<0, int32_t, uint8_t, int8_t, 1, 1, 3, 16, 32, 1, 1, 3, 16, 32, 48, 32, 23>
            <<<1, nullptr, stream>>>(reinterpret_cast<uint8_t *>(out), reinterpret_cast<int8_t *>(src0),
                                     reinterpret_cast<int8_t *>(src1), reinterpret_cast<uint64_t *>(quantTensor));
    } else if constexpr (tilingKey == 4) {
        TStoreAcc2gmVectorNz2nz<0, int32_t, bfloat16_t, int8_t, 3, 2, 5, 16, 16, 3, 2, 5, 16, 16, 80, 96, 49>
            <<<1, nullptr, stream>>>(reinterpret_cast<bfloat16_t *>(out), reinterpret_cast<int8_t *>(src0),
                                     reinterpret_cast<int8_t *>(src1), reinterpret_cast<uint64_t *>(quantTensor));
    } else if constexpr (tilingKey == 5) {
        TStoreAcc2gmVectorNz2nz<0, float, half, half, 2, 3, 8, 16, 16, 2, 3, 8, 16, 16, 128, 96, 31>
            <<<1, nullptr, stream>>>(reinterpret_cast<half *>(out), reinterpret_cast<half *>(src0),
                                     reinterpret_cast<half *>(src1), reinterpret_cast<uint64_t *>(quantTensor));
    } else if constexpr (tilingKey == 6) {
        TStoreAcc2gmVectorNz2nz<0, float, bfloat16_t, half, 2, 3, 2, 16, 16, 2, 3, 2, 16, 16, 32, 96, 17>
            <<<1, nullptr, stream>>>(reinterpret_cast<bfloat16_t *>(out), reinterpret_cast<half *>(src0),
                                     reinterpret_cast<half *>(src1), reinterpret_cast<uint64_t *>(quantTensor));
    } else if constexpr (tilingKey == 7) {
        TStoreAcc2gmVectorNz2nz<0, float, int8_t, half, 2, 1, 2, 16, 32, 2, 1, 2, 16, 32, 32, 64, 25>
            <<<1, nullptr, stream>>>(reinterpret_cast<int8_t *>(out), reinterpret_cast<half *>(src0),
                                     reinterpret_cast<half *>(src1), reinterpret_cast<uint64_t *>(quantTensor));
    } else if constexpr (tilingKey == 8) {
        TStoreAcc2gmVectorNz2nz<0, float, uint8_t, half, 1, 1, 1, 16, 32, 1, 1, 1, 16, 32, 16, 32, 23>
            <<<1, nullptr, stream>>>(reinterpret_cast<uint8_t *>(out), reinterpret_cast<half *>(src0),
                                     reinterpret_cast<half *>(src1), reinterpret_cast<uint64_t *>(quantTensor));
    } else if constexpr (tilingKey == 9) {
        TStoreAcc2gmVectorNz2nz<0, float, hifloat8_t, half, 3, 1, 9, 16, 32, 3, 1, 9, 16, 32, 144, 96, 37>
            <<<1, nullptr, stream>>>(reinterpret_cast<hifloat8_t *>(out), reinterpret_cast<half *>(src0),
                                     reinterpret_cast<half *>(src1), reinterpret_cast<uint64_t *>(quantTensor));
    } else if constexpr (tilingKey == 21) {
        TStoreAcc2gmVectorNz2nz<0, int32_t, int8_t, int8_t, 4, 1, 8, 16, 32, 4, 1, 8, 16, 32, 128, 128, 123, 1>
            <<<1, nullptr, stream>>>(reinterpret_cast<int8_t *>(out), reinterpret_cast<int8_t *>(src0),
                                     reinterpret_cast<int8_t *>(src1), reinterpret_cast<uint64_t *>(quantTensor));
    }
}

template void LaunchTStoreAcc2gmNz2nd<1>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void LaunchTStoreAcc2gmNz2nd<2>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void LaunchTStoreAcc2gmNz2nd<3>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void LaunchTStoreAcc2gmNz2nd<4>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void LaunchTStoreAcc2gmNz2nd<5>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void LaunchTStoreAcc2gmNz2nd<6>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void LaunchTStoreAcc2gmNz2nd<7>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void LaunchTStoreAcc2gmNz2nz<1>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void LaunchTStoreAcc2gmNz2nz<2>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void LaunchTStoreAcc2gmNz2nz<3>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void LaunchTStoreAcc2gmNz2nz<4>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void LaunchTStoreAcc2gmNz2nz<5>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void LaunchTStoreAcc2gmNz2nz<6>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void LaunchTStoreAcc2gmNz2nz<7>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void LaunchTStoreAcc2gmNz2nz<8>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);

template void LaunchTStoreAcc2gmScalarNz2nd<1>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream,
                                               float scalarQuant);
template void LaunchTStoreAcc2gmScalarNz2nd<2>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream,
                                               float scalarQuant);
template void LaunchTStoreAcc2gmScalarNz2nd<3>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream,
                                               float scalarQuant);
template void LaunchTStoreAcc2gmScalarNz2nd<4>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream,
                                               float scalarQuant);
template void LaunchTStoreAcc2gmScalarNz2nd<5>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream,
                                               float scalarQuant);
template void LaunchTStoreAcc2gmScalarNz2nd<6>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream,
                                               float scalarQuant);
template void LaunchTStoreAcc2gmScalarNz2nd<7>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream,
                                               float scalarQuant);
template void LaunchTStoreAcc2gmScalarNz2nd<8>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream,
                                               float scalarQuant);
template void LaunchTStoreAcc2gmScalarNz2nd<9>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream,
                                               float scalarQuant);
template void LaunchTStoreAcc2gmScalarNz2nd<10>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream,
                                                float scalarQuant);

template void LaunchTStoreAcc2gmScalarNz2nz<1>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream,
                                               float scalarQuant);
template void LaunchTStoreAcc2gmScalarNz2nz<2>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream,
                                               float scalarQuant);
template void LaunchTStoreAcc2gmScalarNz2nz<3>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream,
                                               float scalarQuant);
template void LaunchTStoreAcc2gmScalarNz2nz<4>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream,
                                               float scalarQuant);
template void LaunchTStoreAcc2gmScalarNz2nz<5>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream,
                                               float scalarQuant);
template void LaunchTStoreAcc2gmScalarNz2nz<6>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream,
                                               float scalarQuant);
template void LaunchTStoreAcc2gmScalarNz2nz<7>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream,
                                               float scalarQuant);
template void LaunchTStoreAcc2gmScalarNz2nz<8>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream,
                                               float scalarQuant);
template void LaunchTStoreAcc2gmScalarNz2nz<9>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream,
                                               float scalarQuant);

template void LaunchTStoreAcc2gmVectorNz2nd<1>(uint8_t *out, uint8_t *src0, uint8_t *src1, uint8_t *quantTensor,
                                               void *stream);
template void LaunchTStoreAcc2gmVectorNz2nd<2>(uint8_t *out, uint8_t *src0, uint8_t *src1, uint8_t *quantTensor,
                                               void *stream);
template void LaunchTStoreAcc2gmVectorNz2nd<3>(uint8_t *out, uint8_t *src0, uint8_t *src1, uint8_t *quantTensor,
                                               void *stream);
template void LaunchTStoreAcc2gmVectorNz2nd<4>(uint8_t *out, uint8_t *src0, uint8_t *src1, uint8_t *quantTensor,
                                               void *stream);
template void LaunchTStoreAcc2gmVectorNz2nd<5>(uint8_t *out, uint8_t *src0, uint8_t *src1, uint8_t *quantTensor,
                                               void *stream);
template void LaunchTStoreAcc2gmVectorNz2nd<6>(uint8_t *out, uint8_t *src0, uint8_t *src1, uint8_t *quantTensor,
                                               void *stream);
template void LaunchTStoreAcc2gmVectorNz2nd<7>(uint8_t *out, uint8_t *src0, uint8_t *src1, uint8_t *quantTensor,
                                               void *stream);
template void LaunchTStoreAcc2gmVectorNz2nd<8>(uint8_t *out, uint8_t *src0, uint8_t *src1, uint8_t *quantTensor,
                                               void *stream);
template void LaunchTStoreAcc2gmVectorNz2nd<9>(uint8_t *out, uint8_t *src0, uint8_t *src1, uint8_t *quantTensor,
                                               void *stream);
template void LaunchTStoreAcc2gmVectorNz2nz<1>(uint8_t *out, uint8_t *src0, uint8_t *src1, uint8_t *quantTensor,
                                               void *stream);
template void LaunchTStoreAcc2gmVectorNz2nz<2>(uint8_t *out, uint8_t *src0, uint8_t *src1, uint8_t *quantTensor,
                                               void *stream);
template void LaunchTStoreAcc2gmVectorNz2nz<3>(uint8_t *out, uint8_t *src0, uint8_t *src1, uint8_t *quantTensor,
                                               void *stream);
template void LaunchTStoreAcc2gmVectorNz2nz<4>(uint8_t *out, uint8_t *src0, uint8_t *src1, uint8_t *quantTensor,
                                               void *stream);
template void LaunchTStoreAcc2gmVectorNz2nz<5>(uint8_t *out, uint8_t *src0, uint8_t *src1, uint8_t *quantTensor,
                                               void *stream);
template void LaunchTStoreAcc2gmVectorNz2nz<6>(uint8_t *out, uint8_t *src0, uint8_t *src1, uint8_t *quantTensor,
                                               void *stream);
template void LaunchTStoreAcc2gmVectorNz2nz<7>(uint8_t *out, uint8_t *src0, uint8_t *src1, uint8_t *quantTensor,
                                               void *stream);
template void LaunchTStoreAcc2gmVectorNz2nz<8>(uint8_t *out, uint8_t *src0, uint8_t *src1, uint8_t *quantTensor,
                                               void *stream);
template void LaunchTStoreAcc2gmVectorNz2nz<9>(uint8_t *out, uint8_t *src0, uint8_t *src1, uint8_t *quantTensor,
                                               void *stream);

template void LaunchTStoreAcc2gmNz2nd<21>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void LaunchTStoreAcc2gmNz2nz<21>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);

template void LaunchTStoreAcc2gmScalarNz2nd<21>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream,
                                                float scalarQuant);
template void LaunchTStoreAcc2gmScalarNz2nz<21>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream,
                                                float scalarQuant);

template void LaunchTStoreAcc2gmVectorNz2nd<21>(uint8_t *out, uint8_t *src0, uint8_t *src1, uint8_t *quantTensor,
                                                void *stream);
template void LaunchTStoreAcc2gmVectorNz2nz<21>(uint8_t *out, uint8_t *src0, uint8_t *src1, uint8_t *quantTensor,
                                                void *stream);

template void LaunchTStoreAcc2gmNz2nd<31>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void LaunchTStoreAcc2gmNz2nd<32>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void LaunchTStoreAcc2gmNz2nd<33>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void LaunchTStoreAcc2gmNz2nd<34>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void LaunchTStoreAcc2gmScalarNz2nd<31>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream,
                                                float scalarQuant);
template void LaunchTStoreAcc2gmVectorNz2nd<31>(uint8_t *out, uint8_t *src0, uint8_t *src1, uint8_t *quantTensor,
                                                void *stream);
template void LaunchTStoreAcc2gmVectorNz2nd<32>(uint8_t *out, uint8_t *src0, uint8_t *src1, uint8_t *quantTensor,
                                                void *stream);

template void LaunchTStoreAcc2gmNz2nd<41>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void LaunchTStoreAcc2gmNz2nd<42>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void LaunchTStoreAcc2gmNz2nd<43>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void LaunchTStoreAcc2gmNz2nd<44>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void LaunchTStoreAcc2gmScalarNz2nd<41>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream,
                                                float scalarQuant);
template void LaunchTStoreAcc2gmScalarNz2nd<51>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream,
                                                float scalarQuant);
template void LaunchTStoreAcc2gmVectorNz2nd<41>(uint8_t *out, uint8_t *src0, uint8_t *src1, uint8_t *quantTensor,
                                                void *stream);
template void LaunchTStoreAcc2gmVectorNz2nd<42>(uint8_t *out, uint8_t *src0, uint8_t *src1, uint8_t *quantTensor,
                                                void *stream);
template void LaunchTStoreAcc2gmVectorNz2nd<51>(uint8_t *out, uint8_t *src0, uint8_t *src1, uint8_t *quantTensor,
                                                void *stream);
template void LaunchTStoreAcc2gmVectorNz2nd<52>(uint8_t *out, uint8_t *src0, uint8_t *src1, uint8_t *quantTensor,
                                                void *stream);
template void LaunchTStoreAcc2gmNz2nd<51>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void LaunchTStoreAcc2gmNz2nd<52>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void LaunchTStoreAcc2gmNz2nd<53>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
template void LaunchTStoreAcc2gmNz2nd<54>(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);
