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
          int gWholeShape4, int validM, int validN, int validK, int reluMode = 0>
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
    using DynShapeDim5 = pto::Shape<gShape0, gShape1, gShape2, gShape3, gShape4>;
    using DynStridDim5 = pto::Stride<gStride[0], gStride[1], gStride[2], gStride[3], gStride[4]>;
    using GlobalDataSrc0 =
        GlobalTensor<srcDataType, pto::Shape<1, 1, 1, validM, validK>,
                     pto::Stride<1 * validM * validK, 1 * validM * validK, validM * validK, validK, 1>>;
    using GlobalDataSrc1 =
        GlobalTensor<srcDataType, pto::Shape<1, 1, 1, validK, validN>,
                     pto::Stride<1 * validK * validN, 1 * validK * validN, validK * validN, validN, 1>>;
    using GlobalDataOut = GlobalTensor<dstDataType, DynShapeDim5, DynStridDim5>;

    GlobalDataSrc0 src0Global(src0);
    GlobalDataSrc1 src1Global(src1);
    GlobalDataOut dstGlobal(out);

    using TileMatAData = Tile<TileType::Mat, srcDataType, M, K, BLayout::ColMajor, M, K, SLayout::RowMajor, 512>;
    using TileMatBData = Tile<TileType::Mat, srcDataType, K, N, BLayout::ColMajor, K, N, SLayout::RowMajor, 512>;
    using LeftTile = Tile<TileType::Left, srcDataType, M, K, BLayout::RowMajor, validM, validK, SLayout::RowMajor, 512>;
    using RightTile = TileRight<srcDataType, K, N, validK, validN>;
    using AccTile = Tile<TileType::Acc, accDataType, Rows, Cols, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024>;

    uint32_t aMatSize = M * K * sizeof(srcDataType);
    TileMatAData aMatTile;
    TileMatBData bMatTile;
    TASSIGN(aMatTile, 0x0);
    TASSIGN(bMatTile, aMatSize);

    LeftTile aTile;
    RightTile bTile;
    AccTile cTile(gShape3, gShape4);
    TASSIGN(aTile, 0x0);
    TASSIGN(bTile, 0x0);
    TASSIGN(cTile, 0x0);

    TLOAD(aMatTile, src0Global);
    TLOAD(bMatTile, src1Global);

#ifndef __PTO_AUTO__
    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
#endif
    TMOV(aTile, aMatTile);
    TMOV(bTile, bMatTile);
#ifndef __PTO_AUTO__
    set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
    wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
#endif
    TMATMUL(cTile, aTile, bTile);
#ifndef __PTO_AUTO__
    set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
#endif
    constexpr AtomicType atomicTypeEnum = atomicType == 1 ? AtomicType::AtomicAdd : AtomicType::AtomicNone;
    if constexpr (reluMode == 0) {
        TSTORE<AccTile, GlobalDataOut, atomicTypeEnum>(dstGlobal, cTile);
    } else if constexpr (reluMode == 1) {
        constexpr ReluPreMode reluPreMode = ReluPreMode::NormalRelu;
        TSTORE<AccTile, GlobalDataOut, atomicTypeEnum, reluPreMode>(dstGlobal, cTile);
    }
#ifndef __PTO_AUTO__
    set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
    wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
#endif
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
    using GlobalDataOut = GlobalTensor<dstDataType, DynShapeDim5, DynStridDim5, Layout::NZ>;
    using GlobalDataSrc0 =
        GlobalTensor<srcDataType, pto::Shape<1, 1, 1, validM, validK>,
                     pto::Stride<1 * validM * validK, 1 * validM * validK, validM * validK, validK, 1>>;
    using GlobalDataSrc1 =
        GlobalTensor<srcDataType, pto::Shape<1, 1, 1, validK, validN>,
                     pto::Stride<1 * validK * validN, 1 * validK * validN, validK * validN, validN, 1>>;

    GlobalDataSrc0 src0Global(src0);
    GlobalDataSrc1 src1Global(src1);
    GlobalDataOut dstGlobal(out);

    using TileMatAData = Tile<TileType::Mat, srcDataType, M, K, BLayout::ColMajor, M, K, SLayout::RowMajor, 512>;
    using TileMatBData = Tile<TileType::Mat, srcDataType, K, N, BLayout::ColMajor, K, N, SLayout::RowMajor, 512>;

    using LeftTile = Tile<TileType::Left, srcDataType, M, K, BLayout::RowMajor, validM, validK, SLayout::RowMajor, 512>;
    using RightTile = TileRight<srcDataType, K, N, validK, validN>;
    using AccTile = Tile<TileType::Acc, accDataType, Rows, Cols, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024>;

    TileMatAData aMatTile;
    TileMatBData bMatTile;

    uint32_t aMatSize = M * K * sizeof(srcDataType);

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

#ifndef __PTO_AUTO__
    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
#endif

    TMOV(aTile, aMatTile);
    TMOV(bTile, bMatTile);

#ifndef __PTO_AUTO__
    set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
    wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
#endif
    TMATMUL(cTile, aTile, bTile);
#ifndef __PTO_AUTO__
    set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
#endif
    constexpr AtomicType atomicTypeEnum = atomicType == 1 ? AtomicType::AtomicAdd : AtomicType::AtomicNone;
    if constexpr (reluMode == 0) {
        TSTORE<AccTile, GlobalDataOut, atomicTypeEnum>(dstGlobal, cTile);
    } else if constexpr (reluMode == 1) {
        constexpr ReluPreMode reluPreMode = ReluPreMode::NormalRelu;
        TSTORE<AccTile, GlobalDataOut, atomicTypeEnum, reluPreMode>(dstGlobal, cTile);
    }
#ifndef __PTO_AUTO__
    set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
    wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
#endif
    out = dstGlobal.data();
}

template <int atomicType, typename accDataType, typename dstDataType, typename srcDataType, int gShape0, int gShape1,
          int gShape2, int gShape3, int gShape4, int gWholeShape0, int gWholeShape1, int gWholeShape2, int gWholeShape3,
          int gWholeShape4, int validM, int validN, int validK, int reluMode = 0>
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
    using DynShapeDim5 = pto::Shape<gShape0, gShape1, gShape2, gShape3, gShape4>;
    using DynStridDim5 = pto::Stride<gStride[0], gStride[1], gStride[2], gStride[3], gStride[4]>;
    using GlobalDataSrc0 =
        GlobalTensor<srcDataType, pto::Shape<1, 1, 1, validM, validK>,
                     pto::Stride<1 * validM * validK, 1 * validM * validK, validM * validK, validK, 1>>;
    using GlobalDataSrc1 =
        GlobalTensor<srcDataType, pto::Shape<1, 1, 1, validK, validN>,
                     pto::Stride<1 * validK * validN, 1 * validK * validN, validK * validN, validN, 1>>;
    using GlobalDataOut = GlobalTensor<dstDataType, DynShapeDim5, DynStridDim5>;

    GlobalDataSrc0 src0Global(src0);
    GlobalDataSrc1 src1Global(src1);
    GlobalDataOut dstGlobal(out);

    using TileMatAData = Tile<TileType::Mat, srcDataType, M, K, BLayout::ColMajor, M, K, SLayout::RowMajor, 512>;
    using TileMatBData = Tile<TileType::Mat, srcDataType, K, N, BLayout::ColMajor, K, N, SLayout::RowMajor, 512>;

    using LeftTile = Tile<TileType::Left, srcDataType, M, K, BLayout::RowMajor, validM, validK, SLayout::RowMajor, 512>;
    using RightTile = TileRight<srcDataType, K, N, validK, validN>;
    using AccTile = Tile<TileType::Acc, accDataType, Rows, Cols, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024>;

    uint32_t aMatSize = M * K * sizeof(srcDataType);
    TileMatAData aMatTile;
    TileMatBData bMatTile;
    TASSIGN(aMatTile, 0x0);
    TASSIGN(bMatTile, aMatSize);

    LeftTile aTile;
    RightTile bTile;
    AccTile cTile(gShape3, gShape4);
    TASSIGN(aTile, 0x0);
    TASSIGN(bTile, 0x0);
    TASSIGN(cTile, 0x0);
    TLOAD(aMatTile, src0Global);
    TLOAD(bMatTile, src1Global);

#ifndef __PTO_AUTO__
    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
#endif

    TMOV(aTile, aMatTile);
    TMOV(bTile, bMatTile);

#ifndef __PTO_AUTO__
    set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
    wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
#endif
    TMATMUL(cTile, aTile, bTile);
#ifndef __PTO_AUTO__
    set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
#endif
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
#ifndef __PTO_AUTO__
    set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
    wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
#endif
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
    using GlobalDataOut = GlobalTensor<dstDataType, DynShapeDim5, DynStridDim5, Layout::NZ>;
    using GlobalDataSrc0 =
        GlobalTensor<srcDataType, pto::Shape<1, 1, 1, validM, validK>,
                     pto::Stride<1 * validM * validK, 1 * validM * validK, validM * validK, validK, 1>>;
    using GlobalDataSrc1 =
        GlobalTensor<srcDataType, pto::Shape<1, 1, 1, validK, validN>,
                     pto::Stride<1 * validK * validN, 1 * validK * validN, validK * validN, validN, 1>>;

    int offset = 0;
    GlobalDataSrc0 src0Global(src0);
    GlobalDataSrc1 src1Global(src1);
    GlobalDataOut dstGlobal(out);

    using TileMatAData = Tile<TileType::Mat, srcDataType, M, K, BLayout::ColMajor, M, K, SLayout::RowMajor, 512>;
    using TileMatBData = Tile<TileType::Mat, srcDataType, K, N, BLayout::ColMajor, K, N, SLayout::RowMajor, 512>;

    using LeftTile = Tile<TileType::Left, srcDataType, M, K, BLayout::RowMajor, validM, validK, SLayout::RowMajor, 512>;
    using RightTile = TileRight<srcDataType, K, N, validK, validN>;
    using AccTile = Tile<TileType::Acc, accDataType, Rows, Cols, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024>;

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

#ifndef __PTO_AUTO__
    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
#endif

    TMOV(aTile, aMatTile);
    TMOV(bTile, bMatTile);

#ifndef __PTO_AUTO__
    set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
    wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
#endif
    TMATMUL(cTile, aTile, bTile);
#ifndef __PTO_AUTO__
    set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
#endif
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
#ifndef __PTO_AUTO__
    set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
    wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
#endif
    out = dstGlobal.data();
}

template <int atomicType, typename accDataType, typename dstDataType, typename srcDataType, int gShape0, int gShape1,
          int gShape2, int gShape3, int gShape4, int gWholeShape0, int gWholeShape1, int gWholeShape2, int gWholeShape3,
          int gWholeShape4, int validM, int validN, int validK, int reluMode = 0>
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

    using DynShapeDim5 = pto::Shape<gShape0, gShape1, gShape2, gShape3, gShape4>;
    using DynStridDim5 = pto::Stride<gStride[0], gStride[1], gStride[2], gStride[3], gStride[4]>;
    using GlobalDataSrc0 =
        GlobalTensor<srcDataType, pto::Shape<1, 1, 1, validM, validK>,
                     pto::Stride<1 * validM * validK, 1 * validM * validK, validM * validK, validK, 1>>;
    using GlobalDataSrc1 =
        GlobalTensor<srcDataType, pto::Shape<1, 1, 1, validK, validN>,
                     pto::Stride<1 * validK * validN, 1 * validK * validN, validK * validN, validN, 1>>;
    using GlobalDataSrc2 = GlobalTensor<uint64_t, Shape<1, 1, 1, 1, alignScalingN>,
                                        Stride<1 * alignScalingN, 1 * alignScalingN, alignScalingN, alignScalingN, 1>>;
    using GlobalDataOut = GlobalTensor<dstDataType, DynShapeDim5, DynStridDim5>;

    GlobalDataSrc0 src0Global(src0);
    GlobalDataSrc1 src1Global(src1);
    GlobalDataSrc2 src2Global(quantTensor);
    GlobalDataOut dstGlobal(out);

    using TileMatAData = Tile<TileType::Mat, srcDataType, M, K, BLayout::ColMajor, M, K, SLayout::RowMajor, 512>;
    using TileMatBData = Tile<TileType::Mat, srcDataType, K, N, BLayout::ColMajor, K, N, SLayout::RowMajor, 512>;
    using TileMatScalingData =
        Tile<TileType::Mat, uint64_t, 1, alignScalingN, BLayout::RowMajor, 1, -1, SLayout::NoneBox>;

    using LeftTile = Tile<TileType::Left, srcDataType, M, K, BLayout::RowMajor, validM, validK, SLayout::RowMajor, 512>;
    using RightTile = TileRight<srcDataType, K, N, validK, validN>;
    using AccTile = Tile<TileType::Acc, accDataType, Rows, Cols, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024>;
    using ScalingTile = Tile<TileType::Scaling, uint64_t, 1, alignScalingN, BLayout::RowMajor, 1, -1, SLayout::NoneBox>;

    TileMatAData aMatTile;
    TileMatBData bMatTile;
    TileMatScalingData scalingMatTile(alignScalingN);
    uint32_t aMatSize = M * K * sizeof(srcDataType);
    uint32_t bMatSize = K * N * sizeof(srcDataType);
    TASSIGN(aMatTile, 0x0);
    TASSIGN(bMatTile, 0x0 + aMatSize);
    TASSIGN(scalingMatTile, 0x0 + aMatSize + bMatSize);

    LeftTile aTile;
    RightTile bTile;
    AccTile cTile(gShape3, gShape4);
    ScalingTile scalingTile(validN);
    TASSIGN(aTile, 0x0);
    TASSIGN(bTile, 0x0);
    TASSIGN(cTile, 0x0);
    TASSIGN(scalingTile, 0x0);

    TLOAD(aMatTile, src0Global);
    TLOAD(bMatTile, src1Global);
    TLOAD(scalingMatTile, src2Global);
#ifndef __PTO_AUTO__
    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
#endif

    TMOV(aTile, aMatTile);
    TMOV(bTile, bMatTile);

#ifndef __PTO_AUTO__
    set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
    wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
#endif
    TMATMUL(cTile, aTile, bTile);
#ifndef __PTO_AUTO__
    set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
#endif
    TMOV(scalingTile, scalingMatTile);
    constexpr AtomicType atomicTypeEnum = atomicType == 1 ? AtomicType::AtomicAdd : AtomicType::AtomicNone;
    if constexpr (reluMode == 0) {
        TSTORE_FP<AccTile, GlobalDataOut, ScalingTile, atomicTypeEnum>(dstGlobal, cTile, scalingTile);
    } else if constexpr (reluMode == 1) {
        constexpr ReluPreMode reluPreMode = ReluPreMode::NormalRelu;
        TSTORE_FP<AccTile, GlobalDataOut, ScalingTile, atomicTypeEnum, reluPreMode>(dstGlobal, cTile, scalingTile);
    }
#ifndef __PTO_AUTO__
    set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
    wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
#endif
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
    constexpr int alignScalingN = (validN * sizeof(uint64_t) + 127) / 128 * 128 / sizeof(uint64_t);

    constexpr int Rows = gShape2 * gShape3;
    constexpr int Cols = gShape0 * gShape1 * gShape4;
    int validRow = Rows;
    int validCol = Cols;

    using DynShapeDim5 = pto::Shape<gShape0, gShape1, gShape2, gShape3, gShape4>;
    using DynStridDim5 = pto::Stride<gStride[0], gStride[1], gStride[2], gStride[3], gStride[4]>;
    using GlobalDataOut = GlobalTensor<dstDataType, DynShapeDim5, DynStridDim5, Layout::NZ>;
    using GlobalDataSrc0 =
        GlobalTensor<srcDataType, pto::Shape<1, 1, 1, validM, validK>,
                     pto::Stride<1 * validM * validK, 1 * validM * validK, validM * validK, validK, 1>>;
    using GlobalDataSrc1 =
        GlobalTensor<srcDataType, pto::Shape<1, 1, 1, validK, validN>,
                     pto::Stride<1 * validK * validN, 1 * validK * validN, validK * validN, validN, 1>>;
    using GlobalDataSrc2 = GlobalTensor<uint64_t, Shape<1, 1, 1, 1, alignScalingN>,
                                        Stride<1 * alignScalingN, 1 * alignScalingN, alignScalingN, alignScalingN, 1>>;

    GlobalDataSrc0 src0Global(src0);
    GlobalDataSrc1 src1Global(src1);
    GlobalDataSrc2 src2Global(quantTensor);
    GlobalDataOut dstGlobal(out);

    using TileMatAData = Tile<TileType::Mat, srcDataType, M, K, BLayout::ColMajor, M, K, SLayout::RowMajor, 512>;
    using TileMatBData = Tile<TileType::Mat, srcDataType, K, N, BLayout::ColMajor, K, N, SLayout::RowMajor, 512>;
    using TileMatScalingData =
        Tile<TileType::Mat, uint64_t, 1, alignScalingN, BLayout::RowMajor, 1, -1, SLayout::NoneBox>;

    using LeftTile = Tile<TileType::Left, srcDataType, M, K, BLayout::RowMajor, validM, validK, SLayout::RowMajor, 512>;
    using RightTile = TileRight<srcDataType, K, N, validK, validN>;
    using AccTile = Tile<TileType::Acc, accDataType, Rows, Cols, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024>;
    using ScalingTile = Tile<TileType::Scaling, uint64_t, 1, alignScalingN, BLayout::RowMajor, 1, -1, SLayout::NoneBox>;

    TileMatAData aMatTile;
    TileMatBData bMatTile;
    TileMatScalingData scalingMatTile(alignScalingN);
    uint32_t aMatSize = M * K * sizeof(srcDataType);
    uint32_t bMatSize = K * N * sizeof(srcDataType);
    TASSIGN(aMatTile, 0x0);
    TASSIGN(bMatTile, 0x0 + aMatSize);
    TASSIGN(scalingMatTile, 0x0 + aMatSize + bMatSize);

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
#ifndef __PTO_AUTO__
    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
#endif

    TMOV(aTile, aMatTile);
    TMOV(bTile, bMatTile);

#ifndef __PTO_AUTO__
    set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
    wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
#endif
    TMATMUL(cTile, aTile, bTile);
#ifndef __PTO_AUTO__
    set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
#endif
    TMOV(scalingTile, scalingMatTile);
    constexpr AtomicType atomicTypeEnum = atomicType == 1 ? AtomicType::AtomicAdd : AtomicType::AtomicNone;
    if constexpr (reluMode == 0) {
        TSTORE_FP<AccTile, GlobalDataOut, ScalingTile, atomicTypeEnum>(dstGlobal, cTile, scalingTile);
    } else if constexpr (reluMode == 1) {
        constexpr ReluPreMode reluPreMode = ReluPreMode::NormalRelu;
        TSTORE_FP<AccTile, GlobalDataOut, ScalingTile, atomicTypeEnum, reluPreMode>(dstGlobal, cTile, scalingTile);
    }
#ifndef __PTO_AUTO__
    set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
    wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
#endif
    out = dstGlobal.data();
}

template <int tilingKey>
void LaunchTStoreAcc2gmNz2nd(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream)
{
    if constexpr (tilingKey == 1) {
        TStoreAcc2gmNz2nd<1, float, float, float, 1, 1, 1, 128, 128, 1, 1, 1, 128, 128, 128, 128, 61>
            <<<1, nullptr, stream>>>(reinterpret_cast<float *>(out), reinterpret_cast<float *>(src0),
                                     reinterpret_cast<float *>(src1));
    } else if constexpr (tilingKey == 2) {
        TStoreAcc2gmNz2nd<0, float, float, float, 1, 1, 1, 31, 32, 1, 1, 1, 31, 32, 31, 32, 126>
            <<<1, nullptr, stream>>>(reinterpret_cast<float *>(out), reinterpret_cast<float *>(src0),
                                     reinterpret_cast<float *>(src1));
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
        TStoreAcc2gmNz2nd<1, float, bfloat16_t, bfloat16_t, 1, 1, 1, 100, 222, 1, 1, 1, 100, 222, 100, 222, 60>
            <<<1, nullptr, stream>>>(reinterpret_cast<bfloat16_t *>(out), reinterpret_cast<bfloat16_t *>(src0),
                                     reinterpret_cast<bfloat16_t *>(src1));
    } else if constexpr (tilingKey == 7) {
        TStoreAcc2gmNz2nd<1, int32_t, int32_t, int8_t, 1, 1, 1, 44, 128, 1, 1, 1, 44, 128, 44, 128, 27>
            <<<1, nullptr, stream>>>(reinterpret_cast<int32_t *>(out), reinterpret_cast<int8_t *>(src0),
                                     reinterpret_cast<int8_t *>(src1));
    } else if constexpr (tilingKey == 21) {
        TStoreAcc2gmNz2nd<0, float, float, float, 1, 1, 1, 128, 96, 1, 1, 1, 128, 96, 128, 96, 61, 1>
            <<<1, nullptr, stream>>>(reinterpret_cast<float *>(out), reinterpret_cast<float *>(src0),
                                     reinterpret_cast<float *>(src1));
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
        TStoreAcc2gmNz2nz<1, float, bfloat16_t, bfloat16_t, 4, 4, 3, 16, 16, 4, 4, 3, 16, 16, 48, 256, 32>
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
        TStoreAcc2gmNz2nz<0, float, float, half, 4, 1, 16, 16, 16, 4, 1, 16, 16, 16, 256, 64, 33, 1>
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
        TStoreAcc2gmScalarNz2nd<1, float, int8_t, half, 1, 1, 1, 25, 35, 1, 1, 1, 25, 35, 25, 35, 32>
            <<<1, nullptr, stream>>>(reinterpret_cast<int8_t *>(out), reinterpret_cast<half *>(src0),
                                     reinterpret_cast<half *>(src1), scalarQuant);
    } else if constexpr (tilingKey == 5) {
        TStoreAcc2gmScalarNz2nd<0, float, uint8_t, float, 1, 1, 1, 16, 20, 1, 1, 1, 16, 20, 16, 20, 25>
            <<<1, nullptr, stream>>>(reinterpret_cast<uint8_t *>(out), reinterpret_cast<float *>(src0),
                                     reinterpret_cast<float *>(src1), scalarQuant);
    } else if constexpr (tilingKey == 21) {
        TStoreAcc2gmScalarNz2nd<0, float, int8_t, half, 1, 1, 1, 55, 27, 1, 1, 1, 55, 27, 55, 27, 33, 1>
            <<<1, nullptr, stream>>>(reinterpret_cast<int8_t *>(out), reinterpret_cast<half *>(src0),
                                     reinterpret_cast<half *>(src1), scalarQuant);
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
        TStoreAcc2gmScalarNz2nz<0, int32_t, uint8_t, int8_t, 1, 1, 2, 16, 32, 1, 1, 2, 16, 32, 32, 32, 17>
            <<<1, nullptr, stream>>>(reinterpret_cast<uint8_t *>(out), reinterpret_cast<int8_t *>(src0),
                                     reinterpret_cast<int8_t *>(src1), scalarQuant);
    } else if constexpr (tilingKey == 4) {
        TStoreAcc2gmScalarNz2nz<0, float, int8_t, float, 1, 2, 1, 16, 32, 1, 2, 1, 16, 32, 16, 64, 32>
            <<<1, nullptr, stream>>>(reinterpret_cast<int8_t *>(out), reinterpret_cast<float *>(src0),
                                     reinterpret_cast<float *>(src1), scalarQuant);
    } else if constexpr (tilingKey == 5) {
        TStoreAcc2gmScalarNz2nz<0, float, uint8_t, bfloat16_t, 1, 2, 2, 16, 32, 1, 2, 2, 16, 32, 32, 64, 16>
            <<<1, nullptr, stream>>>(reinterpret_cast<uint8_t *>(out), reinterpret_cast<bfloat16_t *>(src0),
                                     reinterpret_cast<bfloat16_t *>(src1), scalarQuant);
    } else if constexpr (tilingKey == 21) {
        TStoreAcc2gmScalarNz2nz<0, int32_t, int8_t, int8_t, 3, 1, 5, 16, 32, 3, 1, 5, 16, 32, 80, 96, 114, 1>
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
        TStoreAcc2gmVectorNz2nd<0, float, int8_t, half, 1, 1, 1, 33, 65, 1, 1, 1, 33, 65, 33, 65, 15>
            <<<1, nullptr, stream>>>(reinterpret_cast<int8_t *>(out), reinterpret_cast<half *>(src0),
                                     reinterpret_cast<half *>(src1), reinterpret_cast<uint64_t *>(quantTensor));
    } else if constexpr (tilingKey == 5) {
        TStoreAcc2gmVectorNz2nd<0, float, uint8_t, half, 1, 1, 1, 19, 33, 1, 1, 1, 19, 33, 19, 33, 23>
            <<<1, nullptr, stream>>>(reinterpret_cast<uint8_t *>(out), reinterpret_cast<half *>(src0),
                                     reinterpret_cast<half *>(src1), reinterpret_cast<uint64_t *>(quantTensor));
    } else if constexpr (tilingKey == 21) {
        TStoreAcc2gmVectorNz2nd<0, float, int8_t, half, 1, 1, 1, 79, 63, 1, 1, 1, 79, 63, 79, 63, 33, 1>
            <<<1, nullptr, stream>>>(reinterpret_cast<int8_t *>(out), reinterpret_cast<half *>(src0),
                                     reinterpret_cast<half *>(src1), reinterpret_cast<uint64_t *>(quantTensor));
    }
}

template <int tilingKey>
void LaunchTStoreAcc2gmVectorNz2nz(uint8_t *out, uint8_t *src0, uint8_t *src1, uint8_t *quantTensor, void *stream)
{
    if constexpr (tilingKey == 1) {
        TStoreAcc2gmVectorNz2nz<0, int32_t, int8_t, int8_t, 1, 1, 2, 16, 32, 1, 1, 2, 16, 32, 32, 32, 32>
            <<<1, nullptr, stream>>>(reinterpret_cast<int8_t *>(out), reinterpret_cast<int8_t *>(src0),
                                     reinterpret_cast<int8_t *>(src1), reinterpret_cast<uint64_t *>(quantTensor));
    } else if constexpr (tilingKey == 2) {
        TStoreAcc2gmVectorNz2nz<0, int32_t, uint8_t, int8_t, 1, 1, 2, 16, 32, 1, 1, 2, 16, 32, 32, 32, 128>
            <<<1, nullptr, stream>>>(reinterpret_cast<uint8_t *>(out), reinterpret_cast<int8_t *>(src0),
                                     reinterpret_cast<int8_t *>(src1), reinterpret_cast<uint64_t *>(quantTensor));
    } else if constexpr (tilingKey == 3) {
        TStoreAcc2gmVectorNz2nz<0, float, int8_t, half, 2, 1, 3, 16, 32, 2, 1, 3, 16, 32, 48, 64, 25>
            <<<1, nullptr, stream>>>(reinterpret_cast<int8_t *>(out), reinterpret_cast<half *>(src0),
                                     reinterpret_cast<half *>(src1), reinterpret_cast<uint64_t *>(quantTensor));
    } else if constexpr (tilingKey == 4) {
        TStoreAcc2gmVectorNz2nz<0, float, uint8_t, half, 3, 1, 8, 16, 32, 3, 1, 8, 16, 32, 128, 96, 17>
            <<<1, nullptr, stream>>>(reinterpret_cast<uint8_t *>(out), reinterpret_cast<half *>(src0),
                                     reinterpret_cast<half *>(src1), reinterpret_cast<uint64_t *>(quantTensor));
    } else if constexpr (tilingKey == 21) {
        TStoreAcc2gmVectorNz2nz<0, int32_t, int8_t, int8_t, 2, 2, 5, 16, 32, 2, 2, 5, 16, 32, 80, 128, 90, 1>
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
template void LaunchTStoreAcc2gmVectorNz2nz<1>(uint8_t *out, uint8_t *src0, uint8_t *src1, uint8_t *quantTensor,
                                               void *stream);
template void LaunchTStoreAcc2gmVectorNz2nz<2>(uint8_t *out, uint8_t *src0, uint8_t *src1, uint8_t *quantTensor,
                                               void *stream);
template void LaunchTStoreAcc2gmVectorNz2nz<3>(uint8_t *out, uint8_t *src0, uint8_t *src1, uint8_t *quantTensor,
                                               void *stream);
template void LaunchTStoreAcc2gmVectorNz2nz<4>(uint8_t *out, uint8_t *src0, uint8_t *src1, uint8_t *quantTensor,
                                               void *stream);

// relu
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