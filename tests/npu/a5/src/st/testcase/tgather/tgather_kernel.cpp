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
#include <iostream>
#include "acl/acl.h"
#include "tgather_common.h"

using namespace std;
using namespace pto;

template <typename Tsrc0, typename Tsrc1, int kGRows0_, int kGCols0_, int kGRows1_, int kGCols1_, int kTRows_,
          int kTCols_>
__global__ AICORE void runTGather(__gm__ Tsrc0 __out__ *out, __gm__ Tsrc0 __in__ *src0, __gm__ Tsrc1 __in__ *src1)
{
    using DynShapeDim5_src0 = pto::Shape<1, 1, 1, kGRows0_, kGCols0_>;
    using DynStridDim5_src0 = pto::Stride<1, 1, 1, kGCols0_, 1>;
    using GlobalData_src0 = GlobalTensor<Tsrc0, DynShapeDim5_src0, DynStridDim5_src0>;
    using DynShapeDim5_src1 = pto::Shape<1, 1, 1, kGRows1_, kGCols1_>;
    using DynStridDim5_src1 = pto::Stride<1, 1, 1, kGCols1_, 1>;
    using GlobalData_src1 = GlobalTensor<Tsrc1, DynShapeDim5_src1, DynStridDim5_src1>;
    using DynShapeDim5_dst = pto::Shape<1, 1, 1, kGRows1_, kGCols1_>;
    using DynStridDim5_dst = pto::Stride<1, 1, 1, kGCols1_, 1>;
    using GlobalData_dst = GlobalTensor<Tsrc0, DynShapeDim5_dst, DynStridDim5_dst>;

    constexpr int src0_row = kGRows0_;
    constexpr int src0_col = kGCols0_;
    constexpr int src1_row = kGRows1_;
    constexpr int src1_col = kGCols1_;
    constexpr int dst_row = kGRows1_;
    constexpr int dst_col = kGCols1_;

    using TileData_src0 = Tile<TileType::Vec, Tsrc0, kGRows0_, kGCols0_, BLayout::RowMajor, -1, -1>;
    using TileData_src1 = Tile<TileType::Vec, Tsrc1, kGRows1_, kGCols1_, BLayout::RowMajor, -1, -1>;
    using TileData_dst = Tile<TileType::Vec, Tsrc0, kGRows1_, kGCols1_, BLayout::RowMajor, -1, -1>;
    TileData_src0 src0Tile(src0_row, src0_col);
    TileData_src1 src1Tile(src1_row, src1_col);
    TileData_dst dstTile(dst_row, dst_col);

    TASSIGN(src0Tile, 0x0);
    TASSIGN(src1Tile, 0x0 + src0_row * src0_col * sizeof(Tsrc0));
    TASSIGN(dstTile, 0x0 + src0_row * src0_col * sizeof(Tsrc0) + src1_row * src1_col * sizeof(Tsrc1));

    GlobalData_src0 src0Global(src0);
    GlobalData_src1 src1Global(src1);
    GlobalData_dst dstGlobal(out);

    TLOAD(src0Tile, src0Global);
    TLOAD(src1Tile, src1Global);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    TGATHER(dstTile, src0Tile, src1Tile);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    TSTORE(dstGlobal, dstTile);
    out = dstGlobal.data();
}

template <typename src0T, typename src1T, typename dstT, uint32_t SRCROW, uint32_t SRCCOL, uint32_t DSTROW,
          uint32_t DSTCOL>
void launchTGATHER_demo(src0T *src0, src1T *src1, dstT *out, void *stream)
{
    cout << "launch TGATHER index start!" << endl;
    runTGather<src0T, src1T, SRCROW, SRCCOL, DSTROW, DSTCOL, SRCROW, SRCCOL><<<1, nullptr, stream>>>(out, src0, src1);
    cout << "launch TGATHER index end!" << endl;
}

template void launchTGATHER_demo<float, int32_t, float, 32, 1024, 16, 64>(float *src0, int32_t *src1, float *out,
                                                                          void *stream);
template void launchTGATHER_demo<int32_t, int32_t, int32_t, 32, 512, 16, 256>(int32_t *src0, int32_t *src1,
                                                                              int32_t *out, void *stream);
template void launchTGATHER_demo<int16_t, int16_t, int16_t, 16, 1024, 16, 128>(int16_t *src0, int16_t *src1,
                                                                               int16_t *out, void *stream);
template void launchTGATHER_demo<int16_t, int16_t, int16_t, 32, 256, 32, 64>(int16_t *src0, int16_t *src1, int16_t *out,
                                                                             void *stream);

template <typename srcT, typename dstT, int kGRows_, int kGCols_, int kTRows_, int kTCols_, MaskPattern maskPattern>
__global__ AICORE void runTGATHER(__gm__ dstT __out__ *out, __gm__ srcT __in__ *src)
{
    using DynShapeDim5 = pto::Shape<1, 1, 1, kGRows_, kGCols_>;
    using DynStridDim5 = pto::Stride<1, 1, 1, kGCols_, 1>;
    using SrcGlobalData = GlobalTensor<srcT, DynShapeDim5, DynStridDim5>;
    using DstGlobalData = GlobalTensor<dstT, DynShapeDim5, DynStridDim5>;
    using TileData = Tile<TileType::Vec, srcT, (kTRows_ + 5), (kTCols_ + 32), BLayout::RowMajor, -1, -1>;
    using DstTileData = Tile<TileType::Vec, dstT, kTRows_, kTCols_, BLayout::RowMajor, -1, -1>;
    TileData srcTile(kTRows_, kTCols_);
    DstTileData dstTile(kTRows_, kTCols_);
    TASSIGN(srcTile, 0x0);
    TASSIGN(dstTile, 0x0 + (kTRows_ + 5) * (kTCols_ + 32) * sizeof(dstT));

    SrcGlobalData srcGlobal(src);
    DstGlobalData dstGlobal(out);

    TLOAD(srcTile, srcGlobal);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    TGATHER<DstTileData, TileData, maskPattern>(dstTile, srcTile);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
    TSTORE(dstGlobal, dstTile);
    out = dstGlobal.data();
}

template <typename srcT, typename dstT, int kGRows_, int kGCols_, int kTRows_, int kTCols_, MaskPattern maskPattern>
void LaunchTGATHER(dstT *out, srcT *src, void *stream)
{
    if constexpr (std::is_same_v<srcT, uint16_t>) {
        runTGATHER<half, half, kGRows_, kGCols_, kTRows_, kTCols_, maskPattern>
            <<<1, nullptr, stream>>>(reinterpret_cast<half *>(out), reinterpret_cast<half *>(src));
    } else {
        runTGATHER<srcT, dstT, kGRows_, kGCols_, kTRows_, kTCols_, maskPattern><<<1, nullptr, stream>>>(out, src);
    }
}

template void LaunchTGATHER<int8_t, int8_t, HALF_P0101_ROW, HALF_P0101_COL, HALF_P0101_ROW, HALF_P0101_COL,
                            MaskPattern::P0101>(int8_t *out, int8_t *src, void *stream);

template void LaunchTGATHER<uint8_t, uint8_t, HALF_P1010_ROW, HALF_P1010_COL, HALF_P1010_ROW, HALF_P1010_COL,
                            MaskPattern::P1010>(uint8_t *out, uint8_t *src, void *stream);

template void LaunchTGATHER<int8_t, int8_t, HALF_P0001_ROW, HALF_P0001_COL, HALF_P0001_ROW, HALF_P0001_COL,
                            MaskPattern::P0001>(int8_t *out, int8_t *src, void *stream);

template void LaunchTGATHER<uint8_t, uint8_t, HALF_P0010_ROW, HALF_P0010_COL, HALF_P0010_ROW, HALF_P0010_COL,
                            MaskPattern::P0010>(uint8_t *out, uint8_t *src, void *stream);

template void LaunchTGATHER<int8_t, int8_t, HALF_P0100_ROW, HALF_P0100_COL, HALF_P0100_ROW, HALF_P0100_COL,
                            MaskPattern::P0100>(int8_t *out, int8_t *src, void *stream);

template void LaunchTGATHER<uint8_t, uint8_t, HALF_P1000_ROW, HALF_P1000_COL, HALF_P1000_ROW, HALF_P1000_COL,
                            MaskPattern::P1000>(uint8_t *out, uint8_t *src, void *stream);

template void LaunchTGATHER<int8_t, int8_t, HALF_P1111_ROW, HALF_P1111_COL, HALF_P1111_ROW, HALF_P1111_COL,
                            MaskPattern::P1111>(int8_t *out, int8_t *src, void *stream);

template void LaunchTGATHER<uint16_t, uint16_t, HALF_P0101_ROW, HALF_P0101_COL, HALF_P0101_ROW, HALF_P0101_COL,
                            MaskPattern::P0101>(uint16_t *out, uint16_t *src, void *stream);

template void LaunchTGATHER<uint16_t, uint16_t, HALF_P1010_ROW, HALF_P1010_COL, HALF_P1010_ROW, HALF_P1010_COL,
                            MaskPattern::P1010>(uint16_t *out, uint16_t *src, void *stream);

template void LaunchTGATHER<int16_t, int16_t, HALF_P0001_ROW, HALF_P0001_COL, HALF_P0001_ROW, HALF_P0001_COL,
                            MaskPattern::P0001>(int16_t *out, int16_t *src, void *stream);

template void LaunchTGATHER<int16_t, int16_t, HALF_P0010_ROW, HALF_P0010_COL, HALF_P0010_ROW, HALF_P0010_COL,
                            MaskPattern::P0010>(int16_t *out, int16_t *src, void *stream);

template void LaunchTGATHER<uint32_t, uint32_t, FLOAT_P0100_ROW, FLOAT_P0100_COL, FLOAT_P0100_ROW, FLOAT_P0100_COL,
                            MaskPattern::P0100>(uint32_t *out, uint32_t *src, void *stream);

template void LaunchTGATHER<int32_t, int32_t, FLOAT_P1000_ROW, FLOAT_P1000_COL, FLOAT_P1000_ROW, FLOAT_P1000_COL,
                            MaskPattern::P1000>(int32_t *out, int32_t *src, void *stream);

template void LaunchTGATHER<int32_t, int32_t, FLOAT_P1111_ROW, FLOAT_P1111_COL, FLOAT_P1111_ROW, FLOAT_P1111_COL,
                            MaskPattern::P1111>(int32_t *out, int32_t *src, void *stream);

template void LaunchTGATHER<half, half, HALF_P0101_ROW, HALF_P0101_COL, HALF_P0101_ROW, HALF_P0101_COL,
                            MaskPattern::P0101>(half *out, half *src, void *stream);

template void LaunchTGATHER<half, half, HALF_P1010_ROW, HALF_P1010_COL, HALF_P1010_ROW, HALF_P1010_COL,
                            MaskPattern::P1010>(half *out, half *src, void *stream);

template void LaunchTGATHER<half, half, HALF_P0001_ROW, HALF_P0001_COL, HALF_P0001_ROW, HALF_P0001_COL,
                            MaskPattern::P0001>(half *out, half *src, void *stream);

template void LaunchTGATHER<half, half, HALF_P0010_ROW, HALF_P0010_COL, HALF_P0010_ROW, HALF_P0010_COL,
                            MaskPattern::P0010>(half *out, half *src, void *stream);

template void LaunchTGATHER<half, half, HALF_P0100_ROW, HALF_P0100_COL, HALF_P0100_ROW, HALF_P0100_COL,
                            MaskPattern::P0100>(half *out, half *src, void *stream);

template void LaunchTGATHER<half, half, HALF_P1000_ROW, HALF_P1000_COL, HALF_P1000_ROW, HALF_P1000_COL,
                            MaskPattern::P1000>(half *out, half *src, void *stream);

template void LaunchTGATHER<half, half, HALF_P1111_ROW, HALF_P1111_COL, HALF_P1111_ROW, HALF_P1111_COL,
                            MaskPattern::P1111>(half *out, half *src, void *stream);

template void LaunchTGATHER<uint16_t, uint16_t, HALF_P0001_ROW, HALF_P0001_COL, HALF_P0001_ROW, HALF_P0001_COL,
                            MaskPattern::P0001>(uint16_t *out, uint16_t *src, void *stream);

template void LaunchTGATHER<uint16_t, uint16_t, HALF_P0010_ROW, HALF_P0010_COL, HALF_P0010_ROW, HALF_P0010_COL,
                            MaskPattern::P0010>(uint16_t *out, uint16_t *src, void *stream);

template void LaunchTGATHER<uint16_t, uint16_t, HALF_P0100_ROW, HALF_P0100_COL, HALF_P0100_ROW, HALF_P0100_COL,
                            MaskPattern::P0100>(uint16_t *out, uint16_t *src, void *stream);

template void LaunchTGATHER<uint16_t, uint16_t, HALF_P1000_ROW, HALF_P1000_COL, HALF_P1000_ROW, HALF_P1000_COL,
                            MaskPattern::P1000>(uint16_t *out, uint16_t *src, void *stream);

template void LaunchTGATHER<uint16_t, uint16_t, HALF_P1111_ROW, HALF_P1111_COL, HALF_P1111_ROW, HALF_P1111_COL,
                            MaskPattern::P1111>(uint16_t *out, uint16_t *src, void *stream);

template void LaunchTGATHER<float, float, FLOAT_P0101_ROW, FLOAT_P0101_COL, FLOAT_P0101_ROW, FLOAT_P0101_COL,
                            MaskPattern::P0101>(float *out, float *src, void *stream);

template void LaunchTGATHER<float, float, FLOAT_P1010_ROW, FLOAT_P1010_COL, FLOAT_P1010_ROW, FLOAT_P1010_COL,
                            MaskPattern::P1010>(float *out, float *src, void *stream);

template void LaunchTGATHER<float, float, FLOAT_P0001_ROW, FLOAT_P0001_COL, FLOAT_P0001_ROW, FLOAT_P0001_COL,
                            MaskPattern::P0001>(float *out, float *src, void *stream);

template void LaunchTGATHER<float, float, FLOAT_P0010_ROW, FLOAT_P0010_COL, FLOAT_P0010_ROW, FLOAT_P0010_COL,
                            MaskPattern::P0010>(float *out, float *src, void *stream);

template void LaunchTGATHER<float, float, FLOAT_P0100_ROW, FLOAT_P0100_COL, FLOAT_P0100_ROW, FLOAT_P0100_COL,
                            MaskPattern::P0100>(float *out, float *src, void *stream);

template void LaunchTGATHER<float, float, FLOAT_P1000_ROW, FLOAT_P1000_COL, FLOAT_P1000_ROW, FLOAT_P1000_COL,
                            MaskPattern::P1000>(float *out, float *src, void *stream);

template void LaunchTGATHER<float, float, FLOAT_P1111_ROW, FLOAT_P1111_COL, FLOAT_P1111_ROW, FLOAT_P1111_COL,
                            MaskPattern::P1111>(float *out, float *src, void *stream);

template void LaunchTGATHER<float, int32_t, FLOAT_P1010_ROW, FLOAT_P1010_COL, FLOAT_P1010_ROW, FLOAT_P1010_COL,
                            MaskPattern::P1010>(int32_t *out, float *src, void *stream);

template <typename srcT, typename src1T, typename dstT, int kGRows_, int kGCols_, int kTRows_, int kTCols_, int K,
          CmpMode cmpMode, uint32_t offset>
__global__ AICORE void runTGATHER_CMP(__gm__ srcT *src, __gm__ src1T *src1, __gm__ dstT *out)
{
    using DynShapeDim5 = pto::Shape<1, 1, 1, kGRows_, kGCols_>;
    using DynStridDim5 = pto::Stride<1, 1, 1, kGCols_, 1>;
    using SrcGlobalData = GlobalTensor<srcT, DynShapeDim5, DynStridDim5>;
    using DstGlobalData = GlobalTensor<dstT, pto::Shape<1, 1, 1, kGRows_, K>, pto::Stride<1, 1, 1, K, 1>>;
    using TileData = Tile<TileType::Vec, srcT, kTRows_, kTCols_, BLayout::RowMajor, -1, -1>;
    using DstTileData = Tile<TileType::Vec, dstT, kTRows_, K, BLayout::RowMajor, -1, -1>;
    TileData srcTile(kTRows_, kTCols_);
    DstTileData dstTile(kTRows_, K);
    TASSIGN(srcTile, 0x0);
    TASSIGN(dstTile, 0x0 + kTRows_ * kTCols_ * sizeof(srcT));

    // for TCONCAT
    constexpr int concat_row = (kTRows_ * sizeof(srcT)) < 32 ? (32 / sizeof(srcT)) : kTRows_;
    using ConcatTileData = Tile<TileType::Vec, dstT, concat_row, 1, BLayout::ColMajor, -1, -1>;
    ConcatTileData concatTile(kTRows_, 1);
    TASSIGN(concatTile, 0x20000);

    SrcGlobalData srcGlobal(src);
    DstGlobalData dstGlobal(out);
    constexpr int cmpVCol = (kTCols_ + 7) / 8;
    constexpr int cmpCol = (cmpVCol + 31) / 32 * 32;
    using TmpTileData = Tile<TileType::Vec, uint8_t, kTRows_, cmpCol, BLayout::RowMajor, -1, -1>;
    TmpTileData tmpTile(kTRows_, cmpVCol);

    TLOAD(srcTile, srcGlobal);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    TGATHER<DstTileData, TileData, ConcatTileData, TmpTileData, cmpMode, offset>(dstTile, srcTile, src1[0], concatTile,
                                                                                 tmpTile);

    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
    TSTORE(dstGlobal, dstTile);
    out = dstGlobal.data();
}

template <typename srcT, typename src1T, typename dstT, int kGRows_, int kGCols_, int kTRows_, int kTCols_, int K,
          CmpMode cmpMode, uint32_t offset>
void LaunchTGATHER_CMP(srcT *src, src1T *src1, dstT *out, void *stream)
{
    if constexpr (std::is_same_v<srcT, aclFloat16>) {
        runTGATHER_CMP<half, half, dstT, kGRows_, kGCols_, kTRows_, kTCols_, K, cmpMode, offset>
            <<<1, nullptr, stream>>>((half *)(src), (half *)(src1), out);
    } else {
        runTGATHER_CMP<srcT, src1T, dstT, kGRows_, kGCols_, kTRows_, kTCols_, K, cmpMode, offset>
            <<<1, nullptr, stream>>>(src, src1, out);
    }
}

template void LaunchTGATHER_CMP<float, float, uint32_t, 16, 64, 16, 64, 32, CmpMode::GT, 0>(float *src, float *src1,
                                                                                            uint32_t *out,
                                                                                            void *stream);
template void LaunchTGATHER_CMP<uint32_t, uint32_t, uint32_t, 8, 128, 8, 128, 64, CmpMode::GT, 0>(uint32_t *src,
                                                                                                  uint32_t *src1,
                                                                                                  uint32_t *out,
                                                                                                  void *stream);
template void LaunchTGATHER_CMP<float, float, uint32_t, 4, 256, 4, 256, 64, CmpMode::EQ, 0>(float *src, float *src1,
                                                                                            uint32_t *out,
                                                                                            void *stream);
template void LaunchTGATHER_CMP<int16_t, int16_t, uint32_t, 16, 128, 16, 128, 32, CmpMode::GT, 0>(int16_t *src,
                                                                                                  int16_t *src1,
                                                                                                  uint32_t *out,
                                                                                                  void *stream);
template void LaunchTGATHER_CMP<int16_t, int16_t, uint32_t, 4, 64, 4, 64, 32, CmpMode::EQ, 0>(int16_t *src,
                                                                                              int16_t *src1,
                                                                                              uint32_t *out,
                                                                                              void *stream);
template void LaunchTGATHER_CMP<aclFloat16, aclFloat16, uint32_t, 2, 256, 2, 256, 32, CmpMode::GT, 0>(aclFloat16 *src,
                                                                                                      aclFloat16 *src1,
                                                                                                      uint32_t *out,
                                                                                                      void *stream);
template void LaunchTGATHER_CMP<aclFloat16, aclFloat16, uint32_t, 8, 128, 8, 128, 32, CmpMode::EQ, 0>(aclFloat16 *src,
                                                                                                      aclFloat16 *src1,
                                                                                                      uint32_t *out,
                                                                                                      void *stream);