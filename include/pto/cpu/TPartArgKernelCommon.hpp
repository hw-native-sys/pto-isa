/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef TPARTARGKERNELCOMMON_HPP
#define TPARTARGKERNELCOMMON_HPP

#include <pto/pto-inst.hpp>
#include "TPartOp.hpp"

namespace pto {

struct TPartArgDefaultConfig {
    static constexpr int Rows = 64;
    static constexpr int Cols = 64;
    static constexpr int ValidRows1 = 32;
    static constexpr int ValidCols1 = 32;
};

template <int kRows, int kCols, int kValidRows1, int kValidCols1>
struct TPartArgKernelCommon {
    using DynShapeDim5 = Shape<1, 1, 1, kRows, kCols>;
    using DynStridDim5 = Stride<1, 1, 1, kCols, 1>;

    using GlobalDataVal = GlobalTensor<float, DynShapeDim5, DynStridDim5>;
    using GlobalDataIdx = GlobalTensor<uint32_t, DynShapeDim5, DynStridDim5>;

    using GlobalData1Val = GlobalTensor<float, Shape<1, 1, 1, kValidRows1, kValidCols1>, DynStridDim5>;
    using GlobalData1Idx = GlobalTensor<uint32_t, Shape<1, 1, 1, kValidRows1, kValidCols1>, DynStridDim5>;

    using TileVal0 = Tile<TileType::Vec, float, kRows, kCols, BLayout::RowMajor, -1, -1>;
    using TileVal1 = Tile<TileType::Vec, float, kValidRows1, kValidCols1, BLayout::RowMajor, -1, -1>;

    using TileIdx0 = Tile<TileType::Vec, uint32_t, kRows, kCols, BLayout::RowMajor, -1, -1>;
    using TileIdx1 = Tile<TileType::Vec, uint32_t, kValidRows1, kValidCols1, BLayout::RowMajor, -1, -1>;

    static constexpr size_t val0Size = kRows * kCols * sizeof(float);
    static constexpr size_t val1Size = kValidRows1 * kValidCols1 * sizeof(float);
    static constexpr size_t idx0Size = kRows * kCols * sizeof(uint32_t);
    static constexpr size_t idx1Size = kValidRows1 * kValidCols1 * sizeof(uint32_t);

    PTO_INTERNAL static void AssignTiles(TileVal0 &dstValTile, TileVal0 &src0ValTile, TileVal1 &src1ValTile,
                                         TileIdx0 &dstIdxTile, TileIdx0 &src0IdxTile, TileIdx1 &src1IdxTile)
    {
        TASSIGN(src0ValTile, 0);
        TASSIGN(src1ValTile, val0Size);
        TASSIGN(dstValTile, val0Size + val1Size);

        TASSIGN(src0IdxTile, val0Size + val1Size + val0Size);
        TASSIGN(src1IdxTile, val0Size + val1Size + val0Size + idx0Size);
        TASSIGN(dstIdxTile, val0Size + val1Size + val0Size + idx0Size + idx1Size);
    }

    PTO_INTERNAL static void LoadTiles(TileVal0 &src0ValTile, TileVal1 &src1ValTile, TileIdx0 &src0IdxTile,
                                       TileIdx1 &src1IdxTile, GlobalDataVal &src0ValGlobal,
                                       GlobalData1Val &src1ValGlobal, GlobalDataIdx &src0IdxGlobal,
                                       GlobalData1Idx &src1IdxGlobal)
    {
        TLOAD(src0ValTile, src0ValGlobal);
        TLOAD(src1ValTile, src1ValGlobal);
        TLOAD(src0IdxTile, src0IdxGlobal);
        TLOAD(src1IdxTile, src1IdxGlobal);
    }

    PTO_INTERNAL static void StoreTiles(TileVal0 &dstValTile, TileIdx0 &dstIdxTile, GlobalDataVal &dstValGlobal,
                                        GlobalDataIdx &dstIdxGlobal)
    {
        TSTORE(dstValGlobal, dstValTile);
        TSTORE(dstIdxGlobal, dstIdxTile);
    }
};

template <int kRows, int kCols, int kValidRows1, int kValidCols1, typename Runner>
AICORE void RunPartArgKernel(__gm__ float *__out__ outVal, __gm__ float *__in__ src0Val, __gm__ float *__in__ src1Val,
                             __gm__ uint32_t *__out__ outIdx, __gm__ uint32_t *__in__ src0Idx,
                             __gm__ uint32_t *__in__ src1Idx)
{
    using Common = TPartArgKernelCommon<kRows, kCols, kValidRows1, kValidCols1>;
    typename Common::TileVal0 src0ValTile(kRows, kCols);
    typename Common::TileVal1 src1ValTile(kValidRows1, kValidCols1);
    typename Common::TileVal0 dstValTile(kRows, kCols);
    typename Common::TileIdx0 src0IdxTile(kRows, kCols);
    typename Common::TileIdx1 src1IdxTile(kValidRows1, kValidCols1);
    typename Common::TileIdx0 dstIdxTile(kRows, kCols);
    typename Common::GlobalDataVal src0ValGlobal(src0Val);
    typename Common::GlobalData1Val src1ValGlobal(src1Val);
    typename Common::GlobalDataVal dstValGlobal(outVal);
    typename Common::GlobalDataIdx src0IdxGlobal(src0Idx);
    typename Common::GlobalData1Idx src1IdxGlobal(src1Idx);
    typename Common::GlobalDataIdx dstIdxGlobal(outIdx);

    Common::AssignTiles(dstValTile, src0ValTile, src1ValTile, dstIdxTile, src0IdxTile, src1IdxTile);
    Common::LoadTiles(src0ValTile, src1ValTile, src0IdxTile, src1IdxTile, src0ValGlobal, src1ValGlobal, src0IdxGlobal,
                      src1IdxGlobal);
    Runner::Run(dstValTile, src0ValTile, src1ValTile, dstIdxTile, src0IdxTile, src1IdxTile);
    Common::StoreTiles(dstValTile, dstIdxTile, dstValGlobal, dstIdxGlobal);

    outVal = dstValGlobal.data();
    outIdx = dstIdxGlobal.data();
}
} // namespace pto
#endif // TPARTARGKERNELCOMMON_HPP
