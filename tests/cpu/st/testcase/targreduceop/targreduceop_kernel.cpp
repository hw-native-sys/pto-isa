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

using namespace pto;

template <typename TIdx, typename T, int axis, int kTRows, int kTCols, int iRow = kTRows, int iCol = kTCols,
          int oRow = kTRows, int oCol = kTCols, typename LaunchFn>
AICORE void runTARGREDUCEOP(__gm__ TIdx __out__ *out, __gm__ T __in__ *src, LaunchFn fn)
{
    using GlobShapeDim5 = Shape<1, 1, 1, -1, -1>;
    using GlobStridDim5 = Stride<1, 1, -1, -1, 1>;
    using GlobalMatVal = GlobalTensor<T, GlobShapeDim5, GlobStridDim5>;
    using GlobalMatIdx = GlobalTensor<TIdx, GlobShapeDim5, GlobStridDim5>;

    using TileMatDst = Tile<TileType::Vec, TIdx, oRow, oCol, BLayout::RowMajor, -1, -1>;
    using TileMatSrc = Tile<TileType::Vec, T, iRow, iCol, BLayout::RowMajor, -1, -1>;
    constexpr int dstKTRows = axis == 0 ? 1 : kTRows;
    constexpr int dstKTCols = axis == 0 ? kTCols : 1;
    TileMatSrc srcTile(kTRows, kTCols);
    TileMatSrc tmpTile(kTRows, kTCols);
    TileMatDst dstTile(dstKTRows, dstKTCols);

    GlobalMatVal srcGlobal(src, GlobShapeDim5(kTRows, kTCols), GlobStridDim5(iRow, iCol));
    GlobalMatIdx dstGlobal(out, GlobShapeDim5(dstKTRows, dstKTCols), GlobStridDim5(oRow, oCol));

    TASSIGN(srcTile, 0);
    TASSIGN(dstTile, iRow * iCol * sizeof(T));

    TLOAD(srcTile, srcGlobal);
    fn(dstTile, srcTile, tmpTile);
    TSTORE(dstGlobal, dstTile);
    out = dstGlobal.data();
}

template <typename TIdx, typename T, int axis, int kTRows, int kTCols, int iRow = kTRows, int iCol = kTCols,
          int oIdxRow = kTRows, int oIdxCol = kTCols, int oValRow = kTRows, int oValCol = kTCols, typename LaunchFn>
AICORE void runTARGREDUCEOP(__gm__ T __out__ *outVal, __gm__ TIdx __out__ *outIdx, __gm__ T __in__ *src, LaunchFn fn)
{
    using GlobShapeDim5 = Shape<1, 1, 1, -1, -1>;
    using GlobStridDim5 = Stride<1, 1, -1, -1, 1>;
    using GlobalMatVal = GlobalTensor<T, GlobShapeDim5, GlobStridDim5>;
    using GlobalMatIdx = GlobalTensor<TIdx, GlobShapeDim5, GlobStridDim5>;

    using TileValDst = Tile<TileType::Vec, T, oValRow, oValCol, BLayout::RowMajor, -1, -1>;
    using TileIdxDst = Tile<TileType::Vec, TIdx, oIdxRow, oIdxCol, BLayout::RowMajor, -1, -1>;
    using TileMatSrc = Tile<TileType::Vec, T, iRow, iCol, BLayout::RowMajor, -1, -1>;
    constexpr int dstKTRows = axis == 0 ? 1 : kTRows;
    constexpr int dstKTCols = axis == 0 ? kTCols : 1;
    TileMatSrc srcTile(kTRows, kTCols);
    TileMatSrc tmpTile(kTRows, kTCols);
    TileValDst dstVal(dstKTRows, dstKTCols);
    TileIdxDst dstIdx(dstKTRows, dstKTCols);

    GlobalMatVal srcGlobal(src, GlobShapeDim5(kTRows, kTCols), GlobStridDim5(iRow, iCol));
    GlobalMatVal dstValGlobal(outVal, GlobShapeDim5(dstKTRows, dstKTCols), GlobStridDim5(oValRow, oValCol));
    GlobalMatIdx dstIdxGlobal(outIdx, GlobShapeDim5(dstKTRows, dstKTCols), GlobStridDim5(oIdxRow, oIdxCol));

    TASSIGN(srcTile, 0);
    TASSIGN(dstVal, iRow * iCol * sizeof(T));
    TASSIGN(dstIdx, iRow * iCol * sizeof(T) + oValRow * oValCol * sizeof(T));

    TLOAD(srcTile, srcGlobal);
    fn(dstVal, dstIdx, srcTile, tmpTile);
    TSTORE(dstValGlobal, dstVal);
    TSTORE(dstIdxGlobal, dstIdx);
    outVal = dstValGlobal.data();
    outIdx = dstIdxGlobal.data();
}

template <typename TIdx, typename T, int kTRows, int kTCols, int iRow = kTRows, int iCol = kTCols, int oRow = kTRows,
          int oCol = kTCols>
void LaunchTROWARGMAX(TIdx *out, T *src, void *stream)
{
    using TileDst = Tile<TileType::Vec, TIdx, oRow, oCol, BLayout::RowMajor, -1, -1>;
    using TileSrc = Tile<TileType::Vec, T, iRow, iCol, BLayout::RowMajor, -1, -1>;
    using TileTmp = Tile<TileType::Vec, T, iRow, iCol, BLayout::RowMajor, -1, -1>;
    if constexpr (std::is_same_v<T, aclFloat16>) {
        runTARGREDUCEOP<TIdx, half, 1, kTRows, kTCols, iRow, iCol, oRow, oCol>(
            out, (half *)(src), [](TileDst &dst, TileSrc &src, TileTmp &tmp) { TROWARGMAX(dst, src, tmp); });
    } else {
        runTARGREDUCEOP<TIdx, T, 1, kTRows, kTCols, iRow, iCol, oRow, oCol>(
            out, src, [](TileDst &dst, TileSrc &src, TileTmp &tmp) { TROWARGMAX(dst, src, tmp); });
    }
}

template <typename TIdx, typename T, int kTRows, int kTCols, int iRow = kTRows, int iCol = kTCols, int oRow = kTRows,
          int oCol = kTCols>
void LaunchTROWARGMIN(TIdx *out, T *src, void *stream)
{
    using TileDst = Tile<TileType::Vec, TIdx, oRow, oCol, BLayout::RowMajor, -1, -1>;
    using TileSrc = Tile<TileType::Vec, T, iRow, iCol, BLayout::RowMajor, -1, -1>;
    using TileTmp = Tile<TileType::Vec, T, iRow, iCol, BLayout::RowMajor, -1, -1>;
    if constexpr (std::is_same_v<T, aclFloat16>) {
        runTARGREDUCEOP<TIdx, half, 1, kTRows, kTCols, iRow, iCol, oRow, oCol>(
            out, (half *)(src), [](TileDst &dst, TileSrc &src, TileTmp &tmp) { TROWARGMIN(dst, src, tmp); });
    } else {
        runTARGREDUCEOP<TIdx, T, 1, kTRows, kTCols, iRow, iCol, oRow, oCol>(
            out, src, [](TileDst &dst, TileSrc &src, TileTmp &tmp) { TROWARGMIN(dst, src, tmp); });
    }
}

template <typename TIdx, typename T, int kTRows, int kTCols, int iRow = kTRows, int iCol = kTCols, int oIdxRow = kTRows,
          int oIdxCol = kTCols, int oValRow = kTRows, int oValCol = kTCols>
void LaunchTROWARGMAX(T *outVal, TIdx *outIdx, T *src, void *stream)
{
    using TileDstVal = Tile<TileType::Vec, T, oValRow, oValCol, BLayout::RowMajor, -1, -1>;
    using TileDstIdx = Tile<TileType::Vec, TIdx, oIdxRow, oIdxCol, BLayout::RowMajor, -1, -1>;
    using TileSrc = Tile<TileType::Vec, T, iRow, iCol, BLayout::RowMajor, -1, -1>;
    using TileTmp = Tile<TileType::Vec, T, iRow, iCol, BLayout::RowMajor, -1, -1>;
    if constexpr (std::is_same_v<T, aclFloat16>) {
        runTARGREDUCEOP<TIdx, half, 1, kTRows, kTCols, iRow, iCol, oIdxRow, oIdxCol, oValRow, oValCol>(
            (half *)(outVal), outIdx, (half *)(src),
            [](TileDstVal &dstVal, TileDstIdx &dstIdx, TileSrc &src, TileTmp &tmp) {
                TROWARGMAX(dstVal, dstIdx, src, tmp);
            });
    } else {
        runTARGREDUCEOP<TIdx, T, 1, kTRows, kTCols, iRow, iCol, oIdxRow, oIdxCol, oValRow, oValCol>(
            outVal, outIdx, src, [](TileDstVal &dstVal, TileDstIdx &dstIdx, TileSrc &src, TileTmp &tmp) {
                TROWARGMAX(dstVal, dstIdx, src, tmp);
            });
    }
}

template <typename TIdx, typename T, int kTRows, int kTCols, int iRow = kTRows, int iCol = kTCols, int oIdxRow = kTRows,
          int oIdxCol = kTCols, int oValRow = kTRows, int oValCol = kTCols>
void LaunchTROWARGMIN(T *outVal, TIdx *outIdx, T *src, void *stream)
{
    using TileDstVal = Tile<TileType::Vec, T, oValRow, oValCol, BLayout::RowMajor, -1, -1>;
    using TileDstIdx = Tile<TileType::Vec, TIdx, oIdxRow, oIdxCol, BLayout::RowMajor, -1, -1>;
    using TileSrc = Tile<TileType::Vec, T, iRow, iCol, BLayout::RowMajor, -1, -1>;
    using TileTmp = Tile<TileType::Vec, T, iRow, iCol, BLayout::RowMajor, -1, -1>;
    if constexpr (std::is_same_v<T, aclFloat16>) {
        runTARGREDUCEOP<TIdx, half, 1, kTRows, kTCols, iRow, iCol, oIdxRow, oIdxCol, oValRow, oValCol>(
            (half *)(outVal), outIdx, (half *)(src),
            [](TileDstVal &dstVal, TileDstIdx &dstIdx, TileSrc &src, TileTmp &tmp) {
                TROWARGMIN(dstVal, dstIdx, src, tmp);
            });
    } else {
        runTARGREDUCEOP<TIdx, T, 1, kTRows, kTCols, iRow, iCol, oIdxRow, oIdxCol, oValRow, oValCol>(
            outVal, outIdx, src, [](TileDstVal &dstVal, TileDstIdx &dstIdx, TileSrc &src, TileTmp &tmp) {
                TROWARGMIN(dstVal, dstIdx, src, tmp);
            });
    }
}

template <typename TIdx, typename T, int kTRows, int kTCols, int iRow = kTRows, int iCol = kTCols, int oRow = kTRows,
          int oCol = kTCols>
void LaunchTCOLARGMAX(TIdx *out, T *src, void *stream)
{
    using TileDst = Tile<TileType::Vec, TIdx, oRow, oCol, BLayout::RowMajor, -1, -1>;
    using TileSrc = Tile<TileType::Vec, T, iRow, iCol, BLayout::RowMajor, -1, -1>;
    using TileTmp = Tile<TileType::Vec, T, iRow, iCol, BLayout::RowMajor, -1, -1>;
    if constexpr (std::is_same_v<T, aclFloat16>) {
        runTARGREDUCEOP<TIdx, half, 0, kTRows, kTCols, iRow, iCol, oRow, oCol>(
            out, (half *)(src), [](TileDst &dst, TileSrc &src, TileTmp &tmp) { TCOLARGMAX(dst, src, tmp); });
    } else {
        runTARGREDUCEOP<TIdx, T, 0, kTRows, kTCols, iRow, iCol, oRow, oCol>(
            out, src, [](TileDst &dst, TileSrc &src, TileTmp &tmp) { TCOLARGMAX(dst, src, tmp); });
    }
}

template <typename TIdx, typename T, int kTRows, int kTCols, int iRow = kTRows, int iCol = kTCols, int oRow = kTRows,
          int oCol = kTCols>
void LaunchTCOLARGMIN(TIdx *out, T *src, void *stream)
{
    using TileDst = Tile<TileType::Vec, TIdx, oRow, oCol, BLayout::RowMajor, -1, -1>;
    using TileSrc = Tile<TileType::Vec, T, iRow, iCol, BLayout::RowMajor, -1, -1>;
    using TileTmp = Tile<TileType::Vec, T, iRow, iCol, BLayout::RowMajor, -1, -1>;
    if constexpr (std::is_same_v<T, aclFloat16>) {
        runTARGREDUCEOP<TIdx, half, 0, kTRows, kTCols, iRow, iCol, oRow, oCol>(
            out, (half *)(src), [](TileDst &dst, TileSrc &src, TileTmp &tmp) { TCOLARGMIN(dst, src, tmp); });
    } else {
        runTARGREDUCEOP<TIdx, T, 0, kTRows, kTCols, iRow, iCol, oRow, oCol>(
            out, src, [](TileDst &dst, TileSrc &src, TileTmp &tmp) { TCOLARGMIN(dst, src, tmp); });
    }
}

template void LaunchTROWARGMAX<uint32_t, float, 64, 64>(uint32_t *out, float *src, void *stream);
template void LaunchTROWARGMAX<int32_t, aclFloat16, 16, 256>(int32_t *out, aclFloat16 *src, void *stream);
template void LaunchTROWARGMAX<uint32_t, float, 16, 16, 32, 32, 64, 64>(uint32_t *out, float *src, void *stream);
template void LaunchTROWARGMIN<uint32_t, float, 64, 64>(uint32_t *out, float *src, void *stream);
template void LaunchTROWARGMIN<int32_t, aclFloat16, 16, 256>(int32_t *out, aclFloat16 *src, void *stream);
template void LaunchTROWARGMIN<uint32_t, float, 16, 16, 32, 32, 64, 64>(uint32_t *out, float *src, void *stream);
template void LaunchTCOLARGMIN<uint32_t, float, 64, 64>(uint32_t *out, float *src, void *stream);
template void LaunchTCOLARGMIN<int32_t, aclFloat16, 16, 256>(int32_t *out, aclFloat16 *src, void *stream);
template void LaunchTCOLARGMIN<uint32_t, float, 16, 16, 32, 32, 64, 64>(uint32_t *out, float *src, void *stream);
template void LaunchTCOLARGMAX<uint32_t, float, 64, 64>(uint32_t *out, float *src, void *stream);
template void LaunchTCOLARGMAX<int32_t, aclFloat16, 16, 256>(int32_t *out, aclFloat16 *src, void *stream);
template void LaunchTCOLARGMAX<uint32_t, float, 16, 16, 32, 32, 64, 64>(uint32_t *out, float *src, void *stream);
template void LaunchTROWARGMAX<uint32_t, float, 64, 64>(float *outVal, uint32_t *outIdx, float *src, void *stream);
template void LaunchTROWARGMAX<int32_t, aclFloat16, 16, 256>(aclFloat16 *outVal, int32_t *outIdx, aclFloat16 *src,
                                                             void *stream);
template void LaunchTROWARGMAX<uint32_t, float, 16, 16, 32, 32, 64, 64, 32, 32>(float *outVal, uint32_t *outIdx,
                                                                                float *src, void *stream);
template void LaunchTROWARGMIN<uint32_t, float, 64, 64>(float *outVal, uint32_t *outIdx, float *src, void *stream);
template void LaunchTROWARGMIN<int32_t, aclFloat16, 16, 256>(aclFloat16 *outVal, int32_t *outIdx, aclFloat16 *src,
                                                             void *stream);
template void LaunchTROWARGMIN<uint32_t, float, 16, 16, 32, 32, 64, 64, 32, 32>(float *outVal, uint32_t *outIdx,
                                                                                float *src, void *stream);
