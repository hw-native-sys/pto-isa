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

using namespace pto;

template <
    typename T, int kTRows, int kTCols, int iRow = kTRows, int iCol = kTCols, int oRow = kTRows, int oCol = kTCols,
    typename LaunchFn>
AICORE void runTROWEXPANDOP(__gm__ T __out__* out, __gm__ T __in__* src0, __gm__ T __in__* src1, LaunchFn fn)
{
    using GlobShapeDim5 = Shape<1, 1, 1, -1, -1>;
    using GlobStridDim5 = Stride<1, 1, -1, -1, 1>;
    using GlobalMat = GlobalTensor<T, GlobShapeDim5, GlobStridDim5>;

    using TileMatDst = Tile<TileType::Vec, T, oRow, oCol, BLayout::RowMajor, -1, -1>;
    using TileMatSrc0 = Tile<TileType::Vec, T, iRow, iCol, BLayout::RowMajor, -1, -1>;
    using TileVec = Tile<TileType::Vec, T, iRow, 1, BLayout::ColMajor, -1, -1>;
    TileMatSrc0 src0Tile(kTRows, kTCols);
    TileVec src1Tile(kTRows, 1);
    TileMatDst dstTile(kTRows, kTCols);

    GlobalMat src0Global(src0, GlobShapeDim5(kTRows, kTCols), GlobStridDim5(iRow, iCol));
    GlobalMat src1Global(src1, GlobShapeDim5(kTRows, 1), GlobStridDim5(iRow, 1));
    GlobalMat dstGlobal(out, GlobShapeDim5(kTRows, kTCols), GlobStridDim5(oRow, oCol));

    TASSIGN(src0Tile, 0);
    TASSIGN(src1Tile, iRow * iCol * sizeof(typename TileMatSrc0::DType));
    TASSIGN(dstTile, 2 * iRow * iCol * sizeof(typename TileMatSrc0::DType));

    TLOAD(src0Tile, src0Global);
    TLOAD(src1Tile, src1Global);
    fn(dstTile, src0Tile, src1Tile);
    TSTORE(dstGlobal, dstTile);
    out = dstGlobal.data();
}

template <
    typename T, int kTRows, int kTCols, int iRow = kTRows, int iCol = kTCols, int oRow = kTRows, int oCol = kTCols>
void LaunchTROWEXPANDDIV(T* out, T* src0, T* src1, void* stream)
{
    using TileDst = Tile<TileType::Vec, T, oRow, oCol, BLayout::RowMajor, -1, -1>;
    using TileSrc0 = Tile<TileType::Vec, T, iRow, iCol, BLayout::RowMajor, -1, -1>;
    using TileSrc1 = Tile<TileType::Vec, T, iRow, 1, BLayout::ColMajor, -1, -1>;
    if constexpr (std::is_same_v<T, aclFloat16>) {
        runTROWEXPANDOP<half, kTRows, kTCols, iRow, iCol, oRow, oCol>(
            (half*)(out), (half*)(src0), (half*)(src1),
            [](TileDst& dst, TileSrc0& src0, TileSrc1& src1) { TROWEXPANDDIV(dst, src0, src1); });
    } else {
        runTROWEXPANDOP<T, kTRows, kTCols, iRow, iCol, oRow, oCol>(
            out, src0, src1, [](TileDst& dst, TileSrc0& src0, TileSrc1& src1) { TROWEXPANDDIV(dst, src0, src1); });
    }
}

template <
    typename T, int kTRows, int kTCols, int iRow = kTRows, int iCol = kTCols, int oRow = kTRows, int oCol = kTCols>
void LaunchTROWEXPANDMUL(T* out, T* src0, T* src1, void* stream)
{
    using TileDst = Tile<TileType::Vec, T, oRow, oCol, BLayout::RowMajor, -1, -1>;
    using TileSrc0 = Tile<TileType::Vec, T, iRow, iCol, BLayout::RowMajor, -1, -1>;
    using TileSrc1 = Tile<TileType::Vec, T, iRow, 1, BLayout::ColMajor, -1, -1>;
    if constexpr (std::is_same_v<T, aclFloat16>) {
        runTROWEXPANDOP<half, kTRows, kTCols, iRow, iCol, oRow, oCol>(
            (half*)(out), (half*)(src0), (half*)(src1),
            [](TileDst& dst, TileSrc0& src0, TileSrc1& src1) { TROWEXPANDMUL(dst, src0, src1); });
    } else {
        runTROWEXPANDOP<T, kTRows, kTCols, iRow, iCol, oRow, oCol>(
            out, src0, src1, [](TileDst& dst, TileSrc0& src0, TileSrc1& src1) { TROWEXPANDMUL(dst, src0, src1); });
    }
}

template <
    typename T, int kTRows, int kTCols, int iRow = kTRows, int iCol = kTCols, int oRow = kTRows, int oCol = kTCols>
void LaunchTROWEXPANDSUB(T* out, T* src0, T* src1, void* stream)
{
    using TileDst = Tile<TileType::Vec, T, oRow, oCol, BLayout::RowMajor, -1, -1>;
    using TileSrc0 = Tile<TileType::Vec, T, iRow, iCol, BLayout::RowMajor, -1, -1>;
    using TileSrc1 = Tile<TileType::Vec, T, iRow, 1, BLayout::ColMajor, -1, -1>;
    if constexpr (std::is_same_v<T, aclFloat16>) {
        runTROWEXPANDOP<half, kTRows, kTCols, iRow, iCol, oRow, oCol>(
            (half*)(out), (half*)(src0), (half*)(src1),
            [](TileDst& dst, TileSrc0& src0, TileSrc1& src1) { TROWEXPANDSUB(dst, src0, src1); });
    } else {
        runTROWEXPANDOP<T, kTRows, kTCols, iRow, iCol, oRow, oCol>(
            out, src0, src1, [](TileDst& dst, TileSrc0& src0, TileSrc1& src1) { TROWEXPANDSUB(dst, src0, src1); });
    }
}

template <
    typename T, int kTRows, int kTCols, int iRow = kTRows, int iCol = kTCols, int oRow = kTRows, int oCol = kTCols>
void LaunchTROWEXPANDADD(T* out, T* src0, T* src1, void* stream)
{
    using TileDst = Tile<TileType::Vec, T, oRow, oCol, BLayout::RowMajor, -1, -1>;
    using TileSrc0 = Tile<TileType::Vec, T, iRow, iCol, BLayout::RowMajor, -1, -1>;
    using TileSrc1 = Tile<TileType::Vec, T, iRow, 1, BLayout::ColMajor, -1, -1>;
    if constexpr (std::is_same_v<T, aclFloat16>) {
        runTROWEXPANDOP<half, kTRows, kTCols, iRow, iCol, oRow, oCol>(
            (half*)(out), (half*)(src0), (half*)(src1),
            [](TileDst& dst, TileSrc0& src0, TileSrc1& src1) { TROWEXPANDADD(dst, src0, src1); });
    } else {
        runTROWEXPANDOP<T, kTRows, kTCols, iRow, iCol, oRow, oCol>(
            out, src0, src1, [](TileDst& dst, TileSrc0& src0, TileSrc1& src1) { TROWEXPANDADD(dst, src0, src1); });
    }
}

template <
    typename T, int kTRows, int kTCols, int iRow = kTRows, int iCol = kTCols, int oRow = kTRows, int oCol = kTCols>
void LaunchTROWEXPANDMAX(T* out, T* src0, T* src1, void* stream)
{
    using TileDst = Tile<TileType::Vec, T, oRow, oCol, BLayout::RowMajor, -1, -1>;
    using TileSrc0 = Tile<TileType::Vec, T, iRow, iCol, BLayout::RowMajor, -1, -1>;
    using TileSrc1 = Tile<TileType::Vec, T, iRow, 1, BLayout::ColMajor, -1, -1>;
    if constexpr (std::is_same_v<T, aclFloat16>) {
        runTROWEXPANDOP<half, kTRows, kTCols, iRow, iCol, oRow, oCol>(
            (half*)(out), (half*)(src0), (half*)(src1),
            [](TileDst& dst, TileSrc0& src0, TileSrc1& src1) { TROWEXPANDMAX(dst, src0, src1); });
    } else {
        runTROWEXPANDOP<T, kTRows, kTCols, iRow, iCol, oRow, oCol>(
            out, src0, src1, [](TileDst& dst, TileSrc0& src0, TileSrc1& src1) { TROWEXPANDMAX(dst, src0, src1); });
    }
}

template <
    typename T, int kTRows, int kTCols, int iRow = kTRows, int iCol = kTCols, int oRow = kTRows, int oCol = kTCols>
void LaunchTROWEXPANDMIN(T* out, T* src0, T* src1, void* stream)
{
    using TileDst = Tile<TileType::Vec, T, oRow, oCol, BLayout::RowMajor, -1, -1>;
    using TileSrc0 = Tile<TileType::Vec, T, iRow, iCol, BLayout::RowMajor, -1, -1>;
    using TileSrc1 = Tile<TileType::Vec, T, iRow, 1, BLayout::ColMajor, -1, -1>;
    if constexpr (std::is_same_v<T, aclFloat16>) {
        runTROWEXPANDOP<half, kTRows, kTCols, iRow, iCol, oRow, oCol>(
            (half*)(out), (half*)(src0), (half*)(src1),
            [](TileDst& dst, TileSrc0& src0, TileSrc1& src1) { TROWEXPANDMIN(dst, src0, src1); });
    } else {
        runTROWEXPANDOP<T, kTRows, kTCols, iRow, iCol, oRow, oCol>(
            out, src0, src1, [](TileDst& dst, TileSrc0& src0, TileSrc1& src1) { TROWEXPANDMIN(dst, src0, src1); });
    }
}

template <
    typename T, int kTRows, int kTCols, int iRow = kTRows, int iCol = kTCols, int oRow = kTRows, int oCol = kTCols>
void LaunchTROWEXPANDEXPDIF(T* out, T* src0, T* src1, void* stream)
{
    using TileDst = Tile<TileType::Vec, T, oRow, oCol, BLayout::RowMajor, -1, -1>;
    using TileSrc0 = Tile<TileType::Vec, T, iRow, iCol, BLayout::RowMajor, -1, -1>;
    using TileSrc1 = Tile<TileType::Vec, T, iRow, 1, BLayout::ColMajor, -1, -1>;
    if constexpr (std::is_same_v<T, aclFloat16>) {
        runTROWEXPANDOP<half, kTRows, kTCols, iRow, iCol, oRow, oCol>(
            (half*)(out), (half*)(src0), (half*)(src1),
            [](TileDst& dst, TileSrc0& src0, TileSrc1& src1) { TROWEXPANDEXPDIF(dst, src0, src1); });
    } else {
        runTROWEXPANDOP<T, kTRows, kTCols, iRow, iCol, oRow, oCol>(
            out, src0, src1, [](TileDst& dst, TileSrc0& src0, TileSrc1& src1) { TROWEXPANDEXPDIF(dst, src0, src1); });
    }
}

template void LaunchTROWEXPANDDIV<float, 64, 64>(float* out, float* src0, float* src1, void* stream);
template void LaunchTROWEXPANDDIV<aclFloat16, 16, 256>(
    aclFloat16* out, aclFloat16* src0, aclFloat16* src1, void* stream);
template void LaunchTROWEXPANDMUL<float, 64, 64>(float* out, float* src0, float* src1, void* stream);
template void LaunchTROWEXPANDMUL<aclFloat16, 16, 256>(
    aclFloat16* out, aclFloat16* src0, aclFloat16* src1, void* stream);
template void LaunchTROWEXPANDSUB<float, 64, 64>(float* out, float* src0, float* src1, void* stream);
template void LaunchTROWEXPANDSUB<aclFloat16, 16, 256>(
    aclFloat16* out, aclFloat16* src0, aclFloat16* src1, void* stream);
template void LaunchTROWEXPANDADD<float, 64, 64>(float* out, float* src0, float* src1, void* stream);
template void LaunchTROWEXPANDADD<aclFloat16, 16, 256>(
    aclFloat16* out, aclFloat16* src0, aclFloat16* src1, void* stream);
template void LaunchTROWEXPANDMAX<float, 64, 64>(float* out, float* src0, float* src1, void* stream);
template void LaunchTROWEXPANDMAX<aclFloat16, 16, 256>(
    aclFloat16* out, aclFloat16* src0, aclFloat16* src1, void* stream);
template void LaunchTROWEXPANDMIN<float, 64, 64>(float* out, float* src0, float* src1, void* stream);
template void LaunchTROWEXPANDMIN<aclFloat16, 16, 256>(
    aclFloat16* out, aclFloat16* src0, aclFloat16* src1, void* stream);
template void LaunchTROWEXPANDEXPDIF<float, 64, 64>(float* out, float* src0, float* src1, void* stream);
template void LaunchTROWEXPANDEXPDIF<aclFloat16, 16, 256>(
    aclFloat16* out, aclFloat16* src0, aclFloat16* src1, void* stream);
template void LaunchTROWEXPANDDIV<float, 16, 16, 32, 32, 64, 64>(float* out, float* src0, float* src1, void* stream);
template void LaunchTROWEXPANDMUL<float, 16, 16, 32, 32, 64, 64>(float* out, float* src0, float* src1, void* stream);
template void LaunchTROWEXPANDSUB<float, 16, 16, 32, 32, 64, 64>(float* out, float* src0, float* src1, void* stream);
template void LaunchTROWEXPANDADD<float, 16, 16, 32, 32, 64, 64>(float* out, float* src0, float* src1, void* stream);
template void LaunchTROWEXPANDMAX<float, 16, 16, 32, 32, 64, 64>(float* out, float* src0, float* src1, void* stream);
template void LaunchTROWEXPANDMIN<float, 16, 16, 32, 32, 64, 64>(float* out, float* src0, float* src1, void* stream);
template void LaunchTROWEXPANDEXPDIF<float, 16, 16, 32, 32, 64, 64>(float* out, float* src0, float* src1, void* stream);

template void LaunchTROWEXPANDDIV<int16_t, 16, 256>(int16_t* out, int16_t* src0, int16_t* src1, void* stream);
template void LaunchTROWEXPANDMUL<int16_t, 16, 256>(int16_t* out, int16_t* src0, int16_t* src1, void* stream);
template void LaunchTROWEXPANDSUB<int16_t, 16, 256>(int16_t* out, int16_t* src0, int16_t* src1, void* stream);
template void LaunchTROWEXPANDADD<int16_t, 16, 256>(int16_t* out, int16_t* src0, int16_t* src1, void* stream);
template void LaunchTROWEXPANDMAX<int16_t, 16, 256>(int16_t* out, int16_t* src0, int16_t* src1, void* stream);
template void LaunchTROWEXPANDMIN<int16_t, 16, 256>(int16_t* out, int16_t* src0, int16_t* src1, void* stream);

template void LaunchTROWEXPANDDIV<int32_t, 64, 64>(int32_t* out, int32_t* src0, int32_t* src1, void* stream);
template void LaunchTROWEXPANDMUL<int32_t, 64, 64>(int32_t* out, int32_t* src0, int32_t* src1, void* stream);
template void LaunchTROWEXPANDSUB<int32_t, 64, 64>(int32_t* out, int32_t* src0, int32_t* src1, void* stream);
template void LaunchTROWEXPANDADD<int32_t, 64, 64>(int32_t* out, int32_t* src0, int32_t* src1, void* stream);
template void LaunchTROWEXPANDMAX<int32_t, 64, 64>(int32_t* out, int32_t* src0, int32_t* src1, void* stream);
template void LaunchTROWEXPANDMIN<int32_t, 64, 64>(int32_t* out, int32_t* src0, int32_t* src1, void* stream);

template void LaunchTROWEXPANDDIV<uint16_t, 64, 64>(uint16_t* out, uint16_t* src0, uint16_t* src1, void* stream);
template void LaunchTROWEXPANDMUL<uint16_t, 64, 64>(uint16_t* out, uint16_t* src0, uint16_t* src1, void* stream);
template void LaunchTROWEXPANDSUB<uint16_t, 64, 64>(uint16_t* out, uint16_t* src0, uint16_t* src1, void* stream);
template void LaunchTROWEXPANDADD<uint16_t, 64, 64>(uint16_t* out, uint16_t* src0, uint16_t* src1, void* stream);
template void LaunchTROWEXPANDMAX<uint16_t, 64, 64>(uint16_t* out, uint16_t* src0, uint16_t* src1, void* stream);
template void LaunchTROWEXPANDMIN<uint16_t, 64, 64>(uint16_t* out, uint16_t* src0, uint16_t* src1, void* stream);

template void LaunchTROWEXPANDDIV<uint32_t, 64, 64>(uint32_t* out, uint32_t* src0, uint32_t* src1, void* stream);
template void LaunchTROWEXPANDMUL<uint32_t, 64, 64>(uint32_t* out, uint32_t* src0, uint32_t* src1, void* stream);
template void LaunchTROWEXPANDSUB<uint32_t, 64, 64>(uint32_t* out, uint32_t* src0, uint32_t* src1, void* stream);
template void LaunchTROWEXPANDADD<uint32_t, 64, 64>(uint32_t* out, uint32_t* src0, uint32_t* src1, void* stream);
template void LaunchTROWEXPANDMAX<uint32_t, 64, 64>(uint32_t* out, uint32_t* src0, uint32_t* src1, void* stream);
template void LaunchTROWEXPANDMIN<uint32_t, 64, 64>(uint32_t* out, uint32_t* src0, uint32_t* src1, void* stream);
