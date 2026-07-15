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
AICORE void runTCOLEXPANDOP(__gm__ T __out__* out, __gm__ T __in__* src0, __gm__ T __in__* src1, LaunchFn fn)
{
    using DynShapeDim5 = Shape<1, 1, 1, -1, -1>;
    using DynStridDim5 = Stride<1, 1, -1, -1, 1>;
    using GlobalData = GlobalTensor<T, DynShapeDim5, DynStridDim5>;

    using TileSrc0 = Tile<TileType::Vec, T, iRow, iCol, BLayout::RowMajor, -1, -1>;
    using TileDst = Tile<TileType::Vec, T, oRow, oCol, BLayout::RowMajor, -1, -1>;
    using TileVec = Tile<TileType::Vec, T, 1, iCol, BLayout::RowMajor, -1, -1>;
    TileSrc0 src0Tile(kTRows, kTCols);
    TileVec src1Tile(1, kTCols);
    TileDst dstTile(kTRows, kTCols);

    GlobalData src0Global(src0, DynShapeDim5(kTRows, kTCols), DynStridDim5(iRow, iCol));
    GlobalData src1Global(src1, DynShapeDim5(1, kTCols), DynStridDim5(1, iCol));
    GlobalData dstGlobal(out, DynShapeDim5(kTRows, kTCols), DynStridDim5(oRow, oCol));

    TASSIGN(src0Tile, 0);
    TASSIGN(src1Tile, iRow * iCol * sizeof(typename TileSrc0::DType));
    TASSIGN(dstTile, 2 * iRow * iCol * sizeof(typename TileVec::DType));

    TLOAD(src0Tile, src0Global);
    TLOAD(src1Tile, src1Global);
    fn(dstTile, src0Tile, src1Tile);
    TSTORE(dstGlobal, dstTile);
    out = dstGlobal.data();
}

template <
    typename T, int kTRows, int kTCols, int iRow = kTRows, int iCol = kTCols, int oRow = kTRows, int oCol = kTCols>
void LaunchTCOLEXPANDDIV(T* out, T* src0, T* src1, void* stream)
{
    using TileDst = Tile<TileType::Vec, T, oRow, oCol, BLayout::RowMajor, -1, -1>;
    using TileSrc0 = Tile<TileType::Vec, T, iRow, iCol, BLayout::RowMajor, -1, -1>;
    using TileSrc1 = Tile<TileType::Vec, T, 1, iCol, BLayout::RowMajor, -1, -1>;
    if constexpr (std::is_same_v<T, aclFloat16>) {
        runTCOLEXPANDOP<half, kTRows, kTCols, iRow, iCol, oRow, oCol>(
            (half*)(out), (half*)(src0), (half*)(src1),
            [](TileDst& dst, TileSrc0& src0, TileSrc1& src1) { TCOLEXPANDDIV(dst, src0, src1); });
    } else {
        runTCOLEXPANDOP<T, kTRows, kTCols, iRow, iCol, oRow, oCol>(
            out, src0, src1, [](TileDst& dst, TileSrc0& src0, TileSrc1& src1) { TCOLEXPANDDIV(dst, src0, src1); });
    }
}

template <
    typename T, int kTRows, int kTCols, int iRow = kTRows, int iCol = kTCols, int oRow = kTRows, int oCol = kTCols>
void LaunchTCOLEXPANDMUL(T* out, T* src0, T* src1, void* stream)
{
    using TileDst = Tile<TileType::Vec, T, oRow, oCol, BLayout::RowMajor, -1, -1>;
    using TileSrc0 = Tile<TileType::Vec, T, iRow, iCol, BLayout::RowMajor, -1, -1>;
    using TileSrc1 = Tile<TileType::Vec, T, 1, iCol, BLayout::RowMajor, -1, -1>;
    if constexpr (std::is_same_v<T, aclFloat16>) {
        runTCOLEXPANDOP<half, kTRows, kTCols, iRow, iCol, oRow, oCol>(
            (half*)(out), (half*)(src0), (half*)(src1),
            [](TileDst& dst, TileSrc0& src0, TileSrc1& src1) { TCOLEXPANDMUL(dst, src0, src1); });
    } else {
        runTCOLEXPANDOP<T, kTRows, kTCols, iRow, iCol, oRow, oCol>(
            out, src0, src1, [](TileDst& dst, TileSrc0& src0, TileSrc1& src1) { TCOLEXPANDMUL(dst, src0, src1); });
    }
}

template <
    typename T, int kTRows, int kTCols, int iRow = kTRows, int iCol = kTCols, int oRow = kTRows, int oCol = kTCols>
void LaunchTCOLEXPANDSUB(T* out, T* src0, T* src1, void* stream)
{
    using TileDst = Tile<TileType::Vec, T, oRow, oCol, BLayout::RowMajor, -1, -1>;
    using TileSrc0 = Tile<TileType::Vec, T, iRow, iCol, BLayout::RowMajor, -1, -1>;
    using TileSrc1 = Tile<TileType::Vec, T, 1, iCol, BLayout::RowMajor, -1, -1>;
    if constexpr (std::is_same_v<T, aclFloat16>) {
        runTCOLEXPANDOP<half, kTRows, kTCols, iRow, iCol, oRow, oCol>(
            (half*)(out), (half*)(src0), (half*)(src1),
            [](TileDst& dst, TileSrc0& src0, TileSrc1& src1) { TCOLEXPANDSUB(dst, src0, src1); });
    } else {
        runTCOLEXPANDOP<T, kTRows, kTCols, iRow, iCol, oRow, oCol>(
            out, src0, src1, [](TileDst& dst, TileSrc0& src0, TileSrc1& src1) { TCOLEXPANDSUB(dst, src0, src1); });
    }
}

template <
    typename T, int kTRows, int kTCols, int iRow = kTRows, int iCol = kTCols, int oRow = kTRows, int oCol = kTCols>
void LaunchTCOLEXPANDADD(T* out, T* src0, T* src1, void* stream)
{
    using TileDst = Tile<TileType::Vec, T, oRow, oCol, BLayout::RowMajor, -1, -1>;
    using TileSrc0 = Tile<TileType::Vec, T, iRow, iCol, BLayout::RowMajor, -1, -1>;
    using TileSrc1 = Tile<TileType::Vec, T, 1, iCol, BLayout::RowMajor, -1, -1>;
    if constexpr (std::is_same_v<T, aclFloat16>) {
        runTCOLEXPANDOP<half, kTRows, kTCols, iRow, iCol, oRow, oCol>(
            (half*)(out), (half*)(src0), (half*)(src1),
            [](TileDst& dst, TileSrc0& src0, TileSrc1& src1) { TCOLEXPANDADD(dst, src0, src1); });
    } else {
        runTCOLEXPANDOP<T, kTRows, kTCols, iRow, iCol, oRow, oCol>(
            out, src0, src1, [](TileDst& dst, TileSrc0& src0, TileSrc1& src1) { TCOLEXPANDADD(dst, src0, src1); });
    }
}

template <
    typename T, int kTRows, int kTCols, int iRow = kTRows, int iCol = kTCols, int oRow = kTRows, int oCol = kTCols>
void LaunchTCOLEXPANDMAX(T* out, T* src0, T* src1, void* stream)
{
    using TileDst = Tile<TileType::Vec, T, oRow, oCol, BLayout::RowMajor, -1, -1>;
    using TileSrc0 = Tile<TileType::Vec, T, iRow, iCol, BLayout::RowMajor, -1, -1>;
    using TileSrc1 = Tile<TileType::Vec, T, 1, iCol, BLayout::RowMajor, -1, -1>;
    if constexpr (std::is_same_v<T, aclFloat16>) {
        runTCOLEXPANDOP<half, kTRows, kTCols, iRow, iCol, oRow, oCol>(
            (half*)(out), (half*)(src0), (half*)(src1),
            [](TileDst& dst, TileSrc0& src0, TileSrc1& src1) { TCOLEXPANDMAX(dst, src0, src1); });
    } else {
        runTCOLEXPANDOP<T, kTRows, kTCols, iRow, iCol, oRow, oCol>(
            out, src0, src1, [](TileDst& dst, TileSrc0& src0, TileSrc1& src1) { TCOLEXPANDMAX(dst, src0, src1); });
    }
}

template <
    typename T, int kTRows, int kTCols, int iRow = kTRows, int iCol = kTCols, int oRow = kTRows, int oCol = kTCols>
void LaunchTCOLEXPANDMIN(T* out, T* src0, T* src1, void* stream)
{
    using TileDst = Tile<TileType::Vec, T, oRow, oCol, BLayout::RowMajor, -1, -1>;
    using TileSrc0 = Tile<TileType::Vec, T, iRow, iCol, BLayout::RowMajor, -1, -1>;
    using TileSrc1 = Tile<TileType::Vec, T, 1, iCol, BLayout::RowMajor, -1, -1>;
    if constexpr (std::is_same_v<T, aclFloat16>) {
        runTCOLEXPANDOP<half, kTRows, kTCols, iRow, iCol, oRow, oCol>(
            (half*)(out), (half*)(src0), (half*)(src1),
            [](TileDst& dst, TileSrc0& src0, TileSrc1& src1) { TCOLEXPANDMIN(dst, src0, src1); });
    } else {
        runTCOLEXPANDOP<T, kTRows, kTCols, iRow, iCol, oRow, oCol>(
            out, src0, src1, [](TileDst& dst, TileSrc0& src0, TileSrc1& src1) { TCOLEXPANDMIN(dst, src0, src1); });
    }
}

template <
    typename T, int kTRows, int kTCols, int iRow = kTRows, int iCol = kTCols, int oRow = kTRows, int oCol = kTCols>
void LaunchTCOLEXPANDEXPDIF(T* out, T* src0, T* src1, void* stream)
{
    using TileDst = Tile<TileType::Vec, T, oRow, oCol, BLayout::RowMajor, -1, -1>;
    using TileSrc0 = Tile<TileType::Vec, T, iRow, iCol, BLayout::RowMajor, -1, -1>;
    using TileSrc1 = Tile<TileType::Vec, T, 1, iCol, BLayout::RowMajor, -1, -1>;
    if constexpr (std::is_same_v<T, aclFloat16>) {
        runTCOLEXPANDOP<half, kTRows, kTCols, iRow, iCol, oRow, oCol>(
            (half*)(out), (half*)(src0), (half*)(src1),
            [](TileDst& dst, TileSrc0& src0, TileSrc1& src1) { TCOLEXPANDEXPDIF(dst, src0, src1); });
    } else {
        runTCOLEXPANDOP<T, kTRows, kTCols, iRow, iCol, oRow, oCol>(
            out, src0, src1, [](TileDst& dst, TileSrc0& src0, TileSrc1& src1) { TCOLEXPANDEXPDIF(dst, src0, src1); });
    }
}

template void LaunchTCOLEXPANDDIV<float, 64, 64>(float* out, float* src0, float* src1, void* stream);
template void LaunchTCOLEXPANDDIV<aclFloat16, 16, 256>(
    aclFloat16* out, aclFloat16* src0, aclFloat16* src1, void* stream);
template void LaunchTCOLEXPANDMUL<float, 64, 64>(float* out, float* src0, float* src1, void* stream);
template void LaunchTCOLEXPANDMUL<aclFloat16, 16, 256>(
    aclFloat16* out, aclFloat16* src0, aclFloat16* src1, void* stream);
template void LaunchTCOLEXPANDSUB<float, 64, 64>(float* out, float* src0, float* src1, void* stream);
template void LaunchTCOLEXPANDSUB<aclFloat16, 16, 256>(
    aclFloat16* out, aclFloat16* src0, aclFloat16* src1, void* stream);
template void LaunchTCOLEXPANDADD<float, 64, 64>(float* out, float* src0, float* src1, void* stream);
template void LaunchTCOLEXPANDADD<aclFloat16, 16, 256>(
    aclFloat16* out, aclFloat16* src0, aclFloat16* src1, void* stream);
template void LaunchTCOLEXPANDMAX<float, 64, 64>(float* out, float* src0, float* src1, void* stream);
template void LaunchTCOLEXPANDMAX<aclFloat16, 16, 256>(
    aclFloat16* out, aclFloat16* src0, aclFloat16* src1, void* stream);
template void LaunchTCOLEXPANDMIN<float, 64, 64>(float* out, float* src0, float* src1, void* stream);
template void LaunchTCOLEXPANDMIN<aclFloat16, 16, 256>(
    aclFloat16* out, aclFloat16* src0, aclFloat16* src1, void* stream);
template void LaunchTCOLEXPANDEXPDIF<float, 64, 64>(float* out, float* src0, float* src1, void* stream);
template void LaunchTCOLEXPANDEXPDIF<aclFloat16, 16, 256>(
    aclFloat16* out, aclFloat16* src0, aclFloat16* src1, void* stream);
template void LaunchTCOLEXPANDDIV<float, 16, 16, 32, 32, 64, 64>(float* out, float* src0, float* src1, void* stream);
template void LaunchTCOLEXPANDMUL<float, 16, 16, 32, 32, 64, 64>(float* out, float* src0, float* src1, void* stream);
template void LaunchTCOLEXPANDSUB<float, 16, 16, 32, 32, 64, 64>(float* out, float* src0, float* src1, void* stream);
template void LaunchTCOLEXPANDADD<float, 16, 16, 32, 32, 64, 64>(float* out, float* src0, float* src1, void* stream);
template void LaunchTCOLEXPANDMAX<float, 16, 16, 32, 32, 64, 64>(float* out, float* src0, float* src1, void* stream);
template void LaunchTCOLEXPANDMIN<float, 16, 16, 32, 32, 64, 64>(float* out, float* src0, float* src1, void* stream);
template void LaunchTCOLEXPANDEXPDIF<float, 16, 16, 32, 32, 64, 64>(float* out, float* src0, float* src1, void* stream);

template void LaunchTCOLEXPANDDIV<int16_t, 16, 256>(int16_t* out, int16_t* src0, int16_t* src1, void* stream);
template void LaunchTCOLEXPANDMUL<int16_t, 16, 256>(int16_t* out, int16_t* src0, int16_t* src1, void* stream);
template void LaunchTCOLEXPANDSUB<int16_t, 16, 256>(int16_t* out, int16_t* src0, int16_t* src1, void* stream);
template void LaunchTCOLEXPANDADD<int16_t, 16, 256>(int16_t* out, int16_t* src0, int16_t* src1, void* stream);
template void LaunchTCOLEXPANDMAX<int16_t, 16, 256>(int16_t* out, int16_t* src0, int16_t* src1, void* stream);
template void LaunchTCOLEXPANDMIN<int16_t, 16, 256>(int16_t* out, int16_t* src0, int16_t* src1, void* stream);

template void LaunchTCOLEXPANDDIV<int32_t, 64, 64>(int32_t* out, int32_t* src0, int32_t* src1, void* stream);
template void LaunchTCOLEXPANDMUL<int32_t, 64, 64>(int32_t* out, int32_t* src0, int32_t* src1, void* stream);
template void LaunchTCOLEXPANDSUB<int32_t, 64, 64>(int32_t* out, int32_t* src0, int32_t* src1, void* stream);
template void LaunchTCOLEXPANDADD<int32_t, 64, 64>(int32_t* out, int32_t* src0, int32_t* src1, void* stream);
template void LaunchTCOLEXPANDMAX<int32_t, 64, 64>(int32_t* out, int32_t* src0, int32_t* src1, void* stream);
template void LaunchTCOLEXPANDMIN<int32_t, 64, 64>(int32_t* out, int32_t* src0, int32_t* src1, void* stream);

template void LaunchTCOLEXPANDDIV<uint16_t, 64, 64>(uint16_t* out, uint16_t* src0, uint16_t* src1, void* stream);
template void LaunchTCOLEXPANDMUL<uint16_t, 64, 64>(uint16_t* out, uint16_t* src0, uint16_t* src1, void* stream);
template void LaunchTCOLEXPANDSUB<uint16_t, 64, 64>(uint16_t* out, uint16_t* src0, uint16_t* src1, void* stream);
template void LaunchTCOLEXPANDADD<uint16_t, 64, 64>(uint16_t* out, uint16_t* src0, uint16_t* src1, void* stream);
template void LaunchTCOLEXPANDMAX<uint16_t, 64, 64>(uint16_t* out, uint16_t* src0, uint16_t* src1, void* stream);
template void LaunchTCOLEXPANDMIN<uint16_t, 64, 64>(uint16_t* out, uint16_t* src0, uint16_t* src1, void* stream);

template void LaunchTCOLEXPANDDIV<uint32_t, 64, 64>(uint32_t* out, uint32_t* src0, uint32_t* src1, void* stream);
template void LaunchTCOLEXPANDMUL<uint32_t, 64, 64>(uint32_t* out, uint32_t* src0, uint32_t* src1, void* stream);
template void LaunchTCOLEXPANDSUB<uint32_t, 64, 64>(uint32_t* out, uint32_t* src0, uint32_t* src1, void* stream);
template void LaunchTCOLEXPANDADD<uint32_t, 64, 64>(uint32_t* out, uint32_t* src0, uint32_t* src1, void* stream);
template void LaunchTCOLEXPANDMAX<uint32_t, 64, 64>(uint32_t* out, uint32_t* src0, uint32_t* src1, void* stream);
template void LaunchTCOLEXPANDMIN<uint32_t, 64, 64>(uint32_t* out, uint32_t* src0, uint32_t* src1, void* stream);
