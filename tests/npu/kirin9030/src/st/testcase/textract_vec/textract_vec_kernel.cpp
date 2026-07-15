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

using namespace pto;

template <
    typename T, uint32_t SrcRows, uint32_t SrcCols, uint32_t DstStaticRows, uint32_t DstStaticCols,
    uint32_t DstValidRows, uint32_t DstValidCols, uint32_t IdxRow, uint32_t IdxCol>
__global__ AICORE void RunTExtractNDVec(__gm__ T* out, __gm__ T* srcIn, __gm__ T* dstInitIn)
{
    constexpr bool isFullValid = (DstValidRows == DstStaticRows) && (DstValidCols == DstStaticCols);

    using SrcShape = pto::Shape<1, 1, 1, SrcRows, SrcCols>;
    using SrcStride = pto::Stride<SrcRows * SrcCols, SrcRows * SrcCols, SrcRows * SrcCols, SrcCols, 1>;
    using SrcGlobal = GlobalTensor<T, SrcShape, SrcStride>;

    using DstShape = pto::Shape<1, 1, 1, DstStaticRows, DstStaticCols>;
    using DstStride = pto::Stride<
        DstStaticRows * DstStaticCols, DstStaticRows * DstStaticCols, DstStaticRows * DstStaticCols, DstStaticCols, 1>;
    using DstGlobal = GlobalTensor<T, DstShape, DstStride>;

    using SrcVec = Tile<TileType::Vec, T, SrcRows, SrcCols, BLayout::RowMajor>;
    using DstFullVec = Tile<TileType::Vec, T, DstStaticRows, DstStaticCols, BLayout::RowMajor>;
    using DstExtractVec =
        Tile<TileType::Vec, T, DstStaticRows, DstStaticCols, BLayout::RowMajor, DstValidRows, DstValidCols>;

    SrcVec srcTile;
    DstFullVec dstFull;
    DstExtractVec dstExtract;

    TASSIGN<0x0>(srcTile);
    constexpr uint32_t srcBytes = SrcRows * SrcCols * sizeof(T);
    constexpr uint32_t dstAssignAddr = ((srcBytes + 0xFF) / 0x100) * 0x100;
    TASSIGN<dstAssignAddr>(dstFull);

    SrcGlobal srcGlobal(srcIn);
    DstGlobal dstInitGlobal(dstInitIn);
    DstGlobal outGlobal(out);

#if defined(__DAV_VEC__)
    TLOAD(srcTile, srcGlobal);
    if constexpr (!isFullValid) {
        TLOAD(dstFull, dstInitGlobal);
    }
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

    if constexpr (isFullValid) {
        TEXTRACT(dstFull, srcTile, static_cast<uint16_t>(IdxRow), static_cast<uint16_t>(IdxCol));
    } else {
        TSUBVIEW(dstExtract, dstFull, 0, 0);
        TEXTRACT(dstExtract, srcTile, static_cast<uint16_t>(IdxRow), static_cast<uint16_t>(IdxCol));
    }

    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);

    TSTORE(outGlobal, dstFull);
#endif
}

template <typename T, uint32_t SrcRows, uint32_t SrcCols, uint32_t IdxRow, uint32_t IdxCol>
__global__ AICORE void RunTExtractNDVecScalar(__gm__ T* out, __gm__ T* srcIn, __gm__ T* dstInitIn)
{
    constexpr uint32_t MinAlignedCols = BLOCK_BYTE_SIZE / sizeof(T);
    using SrcShape = pto::Shape<1, 1, 1, SrcRows, SrcCols>;
    using SrcStride = pto::Stride<SrcRows * SrcCols, SrcRows * SrcCols, SrcRows * SrcCols, SrcCols, 1>;
    using SrcGlobal = GlobalTensor<T, SrcShape, SrcStride>;

    using DstShape = pto::Shape<1, 1, 1, 1, MinAlignedCols>;
    using DstStride = pto::Stride<MinAlignedCols, MinAlignedCols, MinAlignedCols, MinAlignedCols, 1>;
    using DstGlobal = GlobalTensor<T, DstShape, DstStride>;

    using SrcVec = Tile<TileType::Vec, T, SrcRows, SrcCols, BLayout::RowMajor>;
    using DstFullVec = Tile<TileType::Vec, T, 1, MinAlignedCols, BLayout::RowMajor>;
    using DstExtractVec = Tile<TileType::Vec, T, 1, MinAlignedCols, BLayout::RowMajor, 1, 1>;

    SrcVec srcTile;
    DstFullVec dstFull;
    DstExtractVec dstExtract;

    TASSIGN<0x0>(srcTile);
    constexpr uint32_t srcBytes = SrcRows * SrcCols * sizeof(T);
    constexpr uint32_t dstAssignAddr = ((srcBytes + 0xFF) / 0x100) * 0x100;
    TASSIGN<dstAssignAddr>(dstFull);

    SrcGlobal srcGlobal(srcIn);
    DstGlobal dstInitGlobal(dstInitIn);
    DstGlobal outGlobal(out);

#if defined(__DAV_VEC__)
    TLOAD(srcTile, srcGlobal);
    TLOAD(dstFull, dstInitGlobal);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

    TSUBVIEW(dstExtract, dstFull, 0, 0);
    TEXTRACT(dstExtract, srcTile, static_cast<uint16_t>(IdxRow), static_cast<uint16_t>(IdxCol));

    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);

    TSTORE(outGlobal, dstFull);
#endif
}

template <
    typename T, uint32_t SrcRows, uint32_t SrcCols, uint32_t DstRows, uint32_t DstCols, uint32_t IdxRow,
    uint32_t IdxCol = 0, uint32_t DstValidRows = DstRows, uint32_t DstValidCols = DstCols>
__global__ AICORE void RunTExtractNZVec(__gm__ T* out, __gm__ T* srcIn, __gm__ T* dstInitIn)
{
    constexpr uint32_t typeSize = sizeof(T);
    constexpr uint32_t c0Size = (BLOCK_BYTE_SIZE / typeSize);
    constexpr bool isFullValid = (DstValidRows == DstRows) && (DstValidCols == DstCols);

    using SrcShapeND = pto::Shape<1, 1, 1, SrcRows, SrcCols>;
    using SrcStrideND = pto::Stride<SrcRows * SrcCols, SrcRows * SrcCols, SrcRows * SrcCols, SrcCols, 1>;
    using SrcGlobalND = GlobalTensor<T, SrcShapeND, SrcStrideND>;

    using OutShapeNZ = pto::Shape<1, DstCols / c0Size, DstRows / FRACTAL_NZ_ROW, FRACTAL_NZ_ROW, c0Size>;
    using OutStrideNZ =
        pto::Stride<(DstCols / c0Size) * c0Size * DstRows, DstRows * c0Size, FRACTAL_NZ_ROW * c0Size, c0Size, 1>;
    using OutGlobalNZ = GlobalTensor<T, OutShapeNZ, OutStrideNZ, Layout::NZ>;

    using SrcNDTile = Tile<TileType::Vec, T, SrcRows, SrcCols, BLayout::RowMajor>;
    using SrcNZTile = Tile<TileType::Vec, T, SrcRows, SrcCols, BLayout::ColMajor, SrcRows, SrcCols, SLayout::RowMajor>;
    using DstNZFullTile =
        Tile<TileType::Vec, T, DstRows, DstCols, BLayout::ColMajor, DstRows, DstCols, SLayout::RowMajor>;
    using DstNZExtractTile =
        Tile<TileType::Vec, T, DstRows, DstCols, BLayout::ColMajor, DstValidRows, DstValidCols, SLayout::RowMajor>;

    SrcNDTile srcNDTile;
    SrcNZTile srcNZTile;
    DstNZFullTile dstFullTile;
    DstNZExtractTile dstExtractTile;

    TASSIGN<0x0>(srcNDTile);
    TASSIGN<SrcNDTile::Numel * sizeof(T)>(srcNZTile);
    TASSIGN<(SrcNDTile::Numel + SrcNZTile::Numel) * sizeof(T)>(dstFullTile);

    SrcGlobalND srcGlobal(srcIn);
    OutGlobalNZ dstInitGlobal(dstInitIn);
    OutGlobalNZ dstGlobal(out);

#if defined(__DAV_VEC__)
    TLOAD(srcNDTile, srcGlobal);
    if constexpr (!isFullValid) {
        TLOAD(dstFullTile, dstInitGlobal);
    }
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

    TMOV(srcNZTile, srcNDTile);
    if constexpr (isFullValid) {
        TEXTRACT(dstFullTile, srcNZTile, static_cast<uint16_t>(IdxRow), static_cast<uint16_t>(IdxCol));
    } else {
        TSUBVIEW(dstExtractTile, dstFullTile, 0, 0);
        TEXTRACT(dstExtractTile, srcNZTile, static_cast<uint16_t>(IdxRow), static_cast<uint16_t>(IdxCol));
    }

    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);

    TSTORE(dstGlobal, dstFullTile);
#endif
}

// 1-byte-dtype NZ vector kernel via int8-alias.
template <
    typename TByteType, uint32_t SrcRows, uint32_t SrcCols, uint32_t DstRows, uint32_t DstCols, uint32_t IdxRow,
    uint32_t IdxCol = 0, uint32_t DstValidRows = DstRows, uint32_t DstValidCols = DstCols>
__global__ AICORE void RunTExtractNZVecByteAlias(__gm__ uint8_t* out, __gm__ uint8_t* srcIn, __gm__ uint8_t* dstInitIn)
{
    constexpr uint32_t c0SizeI8 = BLOCK_BYTE_SIZE;
    constexpr bool isFullValid = (DstValidRows == DstRows) && (DstValidCols == DstCols);

    using SrcShapeND = pto::Shape<1, 1, 1, SrcRows, SrcCols>;
    using SrcStrideND = pto::Stride<SrcRows * SrcCols, SrcRows * SrcCols, SrcRows * SrcCols, SrcCols, 1>;
    using SrcGlobalI8ND = GlobalTensor<int8_t, SrcShapeND, SrcStrideND>;

    using NZShape = pto::Shape<1, DstCols / c0SizeI8, DstRows / FRACTAL_NZ_ROW, FRACTAL_NZ_ROW, c0SizeI8>;
    using NZStride = pto::Stride<
        (DstCols / c0SizeI8) * c0SizeI8 * DstRows, DstRows * c0SizeI8, FRACTAL_NZ_ROW * c0SizeI8, c0SizeI8, 1>;
    using NZGlobalI8 = GlobalTensor<int8_t, NZShape, NZStride, Layout::NZ>;

    using SrcNDTileI8 = Tile<TileType::Vec, int8_t, SrcRows, SrcCols, BLayout::RowMajor>;
    using SrcNZTileI8 =
        Tile<TileType::Vec, int8_t, SrcRows, SrcCols, BLayout::ColMajor, SrcRows, SrcCols, SLayout::RowMajor>;
    using DstNZFullTileI8 =
        Tile<TileType::Vec, int8_t, DstRows, DstCols, BLayout::ColMajor, DstRows, DstCols, SLayout::RowMajor>;
    using SrcNZTileT =
        Tile<TileType::Vec, TByteType, SrcRows, SrcCols, BLayout::ColMajor, SrcRows, SrcCols, SLayout::RowMajor>;
    using DstNZFullTileT =
        Tile<TileType::Vec, TByteType, DstRows, DstCols, BLayout::ColMajor, DstRows, DstCols, SLayout::RowMajor>;
    using DstNZExtractTileT = Tile<
        TileType::Vec, TByteType, DstRows, DstCols, BLayout::ColMajor, DstValidRows, DstValidCols, SLayout::RowMajor>;

    SrcNDTileI8 srcNDTile;
    SrcNZTileI8 srcNZTile;
    DstNZFullTileI8 dstFullTile;
    SrcNZTileT srcExtractTile;
    DstNZFullTileT dstFullExtractTile;
    DstNZExtractTileT dstExtractTile;

    TASSIGN<0x0>(srcNDTile);
    TASSIGN<SrcNDTileI8::Numel>(srcNZTile);
    TASSIGN<SrcNDTileI8::Numel + SrcNZTileI8::Numel>(dstFullTile);

    SrcGlobalI8ND srcGlobal(reinterpret_cast<__gm__ int8_t*>(srcIn));
    NZGlobalI8 dstInitGlobal(reinterpret_cast<__gm__ int8_t*>(dstInitIn));
    NZGlobalI8 dstGlobal(reinterpret_cast<__gm__ int8_t*>(out));

#if defined(__DAV_VEC__)
    TLOAD(srcNDTile, srcGlobal);
    if constexpr (!isFullValid) {
        TLOAD(dstFullTile, dstInitGlobal);
    }
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

    TMOV(srcNZTile, srcNDTile);
    TSUBVIEW(srcExtractTile, srcNZTile, 0, 0);
    if constexpr (isFullValid) {
        TSUBVIEW(dstFullExtractTile, dstFullTile, 0, 0);
        TEXTRACT(dstFullExtractTile, srcExtractTile, static_cast<uint16_t>(IdxRow), static_cast<uint16_t>(IdxCol));
    } else {
        TSUBVIEW(dstExtractTile, dstFullTile, 0, 0);
        TEXTRACT(dstExtractTile, srcExtractTile, static_cast<uint16_t>(IdxRow), static_cast<uint16_t>(IdxCol));
    }

    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);

    TSTORE(dstGlobal, dstFullTile);
#endif
}

template <
    typename T, uint32_t SrcRows, uint32_t SrcCols, uint32_t DstRows, uint32_t DstCols, uint32_t IdxRow,
    uint32_t IdxCol>
__global__ AICORE void RunTExtractNZVecScalar(__gm__ T* out, __gm__ T* srcIn, __gm__ T* dstInitIn)
{
    constexpr uint32_t c0Size = BLOCK_BYTE_SIZE / sizeof(T);

    using SrcShapeND = pto::Shape<1, 1, 1, SrcRows, SrcCols>;
    using SrcStrideND = pto::Stride<SrcRows * SrcCols, SrcRows * SrcCols, SrcRows * SrcCols, SrcCols, 1>;
    using SrcGlobalND = GlobalTensor<T, SrcShapeND, SrcStrideND>;

    using NZShape = pto::Shape<1, DstCols / c0Size, DstRows / FRACTAL_NZ_ROW, FRACTAL_NZ_ROW, c0Size>;
    using NZStride =
        pto::Stride<(DstCols / c0Size) * c0Size * DstRows, DstRows * c0Size, FRACTAL_NZ_ROW * c0Size, c0Size, 1>;
    using NZGlobal = GlobalTensor<T, NZShape, NZStride, Layout::NZ>;

    using SrcNDTile = Tile<TileType::Vec, T, SrcRows, SrcCols, BLayout::RowMajor>;
    using SrcNZTile = Tile<TileType::Vec, T, SrcRows, SrcCols, BLayout::ColMajor, SrcRows, SrcCols, SLayout::RowMajor>;
    using DstNZFullTile =
        Tile<TileType::Vec, T, DstRows, DstCols, BLayout::ColMajor, DstRows, DstCols, SLayout::RowMajor>;
    using DstNZScalarTile = Tile<TileType::Vec, T, DstRows, DstCols, BLayout::ColMajor, 1, 1, SLayout::RowMajor>;

    SrcNDTile srcNDTile;
    SrcNZTile srcNZTile;
    DstNZFullTile dstFullTile;
    DstNZScalarTile dstScalarTile;

    TASSIGN<0x0>(srcNDTile);
    TASSIGN<SrcNDTile::Numel * sizeof(T)>(srcNZTile);
    TASSIGN<(SrcNDTile::Numel + SrcNZTile::Numel) * sizeof(T)>(dstFullTile);

    SrcGlobalND srcGlobal(srcIn);
    NZGlobal dstInitGlobal(dstInitIn);
    NZGlobal dstGlobal(out);

#if defined(__DAV_VEC__)
    TLOAD(srcNDTile, srcGlobal);
    TLOAD(dstFullTile, dstInitGlobal);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

    TMOV(srcNZTile, srcNDTile);
    TSUBVIEW(dstScalarTile, dstFullTile, 0, 0);
    TEXTRACT(dstScalarTile, srcNZTile, static_cast<uint16_t>(IdxRow), static_cast<uint16_t>(IdxCol));

    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);

    TSTORE(dstGlobal, dstFullTile);
#endif
}

template <int32_t testKey>
void launchTExtractVecND(uint8_t* out, uint8_t* srcIn, uint8_t* dstInitIn, void* stream)
{
    if constexpr (testKey == 1) {
        RunTExtractNDVec<float, 16, 16, 8, 8, 8, 8, 0, 0><<<1, nullptr, stream>>>(
            reinterpret_cast<float*>(out), reinterpret_cast<float*>(srcIn), reinterpret_cast<float*>(dstInitIn));
    } else if constexpr (testKey == 2) {
        RunTExtractNDVec<float, 16, 16, 8, 8, 8, 8, 4, 8><<<1, nullptr, stream>>>(
            reinterpret_cast<float*>(out), reinterpret_cast<float*>(srcIn), reinterpret_cast<float*>(dstInitIn));
    } else if constexpr (testKey == 3) {
        RunTExtractNDVec<half, 32, 32, 16, 16, 16, 16, 8, 16><<<1, nullptr, stream>>>(
            reinterpret_cast<half*>(out), reinterpret_cast<half*>(srcIn), reinterpret_cast<half*>(dstInitIn));
    } else if constexpr (testKey == 5) {
        RunTExtractNDVec<int32_t, 16, 16, 8, 8, 8, 8, 4, 0><<<1, nullptr, stream>>>(
            reinterpret_cast<int32_t*>(out), reinterpret_cast<int32_t*>(srcIn), reinterpret_cast<int32_t*>(dstInitIn));
    } else if constexpr (testKey == 6) {
        RunTExtractNDVec<int8_t, 64, 64, 32, 32, 32, 32, 0, 32><<<1, nullptr, stream>>>(
            reinterpret_cast<int8_t*>(out), reinterpret_cast<int8_t*>(srcIn), reinterpret_cast<int8_t*>(dstInitIn));
    } else if constexpr (testKey == 7) {
        RunTExtractNDVec<float, 16, 16, 8, 8, 8, 6, 0, 0><<<1, nullptr, stream>>>(
            reinterpret_cast<float*>(out), reinterpret_cast<float*>(srcIn), reinterpret_cast<float*>(dstInitIn));
    } else if constexpr (testKey == 8) {
        RunTExtractNDVec<half, 16, 32, 8, 16, 8, 12, 4, 0><<<1, nullptr, stream>>>(
            reinterpret_cast<half*>(out), reinterpret_cast<half*>(srcIn), reinterpret_cast<half*>(dstInitIn));
    } else if constexpr (testKey == 9) {
        RunTExtractNDVec<float, 16, 16, 8, 8, 8, 8, 0, 3><<<1, nullptr, stream>>>(
            reinterpret_cast<float*>(out), reinterpret_cast<float*>(srcIn), reinterpret_cast<float*>(dstInitIn));
    } else if constexpr (testKey == 10) {
        RunTExtractNDVec<half, 16, 48, 8, 16, 8, 16, 2, 5><<<1, nullptr, stream>>>(
            reinterpret_cast<half*>(out), reinterpret_cast<half*>(srcIn), reinterpret_cast<half*>(dstInitIn));
    } else if constexpr (testKey == 11) {
        RunTExtractNDVec<int8_t, 64, 64, 32, 32, 32, 32, 0, 7><<<1, nullptr, stream>>>(
            reinterpret_cast<int8_t*>(out), reinterpret_cast<int8_t*>(srcIn), reinterpret_cast<int8_t*>(dstInitIn));
    } else if constexpr (testKey == 12) {
        RunTExtractNDVec<int8_t, 64, 64, 32, 32, 32, 24, 8, 0><<<1, nullptr, stream>>>(
            reinterpret_cast<int8_t*>(out), reinterpret_cast<int8_t*>(srcIn), reinterpret_cast<int8_t*>(dstInitIn));
    } else if constexpr (testKey == 16) {
        // ND partial valid: half src 32x32, dst static 16x16 valid 4x16, idxRow=2, idxCol=8
        RunTExtractNDVec<half, 32, 32, 16, 16, 4, 16, 2, 8><<<1, nullptr, stream>>>(
            reinterpret_cast<half*>(out), reinterpret_cast<half*>(srcIn), reinterpret_cast<half*>(dstInitIn));
    }
}

template <int32_t testKey>
void launchTExtractVecNDScalar(uint8_t* out, uint8_t* srcIn, uint8_t* dstInitIn, void* stream)
{
    if constexpr (testKey == 1) {
        RunTExtractNDVecScalar<float, 16, 16, 5, 7><<<1, nullptr, stream>>>(
            reinterpret_cast<float*>(out), reinterpret_cast<float*>(srcIn), reinterpret_cast<float*>(dstInitIn));
    } else if constexpr (testKey == 2) {
        RunTExtractNDVecScalar<half, 32, 32, 10, 15><<<1, nullptr, stream>>>(
            reinterpret_cast<half*>(out), reinterpret_cast<half*>(srcIn), reinterpret_cast<half*>(dstInitIn));
    } else if constexpr (testKey == 4) {
        RunTExtractNDVecScalar<int8_t, 64, 64, 20, 30><<<1, nullptr, stream>>>(
            reinterpret_cast<int8_t*>(out), reinterpret_cast<int8_t*>(srcIn), reinterpret_cast<int8_t*>(dstInitIn));
    } else if constexpr (testKey == 5) {
        RunTExtractNDVecScalar<int32_t, 16, 16, 7, 9><<<1, nullptr, stream>>>(
            reinterpret_cast<int32_t*>(out), reinterpret_cast<int32_t*>(srcIn), reinterpret_cast<int32_t*>(dstInitIn));
    }
}

template <int32_t testKey>
void launchTExtractVecNZ(uint8_t* out, uint8_t* srcIn, uint8_t* dstInitIn, void* stream)
{
    if constexpr (testKey == 1) {
        RunTExtractNZVec<float, 32, 32, 16, 32, 0><<<1, nullptr, stream>>>(
            reinterpret_cast<float*>(out), reinterpret_cast<float*>(srcIn), reinterpret_cast<float*>(dstInitIn));
    } else if constexpr (testKey == 2) {
        RunTExtractNZVec<float, 32, 32, 16, 32, 16><<<1, nullptr, stream>>>(
            reinterpret_cast<float*>(out), reinterpret_cast<float*>(srcIn), reinterpret_cast<float*>(dstInitIn));
    } else if constexpr (testKey == 3) {
        RunTExtractNZVec<half, 32, 32, 16, 32, 0><<<1, nullptr, stream>>>(
            reinterpret_cast<half*>(out), reinterpret_cast<half*>(srcIn), reinterpret_cast<half*>(dstInitIn));
    } else if constexpr (testKey == 5) {
        RunTExtractNZVec<int8_t, 32, 64, 16, 64, 0><<<1, nullptr, stream>>>(
            reinterpret_cast<int8_t*>(out), reinterpret_cast<int8_t*>(srcIn), reinterpret_cast<int8_t*>(dstInitIn));
    } else if constexpr (testKey == 6) {
        RunTExtractNZVec<int8_t, 32, 64, 16, 64, 16><<<1, nullptr, stream>>>(
            reinterpret_cast<int8_t*>(out), reinterpret_cast<int8_t*>(srcIn), reinterpret_cast<int8_t*>(dstInitIn));
    } else if constexpr (testKey == 7) {
        // indexCol != 0 (one fractal block over): src 32x64, dst 16x32, idxRow=8, idxCol=32
        RunTExtractNZVec<int8_t, 32, 64, 16, 32, 8, 32><<<1, nullptr, stream>>>(
            reinterpret_cast<int8_t*>(out), reinterpret_cast<int8_t*>(srcIn), reinterpret_cast<int8_t*>(dstInitIn));
    } else if constexpr (testKey == 8) {
        // partial valid: src 32x32, dst static 16x32 valid 8x16, idxRow=4, idxCol=0
        RunTExtractNZVec<half, 32, 32, 16, 32, 4, 0, 8, 16><<<1, nullptr, stream>>>(
            reinterpret_cast<half*>(out), reinterpret_cast<half*>(srcIn), reinterpret_cast<half*>(dstInitIn));
    } else if constexpr (testKey == 9) {
        // multi-fractal-row dst: src 64x32, dst 32x32 (2 fractal blocks of 16 rows), idxRow=0
        RunTExtractNZVec<half, 64, 32, 32, 32, 0><<<1, nullptr, stream>>>(
            reinterpret_cast<half*>(out), reinterpret_cast<half*>(srcIn), reinterpret_cast<half*>(dstInitIn));
    } else if constexpr (testKey == 13) {
        // int32 NZ vector with idxCol=8 (one fractal block over for int32 c0=8)
        RunTExtractNZVec<int32_t, 32, 16, 16, 8, 4, 8><<<1, nullptr, stream>>>(
            reinterpret_cast<int32_t*>(out), reinterpret_cast<int32_t*>(srcIn), reinterpret_cast<int32_t*>(dstInitIn));
    }
}

template <int32_t testKey>
void launchTExtractVecNZScalar(uint8_t* out, uint8_t* srcIn, uint8_t* dstInitIn, void* stream)
{
    if constexpr (testKey == 1) {
        RunTExtractNZVecScalar<float, 32, 32, 16, 32, 5, 9><<<1, nullptr, stream>>>(
            reinterpret_cast<float*>(out), reinterpret_cast<float*>(srcIn), reinterpret_cast<float*>(dstInitIn));
    } else if constexpr (testKey == 2) {
        RunTExtractNZVecScalar<half, 32, 32, 16, 32, 7, 14><<<1, nullptr, stream>>>(
            reinterpret_cast<half*>(out), reinterpret_cast<half*>(srcIn), reinterpret_cast<half*>(dstInitIn));
    } else if constexpr (testKey == 4) {
        RunTExtractNZVecScalar<int8_t, 32, 64, 16, 64, 20, 33><<<1, nullptr, stream>>>(
            reinterpret_cast<int8_t*>(out), reinterpret_cast<int8_t*>(srcIn), reinterpret_cast<int8_t*>(dstInitIn));
    } else if constexpr (testKey == 5) {
        RunTExtractNZVecScalar<int32_t, 32, 16, 16, 16, 4, 7><<<1, nullptr, stream>>>(
            reinterpret_cast<int32_t*>(out), reinterpret_cast<int32_t*>(srcIn), reinterpret_cast<int32_t*>(dstInitIn));
    }
}

template void launchTExtractVecND<1>(uint8_t*, uint8_t*, uint8_t*, void*);
template void launchTExtractVecND<2>(uint8_t*, uint8_t*, uint8_t*, void*);
template void launchTExtractVecND<3>(uint8_t*, uint8_t*, uint8_t*, void*);
template void launchTExtractVecND<5>(uint8_t*, uint8_t*, uint8_t*, void*);
template void launchTExtractVecND<6>(uint8_t*, uint8_t*, uint8_t*, void*);
template void launchTExtractVecND<7>(uint8_t*, uint8_t*, uint8_t*, void*);
template void launchTExtractVecND<8>(uint8_t*, uint8_t*, uint8_t*, void*);
template void launchTExtractVecND<9>(uint8_t*, uint8_t*, uint8_t*, void*);
template void launchTExtractVecND<10>(uint8_t*, uint8_t*, uint8_t*, void*);
template void launchTExtractVecND<11>(uint8_t*, uint8_t*, uint8_t*, void*);
template void launchTExtractVecND<12>(uint8_t*, uint8_t*, uint8_t*, void*);
template void launchTExtractVecND<16>(uint8_t*, uint8_t*, uint8_t*, void*);

template void launchTExtractVecNDScalar<1>(uint8_t*, uint8_t*, uint8_t*, void*);
template void launchTExtractVecNDScalar<2>(uint8_t*, uint8_t*, uint8_t*, void*);
template void launchTExtractVecNDScalar<4>(uint8_t*, uint8_t*, uint8_t*, void*);
template void launchTExtractVecNDScalar<5>(uint8_t*, uint8_t*, uint8_t*, void*);

template void launchTExtractVecNZ<1>(uint8_t*, uint8_t*, uint8_t*, void*);
template void launchTExtractVecNZ<2>(uint8_t*, uint8_t*, uint8_t*, void*);
template void launchTExtractVecNZ<3>(uint8_t*, uint8_t*, uint8_t*, void*);
template void launchTExtractVecNZ<5>(uint8_t*, uint8_t*, uint8_t*, void*);
template void launchTExtractVecNZ<6>(uint8_t*, uint8_t*, uint8_t*, void*);
template void launchTExtractVecNZ<7>(uint8_t*, uint8_t*, uint8_t*, void*);
template void launchTExtractVecNZ<8>(uint8_t*, uint8_t*, uint8_t*, void*);
template void launchTExtractVecNZ<9>(uint8_t*, uint8_t*, uint8_t*, void*);
template void launchTExtractVecNZ<13>(uint8_t*, uint8_t*, uint8_t*, void*);

template void launchTExtractVecNZScalar<1>(uint8_t*, uint8_t*, uint8_t*, void*);
template void launchTExtractVecNZScalar<2>(uint8_t*, uint8_t*, uint8_t*, void*);
template void launchTExtractVecNZScalar<4>(uint8_t*, uint8_t*, uint8_t*, void*);
template void launchTExtractVecNZScalar<5>(uint8_t*, uint8_t*, uint8_t*, void*);
