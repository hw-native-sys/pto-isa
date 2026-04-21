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
#include <pto/common/pto_tile.hpp>
#include <pto/common/constants.hpp>

using namespace pto;

template <typename T, uint32_t SrcRows, uint32_t SrcCols, uint32_t DstStaticRows, uint32_t DstStaticCols,
          uint32_t DstValidRows, uint32_t DstValidCols, uint32_t IdxRow, uint32_t IdxCol>
__global__ AICORE void RunTExtractNDVec(__gm__ T *out, __gm__ T *srcIn, __gm__ T *dstInitIn)
{
    constexpr bool isFullValid = (DstValidRows == DstStaticRows) && (DstValidCols == DstStaticCols);

    using SrcShape = pto::Shape<1, 1, 1, SrcRows, SrcCols>;
    using SrcStride = pto::Stride<SrcRows * SrcCols, SrcRows * SrcCols, SrcRows * SrcCols, SrcCols, 1>;
    using SrcGlobal = GlobalTensor<T, SrcShape, SrcStride>;

    using DstShape = pto::Shape<1, 1, 1, DstStaticRows, DstStaticCols>;
    using DstStride = pto::Stride<DstStaticRows * DstStaticCols, DstStaticRows * DstStaticCols,
                                  DstStaticRows * DstStaticCols, DstStaticCols, 1>;
    using DstGlobal = GlobalTensor<T, DstShape, DstStride>;

    using SrcVec = Tile<TileType::Vec, T, SrcRows, SrcCols, BLayout::RowMajor>;
    using DstFullVec = Tile<TileType::Vec, T, DstStaticRows, DstStaticCols, BLayout::RowMajor>;
    using DstExtractVec =
        Tile<TileType::Vec, T, DstStaticRows, DstStaticCols, BLayout::RowMajor, DstValidRows, DstValidCols>;

    SrcVec srcTile;
    DstFullVec dstFull;
    DstExtractVec dstExtract;

    TASSIGN(srcTile, 0x0);
    constexpr uint32_t srcBytes = SrcRows * SrcCols * sizeof(T);
    constexpr uint32_t dstAssignAddr = ((srcBytes + 0xFF) / 0x100) * 0x100;
    TASSIGN(dstFull, dstAssignAddr);

    SrcGlobal srcGlobal(srcIn);
    DstGlobal dstInitGlobal(dstInitIn);
    DstGlobal outGlobal(out);

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
}

template <typename T, uint32_t SrcRows, uint32_t SrcCols, uint32_t IdxRow, uint32_t IdxCol>
__global__ AICORE void RunTExtractNDVecScalar(__gm__ T *out, __gm__ T *srcIn, __gm__ T *dstInitIn)
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

    TASSIGN(srcTile, 0x0);
    constexpr uint32_t srcBytes = SrcRows * SrcCols * sizeof(T);
    constexpr uint32_t dstAssignAddr = ((srcBytes + 0xFF) / 0x100) * 0x100;
    TASSIGN(dstFull, dstAssignAddr);

    SrcGlobal srcGlobal(srcIn);
    DstGlobal dstInitGlobal(dstInitIn);
    DstGlobal outGlobal(out);

    TLOAD(srcTile, srcGlobal);
    TLOAD(dstFull, dstInitGlobal);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

    TSUBVIEW(dstExtract, dstFull, 0, 0);
    TEXTRACT(dstExtract, srcTile, static_cast<uint16_t>(IdxRow), static_cast<uint16_t>(IdxCol));

    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);

    TSTORE(outGlobal, dstFull);
}

template <typename T, uint32_t SrcRows, uint32_t SrcCols, uint32_t DstRows, uint32_t DstCols, uint32_t IdxRow,
          uint32_t IdxCol = 0, uint32_t DstValidRows = DstRows, uint32_t DstValidCols = DstCols>
__global__ AICORE void RunTExtractNZVec(__gm__ T *out, __gm__ T *srcIn, __gm__ T *dstInitIn)
{
    constexpr uint32_t typeSize = sizeof(T);
    constexpr uint32_t c0Size = BLOCK_BYTE_SIZE / typeSize;
    constexpr bool isFullValid = (DstValidRows == DstRows) && (DstValidCols == DstCols);

    using SrcShapeNZ = pto::Shape<1, SrcCols / c0Size, SrcRows / FRACTAL_NZ_ROW, FRACTAL_NZ_ROW, c0Size>;
    using SrcStrideNZ =
        pto::Stride<(SrcCols / c0Size) * c0Size * SrcRows, SrcRows * c0Size, FRACTAL_NZ_ROW * c0Size, c0Size, 1>;
    using SrcGlobalNZ = GlobalTensor<T, SrcShapeNZ, SrcStrideNZ, Layout::NZ>;

    using OutShapeNZ = pto::Shape<1, DstCols / c0Size, DstRows / FRACTAL_NZ_ROW, FRACTAL_NZ_ROW, c0Size>;
    using OutStrideNZ =
        pto::Stride<(DstCols / c0Size) * c0Size * DstRows, DstRows * c0Size, FRACTAL_NZ_ROW * c0Size, c0Size, 1>;
    using OutGlobalNZ = GlobalTensor<T, OutShapeNZ, OutStrideNZ, Layout::NZ>;

    using SrcNZTile = Tile<TileType::Vec, T, SrcRows, SrcCols, BLayout::ColMajor, SrcRows, SrcCols, SLayout::RowMajor>;
    using DstNZFullTile =
        Tile<TileType::Vec, T, DstRows, DstCols, BLayout::ColMajor, DstRows, DstCols, SLayout::RowMajor>;
    using DstNZExtractTile =
        Tile<TileType::Vec, T, DstRows, DstCols, BLayout::ColMajor, DstValidRows, DstValidCols, SLayout::RowMajor>;

    SrcNZTile srcNZTile;
    DstNZFullTile dstFullTile;
    DstNZExtractTile dstExtractTile;

    TASSIGN(srcNZTile, 0x0);
    constexpr uint32_t srcBytes = SrcRows * SrcCols * sizeof(T);
    constexpr uint32_t dstAssignAddr = ((srcBytes + 0xFF) / 0x100) * 0x100;
    TASSIGN(dstFullTile, dstAssignAddr);

    SrcGlobalNZ srcGlobal(srcIn);
    OutGlobalNZ dstInitGlobal(dstInitIn);
    OutGlobalNZ dstGlobal(out);

    TLOAD(srcNZTile, srcGlobal);
    if constexpr (!isFullValid) {
        TLOAD(dstFullTile, dstInitGlobal);
    }
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

    if constexpr (isFullValid) {
        TEXTRACT(dstFullTile, srcNZTile, static_cast<uint16_t>(IdxRow), static_cast<uint16_t>(IdxCol));
    } else {
        TSUBVIEW(dstExtractTile, dstFullTile, 0, 0);
        TEXTRACT(dstExtractTile, srcNZTile, static_cast<uint16_t>(IdxRow), static_cast<uint16_t>(IdxCol));
    }

    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);

    TSTORE(dstGlobal, dstFullTile);
}

template <typename T, uint32_t SrcRows, uint32_t SrcCols, uint32_t DstRows, uint32_t DstCols, uint32_t IdxRow,
          uint32_t IdxCol>
__global__ AICORE void RunTExtractNZVecScalar(__gm__ T *out, __gm__ T *srcIn, __gm__ T *dstInitIn)
{
    constexpr uint32_t typeSize = sizeof(T);
    constexpr uint32_t c0Size = BLOCK_BYTE_SIZE / typeSize;

    using SrcShapeNZ = pto::Shape<1, SrcCols / c0Size, SrcRows / FRACTAL_NZ_ROW, FRACTAL_NZ_ROW, c0Size>;
    using SrcStrideNZ =
        pto::Stride<(SrcCols / c0Size) * c0Size * SrcRows, SrcRows * c0Size, FRACTAL_NZ_ROW * c0Size, c0Size, 1>;
    using SrcGlobalNZ = GlobalTensor<T, SrcShapeNZ, SrcStrideNZ, Layout::NZ>;

    using OutShapeNZ = pto::Shape<1, DstCols / c0Size, DstRows / FRACTAL_NZ_ROW, FRACTAL_NZ_ROW, c0Size>;
    using OutStrideNZ =
        pto::Stride<(DstCols / c0Size) * c0Size * DstRows, DstRows * c0Size, FRACTAL_NZ_ROW * c0Size, c0Size, 1>;
    using OutGlobalNZ = GlobalTensor<T, OutShapeNZ, OutStrideNZ, Layout::NZ>;

    using SrcNZTile = Tile<TileType::Vec, T, SrcRows, SrcCols, BLayout::ColMajor, SrcRows, SrcCols, SLayout::RowMajor>;
    using DstNZFullTile =
        Tile<TileType::Vec, T, DstRows, DstCols, BLayout::ColMajor, DstRows, DstCols, SLayout::RowMajor>;
    using DstNZScalarTile = Tile<TileType::Vec, T, DstRows, DstCols, BLayout::ColMajor, 1, 1, SLayout::RowMajor>;

    SrcNZTile srcNZTile;
    DstNZFullTile dstFullTile;
    DstNZScalarTile dstScalarTile;

    TASSIGN(srcNZTile, 0x0);
    constexpr uint32_t srcBytes = SrcRows * SrcCols * sizeof(T);
    constexpr uint32_t dstAssignAddr = ((srcBytes + 0xFF) / 0x100) * 0x100;
    TASSIGN(dstFullTile, dstAssignAddr);

    SrcGlobalNZ srcGlobal(srcIn);
    OutGlobalNZ dstInitGlobal(dstInitIn);
    OutGlobalNZ dstGlobal(out);

    TLOAD(srcNZTile, srcGlobal);
    TLOAD(dstFullTile, dstInitGlobal);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

    TSUBVIEW(dstScalarTile, dstFullTile, 0, 0);
    TEXTRACT(dstScalarTile, srcNZTile, static_cast<uint16_t>(IdxRow), static_cast<uint16_t>(IdxCol));

    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);

    TSTORE(dstGlobal, dstFullTile);
}

template <int32_t testKey>
void launchTExtractVecND(uint8_t *out, uint8_t *srcIn, uint8_t *dstInitIn, void *stream)
{
    if constexpr (testKey == 1) {
        RunTExtractNDVec<float, 16, 16, 8, 8, 8, 8, 0, 0>
            <<<1, nullptr, stream>>>((__gm__ float *)out, (__gm__ float *)srcIn, (__gm__ float *)dstInitIn);
    } else if constexpr (testKey == 2) {
        RunTExtractNDVec<float, 16, 16, 8, 8, 8, 8, 4, 8>
            <<<1, nullptr, stream>>>((__gm__ float *)out, (__gm__ float *)srcIn, (__gm__ float *)dstInitIn);
    } else if constexpr (testKey == 3) {
        RunTExtractNDVec<half, 32, 32, 16, 16, 16, 16, 8, 16>
            <<<1, nullptr, stream>>>((__gm__ half *)out, (__gm__ half *)srcIn, (__gm__ half *)dstInitIn);
    } else if constexpr (testKey == 4) {
        RunTExtractNDVec<bfloat16_t, 32, 32, 16, 16, 16, 16, 0, 16><<<1, nullptr, stream>>>(
            (__gm__ bfloat16_t *)out, (__gm__ bfloat16_t *)srcIn, (__gm__ bfloat16_t *)dstInitIn);
    } else if constexpr (testKey == 5) {
        RunTExtractNDVec<int32_t, 16, 16, 8, 8, 8, 8, 4, 0>
            <<<1, nullptr, stream>>>((__gm__ int32_t *)out, (__gm__ int32_t *)srcIn, (__gm__ int32_t *)dstInitIn);
    } else if constexpr (testKey == 6) {
        RunTExtractNDVec<int8_t, 64, 64, 32, 32, 32, 32, 0, 32>
            <<<1, nullptr, stream>>>((__gm__ int8_t *)out, (__gm__ int8_t *)srcIn, (__gm__ int8_t *)dstInitIn);
    } else if constexpr (testKey == 7) {
        RunTExtractNDVec<half, 32, 32, 16, 16, 4, 16, 2, 16>
            <<<1, nullptr, stream>>>((__gm__ half *)out, (__gm__ half *)srcIn, (__gm__ half *)dstInitIn);
    } else if constexpr (testKey == 8) {
        RunTExtractNDVec<float, 16, 32, 8, 16, 8, 16, 0, 16>
            <<<1, nullptr, stream>>>((__gm__ float *)out, (__gm__ float *)srcIn, (__gm__ float *)dstInitIn);
    } else if constexpr (testKey == 9) {
        RunTExtractNDVec<bfloat16_t, 32, 64, 16, 32, 16, 32, 8, 32><<<1, nullptr, stream>>>(
            (__gm__ bfloat16_t *)out, (__gm__ bfloat16_t *)srcIn, (__gm__ bfloat16_t *)dstInitIn);
    } else if constexpr (testKey == 10) {
        RunTExtractNDVec<uint8_t, 64, 64, 32, 32, 32, 32, 0, 32>
            <<<1, nullptr, stream>>>((__gm__ uint8_t *)out, (__gm__ uint8_t *)srcIn, (__gm__ uint8_t *)dstInitIn);
    } else if constexpr (testKey == 11) {
        RunTExtractNDVec<int16_t, 32, 32, 16, 16, 16, 16, 8, 16>
            <<<1, nullptr, stream>>>((__gm__ int16_t *)out, (__gm__ int16_t *)srcIn, (__gm__ int16_t *)dstInitIn);
    } else if constexpr (testKey == 12) {
        RunTExtractNDVec<uint16_t, 32, 32, 16, 16, 16, 16, 0, 0>
            <<<1, nullptr, stream>>>((__gm__ uint16_t *)out, (__gm__ uint16_t *)srcIn, (__gm__ uint16_t *)dstInitIn);
    } else if constexpr (testKey == 13) {
        RunTExtractNDVec<uint32_t, 16, 16, 8, 8, 8, 8, 4, 8>
            <<<1, nullptr, stream>>>((__gm__ uint32_t *)out, (__gm__ uint32_t *)srcIn, (__gm__ uint32_t *)dstInitIn);
    } else if constexpr (testKey == 14) {
        RunTExtractNDVec<float, 32, 64, 16, 32, 8, 16, 8, 16>
            <<<1, nullptr, stream>>>((__gm__ float *)out, (__gm__ float *)srcIn, (__gm__ float *)dstInitIn);
    } else if constexpr (testKey == 15) {
        RunTExtractNDVec<int8_t, 32, 64, 16, 32, 16, 32, 16, 32>
            <<<1, nullptr, stream>>>((__gm__ int8_t *)out, (__gm__ int8_t *)srcIn, (__gm__ int8_t *)dstInitIn);
    } else if constexpr (testKey == 16) {
        RunTExtractNDVec<half, 64, 64, 32, 32, 8, 32, 24, 16>
            <<<1, nullptr, stream>>>((__gm__ half *)out, (__gm__ half *)srcIn, (__gm__ half *)dstInitIn);
    } else if constexpr (testKey == 17) {
        RunTExtractNDVec<float, 24, 48, 12, 32, 12, 32, 5, 8>
            <<<1, nullptr, stream>>>((__gm__ float *)out, (__gm__ float *)srcIn, (__gm__ float *)dstInitIn);
    } else if constexpr (testKey == 18) {
        RunTExtractNDVec<half, 30, 80, 14, 48, 14, 48, 3, 16>
            <<<1, nullptr, stream>>>((__gm__ half *)out, (__gm__ half *)srcIn, (__gm__ half *)dstInitIn);
    } else if constexpr (testKey == 19) {
        RunTExtractNDVec<int8_t, 48, 96, 24, 64, 24, 64, 7, 32>
            <<<1, nullptr, stream>>>((__gm__ int8_t *)out, (__gm__ int8_t *)srcIn, (__gm__ int8_t *)dstInitIn);
    } else if constexpr (testKey == 20) {
        RunTExtractNDVec<bfloat16_t, 24, 48, 12, 32, 6, 16, 11, 16><<<1, nullptr, stream>>>(
            (__gm__ bfloat16_t *)out, (__gm__ bfloat16_t *)srcIn, (__gm__ bfloat16_t *)dstInitIn);
    } else if constexpr (testKey == 21) {
        RunTExtractNDVec<int32_t, 20, 16, 10, 8, 10, 8, 9, 8>
            <<<1, nullptr, stream>>>((__gm__ int32_t *)out, (__gm__ int32_t *)srcIn, (__gm__ int32_t *)dstInitIn);
    } else if constexpr (testKey == 22) {
        RunTExtractNDVec<float, 24, 48, 12, 32, 6, 8, 15, 24>
            <<<1, nullptr, stream>>>((__gm__ float *)out, (__gm__ float *)srcIn, (__gm__ float *)dstInitIn);
    } else if constexpr (testKey == 23) {
        RunTExtractNDVec<float, 32, 32, 8, 16, 8, 10, 0, 0>
            <<<1, nullptr, stream>>>((__gm__ float *)out, (__gm__ float *)srcIn, (__gm__ float *)dstInitIn);
    } else if constexpr (testKey == 24) {
        RunTExtractNDVec<half, 32, 32, 8, 16, 8, 10, 0, 0>
            <<<1, nullptr, stream>>>((__gm__ half *)out, (__gm__ half *)srcIn, (__gm__ half *)dstInitIn);
    } else if constexpr (testKey == 25) {
        RunTExtractNDVec<bfloat16_t, 32, 32, 8, 16, 8, 14, 0, 0><<<1, nullptr, stream>>>(
            (__gm__ bfloat16_t *)out, (__gm__ bfloat16_t *)srcIn, (__gm__ bfloat16_t *)dstInitIn);
    } else if constexpr (testKey == 26) {
        RunTExtractNDVec<int16_t, 32, 32, 8, 16, 8, 11, 0, 0>
            <<<1, nullptr, stream>>>((__gm__ int16_t *)out, (__gm__ int16_t *)srcIn, (__gm__ int16_t *)dstInitIn);
    } else if constexpr (testKey == 27) {
        RunTExtractNDVec<int32_t, 32, 32, 8, 16, 8, 14, 0, 0>
            <<<1, nullptr, stream>>>((__gm__ int32_t *)out, (__gm__ int32_t *)srcIn, (__gm__ int32_t *)dstInitIn);
    } else if constexpr (testKey == 28) {
        RunTExtractNDVec<uint16_t, 32, 64, 16, 32, 16, 22, 0, 0>
            <<<1, nullptr, stream>>>((__gm__ uint16_t *)out, (__gm__ uint16_t *)srcIn, (__gm__ uint16_t *)dstInitIn);
    } else if constexpr (testKey == 29) {
        RunTExtractNDVec<int8_t, 32, 64, 16, 64, 16, 34, 0, 0>
            <<<1, nullptr, stream>>>((__gm__ int8_t *)out, (__gm__ int8_t *)srcIn, (__gm__ int8_t *)dstInitIn);
    } else if constexpr (testKey == 30) {
        RunTExtractNDVec<float, 32, 32, 8, 16, 8, 10, 4, 8>
            <<<1, nullptr, stream>>>((__gm__ float *)out, (__gm__ float *)srcIn, (__gm__ float *)dstInitIn);
    } else if constexpr (testKey == 31) {
        RunTExtractNDVec<uint16_t, 32, 32, 8, 16, 8, 15, 0, 0>
            <<<1, nullptr, stream>>>((__gm__ uint16_t *)out, (__gm__ uint16_t *)srcIn, (__gm__ uint16_t *)dstInitIn);
    } else if constexpr (testKey == 32) {
        RunTExtractNDVec<float, 32, 32, 8, 16, 8, 6, 0, 0>
            <<<1, nullptr, stream>>>((__gm__ float *)out, (__gm__ float *)srcIn, (__gm__ float *)dstInitIn);
    }
}

template <int32_t testKey>
void launchTExtractVecNDScalar(uint8_t *out, uint8_t *srcIn, uint8_t *dstInitIn, void *stream)
{
    if constexpr (testKey == 1) {
        RunTExtractNDVecScalar<float, 16, 16, 5, 7>
            <<<1, nullptr, stream>>>((__gm__ float *)out, (__gm__ float *)srcIn, (__gm__ float *)dstInitIn);
    } else if constexpr (testKey == 2) {
        RunTExtractNDVecScalar<half, 32, 32, 10, 15>
            <<<1, nullptr, stream>>>((__gm__ half *)out, (__gm__ half *)srcIn, (__gm__ half *)dstInitIn);
    } else if constexpr (testKey == 3) {
        RunTExtractNDVecScalar<bfloat16_t, 32, 32, 3, 11><<<1, nullptr, stream>>>(
            (__gm__ bfloat16_t *)out, (__gm__ bfloat16_t *)srcIn, (__gm__ bfloat16_t *)dstInitIn);
    } else if constexpr (testKey == 4) {
        RunTExtractNDVecScalar<int8_t, 64, 64, 20, 30>
            <<<1, nullptr, stream>>>((__gm__ int8_t *)out, (__gm__ int8_t *)srcIn, (__gm__ int8_t *)dstInitIn);
    } else if constexpr (testKey == 5) {
        RunTExtractNDVecScalar<int32_t, 16, 16, 7, 9>
            <<<1, nullptr, stream>>>((__gm__ int32_t *)out, (__gm__ int32_t *)srcIn, (__gm__ int32_t *)dstInitIn);
    } else if constexpr (testKey == 6) {
        RunTExtractNDVecScalar<int16_t, 32, 32, 0, 0>
            <<<1, nullptr, stream>>>((__gm__ int16_t *)out, (__gm__ int16_t *)srcIn, (__gm__ int16_t *)dstInitIn);
    } else if constexpr (testKey == 7) {
        RunTExtractNDVecScalar<uint16_t, 32, 32, 31, 31>
            <<<1, nullptr, stream>>>((__gm__ uint16_t *)out, (__gm__ uint16_t *)srcIn, (__gm__ uint16_t *)dstInitIn);
    } else if constexpr (testKey == 8) {
        RunTExtractNDVecScalar<uint32_t, 16, 16, 15, 15>
            <<<1, nullptr, stream>>>((__gm__ uint32_t *)out, (__gm__ uint32_t *)srcIn, (__gm__ uint32_t *)dstInitIn);
    } else if constexpr (testKey == 9) {
        RunTExtractNDVecScalar<half, 30, 80, 17, 23>
            <<<1, nullptr, stream>>>((__gm__ half *)out, (__gm__ half *)srcIn, (__gm__ half *)dstInitIn);
    } else if constexpr (testKey == 10) {
        RunTExtractNDVecScalar<int8_t, 48, 96, 41, 73>
            <<<1, nullptr, stream>>>((__gm__ int8_t *)out, (__gm__ int8_t *)srcIn, (__gm__ int8_t *)dstInitIn);
    } else if constexpr (testKey == 11) {
        RunTExtractNDVecScalar<float, 24, 48, 11, 13>
            <<<1, nullptr, stream>>>((__gm__ float *)out, (__gm__ float *)srcIn, (__gm__ float *)dstInitIn);
    }
}

template <int32_t testKey>
void launchTExtractVecNZ(uint8_t *out, uint8_t *srcIn, uint8_t *dstInitIn, void *stream)
{
    if constexpr (testKey == 1) {
        RunTExtractNZVec<float, 32, 32, 16, 32, 0>
            <<<1, nullptr, stream>>>((__gm__ float *)out, (__gm__ float *)srcIn, (__gm__ float *)dstInitIn);
    } else if constexpr (testKey == 2) {
        RunTExtractNZVec<float, 32, 32, 16, 32, 16>
            <<<1, nullptr, stream>>>((__gm__ float *)out, (__gm__ float *)srcIn, (__gm__ float *)dstInitIn);
    } else if constexpr (testKey == 3) {
        RunTExtractNZVec<half, 32, 32, 16, 32, 0>
            <<<1, nullptr, stream>>>((__gm__ half *)out, (__gm__ half *)srcIn, (__gm__ half *)dstInitIn);
    } else if constexpr (testKey == 4) {
        RunTExtractNZVec<bfloat16_t, 32, 32, 16, 32, 16><<<1, nullptr, stream>>>(
            (__gm__ bfloat16_t *)out, (__gm__ bfloat16_t *)srcIn, (__gm__ bfloat16_t *)dstInitIn);
    } else if constexpr (testKey == 5) {
        RunTExtractNZVec<int8_t, 32, 64, 16, 64, 0>
            <<<1, nullptr, stream>>>((__gm__ int8_t *)out, (__gm__ int8_t *)srcIn, (__gm__ int8_t *)dstInitIn);
    } else if constexpr (testKey == 6) {
        RunTExtractNZVec<int8_t, 32, 64, 16, 64, 16>
            <<<1, nullptr, stream>>>((__gm__ int8_t *)out, (__gm__ int8_t *)srcIn, (__gm__ int8_t *)dstInitIn);
    } else if constexpr (testKey == 7) {
        RunTExtractNZVec<half, 64, 32, 32, 32, 0>
            <<<1, nullptr, stream>>>((__gm__ half *)out, (__gm__ half *)srcIn, (__gm__ half *)dstInitIn);
    } else if constexpr (testKey == 8) {
        RunTExtractNZVec<int8_t, 32, 64, 16, 32, 16, 32, 16, 32>
            <<<1, nullptr, stream>>>((__gm__ int8_t *)out, (__gm__ int8_t *)srcIn, (__gm__ int8_t *)dstInitIn);
    } else if constexpr (testKey == 9) {
        RunTExtractNZVec<bfloat16_t, 32, 32, 16, 32, 0, 0, 16, 16><<<1, nullptr, stream>>>(
            (__gm__ bfloat16_t *)out, (__gm__ bfloat16_t *)srcIn, (__gm__ bfloat16_t *)dstInitIn);
    } else if constexpr (testKey == 10) {
        RunTExtractNZVec<uint8_t, 32, 64, 16, 64, 16>
            <<<1, nullptr, stream>>>((__gm__ uint8_t *)out, (__gm__ uint8_t *)srcIn, (__gm__ uint8_t *)dstInitIn);
    } else if constexpr (testKey == 11) {
        RunTExtractNZVec<int32_t, 32, 16, 16, 8, 16, 8, 16, 8>
            <<<1, nullptr, stream>>>((__gm__ int32_t *)out, (__gm__ int32_t *)srcIn, (__gm__ int32_t *)dstInitIn);
    } else if constexpr (testKey == 12) {
        RunTExtractNZVec<int16_t, 32, 32, 16, 32, 16>
            <<<1, nullptr, stream>>>((__gm__ int16_t *)out, (__gm__ int16_t *)srcIn, (__gm__ int16_t *)dstInitIn);
    } else if constexpr (testKey == 13) {
        RunTExtractNZVec<uint16_t, 32, 32, 16, 32, 0>
            <<<1, nullptr, stream>>>((__gm__ uint16_t *)out, (__gm__ uint16_t *)srcIn, (__gm__ uint16_t *)dstInitIn);
    } else if constexpr (testKey == 14) {
        RunTExtractNZVec<uint32_t, 32, 16, 16, 16, 0>
            <<<1, nullptr, stream>>>((__gm__ uint32_t *)out, (__gm__ uint32_t *)srcIn, (__gm__ uint32_t *)dstInitIn);
    } else if constexpr (testKey == 15) {
        RunTExtractNZVec<half, 64, 64, 32, 64, 32>
            <<<1, nullptr, stream>>>((__gm__ half *)out, (__gm__ half *)srcIn, (__gm__ half *)dstInitIn);
    } else if constexpr (testKey == 16) {
        RunTExtractNZVec<float, 64, 32, 32, 16, 16, 16, 16, 8>
            <<<1, nullptr, stream>>>((__gm__ float *)out, (__gm__ float *)srcIn, (__gm__ float *)dstInitIn);
    } else if constexpr (testKey == 17) {
        RunTExtractNZVec<bfloat16_t, 64, 64, 32, 32, 32, 32, 16, 16><<<1, nullptr, stream>>>(
            (__gm__ bfloat16_t *)out, (__gm__ bfloat16_t *)srcIn, (__gm__ bfloat16_t *)dstInitIn);
    } else if constexpr (testKey == 18) {
        RunTExtractNZVec<half, 48, 32, 32, 32, 16>
            <<<1, nullptr, stream>>>((__gm__ half *)out, (__gm__ half *)srcIn, (__gm__ half *)dstInitIn);
    } else if constexpr (testKey == 19) {
        RunTExtractNZVec<float, 48, 16, 16, 16, 32>
            <<<1, nullptr, stream>>>((__gm__ float *)out, (__gm__ float *)srcIn, (__gm__ float *)dstInitIn);
    } else if constexpr (testKey == 20) {
        RunTExtractNZVec<bfloat16_t, 48, 48, 32, 32, 16, 16, 16, 16><<<1, nullptr, stream>>>(
            (__gm__ bfloat16_t *)out, (__gm__ bfloat16_t *)srcIn, (__gm__ bfloat16_t *)dstInitIn);
    } else if constexpr (testKey == 21) {
        RunTExtractNZVec<int8_t, 48, 96, 32, 64, 16, 32, 16, 32>
            <<<1, nullptr, stream>>>((__gm__ int8_t *)out, (__gm__ int8_t *)srcIn, (__gm__ int8_t *)dstInitIn);
    } else if constexpr (testKey == 22) {
        RunTExtractNZVec<float, 32, 32, 16, 32, 0, 0, 8, 10>
            <<<1, nullptr, stream>>>((__gm__ float *)out, (__gm__ float *)srcIn, (__gm__ float *)dstInitIn);
    } else if constexpr (testKey == 23) {
        RunTExtractNZVec<half, 32, 64, 16, 32, 0, 0, 16, 22>
            <<<1, nullptr, stream>>>((__gm__ half *)out, (__gm__ half *)srcIn, (__gm__ half *)dstInitIn);
    } else if constexpr (testKey == 24) {
        RunTExtractNZVec<bfloat16_t, 32, 32, 16, 32, 0, 0, 16, 10><<<1, nullptr, stream>>>(
            (__gm__ bfloat16_t *)out, (__gm__ bfloat16_t *)srcIn, (__gm__ bfloat16_t *)dstInitIn);
    } else if constexpr (testKey == 25) {
        RunTExtractNZVec<int16_t, 32, 32, 16, 32, 0, 0, 16, 15>
            <<<1, nullptr, stream>>>((__gm__ int16_t *)out, (__gm__ int16_t *)srcIn, (__gm__ int16_t *)dstInitIn);
    } else if constexpr (testKey == 26) {
        RunTExtractNZVec<int32_t, 32, 16, 16, 16, 0, 0, 16, 14>
            <<<1, nullptr, stream>>>((__gm__ int32_t *)out, (__gm__ int32_t *)srcIn, (__gm__ int32_t *)dstInitIn);
    } else if constexpr (testKey == 27) {
        RunTExtractNZVec<int8_t, 32, 64, 16, 64, 0, 0, 16, 42>
            <<<1, nullptr, stream>>>((__gm__ int8_t *)out, (__gm__ int8_t *)srcIn, (__gm__ int8_t *)dstInitIn);
    } else if constexpr (testKey == 28) {
        RunTExtractNZVec<half, 32, 32, 16, 32, 0, 0, 10, 16>
            <<<1, nullptr, stream>>>((__gm__ half *)out, (__gm__ half *)srcIn, (__gm__ half *)dstInitIn);
    } else if constexpr (testKey == 29) {
        RunTExtractNZVec<float, 32, 32, 16, 32, 0, 0, 5, 10>
            <<<1, nullptr, stream>>>((__gm__ float *)out, (__gm__ float *)srcIn, (__gm__ float *)dstInitIn);
    } else if constexpr (testKey == 30) {
        RunTExtractNZVec<half, 64, 64, 16, 32, 16, 16, 16, 22>
            <<<1, nullptr, stream>>>((__gm__ half *)out, (__gm__ half *)srcIn, (__gm__ half *)dstInitIn);
    } else if constexpr (testKey == 31) {
        RunTExtractNZVec<bfloat16_t, 32, 32, 16, 32, 16, 16, 10, 10><<<1, nullptr, stream>>>(
            (__gm__ bfloat16_t *)out, (__gm__ bfloat16_t *)srcIn, (__gm__ bfloat16_t *)dstInitIn);
    }
}

template <int32_t testKey>
void launchTExtractVecNZScalar(uint8_t *out, uint8_t *srcIn, uint8_t *dstInitIn, void *stream)
{
    if constexpr (testKey == 1) {
        RunTExtractNZVecScalar<float, 32, 32, 16, 32, 5, 9>
            <<<1, nullptr, stream>>>((__gm__ float *)out, (__gm__ float *)srcIn, (__gm__ float *)dstInitIn);
    } else if constexpr (testKey == 2) {
        RunTExtractNZVecScalar<half, 32, 32, 16, 32, 7, 14>
            <<<1, nullptr, stream>>>((__gm__ half *)out, (__gm__ half *)srcIn, (__gm__ half *)dstInitIn);
    } else if constexpr (testKey == 3) {
        RunTExtractNZVecScalar<bfloat16_t, 32, 32, 16, 32, 11, 3><<<1, nullptr, stream>>>(
            (__gm__ bfloat16_t *)out, (__gm__ bfloat16_t *)srcIn, (__gm__ bfloat16_t *)dstInitIn);
    } else if constexpr (testKey == 4) {
        RunTExtractNZVecScalar<int8_t, 32, 64, 16, 64, 20, 33>
            <<<1, nullptr, stream>>>((__gm__ int8_t *)out, (__gm__ int8_t *)srcIn, (__gm__ int8_t *)dstInitIn);
    } else if constexpr (testKey == 5) {
        RunTExtractNZVecScalar<int32_t, 32, 16, 16, 16, 4, 7>
            <<<1, nullptr, stream>>>((__gm__ int32_t *)out, (__gm__ int32_t *)srcIn, (__gm__ int32_t *)dstInitIn);
    } else if constexpr (testKey == 6) {
        RunTExtractNZVecScalar<int16_t, 32, 32, 16, 32, 0, 0>
            <<<1, nullptr, stream>>>((__gm__ int16_t *)out, (__gm__ int16_t *)srcIn, (__gm__ int16_t *)dstInitIn);
    } else if constexpr (testKey == 7) {
        RunTExtractNZVecScalar<uint16_t, 32, 32, 16, 32, 31, 31>
            <<<1, nullptr, stream>>>((__gm__ uint16_t *)out, (__gm__ uint16_t *)srcIn, (__gm__ uint16_t *)dstInitIn);
    } else if constexpr (testKey == 8) {
        RunTExtractNZVecScalar<uint32_t, 32, 16, 16, 16, 30, 15>
            <<<1, nullptr, stream>>>((__gm__ uint32_t *)out, (__gm__ uint32_t *)srcIn, (__gm__ uint32_t *)dstInitIn);
    } else if constexpr (testKey == 9) {
        RunTExtractNZVecScalar<uint8_t, 32, 64, 16, 64, 0, 63>
            <<<1, nullptr, stream>>>((__gm__ uint8_t *)out, (__gm__ uint8_t *)srcIn, (__gm__ uint8_t *)dstInitIn);
    } else if constexpr (testKey == 10) {
        RunTExtractNZVecScalar<half, 48, 32, 32, 32, 33, 17>
            <<<1, nullptr, stream>>>((__gm__ half *)out, (__gm__ half *)srcIn, (__gm__ half *)dstInitIn);
    } else if constexpr (testKey == 11) {
        RunTExtractNZVecScalar<float, 48, 16, 16, 16, 41, 9>
            <<<1, nullptr, stream>>>((__gm__ float *)out, (__gm__ float *)srcIn, (__gm__ float *)dstInitIn);
    } else if constexpr (testKey == 12) {
        RunTExtractNZVecScalar<int8_t, 48, 96, 32, 64, 47, 73>
            <<<1, nullptr, stream>>>((__gm__ int8_t *)out, (__gm__ int8_t *)srcIn, (__gm__ int8_t *)dstInitIn);
    }
}

template void launchTExtractVecND<1>(uint8_t *, uint8_t *, uint8_t *, void *);
template void launchTExtractVecND<2>(uint8_t *, uint8_t *, uint8_t *, void *);
template void launchTExtractVecND<3>(uint8_t *, uint8_t *, uint8_t *, void *);
template void launchTExtractVecND<4>(uint8_t *, uint8_t *, uint8_t *, void *);
template void launchTExtractVecND<5>(uint8_t *, uint8_t *, uint8_t *, void *);
template void launchTExtractVecND<6>(uint8_t *, uint8_t *, uint8_t *, void *);
template void launchTExtractVecND<7>(uint8_t *, uint8_t *, uint8_t *, void *);
template void launchTExtractVecND<8>(uint8_t *, uint8_t *, uint8_t *, void *);
template void launchTExtractVecND<9>(uint8_t *, uint8_t *, uint8_t *, void *);
template void launchTExtractVecND<10>(uint8_t *, uint8_t *, uint8_t *, void *);
template void launchTExtractVecND<11>(uint8_t *, uint8_t *, uint8_t *, void *);
template void launchTExtractVecND<12>(uint8_t *, uint8_t *, uint8_t *, void *);
template void launchTExtractVecND<13>(uint8_t *, uint8_t *, uint8_t *, void *);
template void launchTExtractVecND<14>(uint8_t *, uint8_t *, uint8_t *, void *);
template void launchTExtractVecND<15>(uint8_t *, uint8_t *, uint8_t *, void *);
template void launchTExtractVecND<16>(uint8_t *, uint8_t *, uint8_t *, void *);
template void launchTExtractVecND<17>(uint8_t *, uint8_t *, uint8_t *, void *);
template void launchTExtractVecND<18>(uint8_t *, uint8_t *, uint8_t *, void *);
template void launchTExtractVecND<19>(uint8_t *, uint8_t *, uint8_t *, void *);
template void launchTExtractVecND<20>(uint8_t *, uint8_t *, uint8_t *, void *);
template void launchTExtractVecND<21>(uint8_t *, uint8_t *, uint8_t *, void *);
template void launchTExtractVecND<22>(uint8_t *, uint8_t *, uint8_t *, void *);
template void launchTExtractVecND<23>(uint8_t *, uint8_t *, uint8_t *, void *);
template void launchTExtractVecND<24>(uint8_t *, uint8_t *, uint8_t *, void *);
template void launchTExtractVecND<25>(uint8_t *, uint8_t *, uint8_t *, void *);
template void launchTExtractVecND<26>(uint8_t *, uint8_t *, uint8_t *, void *);
template void launchTExtractVecND<27>(uint8_t *, uint8_t *, uint8_t *, void *);
template void launchTExtractVecND<28>(uint8_t *, uint8_t *, uint8_t *, void *);
template void launchTExtractVecND<29>(uint8_t *, uint8_t *, uint8_t *, void *);
template void launchTExtractVecND<30>(uint8_t *, uint8_t *, uint8_t *, void *);
template void launchTExtractVecND<31>(uint8_t *, uint8_t *, uint8_t *, void *);
template void launchTExtractVecND<32>(uint8_t *, uint8_t *, uint8_t *, void *);

template void launchTExtractVecNDScalar<1>(uint8_t *, uint8_t *, uint8_t *, void *);
template void launchTExtractVecNDScalar<2>(uint8_t *, uint8_t *, uint8_t *, void *);
template void launchTExtractVecNDScalar<3>(uint8_t *, uint8_t *, uint8_t *, void *);
template void launchTExtractVecNDScalar<4>(uint8_t *, uint8_t *, uint8_t *, void *);
template void launchTExtractVecNDScalar<5>(uint8_t *, uint8_t *, uint8_t *, void *);
template void launchTExtractVecNDScalar<6>(uint8_t *, uint8_t *, uint8_t *, void *);
template void launchTExtractVecNDScalar<7>(uint8_t *, uint8_t *, uint8_t *, void *);
template void launchTExtractVecNDScalar<8>(uint8_t *, uint8_t *, uint8_t *, void *);
template void launchTExtractVecNDScalar<9>(uint8_t *, uint8_t *, uint8_t *, void *);
template void launchTExtractVecNDScalar<10>(uint8_t *, uint8_t *, uint8_t *, void *);
template void launchTExtractVecNDScalar<11>(uint8_t *, uint8_t *, uint8_t *, void *);

template void launchTExtractVecNZ<1>(uint8_t *, uint8_t *, uint8_t *, void *);
template void launchTExtractVecNZ<2>(uint8_t *, uint8_t *, uint8_t *, void *);
template void launchTExtractVecNZ<3>(uint8_t *, uint8_t *, uint8_t *, void *);
template void launchTExtractVecNZ<4>(uint8_t *, uint8_t *, uint8_t *, void *);
template void launchTExtractVecNZ<5>(uint8_t *, uint8_t *, uint8_t *, void *);
template void launchTExtractVecNZ<6>(uint8_t *, uint8_t *, uint8_t *, void *);
template void launchTExtractVecNZ<7>(uint8_t *, uint8_t *, uint8_t *, void *);
template void launchTExtractVecNZ<8>(uint8_t *, uint8_t *, uint8_t *, void *);
template void launchTExtractVecNZ<9>(uint8_t *, uint8_t *, uint8_t *, void *);
template void launchTExtractVecNZ<10>(uint8_t *, uint8_t *, uint8_t *, void *);
template void launchTExtractVecNZ<11>(uint8_t *, uint8_t *, uint8_t *, void *);
template void launchTExtractVecNZ<12>(uint8_t *, uint8_t *, uint8_t *, void *);
template void launchTExtractVecNZ<13>(uint8_t *, uint8_t *, uint8_t *, void *);
template void launchTExtractVecNZ<14>(uint8_t *, uint8_t *, uint8_t *, void *);
template void launchTExtractVecNZ<15>(uint8_t *, uint8_t *, uint8_t *, void *);
template void launchTExtractVecNZ<16>(uint8_t *, uint8_t *, uint8_t *, void *);
template void launchTExtractVecNZ<17>(uint8_t *, uint8_t *, uint8_t *, void *);
template void launchTExtractVecNZ<18>(uint8_t *, uint8_t *, uint8_t *, void *);
template void launchTExtractVecNZ<19>(uint8_t *, uint8_t *, uint8_t *, void *);
template void launchTExtractVecNZ<20>(uint8_t *, uint8_t *, uint8_t *, void *);
template void launchTExtractVecNZ<21>(uint8_t *, uint8_t *, uint8_t *, void *);
template void launchTExtractVecNZ<22>(uint8_t *, uint8_t *, uint8_t *, void *);
template void launchTExtractVecNZ<23>(uint8_t *, uint8_t *, uint8_t *, void *);
template void launchTExtractVecNZ<24>(uint8_t *, uint8_t *, uint8_t *, void *);
template void launchTExtractVecNZ<25>(uint8_t *, uint8_t *, uint8_t *, void *);
template void launchTExtractVecNZ<26>(uint8_t *, uint8_t *, uint8_t *, void *);
template void launchTExtractVecNZ<27>(uint8_t *, uint8_t *, uint8_t *, void *);
template void launchTExtractVecNZ<28>(uint8_t *, uint8_t *, uint8_t *, void *);
template void launchTExtractVecNZ<29>(uint8_t *, uint8_t *, uint8_t *, void *);
template void launchTExtractVecNZ<30>(uint8_t *, uint8_t *, uint8_t *, void *);
template void launchTExtractVecNZ<31>(uint8_t *, uint8_t *, uint8_t *, void *);

template void launchTExtractVecNZScalar<1>(uint8_t *, uint8_t *, uint8_t *, void *);
template void launchTExtractVecNZScalar<2>(uint8_t *, uint8_t *, uint8_t *, void *);
template void launchTExtractVecNZScalar<3>(uint8_t *, uint8_t *, uint8_t *, void *);
template void launchTExtractVecNZScalar<4>(uint8_t *, uint8_t *, uint8_t *, void *);
template void launchTExtractVecNZScalar<5>(uint8_t *, uint8_t *, uint8_t *, void *);
template void launchTExtractVecNZScalar<6>(uint8_t *, uint8_t *, uint8_t *, void *);
template void launchTExtractVecNZScalar<7>(uint8_t *, uint8_t *, uint8_t *, void *);
template void launchTExtractVecNZScalar<8>(uint8_t *, uint8_t *, uint8_t *, void *);
template void launchTExtractVecNZScalar<9>(uint8_t *, uint8_t *, uint8_t *, void *);
template void launchTExtractVecNZScalar<10>(uint8_t *, uint8_t *, uint8_t *, void *);
template void launchTExtractVecNZScalar<11>(uint8_t *, uint8_t *, uint8_t *, void *);
template void launchTExtractVecNZScalar<12>(uint8_t *, uint8_t *, uint8_t *, void *);
