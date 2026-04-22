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

template <typename T, uint32_t SrcStaticRows, uint32_t SrcStaticCols, uint32_t DstStaticRows, uint32_t DstStaticCols,
          uint32_t SrcValidRows, uint32_t SrcValidCols, uint32_t IdxRow, uint32_t IdxCol>
__global__ AICORE void RunTInsertNDVec(__gm__ T *out, __gm__ T *srcIn, __gm__ T *dstInitIn)
{
    constexpr bool isFullValid = (SrcValidRows == SrcStaticRows) && (SrcValidCols == SrcStaticCols);

    using SrcShape = pto::Shape<1, 1, 1, SrcStaticRows, SrcStaticCols>;
    using SrcStride = pto::Stride<SrcStaticRows * SrcStaticCols, SrcStaticRows * SrcStaticCols,
                                  SrcStaticRows * SrcStaticCols, SrcStaticCols, 1>;
    using SrcGlobal = GlobalTensor<T, SrcShape, SrcStride>;

    using DstShape = pto::Shape<1, 1, 1, DstStaticRows, DstStaticCols>;
    using DstStride = pto::Stride<DstStaticRows * DstStaticCols, DstStaticRows * DstStaticCols,
                                  DstStaticRows * DstStaticCols, DstStaticCols, 1>;
    using DstGlobal = GlobalTensor<T, DstShape, DstStride>;

    using SrcFullVec = Tile<TileType::Vec, T, SrcStaticRows, SrcStaticCols, BLayout::RowMajor>;
    using SrcInsertVec =
        Tile<TileType::Vec, T, SrcStaticRows, SrcStaticCols, BLayout::RowMajor, SrcValidRows, SrcValidCols>;
    using DstFullVec = Tile<TileType::Vec, T, DstStaticRows, DstStaticCols, BLayout::RowMajor>;

    SrcFullVec srcFull;
    SrcInsertVec srcInsert;
    DstFullVec dstFull;

    TASSIGN(srcFull, 0x0);
    constexpr uint32_t srcBytes = SrcStaticRows * SrcStaticCols * sizeof(T);
    constexpr uint32_t dstAssignAddr = ((srcBytes + 0xFF) / 0x100) * 0x100;
    TASSIGN(dstFull, dstAssignAddr);

    SrcGlobal srcGlobal(srcIn);
    DstGlobal dstInitGlobal(dstInitIn);
    DstGlobal outGlobal(out);

    TLOAD(srcFull, srcGlobal);
    TLOAD(dstFull, dstInitGlobal);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

    if constexpr (isFullValid) {
        TINSERT(dstFull, srcFull, static_cast<uint16_t>(IdxRow), static_cast<uint16_t>(IdxCol));
    } else {
        TSUBVIEW(srcInsert, srcFull, 0, 0);
        TINSERT(dstFull, srcInsert, static_cast<uint16_t>(IdxRow), static_cast<uint16_t>(IdxCol));
    }

    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);

    TSTORE(outGlobal, dstFull);
}

template <typename T, uint32_t DstStaticRows, uint32_t DstStaticCols, uint32_t IdxRow, uint32_t IdxCol>
__global__ AICORE void RunTInsertNDVecScalar(__gm__ T *out, __gm__ T *srcIn, __gm__ T *dstInitIn)
{
    constexpr uint32_t MinAlignedCols = BLOCK_BYTE_SIZE / sizeof(T);

    using SrcShape = pto::Shape<1, 1, 1, 1, MinAlignedCols>;
    using SrcStride = pto::Stride<MinAlignedCols, MinAlignedCols, MinAlignedCols, MinAlignedCols, 1>;
    using SrcGlobal = GlobalTensor<T, SrcShape, SrcStride>;

    using DstShape = pto::Shape<1, 1, 1, DstStaticRows, DstStaticCols>;
    using DstStride = pto::Stride<DstStaticRows * DstStaticCols, DstStaticRows * DstStaticCols,
                                  DstStaticRows * DstStaticCols, DstStaticCols, 1>;
    using DstGlobal = GlobalTensor<T, DstShape, DstStride>;

    using SrcFullVec = Tile<TileType::Vec, T, 1, MinAlignedCols, BLayout::RowMajor>;
    using SrcInsertVec = Tile<TileType::Vec, T, 1, MinAlignedCols, BLayout::RowMajor, 1, 1>;
    using DstFullVec = Tile<TileType::Vec, T, DstStaticRows, DstStaticCols, BLayout::RowMajor>;

    SrcFullVec srcFull;
    SrcInsertVec srcInsert;
    DstFullVec dstFull;

    TASSIGN(srcFull, 0x0);
    constexpr uint32_t srcBytes = 1 * MinAlignedCols * sizeof(T);
    constexpr uint32_t dstAssignAddr = ((srcBytes + 0xFF) / 0x100) * 0x100;
    TASSIGN(dstFull, dstAssignAddr);

    SrcGlobal srcGlobal(srcIn);
    DstGlobal dstInitGlobal(dstInitIn);
    DstGlobal outGlobal(out);

    TLOAD(srcFull, srcGlobal);
    TLOAD(dstFull, dstInitGlobal);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

    TSUBVIEW(srcInsert, srcFull, 0, 0);
    TINSERT(dstFull, srcInsert, static_cast<uint16_t>(IdxRow), static_cast<uint16_t>(IdxCol));

    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);

    TSTORE(outGlobal, dstFull);
}

template <typename T, uint32_t SrcRows, uint32_t SrcCols, uint32_t DstRows, uint32_t DstCols, uint32_t SrcValidRows,
          uint32_t SrcValidCols, uint32_t IdxRow, uint32_t IdxCol>
__global__ AICORE void RunTInsertNZVec(__gm__ T *out, __gm__ T *srcIn, __gm__ T *dstInitIn)
{
    constexpr uint32_t typeSize = sizeof(T);
    constexpr uint32_t c0Size = BLOCK_BYTE_SIZE / typeSize;
    constexpr bool isFullValid = (SrcValidRows == SrcRows) && (SrcValidCols == SrcCols);

    using SrcShapeNZ = pto::Shape<1, SrcCols / c0Size, SrcRows / FRACTAL_NZ_ROW, FRACTAL_NZ_ROW, c0Size>;
    using SrcStrideNZ =
        pto::Stride<(SrcCols / c0Size) * c0Size * SrcRows, SrcRows * c0Size, FRACTAL_NZ_ROW * c0Size, c0Size, 1>;
    using SrcGlobalNZ = GlobalTensor<T, SrcShapeNZ, SrcStrideNZ, Layout::NZ>;

    using DstShapeNZ = pto::Shape<1, DstCols / c0Size, DstRows / FRACTAL_NZ_ROW, FRACTAL_NZ_ROW, c0Size>;
    using DstStrideNZ =
        pto::Stride<(DstCols / c0Size) * c0Size * DstRows, DstRows * c0Size, FRACTAL_NZ_ROW * c0Size, c0Size, 1>;
    using DstGlobalNZ = GlobalTensor<T, DstShapeNZ, DstStrideNZ, Layout::NZ>;

    using SrcFullTile =
        Tile<TileType::Vec, T, SrcRows, SrcCols, BLayout::ColMajor, SrcRows, SrcCols, SLayout::RowMajor>;
    using SrcInsertTile =
        Tile<TileType::Vec, T, SrcRows, SrcCols, BLayout::ColMajor, SrcValidRows, SrcValidCols, SLayout::RowMajor>;
    using DstFullTile =
        Tile<TileType::Vec, T, DstRows, DstCols, BLayout::ColMajor, DstRows, DstCols, SLayout::RowMajor>;

    SrcFullTile srcFull;
    SrcInsertTile srcInsert;
    DstFullTile dstFull;

    TASSIGN(srcFull, 0x0);
    constexpr uint32_t srcBytes = SrcRows * SrcCols * sizeof(T);
    constexpr uint32_t dstAssignAddr = ((srcBytes + 0xFF) / 0x100) * 0x100;
    TASSIGN(dstFull, dstAssignAddr);

    SrcGlobalNZ srcGlobal(srcIn);
    DstGlobalNZ dstInitGlobal(dstInitIn);
    DstGlobalNZ outGlobal(out);

    TLOAD(srcFull, srcGlobal);
    TLOAD(dstFull, dstInitGlobal);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

    if constexpr (isFullValid) {
        TINSERT(dstFull, srcFull, static_cast<uint16_t>(IdxRow), static_cast<uint16_t>(IdxCol));
    } else {
        TSUBVIEW(srcInsert, srcFull, 0, 0);
        TINSERT(dstFull, srcInsert, static_cast<uint16_t>(IdxRow), static_cast<uint16_t>(IdxCol));
    }

    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);

    TSTORE(outGlobal, dstFull);
}

template <typename T, uint32_t SrcRows, uint32_t SrcCols, uint32_t DstRows, uint32_t DstCols, uint32_t IdxRow,
          uint32_t IdxCol>
__global__ AICORE void RunTInsertNZVecScalar(__gm__ T *out, __gm__ T *srcIn, __gm__ T *dstInitIn)
{
    constexpr uint32_t typeSize = sizeof(T);
    constexpr uint32_t c0Size = BLOCK_BYTE_SIZE / typeSize;

    using SrcShapeNZ = pto::Shape<1, SrcCols / c0Size, SrcRows / FRACTAL_NZ_ROW, FRACTAL_NZ_ROW, c0Size>;
    using SrcStrideNZ =
        pto::Stride<(SrcCols / c0Size) * c0Size * SrcRows, SrcRows * c0Size, FRACTAL_NZ_ROW * c0Size, c0Size, 1>;
    using SrcGlobalNZ = GlobalTensor<T, SrcShapeNZ, SrcStrideNZ, Layout::NZ>;

    using DstShapeNZ = pto::Shape<1, DstCols / c0Size, DstRows / FRACTAL_NZ_ROW, FRACTAL_NZ_ROW, c0Size>;
    using DstStrideNZ =
        pto::Stride<(DstCols / c0Size) * c0Size * DstRows, DstRows * c0Size, FRACTAL_NZ_ROW * c0Size, c0Size, 1>;
    using DstGlobalNZ = GlobalTensor<T, DstShapeNZ, DstStrideNZ, Layout::NZ>;

    using SrcFullTile =
        Tile<TileType::Vec, T, SrcRows, SrcCols, BLayout::ColMajor, SrcRows, SrcCols, SLayout::RowMajor>;
    using SrcInsertTile = Tile<TileType::Vec, T, SrcRows, SrcCols, BLayout::ColMajor, 1, 1, SLayout::RowMajor>;
    using DstFullTile =
        Tile<TileType::Vec, T, DstRows, DstCols, BLayout::ColMajor, DstRows, DstCols, SLayout::RowMajor>;

    SrcFullTile srcFull;
    SrcInsertTile srcInsert;
    DstFullTile dstFull;

    TASSIGN(srcFull, 0x0);
    constexpr uint32_t srcBytes = SrcRows * SrcCols * sizeof(T);
    constexpr uint32_t dstAssignAddr = ((srcBytes + 0xFF) / 0x100) * 0x100;
    TASSIGN(dstFull, dstAssignAddr);

    SrcGlobalNZ srcGlobal(srcIn);
    DstGlobalNZ dstInitGlobal(dstInitIn);
    DstGlobalNZ outGlobal(out);

    TLOAD(srcFull, srcGlobal);
    TLOAD(dstFull, dstInitGlobal);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

    TSUBVIEW(srcInsert, srcFull, 0, 0);
    TINSERT(dstFull, srcInsert, static_cast<uint16_t>(IdxRow), static_cast<uint16_t>(IdxCol));

    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);

    TSTORE(outGlobal, dstFull);
}

template <int32_t testKey>
void launchTInsertVecND(uint8_t *out, uint8_t *srcIn, uint8_t *dstInitIn, void *stream)
{
    if constexpr (testKey == 1) {
        RunTInsertNDVec<float, 8, 8, 16, 16, 8, 8, 0, 0>
            <<<1, nullptr, stream>>>((__gm__ float *)out, (__gm__ float *)srcIn, (__gm__ float *)dstInitIn);
    } else if constexpr (testKey == 2) {
        RunTInsertNDVec<float, 8, 8, 16, 16, 8, 8, 4, 8>
            <<<1, nullptr, stream>>>((__gm__ float *)out, (__gm__ float *)srcIn, (__gm__ float *)dstInitIn);
    } else if constexpr (testKey == 3) {
        RunTInsertNDVec<half, 16, 16, 32, 32, 16, 16, 8, 16>
            <<<1, nullptr, stream>>>((__gm__ half *)out, (__gm__ half *)srcIn, (__gm__ half *)dstInitIn);
    } else if constexpr (testKey == 4) {
        RunTInsertNDVec<bfloat16_t, 16, 16, 32, 32, 16, 16, 0, 16><<<1, nullptr, stream>>>(
            (__gm__ bfloat16_t *)out, (__gm__ bfloat16_t *)srcIn, (__gm__ bfloat16_t *)dstInitIn);
    } else if constexpr (testKey == 5) {
        RunTInsertNDVec<int32_t, 8, 8, 16, 16, 8, 8, 4, 0>
            <<<1, nullptr, stream>>>((__gm__ int32_t *)out, (__gm__ int32_t *)srcIn, (__gm__ int32_t *)dstInitIn);
    } else if constexpr (testKey == 6) {
        RunTInsertNDVec<int8_t, 32, 32, 64, 64, 32, 32, 0, 32>
            <<<1, nullptr, stream>>>((__gm__ int8_t *)out, (__gm__ int8_t *)srcIn, (__gm__ int8_t *)dstInitIn);
    } else if constexpr (testKey == 7) {
        RunTInsertNDVec<half, 16, 16, 32, 32, 4, 16, 2, 16>
            <<<1, nullptr, stream>>>((__gm__ half *)out, (__gm__ half *)srcIn, (__gm__ half *)dstInitIn);
    } else if constexpr (testKey == 8) {
        RunTInsertNDVec<float, 8, 16, 16, 32, 8, 16, 0, 16>
            <<<1, nullptr, stream>>>((__gm__ float *)out, (__gm__ float *)srcIn, (__gm__ float *)dstInitIn);
    } else if constexpr (testKey == 9) {
        RunTInsertNDVec<bfloat16_t, 16, 32, 32, 64, 16, 32, 8, 32><<<1, nullptr, stream>>>(
            (__gm__ bfloat16_t *)out, (__gm__ bfloat16_t *)srcIn, (__gm__ bfloat16_t *)dstInitIn);
    } else if constexpr (testKey == 10) {
        RunTInsertNDVec<uint8_t, 32, 32, 64, 64, 32, 32, 0, 32>
            <<<1, nullptr, stream>>>((__gm__ uint8_t *)out, (__gm__ uint8_t *)srcIn, (__gm__ uint8_t *)dstInitIn);
    } else if constexpr (testKey == 11) {
        RunTInsertNDVec<int16_t, 16, 16, 32, 32, 16, 16, 8, 16>
            <<<1, nullptr, stream>>>((__gm__ int16_t *)out, (__gm__ int16_t *)srcIn, (__gm__ int16_t *)dstInitIn);
    } else if constexpr (testKey == 12) {
        RunTInsertNDVec<uint16_t, 16, 16, 32, 32, 16, 16, 0, 0>
            <<<1, nullptr, stream>>>((__gm__ uint16_t *)out, (__gm__ uint16_t *)srcIn, (__gm__ uint16_t *)dstInitIn);
    } else if constexpr (testKey == 13) {
        RunTInsertNDVec<uint32_t, 8, 8, 16, 16, 8, 8, 4, 8>
            <<<1, nullptr, stream>>>((__gm__ uint32_t *)out, (__gm__ uint32_t *)srcIn, (__gm__ uint32_t *)dstInitIn);
    } else if constexpr (testKey == 14) {
        RunTInsertNDVec<float, 16, 32, 32, 64, 8, 16, 8, 16>
            <<<1, nullptr, stream>>>((__gm__ float *)out, (__gm__ float *)srcIn, (__gm__ float *)dstInitIn);
    } else if constexpr (testKey == 15) {
        RunTInsertNDVec<int8_t, 16, 32, 32, 64, 16, 32, 16, 32>
            <<<1, nullptr, stream>>>((__gm__ int8_t *)out, (__gm__ int8_t *)srcIn, (__gm__ int8_t *)dstInitIn);
    } else if constexpr (testKey == 16) {
        RunTInsertNDVec<half, 32, 32, 64, 64, 8, 32, 24, 16>
            <<<1, nullptr, stream>>>((__gm__ half *)out, (__gm__ half *)srcIn, (__gm__ half *)dstInitIn);
    } else if constexpr (testKey == 17) {
        RunTInsertNDVec<float, 12, 32, 24, 48, 12, 32, 5, 8>
            <<<1, nullptr, stream>>>((__gm__ float *)out, (__gm__ float *)srcIn, (__gm__ float *)dstInitIn);
    } else if constexpr (testKey == 18) {
        RunTInsertNDVec<half, 14, 48, 30, 80, 14, 48, 3, 16>
            <<<1, nullptr, stream>>>((__gm__ half *)out, (__gm__ half *)srcIn, (__gm__ half *)dstInitIn);
    } else if constexpr (testKey == 19) {
        RunTInsertNDVec<int8_t, 24, 64, 48, 96, 24, 64, 7, 32>
            <<<1, nullptr, stream>>>((__gm__ int8_t *)out, (__gm__ int8_t *)srcIn, (__gm__ int8_t *)dstInitIn);
    } else if constexpr (testKey == 20) {
        RunTInsertNDVec<bfloat16_t, 12, 32, 24, 48, 6, 16, 11, 16><<<1, nullptr, stream>>>(
            (__gm__ bfloat16_t *)out, (__gm__ bfloat16_t *)srcIn, (__gm__ bfloat16_t *)dstInitIn);
    } else if constexpr (testKey == 21) {
        RunTInsertNDVec<int32_t, 10, 8, 20, 16, 10, 8, 9, 8>
            <<<1, nullptr, stream>>>((__gm__ int32_t *)out, (__gm__ int32_t *)srcIn, (__gm__ int32_t *)dstInitIn);
    } else if constexpr (testKey == 22) {
        RunTInsertNDVec<float, 12, 32, 24, 48, 6, 8, 15, 24>
            <<<1, nullptr, stream>>>((__gm__ float *)out, (__gm__ float *)srcIn, (__gm__ float *)dstInitIn);
    } else if constexpr (testKey == 23) {
        RunTInsertNDVec<float, 8, 16, 32, 32, 8, 10, 0, 0>
            <<<1, nullptr, stream>>>((__gm__ float *)out, (__gm__ float *)srcIn, (__gm__ float *)dstInitIn);
    } else if constexpr (testKey == 24) {
        RunTInsertNDVec<half, 8, 16, 32, 32, 8, 10, 0, 0>
            <<<1, nullptr, stream>>>((__gm__ half *)out, (__gm__ half *)srcIn, (__gm__ half *)dstInitIn);
    } else if constexpr (testKey == 25) {
        RunTInsertNDVec<bfloat16_t, 8, 16, 32, 32, 8, 14, 0, 0><<<1, nullptr, stream>>>(
            (__gm__ bfloat16_t *)out, (__gm__ bfloat16_t *)srcIn, (__gm__ bfloat16_t *)dstInitIn);
    } else if constexpr (testKey == 26) {
        RunTInsertNDVec<int16_t, 8, 16, 32, 32, 8, 11, 0, 0>
            <<<1, nullptr, stream>>>((__gm__ int16_t *)out, (__gm__ int16_t *)srcIn, (__gm__ int16_t *)dstInitIn);
    } else if constexpr (testKey == 27) {
        RunTInsertNDVec<int32_t, 8, 16, 32, 32, 8, 14, 0, 0>
            <<<1, nullptr, stream>>>((__gm__ int32_t *)out, (__gm__ int32_t *)srcIn, (__gm__ int32_t *)dstInitIn);
    } else if constexpr (testKey == 28) {
        RunTInsertNDVec<uint16_t, 16, 32, 32, 64, 16, 22, 0, 0>
            <<<1, nullptr, stream>>>((__gm__ uint16_t *)out, (__gm__ uint16_t *)srcIn, (__gm__ uint16_t *)dstInitIn);
    } else if constexpr (testKey == 29) {
        RunTInsertNDVec<int8_t, 16, 64, 32, 64, 16, 34, 0, 0>
            <<<1, nullptr, stream>>>((__gm__ int8_t *)out, (__gm__ int8_t *)srcIn, (__gm__ int8_t *)dstInitIn);
    } else if constexpr (testKey == 30) {
        RunTInsertNDVec<float, 8, 16, 32, 32, 8, 10, 4, 8>
            <<<1, nullptr, stream>>>((__gm__ float *)out, (__gm__ float *)srcIn, (__gm__ float *)dstInitIn);
    } else if constexpr (testKey == 31) {
        RunTInsertNDVec<uint16_t, 8, 16, 32, 32, 8, 15, 0, 0>
            <<<1, nullptr, stream>>>((__gm__ uint16_t *)out, (__gm__ uint16_t *)srcIn, (__gm__ uint16_t *)dstInitIn);
    } else if constexpr (testKey == 32) {
        RunTInsertNDVec<float, 8, 16, 32, 32, 8, 6, 0, 0>
            <<<1, nullptr, stream>>>((__gm__ float *)out, (__gm__ float *)srcIn, (__gm__ float *)dstInitIn);
    }
}

template <int32_t testKey>
void launchTInsertVecNDScalar(uint8_t *out, uint8_t *srcIn, uint8_t *dstInitIn, void *stream)
{
    if constexpr (testKey == 1) {
        RunTInsertNDVecScalar<float, 16, 16, 5, 7>
            <<<1, nullptr, stream>>>((__gm__ float *)out, (__gm__ float *)srcIn, (__gm__ float *)dstInitIn);
    } else if constexpr (testKey == 2) {
        RunTInsertNDVecScalar<half, 32, 32, 10, 15>
            <<<1, nullptr, stream>>>((__gm__ half *)out, (__gm__ half *)srcIn, (__gm__ half *)dstInitIn);
    } else if constexpr (testKey == 3) {
        RunTInsertNDVecScalar<bfloat16_t, 32, 32, 3, 11><<<1, nullptr, stream>>>(
            (__gm__ bfloat16_t *)out, (__gm__ bfloat16_t *)srcIn, (__gm__ bfloat16_t *)dstInitIn);
    } else if constexpr (testKey == 4) {
        RunTInsertNDVecScalar<int8_t, 64, 64, 20, 30>
            <<<1, nullptr, stream>>>((__gm__ int8_t *)out, (__gm__ int8_t *)srcIn, (__gm__ int8_t *)dstInitIn);
    } else if constexpr (testKey == 5) {
        RunTInsertNDVecScalar<int32_t, 16, 16, 7, 9>
            <<<1, nullptr, stream>>>((__gm__ int32_t *)out, (__gm__ int32_t *)srcIn, (__gm__ int32_t *)dstInitIn);
    } else if constexpr (testKey == 6) {
        RunTInsertNDVecScalar<int16_t, 32, 32, 0, 0>
            <<<1, nullptr, stream>>>((__gm__ int16_t *)out, (__gm__ int16_t *)srcIn, (__gm__ int16_t *)dstInitIn);
    } else if constexpr (testKey == 7) {
        RunTInsertNDVecScalar<uint16_t, 32, 32, 31, 31>
            <<<1, nullptr, stream>>>((__gm__ uint16_t *)out, (__gm__ uint16_t *)srcIn, (__gm__ uint16_t *)dstInitIn);
    } else if constexpr (testKey == 8) {
        RunTInsertNDVecScalar<uint32_t, 16, 16, 15, 15>
            <<<1, nullptr, stream>>>((__gm__ uint32_t *)out, (__gm__ uint32_t *)srcIn, (__gm__ uint32_t *)dstInitIn);
    } else if constexpr (testKey == 9) {
        RunTInsertNDVecScalar<half, 30, 80, 17, 23>
            <<<1, nullptr, stream>>>((__gm__ half *)out, (__gm__ half *)srcIn, (__gm__ half *)dstInitIn);
    } else if constexpr (testKey == 10) {
        RunTInsertNDVecScalar<int8_t, 48, 96, 41, 73>
            <<<1, nullptr, stream>>>((__gm__ int8_t *)out, (__gm__ int8_t *)srcIn, (__gm__ int8_t *)dstInitIn);
    } else if constexpr (testKey == 11) {
        RunTInsertNDVecScalar<float, 24, 48, 11, 13>
            <<<1, nullptr, stream>>>((__gm__ float *)out, (__gm__ float *)srcIn, (__gm__ float *)dstInitIn);
    }
}

template <int32_t testKey>
void launchTInsertVecNZ(uint8_t *out, uint8_t *srcIn, uint8_t *dstInitIn, void *stream)
{
    if constexpr (testKey == 1) {
        RunTInsertNZVec<float, 16, 32, 32, 32, 16, 32, 0, 0>
            <<<1, nullptr, stream>>>((__gm__ float *)out, (__gm__ float *)srcIn, (__gm__ float *)dstInitIn);
    } else if constexpr (testKey == 2) {
        RunTInsertNZVec<float, 16, 32, 32, 32, 16, 32, 16, 0>
            <<<1, nullptr, stream>>>((__gm__ float *)out, (__gm__ float *)srcIn, (__gm__ float *)dstInitIn);
    } else if constexpr (testKey == 3) {
        RunTInsertNZVec<half, 16, 32, 32, 32, 16, 32, 0, 0>
            <<<1, nullptr, stream>>>((__gm__ half *)out, (__gm__ half *)srcIn, (__gm__ half *)dstInitIn);
    } else if constexpr (testKey == 4) {
        RunTInsertNZVec<bfloat16_t, 16, 32, 32, 32, 16, 32, 16, 0><<<1, nullptr, stream>>>(
            (__gm__ bfloat16_t *)out, (__gm__ bfloat16_t *)srcIn, (__gm__ bfloat16_t *)dstInitIn);
    } else if constexpr (testKey == 5) {
        RunTInsertNZVec<int8_t, 16, 64, 32, 64, 16, 64, 0, 0>
            <<<1, nullptr, stream>>>((__gm__ int8_t *)out, (__gm__ int8_t *)srcIn, (__gm__ int8_t *)dstInitIn);
    } else if constexpr (testKey == 6) {
        RunTInsertNZVec<int8_t, 16, 64, 32, 64, 16, 64, 16, 0>
            <<<1, nullptr, stream>>>((__gm__ int8_t *)out, (__gm__ int8_t *)srcIn, (__gm__ int8_t *)dstInitIn);
    } else if constexpr (testKey == 7) {
        RunTInsertNZVec<half, 32, 32, 64, 32, 32, 32, 0, 0>
            <<<1, nullptr, stream>>>((__gm__ half *)out, (__gm__ half *)srcIn, (__gm__ half *)dstInitIn);
    } else if constexpr (testKey == 8) {
        RunTInsertNZVec<int8_t, 16, 32, 32, 64, 16, 32, 16, 32>
            <<<1, nullptr, stream>>>((__gm__ int8_t *)out, (__gm__ int8_t *)srcIn, (__gm__ int8_t *)dstInitIn);
    } else if constexpr (testKey == 9) {
        RunTInsertNZVec<bfloat16_t, 16, 32, 32, 32, 16, 16, 0, 0><<<1, nullptr, stream>>>(
            (__gm__ bfloat16_t *)out, (__gm__ bfloat16_t *)srcIn, (__gm__ bfloat16_t *)dstInitIn);
    } else if constexpr (testKey == 10) {
        RunTInsertNZVec<uint8_t, 16, 64, 32, 64, 16, 64, 16, 0>
            <<<1, nullptr, stream>>>((__gm__ uint8_t *)out, (__gm__ uint8_t *)srcIn, (__gm__ uint8_t *)dstInitIn);
    } else if constexpr (testKey == 11) {
        RunTInsertNZVec<int32_t, 16, 8, 32, 16, 16, 8, 16, 8>
            <<<1, nullptr, stream>>>((__gm__ int32_t *)out, (__gm__ int32_t *)srcIn, (__gm__ int32_t *)dstInitIn);
    } else if constexpr (testKey == 12) {
        RunTInsertNZVec<int16_t, 16, 32, 32, 32, 16, 32, 16, 0>
            <<<1, nullptr, stream>>>((__gm__ int16_t *)out, (__gm__ int16_t *)srcIn, (__gm__ int16_t *)dstInitIn);
    } else if constexpr (testKey == 13) {
        RunTInsertNZVec<uint16_t, 16, 32, 32, 32, 16, 32, 0, 0>
            <<<1, nullptr, stream>>>((__gm__ uint16_t *)out, (__gm__ uint16_t *)srcIn, (__gm__ uint16_t *)dstInitIn);
    } else if constexpr (testKey == 14) {
        RunTInsertNZVec<uint32_t, 16, 16, 32, 16, 16, 16, 0, 0>
            <<<1, nullptr, stream>>>((__gm__ uint32_t *)out, (__gm__ uint32_t *)srcIn, (__gm__ uint32_t *)dstInitIn);
    } else if constexpr (testKey == 15) {
        RunTInsertNZVec<half, 32, 64, 64, 64, 32, 64, 32, 0>
            <<<1, nullptr, stream>>>((__gm__ half *)out, (__gm__ half *)srcIn, (__gm__ half *)dstInitIn);
    } else if constexpr (testKey == 16) {
        RunTInsertNZVec<float, 16, 8, 32, 16, 16, 8, 16, 8>
            <<<1, nullptr, stream>>>((__gm__ float *)out, (__gm__ float *)srcIn, (__gm__ float *)dstInitIn);
    } else if constexpr (testKey == 17) {
        RunTInsertNZVec<bfloat16_t, 32, 32, 64, 64, 16, 16, 32, 32><<<1, nullptr, stream>>>(
            (__gm__ bfloat16_t *)out, (__gm__ bfloat16_t *)srcIn, (__gm__ bfloat16_t *)dstInitIn);
    } else if constexpr (testKey == 18) {
        RunTInsertNZVec<half, 32, 32, 48, 32, 32, 32, 16, 0>
            <<<1, nullptr, stream>>>((__gm__ half *)out, (__gm__ half *)srcIn, (__gm__ half *)dstInitIn);
    } else if constexpr (testKey == 19) {
        RunTInsertNZVec<float, 16, 16, 48, 16, 16, 16, 32, 0>
            <<<1, nullptr, stream>>>((__gm__ float *)out, (__gm__ float *)srcIn, (__gm__ float *)dstInitIn);
    } else if constexpr (testKey == 20) {
        RunTInsertNZVec<bfloat16_t, 32, 32, 48, 48, 16, 16, 16, 16><<<1, nullptr, stream>>>(
            (__gm__ bfloat16_t *)out, (__gm__ bfloat16_t *)srcIn, (__gm__ bfloat16_t *)dstInitIn);
    } else if constexpr (testKey == 21) {
        RunTInsertNZVec<int8_t, 32, 64, 48, 96, 16, 32, 16, 32>
            <<<1, nullptr, stream>>>((__gm__ int8_t *)out, (__gm__ int8_t *)srcIn, (__gm__ int8_t *)dstInitIn);
    } else if constexpr (testKey == 22) {
        RunTInsertNZVec<float, 16, 32, 32, 32, 8, 10, 0, 0>
            <<<1, nullptr, stream>>>((__gm__ float *)out, (__gm__ float *)srcIn, (__gm__ float *)dstInitIn);
    } else if constexpr (testKey == 23) {
        RunTInsertNZVec<half, 16, 32, 32, 64, 16, 22, 0, 0>
            <<<1, nullptr, stream>>>((__gm__ half *)out, (__gm__ half *)srcIn, (__gm__ half *)dstInitIn);
    } else if constexpr (testKey == 24) {
        RunTInsertNZVec<bfloat16_t, 16, 32, 32, 32, 16, 10, 0, 0><<<1, nullptr, stream>>>(
            (__gm__ bfloat16_t *)out, (__gm__ bfloat16_t *)srcIn, (__gm__ bfloat16_t *)dstInitIn);
    } else if constexpr (testKey == 25) {
        RunTInsertNZVec<int16_t, 16, 32, 32, 32, 16, 15, 0, 0>
            <<<1, nullptr, stream>>>((__gm__ int16_t *)out, (__gm__ int16_t *)srcIn, (__gm__ int16_t *)dstInitIn);
    } else if constexpr (testKey == 26) {
        RunTInsertNZVec<int32_t, 16, 16, 32, 16, 16, 14, 0, 0>
            <<<1, nullptr, stream>>>((__gm__ int32_t *)out, (__gm__ int32_t *)srcIn, (__gm__ int32_t *)dstInitIn);
    } else if constexpr (testKey == 27) {
        RunTInsertNZVec<int8_t, 16, 64, 32, 64, 16, 42, 0, 0>
            <<<1, nullptr, stream>>>((__gm__ int8_t *)out, (__gm__ int8_t *)srcIn, (__gm__ int8_t *)dstInitIn);
    } else if constexpr (testKey == 28) {
        RunTInsertNZVec<half, 16, 32, 32, 32, 10, 16, 0, 0>
            <<<1, nullptr, stream>>>((__gm__ half *)out, (__gm__ half *)srcIn, (__gm__ half *)dstInitIn);
    } else if constexpr (testKey == 29) {
        RunTInsertNZVec<float, 16, 32, 32, 32, 5, 10, 0, 0>
            <<<1, nullptr, stream>>>((__gm__ float *)out, (__gm__ float *)srcIn, (__gm__ float *)dstInitIn);
    } else if constexpr (testKey == 30) {
        RunTInsertNZVec<half, 16, 32, 64, 64, 16, 22, 16, 16>
            <<<1, nullptr, stream>>>((__gm__ half *)out, (__gm__ half *)srcIn, (__gm__ half *)dstInitIn);
    } else if constexpr (testKey == 31) {
        RunTInsertNZVec<bfloat16_t, 16, 32, 32, 32, 10, 10, 16, 16><<<1, nullptr, stream>>>(
            (__gm__ bfloat16_t *)out, (__gm__ bfloat16_t *)srcIn, (__gm__ bfloat16_t *)dstInitIn);
    }
}

template <int32_t testKey>
void launchTInsertVecNZScalar(uint8_t *out, uint8_t *srcIn, uint8_t *dstInitIn, void *stream)
{
    if constexpr (testKey == 1) {
        RunTInsertNZVecScalar<float, 16, 8, 32, 32, 5, 9>
            <<<1, nullptr, stream>>>((__gm__ float *)out, (__gm__ float *)srcIn, (__gm__ float *)dstInitIn);
    } else if constexpr (testKey == 2) {
        RunTInsertNZVecScalar<half, 16, 16, 32, 32, 7, 14>
            <<<1, nullptr, stream>>>((__gm__ half *)out, (__gm__ half *)srcIn, (__gm__ half *)dstInitIn);
    } else if constexpr (testKey == 3) {
        RunTInsertNZVecScalar<bfloat16_t, 16, 16, 32, 32, 11, 3><<<1, nullptr, stream>>>(
            (__gm__ bfloat16_t *)out, (__gm__ bfloat16_t *)srcIn, (__gm__ bfloat16_t *)dstInitIn);
    } else if constexpr (testKey == 4) {
        RunTInsertNZVecScalar<int8_t, 16, 32, 32, 64, 20, 33>
            <<<1, nullptr, stream>>>((__gm__ int8_t *)out, (__gm__ int8_t *)srcIn, (__gm__ int8_t *)dstInitIn);
    } else if constexpr (testKey == 5) {
        RunTInsertNZVecScalar<int32_t, 16, 8, 32, 16, 4, 7>
            <<<1, nullptr, stream>>>((__gm__ int32_t *)out, (__gm__ int32_t *)srcIn, (__gm__ int32_t *)dstInitIn);
    } else if constexpr (testKey == 6) {
        RunTInsertNZVecScalar<int16_t, 16, 16, 32, 32, 0, 0>
            <<<1, nullptr, stream>>>((__gm__ int16_t *)out, (__gm__ int16_t *)srcIn, (__gm__ int16_t *)dstInitIn);
    } else if constexpr (testKey == 7) {
        RunTInsertNZVecScalar<uint16_t, 16, 16, 32, 32, 31, 31>
            <<<1, nullptr, stream>>>((__gm__ uint16_t *)out, (__gm__ uint16_t *)srcIn, (__gm__ uint16_t *)dstInitIn);
    } else if constexpr (testKey == 8) {
        RunTInsertNZVecScalar<uint32_t, 16, 8, 32, 16, 30, 15>
            <<<1, nullptr, stream>>>((__gm__ uint32_t *)out, (__gm__ uint32_t *)srcIn, (__gm__ uint32_t *)dstInitIn);
    } else if constexpr (testKey == 9) {
        RunTInsertNZVecScalar<uint8_t, 16, 32, 32, 64, 0, 63>
            <<<1, nullptr, stream>>>((__gm__ uint8_t *)out, (__gm__ uint8_t *)srcIn, (__gm__ uint8_t *)dstInitIn);
    } else if constexpr (testKey == 10) {
        RunTInsertNZVecScalar<half, 32, 32, 48, 32, 33, 17>
            <<<1, nullptr, stream>>>((__gm__ half *)out, (__gm__ half *)srcIn, (__gm__ half *)dstInitIn);
    } else if constexpr (testKey == 11) {
        RunTInsertNZVecScalar<float, 16, 16, 48, 16, 41, 9>
            <<<1, nullptr, stream>>>((__gm__ float *)out, (__gm__ float *)srcIn, (__gm__ float *)dstInitIn);
    } else if constexpr (testKey == 12) {
        RunTInsertNZVecScalar<int8_t, 32, 64, 48, 96, 47, 73>
            <<<1, nullptr, stream>>>((__gm__ int8_t *)out, (__gm__ int8_t *)srcIn, (__gm__ int8_t *)dstInitIn);
    }
}

template void launchTInsertVecND<1>(uint8_t *, uint8_t *, uint8_t *, void *);
template void launchTInsertVecND<2>(uint8_t *, uint8_t *, uint8_t *, void *);
template void launchTInsertVecND<3>(uint8_t *, uint8_t *, uint8_t *, void *);
template void launchTInsertVecND<4>(uint8_t *, uint8_t *, uint8_t *, void *);
template void launchTInsertVecND<5>(uint8_t *, uint8_t *, uint8_t *, void *);
template void launchTInsertVecND<6>(uint8_t *, uint8_t *, uint8_t *, void *);
template void launchTInsertVecND<7>(uint8_t *, uint8_t *, uint8_t *, void *);
template void launchTInsertVecND<8>(uint8_t *, uint8_t *, uint8_t *, void *);
template void launchTInsertVecND<9>(uint8_t *, uint8_t *, uint8_t *, void *);
template void launchTInsertVecND<10>(uint8_t *, uint8_t *, uint8_t *, void *);
template void launchTInsertVecND<11>(uint8_t *, uint8_t *, uint8_t *, void *);
template void launchTInsertVecND<12>(uint8_t *, uint8_t *, uint8_t *, void *);
template void launchTInsertVecND<13>(uint8_t *, uint8_t *, uint8_t *, void *);
template void launchTInsertVecND<14>(uint8_t *, uint8_t *, uint8_t *, void *);
template void launchTInsertVecND<15>(uint8_t *, uint8_t *, uint8_t *, void *);
template void launchTInsertVecND<16>(uint8_t *, uint8_t *, uint8_t *, void *);
template void launchTInsertVecND<17>(uint8_t *, uint8_t *, uint8_t *, void *);
template void launchTInsertVecND<18>(uint8_t *, uint8_t *, uint8_t *, void *);
template void launchTInsertVecND<19>(uint8_t *, uint8_t *, uint8_t *, void *);
template void launchTInsertVecND<20>(uint8_t *, uint8_t *, uint8_t *, void *);
template void launchTInsertVecND<21>(uint8_t *, uint8_t *, uint8_t *, void *);
template void launchTInsertVecND<22>(uint8_t *, uint8_t *, uint8_t *, void *);
template void launchTInsertVecND<23>(uint8_t *, uint8_t *, uint8_t *, void *);
template void launchTInsertVecND<24>(uint8_t *, uint8_t *, uint8_t *, void *);
template void launchTInsertVecND<25>(uint8_t *, uint8_t *, uint8_t *, void *);
template void launchTInsertVecND<26>(uint8_t *, uint8_t *, uint8_t *, void *);
template void launchTInsertVecND<27>(uint8_t *, uint8_t *, uint8_t *, void *);
template void launchTInsertVecND<28>(uint8_t *, uint8_t *, uint8_t *, void *);
template void launchTInsertVecND<29>(uint8_t *, uint8_t *, uint8_t *, void *);
template void launchTInsertVecND<30>(uint8_t *, uint8_t *, uint8_t *, void *);
template void launchTInsertVecND<31>(uint8_t *, uint8_t *, uint8_t *, void *);
template void launchTInsertVecND<32>(uint8_t *, uint8_t *, uint8_t *, void *);

template void launchTInsertVecNDScalar<1>(uint8_t *, uint8_t *, uint8_t *, void *);
template void launchTInsertVecNDScalar<2>(uint8_t *, uint8_t *, uint8_t *, void *);
template void launchTInsertVecNDScalar<3>(uint8_t *, uint8_t *, uint8_t *, void *);
template void launchTInsertVecNDScalar<4>(uint8_t *, uint8_t *, uint8_t *, void *);
template void launchTInsertVecNDScalar<5>(uint8_t *, uint8_t *, uint8_t *, void *);
template void launchTInsertVecNDScalar<6>(uint8_t *, uint8_t *, uint8_t *, void *);
template void launchTInsertVecNDScalar<7>(uint8_t *, uint8_t *, uint8_t *, void *);
template void launchTInsertVecNDScalar<8>(uint8_t *, uint8_t *, uint8_t *, void *);
template void launchTInsertVecNDScalar<9>(uint8_t *, uint8_t *, uint8_t *, void *);
template void launchTInsertVecNDScalar<10>(uint8_t *, uint8_t *, uint8_t *, void *);
template void launchTInsertVecNDScalar<11>(uint8_t *, uint8_t *, uint8_t *, void *);

template void launchTInsertVecNZ<1>(uint8_t *, uint8_t *, uint8_t *, void *);
template void launchTInsertVecNZ<2>(uint8_t *, uint8_t *, uint8_t *, void *);
template void launchTInsertVecNZ<3>(uint8_t *, uint8_t *, uint8_t *, void *);
template void launchTInsertVecNZ<4>(uint8_t *, uint8_t *, uint8_t *, void *);
template void launchTInsertVecNZ<5>(uint8_t *, uint8_t *, uint8_t *, void *);
template void launchTInsertVecNZ<6>(uint8_t *, uint8_t *, uint8_t *, void *);
template void launchTInsertVecNZ<7>(uint8_t *, uint8_t *, uint8_t *, void *);
template void launchTInsertVecNZ<8>(uint8_t *, uint8_t *, uint8_t *, void *);
template void launchTInsertVecNZ<9>(uint8_t *, uint8_t *, uint8_t *, void *);
template void launchTInsertVecNZ<10>(uint8_t *, uint8_t *, uint8_t *, void *);
template void launchTInsertVecNZ<11>(uint8_t *, uint8_t *, uint8_t *, void *);
template void launchTInsertVecNZ<12>(uint8_t *, uint8_t *, uint8_t *, void *);
template void launchTInsertVecNZ<13>(uint8_t *, uint8_t *, uint8_t *, void *);
template void launchTInsertVecNZ<14>(uint8_t *, uint8_t *, uint8_t *, void *);
template void launchTInsertVecNZ<15>(uint8_t *, uint8_t *, uint8_t *, void *);
template void launchTInsertVecNZ<16>(uint8_t *, uint8_t *, uint8_t *, void *);
template void launchTInsertVecNZ<17>(uint8_t *, uint8_t *, uint8_t *, void *);
template void launchTInsertVecNZ<18>(uint8_t *, uint8_t *, uint8_t *, void *);
template void launchTInsertVecNZ<19>(uint8_t *, uint8_t *, uint8_t *, void *);
template void launchTInsertVecNZ<20>(uint8_t *, uint8_t *, uint8_t *, void *);
template void launchTInsertVecNZ<21>(uint8_t *, uint8_t *, uint8_t *, void *);
template void launchTInsertVecNZ<22>(uint8_t *, uint8_t *, uint8_t *, void *);
template void launchTInsertVecNZ<23>(uint8_t *, uint8_t *, uint8_t *, void *);
template void launchTInsertVecNZ<24>(uint8_t *, uint8_t *, uint8_t *, void *);
template void launchTInsertVecNZ<25>(uint8_t *, uint8_t *, uint8_t *, void *);
template void launchTInsertVecNZ<26>(uint8_t *, uint8_t *, uint8_t *, void *);
template void launchTInsertVecNZ<27>(uint8_t *, uint8_t *, uint8_t *, void *);
template void launchTInsertVecNZ<28>(uint8_t *, uint8_t *, uint8_t *, void *);
template void launchTInsertVecNZ<29>(uint8_t *, uint8_t *, uint8_t *, void *);
template void launchTInsertVecNZ<30>(uint8_t *, uint8_t *, uint8_t *, void *);
template void launchTInsertVecNZ<31>(uint8_t *, uint8_t *, uint8_t *, void *);

template void launchTInsertVecNZScalar<1>(uint8_t *, uint8_t *, uint8_t *, void *);
template void launchTInsertVecNZScalar<2>(uint8_t *, uint8_t *, uint8_t *, void *);
template void launchTInsertVecNZScalar<3>(uint8_t *, uint8_t *, uint8_t *, void *);
template void launchTInsertVecNZScalar<4>(uint8_t *, uint8_t *, uint8_t *, void *);
template void launchTInsertVecNZScalar<5>(uint8_t *, uint8_t *, uint8_t *, void *);
template void launchTInsertVecNZScalar<6>(uint8_t *, uint8_t *, uint8_t *, void *);
template void launchTInsertVecNZScalar<7>(uint8_t *, uint8_t *, uint8_t *, void *);
template void launchTInsertVecNZScalar<8>(uint8_t *, uint8_t *, uint8_t *, void *);
template void launchTInsertVecNZScalar<9>(uint8_t *, uint8_t *, uint8_t *, void *);
template void launchTInsertVecNZScalar<10>(uint8_t *, uint8_t *, uint8_t *, void *);
template void launchTInsertVecNZScalar<11>(uint8_t *, uint8_t *, uint8_t *, void *);
template void launchTInsertVecNZScalar<12>(uint8_t *, uint8_t *, uint8_t *, void *);
