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
#include <pto/npu/a2a3/MGather.hpp>
#include "acl/acl.h"

using namespace pto;

template <pto::GatherOOB Oob, typename T, typename TIdx, uint32_t kDstRows, uint32_t kDstCols, uint32_t kTableRows>
inline AICORE void runRow(__gm__ T *out, __gm__ T *table, __gm__ TIdx *indices)
{
    using TableShape = pto::Shape<1, 1, 1, kTableRows, kDstCols>;
    using TableStride = pto::Stride<1, 1, 1, kDstCols, 1>;
    using IdxShape = pto::Shape<1, 1, 1, 1, kDstRows>;
    using IdxStride = pto::Stride<1, 1, 1, kDstRows, 1>;
    using OutShape = pto::Shape<1, 1, 1, kDstRows, kDstCols>;
    using OutStride = pto::Stride<1, 1, 1, kDstCols, 1>;

    GlobalTensor<T, TableShape, TableStride> tableGlobal(table);
    GlobalTensor<TIdx, IdxShape, IdxStride> idxGlobal(indices);
    GlobalTensor<T, OutShape, OutStride> outGlobal(out);

    using DstTile = Tile<TileType::Vec, T, kDstRows, kDstCols, BLayout::RowMajor, kDstRows, kDstCols>;
    using IdxTile = Tile<TileType::Vec, TIdx, 1, kDstRows, BLayout::RowMajor, 1, kDstRows>;

    DstTile dstTile;
    IdxTile idxTile;

    constexpr uint32_t idxBytes = ((1u * kDstRows * sizeof(TIdx) + 31u) / 32u) * 32u;
    TASSIGN(idxTile, 0x0);
    TASSIGN(dstTile, idxBytes);

    TLOAD(idxTile, idxGlobal);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

    MGATHER<Coalesce::Row, Oob>(dstTile, tableGlobal, idxTile);

    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);

    TSTORE(outGlobal, dstTile);
}

template <pto::GatherOOB Oob, typename T, typename TIdx, uint32_t kValidRows, uint32_t kPadRows, uint32_t kPadCols,
          uint32_t kPadIdxCols, uint32_t kTableRows>
inline AICORE void runRowPadded(__gm__ T *out, __gm__ T *table, __gm__ TIdx *indices)
{
    using TableShape = pto::Shape<1, 1, 1, kTableRows, kPadCols>;
    using TableStride = pto::Stride<1, 1, 1, kPadCols, 1>;
    using IdxShape = pto::Shape<1, 1, 1, 1, kValidRows>;
    using IdxStride = pto::Stride<1, 1, 1, kValidRows, 1>;
    using OutShape = pto::Shape<1, 1, 1, kValidRows, kPadCols>;
    using OutStride = pto::Stride<1, 1, 1, kPadCols, 1>;

    GlobalTensor<T, TableShape, TableStride> tableGlobal(table);
    GlobalTensor<TIdx, IdxShape, IdxStride> idxGlobal(indices);
    GlobalTensor<T, OutShape, OutStride> outGlobal(out);

    using DstTile = Tile<TileType::Vec, T, kPadRows, kPadCols, BLayout::RowMajor, kValidRows, kPadCols>;
    using IdxTile = Tile<TileType::Vec, TIdx, 1, kPadIdxCols, BLayout::RowMajor, 1, kValidRows>;

    DstTile dstTile;
    IdxTile idxTile;

    constexpr uint32_t idxBytes = ((1u * kPadIdxCols * sizeof(TIdx) + 31u) / 32u) * 32u;
    TASSIGN(idxTile, 0x0);
    TASSIGN(dstTile, idxBytes);

    TLOAD(idxTile, idxGlobal);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

    MGATHER<Coalesce::Row, Oob>(dstTile, tableGlobal, idxTile);

    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);

    TSTORE(outGlobal, dstTile);
}

template <pto::GatherOOB Oob, typename T, typename TIdx, uint32_t kDstCols, uint32_t kTableSize>
inline AICORE void runElem(__gm__ T *out, __gm__ T *table, __gm__ TIdx *indices)
{
    using TableShape = pto::Shape<1, 1, 1, 1, kTableSize>;
    using TableStride = pto::Stride<1, 1, 1, kTableSize, 1>;
    using IdxShape = pto::Shape<1, 1, 1, 1, kDstCols>;
    using IdxStride = pto::Stride<1, 1, 1, kDstCols, 1>;
    using OutShape = pto::Shape<1, 1, 1, 1, kDstCols>;
    using OutStride = pto::Stride<1, 1, 1, kDstCols, 1>;

    GlobalTensor<T, TableShape, TableStride> tableGlobal(table);
    GlobalTensor<TIdx, IdxShape, IdxStride> idxGlobal(indices);
    GlobalTensor<T, OutShape, OutStride> outGlobal(out);

    using DstTile = Tile<TileType::Vec, T, 1, kDstCols, BLayout::RowMajor, 1, kDstCols>;
    using IdxTile = Tile<TileType::Vec, TIdx, 1, kDstCols, BLayout::RowMajor, 1, kDstCols>;

    DstTile dstTile;
    IdxTile idxTile;

    constexpr uint32_t idxBytes = ((1u * kDstCols * sizeof(TIdx) + 31u) / 32u) * 32u;
    TASSIGN(idxTile, 0x0);
    TASSIGN(dstTile, idxBytes);

    TLOAD(idxTile, idxGlobal);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

    MGATHER<Coalesce::Elem, Oob>(dstTile, tableGlobal, idxTile);

    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);

    TSTORE(outGlobal, dstTile);
}

template <pto::GatherOOB Oob, typename T, typename TIdx, uint32_t kDstRows, uint32_t kDstCols, uint32_t kTableSize>
inline AICORE void runElem2D(__gm__ T *out, __gm__ T *table, __gm__ TIdx *indices)
{
    using TableShape = pto::Shape<1, 1, 1, 1, kTableSize>;
    using TableStride = pto::Stride<1, 1, 1, kTableSize, 1>;
    using IdxShape = pto::Shape<1, 1, 1, kDstRows, kDstCols>;
    using IdxStride = pto::Stride<1, 1, 1, kDstCols, 1>;
    using OutShape = pto::Shape<1, 1, 1, kDstRows, kDstCols>;
    using OutStride = pto::Stride<1, 1, 1, kDstCols, 1>;

    GlobalTensor<T, TableShape, TableStride> tableGlobal(table);
    GlobalTensor<TIdx, IdxShape, IdxStride> idxGlobal(indices);
    GlobalTensor<T, OutShape, OutStride> outGlobal(out);

    using DstTile = Tile<TileType::Vec, T, kDstRows, kDstCols, BLayout::RowMajor, kDstRows, kDstCols>;
    using IdxTile = Tile<TileType::Vec, TIdx, kDstRows, kDstCols, BLayout::RowMajor, kDstRows, kDstCols>;

    DstTile dstTile;
    IdxTile idxTile;

    constexpr uint32_t idxBytes = ((kDstRows * kDstCols * sizeof(TIdx) + 31u) / 32u) * 32u;
    TASSIGN(idxTile, 0x0);
    TASSIGN(dstTile, idxBytes);

    TLOAD(idxTile, idxGlobal);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

    MGATHER<Coalesce::Elem, Oob>(dstTile, tableGlobal, idxTile);

    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);

    TSTORE(outGlobal, dstTile);
}

template <pto::GatherOOB Oob, typename T, typename TIdx, uint32_t kValidRows, uint32_t kValidCols, uint32_t kPadRows,
          uint32_t kPadCols, uint32_t kTableSize>
inline AICORE void runElem2DPadded(__gm__ T *out, __gm__ T *table, __gm__ TIdx *indices)
{
    using TableShape = pto::Shape<1, 1, 1, 1, kTableSize>;
    using TableStride = pto::Stride<1, 1, 1, kTableSize, 1>;
    using IdxShape = pto::Shape<1, 1, 1, kValidRows, kValidCols>;
    using IdxStride = pto::Stride<1, 1, 1, kValidCols, 1>;
    using OutShape = pto::Shape<1, 1, 1, kValidRows, kValidCols>;
    using OutStride = pto::Stride<1, 1, 1, kValidCols, 1>;

    GlobalTensor<T, TableShape, TableStride> tableGlobal(table);
    GlobalTensor<TIdx, IdxShape, IdxStride> idxGlobal(indices);
    GlobalTensor<T, OutShape, OutStride> outGlobal(out);

    using DstTile = Tile<TileType::Vec, T, kPadRows, kPadCols, BLayout::RowMajor, kValidRows, kValidCols>;
    using IdxTile = Tile<TileType::Vec, TIdx, kPadRows, kPadCols, BLayout::RowMajor, kValidRows, kValidCols>;

    DstTile dstTile;
    IdxTile idxTile;

    constexpr uint32_t idxBytes = ((kPadRows * kPadCols * sizeof(TIdx) + 31u) / 32u) * 32u;
    TASSIGN(idxTile, 0x0);
    TASSIGN(dstTile, idxBytes);

    TLOAD(idxTile, idxGlobal);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

    MGATHER<Coalesce::Elem, Oob>(dstTile, tableGlobal, idxTile);

    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);

    TSTORE(outGlobal, dstTile);
}

template <pto::GatherOOB Oob, typename T, typename TIdx, uint32_t kPadCols, uint32_t kTableSize>
inline AICORE void runElemScalar(__gm__ T *out, __gm__ T *table, __gm__ TIdx *indices)
{
    using TableShape = pto::Shape<1, 1, 1, 1, kTableSize>;
    using TableStride = pto::Stride<1, 1, 1, kTableSize, 1>;
    using IdxShape = pto::Shape<1, 1, 1, 1, 1>;
    using IdxStride = pto::Stride<1, 1, 1, 1, 1>;
    using OutShape = pto::Shape<1, 1, 1, 1, 1>;
    using OutStride = pto::Stride<1, 1, 1, 1, 1>;

    GlobalTensor<T, TableShape, TableStride> tableGlobal(table);
    GlobalTensor<TIdx, IdxShape, IdxStride> idxGlobal(indices);
    GlobalTensor<T, OutShape, OutStride> outGlobal(out);

    using DstTile = Tile<TileType::Vec, T, 1, kPadCols, BLayout::RowMajor, 1, 1>;
    using IdxTile = Tile<TileType::Vec, TIdx, 1, kPadCols, BLayout::RowMajor, 1, 1>;

    DstTile dstTile;
    IdxTile idxTile;

    constexpr uint32_t idxBytes = ((1u * kPadCols * sizeof(TIdx) + 31u) / 32u) * 32u;
    TASSIGN(idxTile, 0x0);
    TASSIGN(dstTile, idxBytes);

    TLOAD(idxTile, idxGlobal);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

    MGATHER<Coalesce::Elem, Oob>(dstTile, tableGlobal, idxTile);

    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);

    TSTORE(outGlobal, dstTile);
}

template <pto::GatherOOB Oob, typename T, typename TIdx, uint32_t kPadRows, uint32_t kPadCols, int64_t kRtValidRows,
          int64_t kRtValidCols, int64_t kRtTableSize>
inline AICORE void runElem2DDyn(__gm__ T *out, __gm__ T *table, __gm__ TIdx *indices)
{
    using TableShape = pto::Shape<1, 1, 1, -1, -1>;
    using TableStride = pto::Stride<1, 1, 1, -1, -1>;
    using IdxShape = pto::Shape<1, 1, 1, -1, -1>;
    using IdxStride = pto::Stride<1, 1, 1, -1, -1>;
    using OutShape = pto::Shape<1, 1, 1, -1, -1>;
    using OutStride = pto::Stride<1, 1, 1, -1, -1>;

    TableShape tableShape((int64_t)1, kRtTableSize);
    TableStride tableStride(kRtTableSize, (int64_t)1);
    IdxShape idxShape(kRtValidRows, kRtValidCols);
    IdxStride idxStride(kRtValidCols, (int64_t)1);
    OutShape outShape(kRtValidRows, kRtValidCols);
    OutStride outStride(kRtValidCols, (int64_t)1);

    GlobalTensor<T, TableShape, TableStride> tableGlobal(table, tableShape, tableStride);
    GlobalTensor<TIdx, IdxShape, IdxStride> idxGlobal(indices, idxShape, idxStride);
    GlobalTensor<T, OutShape, OutStride> outGlobal(out, outShape, outStride);

    using DstTile = Tile<TileType::Vec, T, kPadRows, kPadCols, BLayout::RowMajor, -1, -1>;
    using IdxTile = Tile<TileType::Vec, TIdx, kPadRows, kPadCols, BLayout::RowMajor, -1, -1>;

    DstTile dstTile(static_cast<unsigned>(kRtValidRows), static_cast<unsigned>(kRtValidCols));
    IdxTile idxTile(static_cast<unsigned>(kRtValidRows), static_cast<unsigned>(kRtValidCols));

    constexpr uint32_t idxBytes = ((kPadRows * kPadCols * sizeof(TIdx) + 31u) / 32u) * 32u;
    TASSIGN(idxTile, 0x0);
    TASSIGN(dstTile, idxBytes);

    TLOAD(idxTile, idxGlobal);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

    MGATHER<Coalesce::Elem, Oob>(dstTile, tableGlobal, idxTile);

    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);

    TSTORE(outGlobal, dstTile);
}

template <pto::GatherOOB Oob, typename T, typename TIdx, uint32_t kPadRows, uint32_t kPadCols, uint32_t kPadIdxCols,
          int64_t kRtValidRows, int64_t kRtValidCols, int64_t kRtTableRows>
inline AICORE void runRowDyn(__gm__ T *out, __gm__ T *table, __gm__ TIdx *indices)
{
    using TableShape = pto::Shape<1, 1, 1, -1, -1>;
    using TableStride = pto::Stride<1, 1, 1, -1, -1>;
    using IdxShape = pto::Shape<1, 1, 1, 1, -1>;
    using IdxStride = pto::Stride<1, 1, 1, -1, -1>;
    using OutShape = pto::Shape<1, 1, 1, -1, -1>;
    using OutStride = pto::Stride<1, 1, 1, -1, -1>;

    TableShape tableShape(kRtTableRows, kRtValidCols);
    TableStride tableStride(kRtValidCols, (int64_t)1);
    IdxShape idxShape(kRtValidRows);
    IdxStride idxStride(kRtValidRows, (int64_t)1);
    OutShape outShape(kRtValidRows, kRtValidCols);
    OutStride outStride(kRtValidCols, (int64_t)1);

    GlobalTensor<T, TableShape, TableStride> tableGlobal(table, tableShape, tableStride);
    GlobalTensor<TIdx, IdxShape, IdxStride> idxGlobal(indices, idxShape, idxStride);
    GlobalTensor<T, OutShape, OutStride> outGlobal(out, outShape, outStride);

    using DstTile = Tile<TileType::Vec, T, kPadRows, kPadCols, BLayout::RowMajor, -1, -1>;
    using IdxTile = Tile<TileType::Vec, TIdx, 1, kPadIdxCols, BLayout::RowMajor, 1, -1>;

    DstTile dstTile(static_cast<unsigned>(kRtValidRows), static_cast<unsigned>(kRtValidCols));
    IdxTile idxTile(static_cast<unsigned>(kRtValidRows));

    constexpr uint32_t idxBytes = ((1u * kPadIdxCols * sizeof(TIdx) + 31u) / 32u) * 32u;
    TASSIGN(idxTile, 0x0);
    TASSIGN(dstTile, idxBytes);

    TLOAD(idxTile, idxGlobal);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

    MGATHER<Coalesce::Row, Oob>(dstTile, tableGlobal, idxTile);

    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);

    TSTORE(outGlobal, dstTile);
}

template <pto::GatherOOB Oob, typename T, typename TIdx, uint32_t kDstRows, uint32_t kDstCols, uint32_t kBlockRows,
          uint32_t kBlockCols, uint32_t kC0>
inline AICORE void runRowNz(__gm__ T *out, __gm__ T *table, __gm__ TIdx *indices)
{
    using TableShape = pto::Shape<1, kBlockCols, kBlockRows, 16, kC0>;
    using TableStride = pto::Stride<kBlockCols * kBlockRows * 16 * kC0, kBlockRows * 16 * kC0, 16 * kC0, kC0, 1>;
    using IdxShape = pto::Shape<1, 1, 1, 1, kDstRows>;
    using IdxStride = pto::Stride<1, 1, 1, kDstRows, 1>;
    using OutShape = pto::Shape<1, kBlockCols, kDstRows / 16, 16, kC0>;
    using OutStride = pto::Stride<kBlockCols *(kDstRows / 16) * 16 * kC0, (kDstRows / 16) * 16 * kC0, 16 * kC0, kC0, 1>;

    GlobalTensor<T, TableShape, TableStride, Layout::NZ> tableGlobal(table);
    GlobalTensor<TIdx, IdxShape, IdxStride> idxGlobal(indices);
    GlobalTensor<T, OutShape, OutStride, Layout::NZ> outGlobal(out);

    using DstTile =
        Tile<TileType::Vec, T, kDstRows, kDstCols, BLayout::ColMajor, kDstRows, kDstCols, SLayout::RowMajor, 512>;
    using IdxTile = Tile<TileType::Vec, TIdx, 1, kDstRows, BLayout::RowMajor, 1, kDstRows>;

    DstTile dstTile;
    IdxTile idxTile;

    constexpr uint32_t idxBytes = ((1u * kDstRows * sizeof(TIdx) + 31u) / 32u) * 32u;
    TASSIGN(idxTile, 0x0);
    TASSIGN(dstTile, idxBytes);

    TLOAD(idxTile, idxGlobal);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

    MGATHER<Coalesce::Row, Oob>(dstTile, tableGlobal, idxTile);

    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);

    TSTORE(outGlobal, dstTile);
}

template <pto::GatherOOB Oob, typename T, typename TIdx, uint32_t kDstRows, uint32_t kDstCols, uint32_t kBlockRows,
          uint32_t kBlockCols, uint32_t kC0>
inline AICORE void runElem2DNz(__gm__ T *out, __gm__ T *table, __gm__ TIdx *indices)
{
    using TableShape = pto::Shape<1, kBlockCols, kBlockRows, 16, kC0>;
    using TableStride = pto::Stride<kBlockCols * kBlockRows * 16 * kC0, kBlockRows * 16 * kC0, 16 * kC0, kC0, 1>;
    using IdxShape = pto::Shape<1, 1, 1, kDstRows, kDstCols>;
    using IdxStride = pto::Stride<1, 1, 1, kDstCols, 1>;
    using OutShape = pto::Shape<1, kBlockCols, kDstRows / 16, 16, kC0>;
    using OutStride = pto::Stride<kBlockCols *(kDstRows / 16) * 16 * kC0, (kDstRows / 16) * 16 * kC0, 16 * kC0, kC0, 1>;

    GlobalTensor<T, TableShape, TableStride, Layout::NZ> tableGlobal(table);
    GlobalTensor<TIdx, IdxShape, IdxStride> idxGlobal(indices);
    GlobalTensor<T, OutShape, OutStride, Layout::NZ> outGlobal(out);

    using DstTile =
        Tile<TileType::Vec, T, kDstRows, kDstCols, BLayout::ColMajor, kDstRows, kDstCols, SLayout::RowMajor, 512>;
    using IdxTile = Tile<TileType::Vec, TIdx, kDstRows, kDstCols, BLayout::RowMajor, kDstRows, kDstCols>;

    DstTile dstTile;
    IdxTile idxTile;

    constexpr uint32_t idxBytes = ((kDstRows * kDstCols * sizeof(TIdx) + 31u) / 32u) * 32u;
    TASSIGN(idxTile, 0x0);
    TASSIGN(dstTile, idxBytes);

    TLOAD(idxTile, idxGlobal);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

    MGATHER<Coalesce::Elem, Oob>(dstTile, tableGlobal, idxTile);

    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);

    TSTORE(outGlobal, dstTile);
}

#define DEFINE_ROW(NAME, THOST, T, TIDX, R, C, TR, OOB)                                                               \
    extern "C" __global__ AICORE void runMGATHER_##NAME(__gm__ T *out, __gm__ T *table, __gm__ TIDX *indices)         \
    {                                                                                                                 \
        runRow<pto::GatherOOB::OOB, T, TIDX, R, C, TR>(out, table, indices);                                          \
    }                                                                                                                 \
    void Launch_##NAME(THOST *out, THOST *table, TIDX *indices, void *stream)                                         \
    {                                                                                                                 \
        runMGATHER_##NAME<<<1, nullptr, stream>>>(reinterpret_cast<T *>(out), reinterpret_cast<T *>(table), indices); \
    }

#define DEFINE_ROW_PAD(NAME, THOST, T, TIDX, VR, PR, PC, PIC, TR, OOB)                                                \
    extern "C" __global__ AICORE void runMGATHER_##NAME(__gm__ T *out, __gm__ T *table, __gm__ TIDX *indices)         \
    {                                                                                                                 \
        runRowPadded<pto::GatherOOB::OOB, T, TIDX, VR, PR, PC, PIC, TR>(out, table, indices);                         \
    }                                                                                                                 \
    void Launch_##NAME(THOST *out, THOST *table, TIDX *indices, void *stream)                                         \
    {                                                                                                                 \
        runMGATHER_##NAME<<<1, nullptr, stream>>>(reinterpret_cast<T *>(out), reinterpret_cast<T *>(table), indices); \
    }

#define DEFINE_ELEM(NAME, THOST, T, TIDX, N, TS, OOB)                                                                 \
    extern "C" __global__ AICORE void runMGATHER_##NAME(__gm__ T *out, __gm__ T *table, __gm__ TIDX *indices)         \
    {                                                                                                                 \
        runElem<pto::GatherOOB::OOB, T, TIDX, N, TS>(out, table, indices);                                            \
    }                                                                                                                 \
    void Launch_##NAME(THOST *out, THOST *table, TIDX *indices, void *stream)                                         \
    {                                                                                                                 \
        runMGATHER_##NAME<<<1, nullptr, stream>>>(reinterpret_cast<T *>(out), reinterpret_cast<T *>(table), indices); \
    }

#define DEFINE_ELEM2D(NAME, THOST, T, TIDX, R, C, TS, OOB)                                                            \
    extern "C" __global__ AICORE void runMGATHER_##NAME(__gm__ T *out, __gm__ T *table, __gm__ TIDX *indices)         \
    {                                                                                                                 \
        runElem2D<pto::GatherOOB::OOB, T, TIDX, R, C, TS>(out, table, indices);                                       \
    }                                                                                                                 \
    void Launch_##NAME(THOST *out, THOST *table, TIDX *indices, void *stream)                                         \
    {                                                                                                                 \
        runMGATHER_##NAME<<<1, nullptr, stream>>>(reinterpret_cast<T *>(out), reinterpret_cast<T *>(table), indices); \
    }

#define DEFINE_ELEM2D_PAD(NAME, THOST, T, TIDX, VR, VC, PR, PC, TS, OOB)                                              \
    extern "C" __global__ AICORE void runMGATHER_##NAME(__gm__ T *out, __gm__ T *table, __gm__ TIDX *indices)         \
    {                                                                                                                 \
        runElem2DPadded<pto::GatherOOB::OOB, T, TIDX, VR, VC, PR, PC, TS>(out, table, indices);                       \
    }                                                                                                                 \
    void Launch_##NAME(THOST *out, THOST *table, TIDX *indices, void *stream)                                         \
    {                                                                                                                 \
        runMGATHER_##NAME<<<1, nullptr, stream>>>(reinterpret_cast<T *>(out), reinterpret_cast<T *>(table), indices); \
    }

#define DEFINE_ELEM_SCALAR(NAME, THOST, T, TIDX, PC, TS, OOB)                                                         \
    extern "C" __global__ AICORE void runMGATHER_##NAME(__gm__ T *out, __gm__ T *table, __gm__ TIDX *indices)         \
    {                                                                                                                 \
        runElemScalar<pto::GatherOOB::OOB, T, TIDX, PC, TS>(out, table, indices);                                     \
    }                                                                                                                 \
    void Launch_##NAME(THOST *out, THOST *table, TIDX *indices, void *stream)                                         \
    {                                                                                                                 \
        runMGATHER_##NAME<<<1, nullptr, stream>>>(reinterpret_cast<T *>(out), reinterpret_cast<T *>(table), indices); \
    }

#define DEFINE_ELEM2D_DYN(NAME, THOST, T, TIDX, PR, PC, RVR, RVC, RTS, OOB)                                           \
    extern "C" __global__ AICORE void runMGATHER_##NAME(__gm__ T *out, __gm__ T *table, __gm__ TIDX *indices)         \
    {                                                                                                                 \
        runElem2DDyn<pto::GatherOOB::OOB, T, TIDX, PR, PC, (int64_t)RVR, (int64_t)RVC, (int64_t)RTS>(out, table,      \
                                                                                                     indices);        \
    }                                                                                                                 \
    void Launch_##NAME(THOST *out, THOST *table, TIDX *indices, void *stream)                                         \
    {                                                                                                                 \
        runMGATHER_##NAME<<<1, nullptr, stream>>>(reinterpret_cast<T *>(out), reinterpret_cast<T *>(table), indices); \
    }

#define DEFINE_ROW_DYN(NAME, THOST, T, TIDX, PR, PC, PIC, RVR, RVC, RTR, OOB)                                         \
    extern "C" __global__ AICORE void runMGATHER_##NAME(__gm__ T *out, __gm__ T *table, __gm__ TIDX *indices)         \
    {                                                                                                                 \
        runRowDyn<pto::GatherOOB::OOB, T, TIDX, PR, PC, PIC, (int64_t)RVR, (int64_t)RVC, (int64_t)RTR>(out, table,    \
                                                                                                       indices);      \
    }                                                                                                                 \
    void Launch_##NAME(THOST *out, THOST *table, TIDX *indices, void *stream)                                         \
    {                                                                                                                 \
        runMGATHER_##NAME<<<1, nullptr, stream>>>(reinterpret_cast<T *>(out), reinterpret_cast<T *>(table), indices); \
    }

#define DEFINE_ROW_NZ(NAME, THOST, T, TIDX, R, C, BR, BC, C0, OOB)                                                    \
    extern "C" __global__ AICORE void runMGATHER_##NAME(__gm__ T *out, __gm__ T *table, __gm__ TIDX *indices)         \
    {                                                                                                                 \
        runRowNz<pto::GatherOOB::OOB, T, TIDX, R, C, BR, BC, C0>(out, table, indices);                                \
    }                                                                                                                 \
    void Launch_##NAME(THOST *out, THOST *table, TIDX *indices, void *stream)                                         \
    {                                                                                                                 \
        runMGATHER_##NAME<<<1, nullptr, stream>>>(reinterpret_cast<T *>(out), reinterpret_cast<T *>(table), indices); \
    }

#define DEFINE_ELEM2D_NZ(NAME, THOST, T, TIDX, R, C, BR, BC, C0, OOB)                                                 \
    extern "C" __global__ AICORE void runMGATHER_##NAME(__gm__ T *out, __gm__ T *table, __gm__ TIDX *indices)         \
    {                                                                                                                 \
        runElem2DNz<pto::GatherOOB::OOB, T, TIDX, R, C, BR, BC, C0>(out, table, indices);                             \
    }                                                                                                                 \
    void Launch_##NAME(THOST *out, THOST *table, TIDX *indices, void *stream)                                         \
    {                                                                                                                 \
        runMGATHER_##NAME<<<1, nullptr, stream>>>(reinterpret_cast<T *>(out), reinterpret_cast<T *>(table), indices); \
    }

DEFINE_ROW(row_float_8x32_64rows, float, float, int32_t, 8, 32, 64, Undefined)
DEFINE_ROW(row_half_16x64_64rows, aclFloat16, half, int32_t, 16, 64, 64, Undefined)
DEFINE_ROW(row_bfloat16_16x16_64rows, uint16_t, bfloat16_t, int32_t, 16, 16, 64, Undefined)
DEFINE_ROW(row_int32_8x16_32rows, int32_t, int32_t, int32_t, 8, 16, 32, Undefined)
DEFINE_ROW(row_uint32_8x16_32rows, uint32_t, uint32_t, int32_t, 8, 16, 32, Undefined)
DEFINE_ROW(row_int16_8x16_32rows, int16_t, int16_t, int32_t, 8, 16, 32, Undefined)
DEFINE_ROW(row_uint16_8x16_32rows, uint16_t, uint16_t, int32_t, 8, 16, 32, Undefined)
DEFINE_ROW(row_int8_8x32_32rows, int8_t, int8_t, int32_t, 8, 32, 32, Undefined)
DEFINE_ROW(row_uint8_8x32_32rows, uint8_t, uint8_t, int32_t, 8, 32, 32, Undefined)
DEFINE_ROW(row_float_clamp_8x32_8rows, float, float, int32_t, 8, 32, 8, Clamp)
DEFINE_ROW(row_int32_wrap_8x16_8rows, int32_t, int32_t, int32_t, 8, 16, 8, Wrap)
DEFINE_ROW(row_half_zero_8x32_8rows, aclFloat16, half, int32_t, 8, 32, 8, Zero)

DEFINE_ROW_PAD(row_int32_unaligned_3x8_8rows, int32_t, int32_t, int32_t, 3, 3, 8, 8, 8, Undefined)
DEFINE_ROW_PAD(row_float_partial_4x16_in_8x16, float, float, int32_t, 4, 8, 16, 8, 8, Undefined)
DEFINE_ROW_PAD(row_half_partial_5x32_in_8x32, aclFloat16, half, int32_t, 5, 8, 32, 8, 8, Undefined)
DEFINE_ROW_PAD(row_uint8_unaligned_3x32_32rows, uint8_t, uint8_t, int32_t, 3, 3, 32, 8, 8, Undefined)
DEFINE_ROW_PAD(row_int16_partial_3x16_in_4x16, int16_t, int16_t, int32_t, 3, 4, 16, 8, 8, Clamp)

DEFINE_ELEM(elem_float_64_128size, float, float, int32_t, 64, 128, Undefined)
DEFINE_ELEM(elem_half_64_128size, aclFloat16, half, int32_t, 64, 128, Undefined)
DEFINE_ELEM(elem_bfloat16_64_128size, uint16_t, bfloat16_t, int32_t, 64, 128, Undefined)
DEFINE_ELEM(elem_int32_32_64size, int32_t, int32_t, int32_t, 32, 64, Undefined)
DEFINE_ELEM(elem_uint32_32_64size, uint32_t, uint32_t, int32_t, 32, 64, Undefined)
DEFINE_ELEM(elem_int16_32_64size, int16_t, int16_t, int32_t, 32, 64, Undefined)
DEFINE_ELEM(elem_uint16_32_64size, uint16_t, uint16_t, int32_t, 32, 64, Undefined)
DEFINE_ELEM(elem_int8_64_128size, int8_t, int8_t, int32_t, 64, 128, Undefined)
DEFINE_ELEM(elem_uint8_64_128size, uint8_t, uint8_t, int32_t, 64, 128, Undefined)
DEFINE_ELEM(elem_float_clamp_32_16size, float, float, int32_t, 32, 16, Clamp)
DEFINE_ELEM(elem_int32_wrap_32_16size, int32_t, int32_t, int32_t, 32, 16, Wrap)
DEFINE_ELEM(elem_half_zero_32_16size, aclFloat16, half, int32_t, 32, 16, Zero)

DEFINE_ELEM2D(elem2d_float_8x32_256size, float, float, int32_t, 8, 32, 256, Undefined)
DEFINE_ELEM2D(elem2d_int32_8x16_256size, int32_t, int32_t, int32_t, 8, 16, 256, Undefined)
DEFINE_ELEM2D(elem2d_half_4x32_256size, aclFloat16, half, int32_t, 4, 32, 256, Undefined)
DEFINE_ELEM2D(elem2d_bfloat16_4x32_256size, uint16_t, bfloat16_t, int32_t, 4, 32, 256, Undefined)
DEFINE_ELEM2D(elem2d_uint8_4x64_256size, uint8_t, uint8_t, int32_t, 4, 64, 256, Undefined)
DEFINE_ELEM2D(elem2d_int8_4x64_256size, int8_t, int8_t, int32_t, 4, 64, 256, Undefined)
DEFINE_ELEM2D(elem2d_int16_4x32_256size, int16_t, int16_t, int32_t, 4, 32, 256, Undefined)
DEFINE_ELEM2D(elem2d_uint16_4x32_256size, uint16_t, uint16_t, int32_t, 4, 32, 256, Undefined)
DEFINE_ELEM2D(elem2d_uint32_8x16_256size, uint32_t, uint32_t, int32_t, 8, 16, 256, Undefined)
DEFINE_ELEM2D(elem2d_float_wrap_4x16_64size, float, float, int32_t, 4, 16, 64, Wrap)
DEFINE_ELEM2D(elem2d_int32_clamp_4x8_32size, int32_t, int32_t, int32_t, 4, 8, 32, Clamp)
DEFINE_ELEM2D(elem2d_half_zero_4x32_64size, aclFloat16, half, int32_t, 4, 32, 64, Zero)

DEFINE_ELEM2D_PAD(elem2d_int32_unaligned_3x3_in_3x8_64size, int32_t, int32_t, int32_t, 3, 3, 3, 8, 64, Undefined)
DEFINE_ELEM2D_PAD(elem2d_float_unaligned_5x5_in_5x8_64size, float, float, int32_t, 5, 5, 5, 8, 64, Undefined)
DEFINE_ELEM2D_PAD(elem2d_half_unaligned_3x9_in_3x16_64size, aclFloat16, half, int32_t, 3, 9, 3, 16, 64, Undefined)
DEFINE_ELEM2D_PAD(elem2d_int8_unaligned_3x17_in_3x32_64size, int8_t, int8_t, int32_t, 3, 17, 3, 32, 64, Undefined)

DEFINE_ELEM_SCALAR(elem_scalar_float_1x1_in_1x8_8size, float, float, int32_t, 8, 8, Undefined)
DEFINE_ELEM_SCALAR(elem_scalar_int32_1x1_in_1x8_8size, int32_t, int32_t, int32_t, 8, 8, Undefined)
DEFINE_ELEM_SCALAR(elem_scalar_half_1x1_in_1x16_16size, aclFloat16, half, int32_t, 16, 16, Undefined)

DEFINE_ELEM2D_DYN(elem2d_dyn_float_4x8_64size, float, float, int32_t, 4, 8, 4, 8, 64, Undefined)
DEFINE_ELEM2D_DYN(elem2d_dyn_int32_3x3_in_3x8_64size, int32_t, int32_t, int32_t, 3, 8, 3, 3, 64, Undefined)

DEFINE_ROW_DYN(row_dyn_int32_3x16_8rows, int32_t, int32_t, int32_t, 3, 16, 8, 3, 16, 8, Undefined)
DEFINE_ROW_DYN(row_dyn_half_4x32_16rows, aclFloat16, half, int32_t, 4, 32, 16, 4, 32, 16, Undefined)

DEFINE_ROW_NZ(row_nz_float_16x16_2blk, float, float, int32_t, 16, 16, 2, 2, 8, Undefined)
DEFINE_ROW_NZ(row_nz_half_32x16_2blk, aclFloat16, half, int32_t, 32, 16, 2, 1, 16, Undefined)
DEFINE_ROW_NZ(row_nz_int32_16x16_2blk, int32_t, int32_t, int32_t, 16, 16, 2, 2, 8, Undefined)
DEFINE_ROW_NZ(row_nz_int16_32x16_1blk, int16_t, int16_t, int32_t, 32, 16, 2, 1, 16, Undefined)
DEFINE_ROW_NZ(row_nz_int8_16x32_1blk, int8_t, int8_t, int32_t, 16, 32, 2, 1, 32, Undefined)
DEFINE_ROW_NZ(row_nz_float_clamp_16x8_1blk, float, float, int32_t, 16, 8, 2, 1, 8, Clamp)
DEFINE_ROW_NZ(row_nz_half_zero_16x16_2blk, aclFloat16, half, int32_t, 16, 16, 2, 1, 16, Zero)

DEFINE_ELEM2D_NZ(elem2d_nz_float_16x16_2blk, float, float, int32_t, 16, 16, 2, 2, 8, Undefined)
DEFINE_ELEM2D_NZ(elem2d_nz_half_16x16_1blk, aclFloat16, half, int32_t, 16, 16, 2, 1, 16, Undefined)
DEFINE_ELEM2D_NZ(elem2d_nz_int32_16x8_1blk, int32_t, int32_t, int32_t, 16, 8, 2, 1, 8, Undefined)
DEFINE_ELEM2D_NZ(elem2d_nz_half_zero_16x16_1blk, aclFloat16, half, int32_t, 16, 16, 2, 1, 16, Zero)
