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
#include <pto/npu/a5/MGather.hpp>
#include "acl/acl.h"

using namespace pto;

template <pto::GatherOOB Oob, typename T, typename TIdx, int kDstRows, int kDstCols, int kTableRows>
inline AICORE void runRow(__gm__ T __out__ *out, __gm__ T __in__ *table, __gm__ TIdx __in__ *indices)
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

    constexpr int idxBytes = ((1 * kDstRows * (int)sizeof(TIdx) + 31) / 32) * 32;
    constexpr int dstBytes = ((kDstRows * kDstCols * (int)sizeof(T) + 31) / 32) * 32;
    TASSIGN(idxTile, 0x0);
    TASSIGN(dstTile, idxBytes);

    TLOAD(idxTile, idxGlobal);
#ifndef __PTO_AUTO__
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
#endif
    MGATHER<Coalesce::Row, Oob>(dstTile, tableGlobal, idxTile);
#ifndef __PTO_AUTO__
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
#endif
    TSTORE(outGlobal, dstTile);
    (void)dstBytes;
}

template <pto::GatherOOB Oob, typename T, typename TIdx, int kValidRows, int kPadIdxCols, int kDstCols, int kTableRows>
inline AICORE void runRowPadded(__gm__ T __out__ *out, __gm__ T __in__ *table, __gm__ TIdx __in__ *indices)
{
    using TableShape = pto::Shape<1, 1, 1, kTableRows, kDstCols>;
    using TableStride = pto::Stride<1, 1, 1, kDstCols, 1>;
    using IdxShape = pto::Shape<1, 1, 1, 1, kValidRows>;
    using IdxStride = pto::Stride<1, 1, 1, kValidRows, 1>;
    using OutShape = pto::Shape<1, 1, 1, kValidRows, kDstCols>;
    using OutStride = pto::Stride<1, 1, 1, kDstCols, 1>;

    GlobalTensor<T, TableShape, TableStride> tableGlobal(table);
    GlobalTensor<TIdx, IdxShape, IdxStride> idxGlobal(indices);
    GlobalTensor<T, OutShape, OutStride> outGlobal(out);

    using DstTile = Tile<TileType::Vec, T, kValidRows, kDstCols, BLayout::RowMajor, kValidRows, kDstCols>;
    using IdxTile = Tile<TileType::Vec, TIdx, 1, kPadIdxCols, BLayout::RowMajor, 1, kValidRows>;

    DstTile dstTile;
    IdxTile idxTile;

    constexpr int idxBytes = ((1 * kPadIdxCols * (int)sizeof(TIdx) + 31) / 32) * 32;
    constexpr int dstBytes = ((kValidRows * kDstCols * (int)sizeof(T) + 31) / 32) * 32;
    TASSIGN(idxTile, 0x0);
    TASSIGN(dstTile, idxBytes);

    TLOAD(idxTile, idxGlobal);
#ifndef __PTO_AUTO__
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
#endif
    MGATHER<Coalesce::Row, Oob>(dstTile, tableGlobal, idxTile);
#ifndef __PTO_AUTO__
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
#endif
    TSTORE(outGlobal, dstTile);
    (void)dstBytes;
}

template <pto::GatherOOB Oob, typename T, typename TIdx, int kDstRows, int kDstCols, int kTableRows>
inline AICORE void runRowColIdx(__gm__ T __out__ *out, __gm__ T __in__ *table, __gm__ TIdx __in__ *indices)
{
    using TableShape = pto::Shape<1, 1, 1, kTableRows, kDstCols>;
    using TableStride = pto::Stride<1, 1, 1, kDstCols, 1>;
    using IdxShape = pto::Shape<1, 1, 1, kDstRows, 1>;
    using IdxStride = pto::Stride<1, 1, 1, 1, 1>;
    using OutShape = pto::Shape<1, 1, 1, kDstRows, kDstCols>;
    using OutStride = pto::Stride<1, 1, 1, kDstCols, 1>;

    GlobalTensor<T, TableShape, TableStride> tableGlobal(table);
    GlobalTensor<TIdx, IdxShape, IdxStride, Layout::DN> idxGlobal(indices);
    GlobalTensor<T, OutShape, OutStride> outGlobal(out);

    using DstTile = Tile<TileType::Vec, T, kDstRows, kDstCols, BLayout::RowMajor, kDstRows, kDstCols>;
    using IdxTile = Tile<TileType::Vec, TIdx, kDstRows, 1, BLayout::ColMajor, kDstRows, 1>;

    DstTile dstTile;
    IdxTile idxTile;

    constexpr int idxBytes = ((kDstRows * 1 * (int)sizeof(TIdx) + 31) / 32) * 32;
    constexpr int dstBytes = ((kDstRows * kDstCols * (int)sizeof(T) + 31) / 32) * 32;
    TASSIGN(idxTile, 0x0);
    TASSIGN(dstTile, idxBytes);

    TLOAD(idxTile, idxGlobal);
#ifndef __PTO_AUTO__
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
#endif
    MGATHER<Coalesce::Row, Oob>(dstTile, tableGlobal, idxTile);
#ifndef __PTO_AUTO__
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
#endif
    TSTORE(outGlobal, dstTile);
    (void)dstBytes;
}

template <pto::GatherOOB Oob, typename T, typename TIdx, int kDstCols, int kTableSize>
inline AICORE void runElem(__gm__ T __out__ *out, __gm__ T __in__ *table, __gm__ TIdx __in__ *indices)
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

    constexpr int idxBytes = ((1 * kDstCols * (int)sizeof(TIdx) + 31) / 32) * 32;
    constexpr int dstBytes = ((1 * kDstCols * (int)sizeof(T) + 31) / 32) * 32;
    TASSIGN(idxTile, 0x0);
    TASSIGN(dstTile, idxBytes);

    TLOAD(idxTile, idxGlobal);
#ifndef __PTO_AUTO__
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
#endif
    MGATHER<Coalesce::Elem, Oob>(dstTile, tableGlobal, idxTile);
#ifndef __PTO_AUTO__
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
#endif
    TSTORE(outGlobal, dstTile);
    (void)dstBytes;
}

template <pto::GatherOOB Oob, typename T, typename TIdx, int kValidRows, int kValidCols, int kPadRows, int kPadCols,
          int kTableSize>
inline AICORE void runElem2DPadded(__gm__ T __out__ *out, __gm__ T __in__ *table, __gm__ TIdx __in__ *indices)
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

    constexpr int idxBytes = ((kPadRows * kPadCols * (int)sizeof(TIdx) + 31) / 32) * 32;
    constexpr int dstBytes = ((kPadRows * kPadCols * (int)sizeof(T) + 31) / 32) * 32;
    TASSIGN(idxTile, 0x0);
    TASSIGN(dstTile, idxBytes);

    TLOAD(idxTile, idxGlobal);
#ifndef __PTO_AUTO__
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
#endif
    MGATHER<Coalesce::Elem, Oob>(dstTile, tableGlobal, idxTile);
#ifndef __PTO_AUTO__
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
#endif
    TSTORE(outGlobal, dstTile);
    (void)dstBytes;
}

template <pto::GatherOOB Oob, typename T, typename TIdx, int kPadRows, int kPadCols, int64_t kRtValidRows,
          int64_t kRtValidCols, int64_t kRtTableR, int64_t kRtTableC>
inline AICORE void runElem2DDyn(__gm__ T __out__ *out, __gm__ T __in__ *table, __gm__ TIdx __in__ *indices)
{
    using TableShape = pto::Shape<1, 1, 1, -1, -1>;
    using TableStride = pto::Stride<1, 1, 1, -1, -1>;
    using IdxShape = pto::Shape<1, 1, 1, -1, -1>;
    using IdxStride = pto::Stride<1, 1, 1, -1, -1>;
    using OutShape = pto::Shape<1, 1, 1, -1, -1>;
    using OutStride = pto::Stride<1, 1, 1, -1, -1>;

    TableShape tableShape(kRtTableR, kRtTableC);
    TableStride tableStride(kRtTableC, (int64_t)1);
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

    constexpr int idxBytes = ((kPadRows * kPadCols * (int)sizeof(TIdx) + 31) / 32) * 32;
    constexpr int dstBytes = ((kPadRows * kPadCols * (int)sizeof(T) + 31) / 32) * 32;
    TASSIGN(idxTile, 0x0);
    TASSIGN(dstTile, idxBytes);

    TLOAD(idxTile, idxGlobal);
#ifndef __PTO_AUTO__
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
#endif
    MGATHER<Coalesce::Elem, Oob>(dstTile, tableGlobal, idxTile);
#ifndef __PTO_AUTO__
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
#endif
    TSTORE(outGlobal, dstTile);
    (void)dstBytes;
}

template <pto::GatherOOB Oob, typename T, typename TIdx, int kPadRows, int kPadCols, int kPadIdxCols,
          int64_t kRtValidRows, int64_t kRtValidCols, int64_t kRtTableR>
inline AICORE void runRowDyn(__gm__ T __out__ *out, __gm__ T __in__ *table, __gm__ TIdx __in__ *indices)
{
    using TableShape = pto::Shape<1, 1, 1, -1, -1>;
    using TableStride = pto::Stride<1, 1, 1, -1, -1>;
    using IdxShape = pto::Shape<1, 1, 1, 1, -1>;
    using IdxStride = pto::Stride<1, 1, 1, -1, -1>;
    using OutShape = pto::Shape<1, 1, 1, -1, -1>;
    using OutStride = pto::Stride<1, 1, 1, -1, -1>;

    TableShape tableShape(kRtTableR, kRtValidCols);
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

    constexpr int idxBytes = ((1 * kPadIdxCols * (int)sizeof(TIdx) + 31) / 32) * 32;
    constexpr int dstBytes = ((kPadRows * kPadCols * (int)sizeof(T) + 31) / 32) * 32;
    TASSIGN(idxTile, 0x0);
    TASSIGN(dstTile, idxBytes);

    TLOAD(idxTile, idxGlobal);
#ifndef __PTO_AUTO__
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
#endif
    MGATHER<Coalesce::Row, Oob>(dstTile, tableGlobal, idxTile);
#ifndef __PTO_AUTO__
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
#endif
    TSTORE(outGlobal, dstTile);
    (void)dstBytes;
}

template <pto::GatherOOB Oob, typename T, typename TIdx, int kDstRows, int kDstCols, int kTableSize>
inline AICORE void runElem2D(__gm__ T __out__ *out, __gm__ T __in__ *table, __gm__ TIdx __in__ *indices)
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

    constexpr int idxBytes = ((kDstRows * kDstCols * (int)sizeof(TIdx) + 31) / 32) * 32;
    constexpr int dstBytes = ((kDstRows * kDstCols * (int)sizeof(T) + 31) / 32) * 32;
    TASSIGN(idxTile, 0x0);
    TASSIGN(dstTile, idxBytes);

    TLOAD(idxTile, idxGlobal);
#ifndef __PTO_AUTO__
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
#endif
    MGATHER<Coalesce::Elem, Oob>(dstTile, tableGlobal, idxTile);
#ifndef __PTO_AUTO__
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
#endif
    TSTORE(outGlobal, dstTile);
    (void)dstBytes;
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

#define DEFINE_ROW_PAD(NAME, THOST, T, TIDX, VR, PIC, C, TR, OOB)                                                     \
    extern "C" __global__ AICORE void runMGATHER_##NAME(__gm__ T *out, __gm__ T *table, __gm__ TIDX *indices)         \
    {                                                                                                                 \
        runRowPadded<pto::GatherOOB::OOB, T, TIDX, VR, PIC, C, TR>(out, table, indices);                              \
    }                                                                                                                 \
    void Launch_##NAME(THOST *out, THOST *table, TIDX *indices, void *stream)                                         \
    {                                                                                                                 \
        runMGATHER_##NAME<<<1, nullptr, stream>>>(reinterpret_cast<T *>(out), reinterpret_cast<T *>(table), indices); \
    }

#define DEFINE_ROW_COLIDX(NAME, THOST, T, TIDX, R, C, TR, OOB)                                                        \
    extern "C" __global__ AICORE void runMGATHER_##NAME(__gm__ T *out, __gm__ T *table, __gm__ TIDX *indices)         \
    {                                                                                                                 \
        runRowColIdx<pto::GatherOOB::OOB, T, TIDX, R, C, TR>(out, table, indices);                                    \
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

#define DEFINE_ELEM2D_DYN(NAME, THOST, T, TIDX, PR, PC, RVR, RVC, RTR, RTC, OOB)                                      \
    extern "C" __global__ AICORE void runMGATHER_##NAME(__gm__ T *out, __gm__ T *table, __gm__ TIDX *indices)         \
    {                                                                                                                 \
        runElem2DDyn<pto::GatherOOB::OOB, T, TIDX, PR, PC, (int64_t)RVR, (int64_t)RVC, (int64_t)RTR, (int64_t)RTC>(   \
            out, table, indices);                                                                                     \
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

DEFINE_ROW(row_float_8x32_64rows, float, float, int32_t, 8, 32, 64, Undefined)
DEFINE_ROW(row_half_16x64_64rows, aclFloat16, half, int32_t, 16, 64, 64, Undefined)
DEFINE_ROW(row_int32_8x16_32rows, int32_t, int32_t, int32_t, 8, 16, 32, Undefined)
DEFINE_ROW(row_uint8_8x32_32rows, uint8_t, uint8_t, int32_t, 8, 32, 32, Undefined)
DEFINE_ROW(row_int16_8x16_32rows, int16_t, int16_t, int32_t, 8, 16, 32, Undefined)
DEFINE_ROW(row_float_clamp_8x32_8rows, float, float, int32_t, 8, 32, 8, Clamp)
DEFINE_ROW(row_int32_wrap_8x16_8rows, int32_t, int32_t, int32_t, 8, 16, 8, Wrap)
DEFINE_ROW(row_half_zero_8x32_8rows, aclFloat16, half, int32_t, 8, 32, 8, Zero)

DEFINE_ROW_COLIDX(row_colidx_float_8x32_64rows, float, float, int32_t, 8, 32, 64, Undefined)
DEFINE_ROW_COLIDX(row_colidx_int32_clamp_8x16_8rows, int32_t, int32_t, int32_t, 8, 16, 8, Clamp)
DEFINE_ROW_COLIDX(row_colidx_half_16x64_64rows, aclFloat16, half, int32_t, 16, 64, 64, Undefined)

DEFINE_ELEM(elem_float_64_128size, float, float, int32_t, 64, 128, Undefined)
DEFINE_ELEM(elem_half_64_128size, aclFloat16, half, int32_t, 64, 128, Undefined)
DEFINE_ELEM(elem_int32_32_64size, int32_t, int32_t, int32_t, 32, 64, Undefined)
DEFINE_ELEM(elem_uint8_64_128size, uint8_t, uint8_t, int32_t, 64, 128, Undefined)
DEFINE_ELEM(elem_int16_32_64size, int16_t, int16_t, int32_t, 32, 64, Undefined)
DEFINE_ELEM(elem_float_clamp_32_16size, float, float, int32_t, 32, 16, Clamp)
DEFINE_ELEM(elem_int32_wrap_32_16size, int32_t, int32_t, int32_t, 32, 16, Wrap)
DEFINE_ELEM(elem_half_zero_32_16size, aclFloat16, half, int32_t, 32, 16, Zero)

DEFINE_ELEM2D(elem2d_float_8x32_256size, float, float, int32_t, 8, 32, 256, Undefined)
DEFINE_ELEM2D(elem2d_int32_8x16_256size, int32_t, int32_t, int32_t, 8, 16, 256, Undefined)
DEFINE_ELEM2D(elem2d_half_4x32_256size, aclFloat16, half, int32_t, 4, 32, 256, Undefined)
DEFINE_ELEM2D(elem2d_int32_unaligned_3x8_64size, int32_t, int32_t, int32_t, 3, 8, 64, Undefined)
DEFINE_ELEM2D(elem2d_uint8_unaligned_3x32_256size, uint8_t, uint8_t, int32_t, 3, 32, 256, Undefined)
DEFINE_ELEM2D_PAD(elem2d_int32_unaligned_3x3_in_3x8_64size, int32_t, int32_t, int32_t, 3, 3, 3, 8, 64, Undefined)
DEFINE_ELEM2D_PAD(elem2d_int32_unaligned_9x9_in_9x16_256size, int32_t, int32_t, int32_t, 9, 9, 9, 16, 256, Undefined)
DEFINE_ELEM2D_PAD(elem2d_int32_scalar_1x1_in_1x8_8size, int32_t, int32_t, int32_t, 1, 1, 1, 8, 8, Undefined)
DEFINE_ROW_PAD(row_int32_unaligned_3x8_8rows, int32_t, int32_t, int32_t, 3, 8, 8, 8, Undefined)
DEFINE_ROW_PAD(row_int32_unaligned_9x16_16rows, int32_t, int32_t, int32_t, 9, 16, 16, 16, Undefined)

DEFINE_ELEM2D_DYN(elem2d_dyn_user_float_1x9_in_1x16_3x10, float, float, int32_t, 1, 16, 1, 9, 3, 10, Undefined)
DEFINE_ELEM2D_DYN(elem2d_dyn_int32_4x8_in_4x8_64size, int32_t, int32_t, int32_t, 4, 8, 4, 8, 1, 64, Undefined)
DEFINE_ELEM2D_DYN(elem2d_dyn_float_3x3_in_3x8_64size, float, float, int32_t, 3, 8, 3, 3, 1, 64, Undefined)
DEFINE_ELEM2D_DYN(elem2d_dyn_half_8x16_in_8x16_4x32, aclFloat16, half, int32_t, 8, 16, 8, 16, 4, 32, Undefined)
DEFINE_ROW_DYN(row_dyn_int32_3x16_8rows, int32_t, int32_t, int32_t, 3, 16, 8, 3, 16, 8, Undefined)
DEFINE_ROW_DYN(row_dyn_half_4x32_16rows, aclFloat16, half, int32_t, 4, 32, 8, 4, 32, 16, Undefined)
