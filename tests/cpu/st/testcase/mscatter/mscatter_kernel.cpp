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

template <pto::ScatterAtomicOp Atomic, pto::ScatterOOB Oob, pto::ScatterConflict Conflict, typename T, typename TIdx,
          int kSrcRows, int kSrcCols, int kTableRows>
inline AICORE void runRow(__gm__ T __out__ *out, __gm__ T __in__ *src, __gm__ TIdx __in__ *indices)
{
    using SrcShape = pto::Shape<1, 1, 1, kSrcRows, kSrcCols>;
    using SrcStride = pto::Stride<1, 1, 1, kSrcCols, 1>;
    using IdxShape = pto::Shape<1, 1, 1, 1, kSrcRows>;
    using IdxStride = pto::Stride<1, 1, 1, kSrcRows, 1>;
    using OutShape = pto::Shape<1, 1, 1, kTableRows, kSrcCols>;
    using OutStride = pto::Stride<1, 1, 1, kSrcCols, 1>;

    GlobalTensor<T, SrcShape, SrcStride> srcGlobal(src);
    GlobalTensor<TIdx, IdxShape, IdxStride> idxGlobal(indices);
    GlobalTensor<T, OutShape, OutStride> outGlobal(out);

    using SrcTile = Tile<TileType::Vec, T, kSrcRows, kSrcCols, BLayout::RowMajor, kSrcRows, kSrcCols>;
    using IdxTile = Tile<TileType::Vec, TIdx, 1, kSrcRows, BLayout::RowMajor, 1, kSrcRows>;

    SrcTile srcTile;
    IdxTile idxTile;

    constexpr int idxBytes = ((1 * kSrcRows * (int)sizeof(TIdx) + 31) / 32) * 32;
    constexpr int srcBytes = ((kSrcRows * kSrcCols * (int)sizeof(T) + 31) / 32) * 32;
    TASSIGN(idxTile, 0x0);
    TASSIGN(srcTile, idxBytes);

    TLOAD(idxTile, idxGlobal);
    TLOAD(srcTile, srcGlobal);
#ifndef __PTO_AUTO__
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
#endif
    MSCATTER<Coalesce::Row, Atomic, Oob, Conflict>(outGlobal, srcTile, idxTile);
#ifndef __PTO_AUTO__
    pipe_barrier(PIPE_ALL);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
#endif
}

template <pto::ScatterAtomicOp Atomic, pto::ScatterOOB Oob, pto::ScatterConflict Conflict, typename T, typename TIdx,
          int kValidRows, int kPadIdxCols, int kSrcCols, int kTableRows>
inline AICORE void runRowPadded(__gm__ T __out__ *out, __gm__ T __in__ *src, __gm__ TIdx __in__ *indices)
{
    using SrcShape = pto::Shape<1, 1, 1, kValidRows, kSrcCols>;
    using SrcStride = pto::Stride<1, 1, 1, kSrcCols, 1>;
    using IdxShape = pto::Shape<1, 1, 1, 1, kValidRows>;
    using IdxStride = pto::Stride<1, 1, 1, kValidRows, 1>;
    using OutShape = pto::Shape<1, 1, 1, kTableRows, kSrcCols>;
    using OutStride = pto::Stride<1, 1, 1, kSrcCols, 1>;

    GlobalTensor<T, SrcShape, SrcStride> srcGlobal(src);
    GlobalTensor<TIdx, IdxShape, IdxStride> idxGlobal(indices);
    GlobalTensor<T, OutShape, OutStride> outGlobal(out);

    using SrcTile = Tile<TileType::Vec, T, kValidRows, kSrcCols, BLayout::RowMajor, kValidRows, kSrcCols>;
    using IdxTile = Tile<TileType::Vec, TIdx, 1, kPadIdxCols, BLayout::RowMajor, 1, kValidRows>;

    SrcTile srcTile;
    IdxTile idxTile;

    constexpr int idxBytes = ((1 * kPadIdxCols * (int)sizeof(TIdx) + 31) / 32) * 32;
    constexpr int srcBytes = ((kValidRows * kSrcCols * (int)sizeof(T) + 31) / 32) * 32;
    TASSIGN(idxTile, 0x0);
    TASSIGN(srcTile, idxBytes);

    TLOAD(idxTile, idxGlobal);
    TLOAD(srcTile, srcGlobal);
#ifndef __PTO_AUTO__
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
#endif
    MSCATTER<Coalesce::Row, Atomic, Oob, Conflict>(outGlobal, srcTile, idxTile);
#ifndef __PTO_AUTO__
    pipe_barrier(PIPE_ALL);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
#endif
    (void)srcBytes;
}

template <pto::ScatterAtomicOp Atomic, pto::ScatterOOB Oob, pto::ScatterConflict Conflict, typename T, typename TIdx,
          int kSrcRows, int kSrcCols, int kTableRows>
inline AICORE void runRowColIdx(__gm__ T __out__ *out, __gm__ T __in__ *src, __gm__ TIdx __in__ *indices)
{
    using SrcShape = pto::Shape<1, 1, 1, kSrcRows, kSrcCols>;
    using SrcStride = pto::Stride<1, 1, 1, kSrcCols, 1>;
    using IdxShape = pto::Shape<1, 1, 1, kSrcRows, 1>;
    using IdxStride = pto::Stride<1, 1, 1, 1, 1>;
    using OutShape = pto::Shape<1, 1, 1, kTableRows, kSrcCols>;
    using OutStride = pto::Stride<1, 1, 1, kSrcCols, 1>;

    GlobalTensor<T, SrcShape, SrcStride> srcGlobal(src);
    GlobalTensor<TIdx, IdxShape, IdxStride, Layout::DN> idxGlobal(indices);
    GlobalTensor<T, OutShape, OutStride> outGlobal(out);

    using SrcTile = Tile<TileType::Vec, T, kSrcRows, kSrcCols, BLayout::RowMajor, kSrcRows, kSrcCols>;
    using IdxTile = Tile<TileType::Vec, TIdx, kSrcRows, 1, BLayout::ColMajor, kSrcRows, 1>;

    SrcTile srcTile;
    IdxTile idxTile;

    constexpr int idxBytes = ((kSrcRows * 1 * (int)sizeof(TIdx) + 31) / 32) * 32;
    constexpr int srcBytes = ((kSrcRows * kSrcCols * (int)sizeof(T) + 31) / 32) * 32;
    TASSIGN(idxTile, 0x0);
    TASSIGN(srcTile, idxBytes);

    TLOAD(idxTile, idxGlobal);
    TLOAD(srcTile, srcGlobal);
#ifndef __PTO_AUTO__
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
#endif
    MSCATTER<Coalesce::Row, Atomic, Oob, Conflict>(outGlobal, srcTile, idxTile);
#ifndef __PTO_AUTO__
    pipe_barrier(PIPE_ALL);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
#endif
}

template <pto::ScatterAtomicOp Atomic, pto::ScatterOOB Oob, pto::ScatterConflict Conflict, typename T, typename TIdx,
          int kSrcCols, int kTableSize>
inline AICORE void runElem(__gm__ T __out__ *out, __gm__ T __in__ *src, __gm__ TIdx __in__ *indices)
{
    using SrcShape = pto::Shape<1, 1, 1, 1, kSrcCols>;
    using SrcStride = pto::Stride<1, 1, 1, kSrcCols, 1>;
    using IdxShape = pto::Shape<1, 1, 1, 1, kSrcCols>;
    using IdxStride = pto::Stride<1, 1, 1, kSrcCols, 1>;
    using OutShape = pto::Shape<1, 1, 1, 1, kTableSize>;
    using OutStride = pto::Stride<1, 1, 1, kTableSize, 1>;

    GlobalTensor<T, SrcShape, SrcStride> srcGlobal(src);
    GlobalTensor<TIdx, IdxShape, IdxStride> idxGlobal(indices);
    GlobalTensor<T, OutShape, OutStride> outGlobal(out);

    using SrcTile = Tile<TileType::Vec, T, 1, kSrcCols, BLayout::RowMajor, 1, kSrcCols>;
    using IdxTile = Tile<TileType::Vec, TIdx, 1, kSrcCols, BLayout::RowMajor, 1, kSrcCols>;

    SrcTile srcTile;
    IdxTile idxTile;

    constexpr int idxBytes = ((1 * kSrcCols * (int)sizeof(TIdx) + 31) / 32) * 32;
    constexpr int srcBytes = ((1 * kSrcCols * (int)sizeof(T) + 31) / 32) * 32;
    TASSIGN(idxTile, 0x0);
    TASSIGN(srcTile, idxBytes);

    TLOAD(idxTile, idxGlobal);
    TLOAD(srcTile, srcGlobal);
#ifndef __PTO_AUTO__
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
#endif
    MSCATTER<Coalesce::Elem, Atomic, Oob, Conflict>(outGlobal, srcTile, idxTile);
#ifndef __PTO_AUTO__
    pipe_barrier(PIPE_ALL);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
#endif
}

template <pto::ScatterAtomicOp Atomic, pto::ScatterOOB Oob, pto::ScatterConflict Conflict, typename T, typename TIdx,
          int kValidRows, int kValidCols, int kPadRows, int kPadCols, int kTableSize>
inline AICORE void runElem2DPadded(__gm__ T __out__ *out, __gm__ T __in__ *src, __gm__ TIdx __in__ *indices)
{
    using SrcShape = pto::Shape<1, 1, 1, kValidRows, kValidCols>;
    using SrcStride = pto::Stride<1, 1, 1, kValidCols, 1>;
    using IdxShape = pto::Shape<1, 1, 1, kValidRows, kValidCols>;
    using IdxStride = pto::Stride<1, 1, 1, kValidCols, 1>;
    using OutShape = pto::Shape<1, 1, 1, 1, kTableSize>;
    using OutStride = pto::Stride<1, 1, 1, kTableSize, 1>;

    GlobalTensor<T, SrcShape, SrcStride> srcGlobal(src);
    GlobalTensor<TIdx, IdxShape, IdxStride> idxGlobal(indices);
    GlobalTensor<T, OutShape, OutStride> outGlobal(out);

    using SrcTile = Tile<TileType::Vec, T, kPadRows, kPadCols, BLayout::RowMajor, kValidRows, kValidCols>;
    using IdxTile = Tile<TileType::Vec, TIdx, kPadRows, kPadCols, BLayout::RowMajor, kValidRows, kValidCols>;

    SrcTile srcTile;
    IdxTile idxTile;

    constexpr int idxBytes = ((kPadRows * kPadCols * (int)sizeof(TIdx) + 31) / 32) * 32;
    constexpr int srcBytes = ((kPadRows * kPadCols * (int)sizeof(T) + 31) / 32) * 32;
    TASSIGN(idxTile, 0x0);
    TASSIGN(srcTile, idxBytes);

    TLOAD(idxTile, idxGlobal);
    TLOAD(srcTile, srcGlobal);
#ifndef __PTO_AUTO__
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
#endif
    MSCATTER<Coalesce::Elem, Atomic, Oob, Conflict>(outGlobal, srcTile, idxTile);
#ifndef __PTO_AUTO__
    pipe_barrier(PIPE_ALL);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
#endif
    (void)srcBytes;
}

template <pto::ScatterAtomicOp Atomic, pto::ScatterOOB Oob, pto::ScatterConflict Conflict, typename T, typename TIdx,
          int kPadRows, int kPadCols, int64_t kRtValidRows, int64_t kRtValidCols, int64_t kRtTableR, int64_t kRtTableC>
inline AICORE void runElem2DDyn(__gm__ T __out__ *out, __gm__ T __in__ *src, __gm__ TIdx __in__ *indices)
{
    using SrcShape = pto::Shape<1, 1, 1, -1, -1>;
    using SrcStride = pto::Stride<1, 1, 1, -1, -1>;
    using IdxShape = pto::Shape<1, 1, 1, -1, -1>;
    using IdxStride = pto::Stride<1, 1, 1, -1, -1>;
    using OutShape = pto::Shape<1, 1, 1, -1, -1>;
    using OutStride = pto::Stride<1, 1, 1, -1, -1>;

    SrcShape srcShape(kRtValidRows, kRtValidCols);
    SrcStride srcStride(kRtValidCols, (int64_t)1);
    IdxShape idxShape(kRtValidRows, kRtValidCols);
    IdxStride idxStride(kRtValidCols, (int64_t)1);
    OutShape outShape(kRtTableR, kRtTableC);
    OutStride outStride(kRtTableC, (int64_t)1);

    GlobalTensor<T, SrcShape, SrcStride> srcGlobal(src, srcShape, srcStride);
    GlobalTensor<TIdx, IdxShape, IdxStride> idxGlobal(indices, idxShape, idxStride);
    GlobalTensor<T, OutShape, OutStride> outGlobal(out, outShape, outStride);

    using SrcTile = Tile<TileType::Vec, T, kPadRows, kPadCols, BLayout::RowMajor, -1, -1>;
    using IdxTile = Tile<TileType::Vec, TIdx, kPadRows, kPadCols, BLayout::RowMajor, -1, -1>;

    SrcTile srcTile(static_cast<unsigned>(kRtValidRows), static_cast<unsigned>(kRtValidCols));
    IdxTile idxTile(static_cast<unsigned>(kRtValidRows), static_cast<unsigned>(kRtValidCols));

    constexpr int idxBytes = ((kPadRows * kPadCols * (int)sizeof(TIdx) + 31) / 32) * 32;
    constexpr int srcBytes = ((kPadRows * kPadCols * (int)sizeof(T) + 31) / 32) * 32;
    TASSIGN(idxTile, 0x0);
    TASSIGN(srcTile, idxBytes);

    TLOAD(idxTile, idxGlobal);
    TLOAD(srcTile, srcGlobal);
#ifndef __PTO_AUTO__
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
#endif
    MSCATTER<Coalesce::Elem, Atomic, Oob, Conflict>(outGlobal, srcTile, idxTile);
#ifndef __PTO_AUTO__
    pipe_barrier(PIPE_ALL);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
#endif
    (void)srcBytes;
}

template <pto::ScatterAtomicOp Atomic, pto::ScatterOOB Oob, pto::ScatterConflict Conflict, typename T, typename TIdx,
          int kPadRows, int kPadCols, int kPadIdxCols, int64_t kRtValidRows, int64_t kRtValidCols, int64_t kRtTableR>
inline AICORE void runRowDyn(__gm__ T __out__ *out, __gm__ T __in__ *src, __gm__ TIdx __in__ *indices)
{
    using SrcShape = pto::Shape<1, 1, 1, -1, -1>;
    using SrcStride = pto::Stride<1, 1, 1, -1, -1>;
    using IdxShape = pto::Shape<1, 1, 1, 1, -1>;
    using IdxStride = pto::Stride<1, 1, 1, -1, -1>;
    using OutShape = pto::Shape<1, 1, 1, -1, -1>;
    using OutStride = pto::Stride<1, 1, 1, -1, -1>;

    SrcShape srcShape(kRtValidRows, kRtValidCols);
    SrcStride srcStride(kRtValidCols, (int64_t)1);
    IdxShape idxShape(kRtValidRows);
    IdxStride idxStride(kRtValidRows, (int64_t)1);
    OutShape outShape(kRtTableR, kRtValidCols);
    OutStride outStride(kRtValidCols, (int64_t)1);

    GlobalTensor<T, SrcShape, SrcStride> srcGlobal(src, srcShape, srcStride);
    GlobalTensor<TIdx, IdxShape, IdxStride> idxGlobal(indices, idxShape, idxStride);
    GlobalTensor<T, OutShape, OutStride> outGlobal(out, outShape, outStride);

    using SrcTile = Tile<TileType::Vec, T, kPadRows, kPadCols, BLayout::RowMajor, -1, -1>;
    using IdxTile = Tile<TileType::Vec, TIdx, 1, kPadIdxCols, BLayout::RowMajor, 1, -1>;

    SrcTile srcTile(static_cast<unsigned>(kRtValidRows), static_cast<unsigned>(kRtValidCols));
    IdxTile idxTile(static_cast<unsigned>(kRtValidRows));

    constexpr int idxBytes = ((1 * kPadIdxCols * (int)sizeof(TIdx) + 31) / 32) * 32;
    constexpr int srcBytes = ((kPadRows * kPadCols * (int)sizeof(T) + 31) / 32) * 32;
    TASSIGN(idxTile, 0x0);
    TASSIGN(srcTile, idxBytes);

    TLOAD(idxTile, idxGlobal);
    TLOAD(srcTile, srcGlobal);
#ifndef __PTO_AUTO__
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
#endif
    MSCATTER<Coalesce::Row, Atomic, Oob, Conflict>(outGlobal, srcTile, idxTile);
#ifndef __PTO_AUTO__
    pipe_barrier(PIPE_ALL);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
#endif
    (void)srcBytes;
}

template <pto::ScatterAtomicOp Atomic, pto::ScatterOOB Oob, pto::ScatterConflict Conflict, typename T, typename TIdx,
          int kSrcRows, int kSrcCols, int kTableSize>
inline AICORE void runElem2D(__gm__ T __out__ *out, __gm__ T __in__ *src, __gm__ TIdx __in__ *indices)
{
    using SrcShape = pto::Shape<1, 1, 1, kSrcRows, kSrcCols>;
    using SrcStride = pto::Stride<1, 1, 1, kSrcCols, 1>;
    using IdxShape = pto::Shape<1, 1, 1, kSrcRows, kSrcCols>;
    using IdxStride = pto::Stride<1, 1, 1, kSrcCols, 1>;
    using OutShape = pto::Shape<1, 1, 1, 1, kTableSize>;
    using OutStride = pto::Stride<1, 1, 1, kTableSize, 1>;

    GlobalTensor<T, SrcShape, SrcStride> srcGlobal(src);
    GlobalTensor<TIdx, IdxShape, IdxStride> idxGlobal(indices);
    GlobalTensor<T, OutShape, OutStride> outGlobal(out);

    using SrcTile = Tile<TileType::Vec, T, kSrcRows, kSrcCols, BLayout::RowMajor, kSrcRows, kSrcCols>;
    using IdxTile = Tile<TileType::Vec, TIdx, kSrcRows, kSrcCols, BLayout::RowMajor, kSrcRows, kSrcCols>;

    SrcTile srcTile;
    IdxTile idxTile;

    constexpr int idxBytes = ((kSrcRows * kSrcCols * (int)sizeof(TIdx) + 31) / 32) * 32;
    constexpr int srcBytes = ((kSrcRows * kSrcCols * (int)sizeof(T) + 31) / 32) * 32;
    TASSIGN(idxTile, 0x0);
    TASSIGN(srcTile, idxBytes);

    TLOAD(idxTile, idxGlobal);
    TLOAD(srcTile, srcGlobal);
#ifndef __PTO_AUTO__
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
#endif
    MSCATTER<Coalesce::Elem, Atomic, Oob, Conflict>(outGlobal, srcTile, idxTile);
#ifndef __PTO_AUTO__
    pipe_barrier(PIPE_ALL);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
#endif
}

template <pto::ScatterAtomicOp Atomic, pto::ScatterOOB Oob, pto::ScatterConflict Conflict, typename T, typename TIdx,
          int kTotalRows, int kSrcCols, int kChunkRows, int kTableSize>
inline AICORE void runElem2DChunked(__gm__ T __out__ *out, __gm__ T __in__ *src, __gm__ TIdx __in__ *indices)
{
    static_assert(kTotalRows % kChunkRows == 0, "kTotalRows must be a whole multiple of kChunkRows.");

    using ChunkSrcShape = pto::Shape<1, 1, 1, kChunkRows, kSrcCols>;
    using ChunkSrcStride = pto::Stride<1, 1, 1, kSrcCols, 1>;
    using ChunkIdxShape = pto::Shape<1, 1, 1, kChunkRows, kSrcCols>;
    using ChunkIdxStride = pto::Stride<1, 1, 1, kSrcCols, 1>;
    using OutShape = pto::Shape<1, 1, 1, 1, kTableSize>;
    using OutStride = pto::Stride<1, 1, 1, kTableSize, 1>;

    GlobalTensor<T, OutShape, OutStride> outGlobal(out);

    using SrcTile = Tile<TileType::Vec, T, kChunkRows, kSrcCols, BLayout::RowMajor, kChunkRows, kSrcCols>;
    using IdxTile = Tile<TileType::Vec, TIdx, kChunkRows, kSrcCols, BLayout::RowMajor, kChunkRows, kSrcCols>;

    SrcTile srcTile;
    IdxTile idxTile;

    constexpr int idxBytes = ((kChunkRows * kSrcCols * (int)sizeof(TIdx) + 31) / 32) * 32;
    constexpr int srcBytes = ((kChunkRows * kSrcCols * (int)sizeof(T) + 31) / 32) * 32;
    TASSIGN(idxTile, 0x0);
    TASSIGN(srcTile, idxBytes);

    constexpr int kChunkElems = kChunkRows * kSrcCols;
    constexpr int kNumChunks = kTotalRows / kChunkRows;

    for (int chunk = 0; chunk < kNumChunks; ++chunk) {
        GlobalTensor<T, ChunkSrcShape, ChunkSrcStride> srcGlobal(src + chunk * kChunkElems);
        GlobalTensor<TIdx, ChunkIdxShape, ChunkIdxStride> idxGlobal(indices + chunk * kChunkElems);

        TLOAD(idxTile, idxGlobal);
        TLOAD(srcTile, srcGlobal);
#ifndef __PTO_AUTO__
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
#endif
        MSCATTER<Coalesce::Elem, Atomic, Oob, Conflict>(outGlobal, srcTile, idxTile);
#ifndef __PTO_AUTO__
        pipe_barrier(PIPE_ALL);
        set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
#endif
    }
#ifndef __PTO_AUTO__
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
#endif
    (void)srcBytes;
}

#define DEFINE_ROW(NAME, THOST, T, TIDX, R, C, TR, ATOMIC, OOB, CONFLICT)                                              \
    extern "C" __global__ AICORE void runMSCATTER_##NAME(__gm__ T *out, __gm__ T *src, __gm__ TIDX *indices)           \
    {                                                                                                                  \
        runRow<pto::ScatterAtomicOp::ATOMIC, pto::ScatterOOB::OOB, pto::ScatterConflict::CONFLICT, T, TIDX, R, C, TR>( \
            out, src, indices);                                                                                        \
    }                                                                                                                  \
    void Launch_##NAME(THOST *out, THOST *src, TIDX *indices, void *stream)                                            \
    {                                                                                                                  \
        runMSCATTER_##NAME(reinterpret_cast<T *>(out), reinterpret_cast<T *>(src), indices);                           \
    }

#define DEFINE_ROW_PAD(NAME, THOST, T, TIDX, VR, PIC, C, TR, ATOMIC, OOB, CONFLICT)                                   \
    extern "C" __global__ AICORE void runMSCATTER_##NAME(__gm__ T *out, __gm__ T *src, __gm__ TIDX *indices)          \
    {                                                                                                                 \
        runRowPadded<pto::ScatterAtomicOp::ATOMIC, pto::ScatterOOB::OOB, pto::ScatterConflict::CONFLICT, T, TIDX, VR, \
                     PIC, C, TR>(out, src, indices);                                                                  \
    }                                                                                                                 \
    void Launch_##NAME(THOST *out, THOST *src, TIDX *indices, void *stream)                                           \
    {                                                                                                                 \
        runMSCATTER_##NAME(reinterpret_cast<T *>(out), reinterpret_cast<T *>(src), indices);                          \
    }

#define DEFINE_ROW_COLIDX(NAME, THOST, T, TIDX, R, C, TR, ATOMIC, OOB, CONFLICT)                                     \
    extern "C" __global__ AICORE void runMSCATTER_##NAME(__gm__ T *out, __gm__ T *src, __gm__ TIDX *indices)         \
    {                                                                                                                \
        runRowColIdx<pto::ScatterAtomicOp::ATOMIC, pto::ScatterOOB::OOB, pto::ScatterConflict::CONFLICT, T, TIDX, R, \
                     C, TR>(out, src, indices);                                                                      \
    }                                                                                                                \
    void Launch_##NAME(THOST *out, THOST *src, TIDX *indices, void *stream)                                          \
    {                                                                                                                \
        runMSCATTER_##NAME(reinterpret_cast<T *>(out), reinterpret_cast<T *>(src), indices);                         \
    }

#define DEFINE_ELEM(NAME, THOST, T, TIDX, N, TS, ATOMIC, OOB, CONFLICT)                                              \
    extern "C" __global__ AICORE void runMSCATTER_##NAME(__gm__ T *out, __gm__ T *src, __gm__ TIDX *indices)         \
    {                                                                                                                \
        runElem<pto::ScatterAtomicOp::ATOMIC, pto::ScatterOOB::OOB, pto::ScatterConflict::CONFLICT, T, TIDX, N, TS>( \
            out, src, indices);                                                                                      \
    }                                                                                                                \
    void Launch_##NAME(THOST *out, THOST *src, TIDX *indices, void *stream)                                          \
    {                                                                                                                \
        runMSCATTER_##NAME(reinterpret_cast<T *>(out), reinterpret_cast<T *>(src), indices);                         \
    }

#define DEFINE_ELEM2D(NAME, THOST, T, TIDX, R, C, TS, ATOMIC, OOB, CONFLICT)                                         \
    extern "C" __global__ AICORE void runMSCATTER_##NAME(__gm__ T *out, __gm__ T *src, __gm__ TIDX *indices)         \
    {                                                                                                                \
        runElem2D<pto::ScatterAtomicOp::ATOMIC, pto::ScatterOOB::OOB, pto::ScatterConflict::CONFLICT, T, TIDX, R, C, \
                  TS>(out, src, indices);                                                                            \
    }                                                                                                                \
    void Launch_##NAME(THOST *out, THOST *src, TIDX *indices, void *stream)                                          \
    {                                                                                                                \
        runMSCATTER_##NAME(reinterpret_cast<T *>(out), reinterpret_cast<T *>(src), indices);                         \
    }

#define DEFINE_ELEM2D_PAD(NAME, THOST, T, TIDX, VR, VC, PR, PC, TS, ATOMIC, OOB, CONFLICT)                           \
    extern "C" __global__ AICORE void runMSCATTER_##NAME(__gm__ T *out, __gm__ T *src, __gm__ TIDX *indices)         \
    {                                                                                                                \
        runElem2DPadded<pto::ScatterAtomicOp::ATOMIC, pto::ScatterOOB::OOB, pto::ScatterConflict::CONFLICT, T, TIDX, \
                        VR, VC, PR, PC, TS>(out, src, indices);                                                      \
    }                                                                                                                \
    void Launch_##NAME(THOST *out, THOST *src, TIDX *indices, void *stream)                                          \
    {                                                                                                                \
        runMSCATTER_##NAME(reinterpret_cast<T *>(out), reinterpret_cast<T *>(src), indices);                         \
    }

#define DEFINE_ELEM2D_DYN(NAME, THOST, T, TIDX, PR, PC, RVR, RVC, RTR, RTC, ATOMIC, OOB, CONFLICT)                    \
    extern "C" __global__ AICORE void runMSCATTER_##NAME(__gm__ T *out, __gm__ T *src, __gm__ TIDX *indices)          \
    {                                                                                                                 \
        runElem2DDyn<pto::ScatterAtomicOp::ATOMIC, pto::ScatterOOB::OOB, pto::ScatterConflict::CONFLICT, T, TIDX, PR, \
                     PC, (int64_t)RVR, (int64_t)RVC, (int64_t)RTR, (int64_t)RTC>(out, src, indices);                  \
    }                                                                                                                 \
    void Launch_##NAME(THOST *out, THOST *src, TIDX *indices, void *stream)                                           \
    {                                                                                                                 \
        runMSCATTER_##NAME(reinterpret_cast<T *>(out), reinterpret_cast<T *>(src), indices);                          \
    }

#define DEFINE_ROW_DYN(NAME, THOST, T, TIDX, PR, PC, PIC, RVR, RVC, RTR, ATOMIC, OOB, CONFLICT)                        \
    extern "C" __global__ AICORE void runMSCATTER_##NAME(__gm__ T *out, __gm__ T *src, __gm__ TIDX *indices)           \
    {                                                                                                                  \
        runRowDyn<pto::ScatterAtomicOp::ATOMIC, pto::ScatterOOB::OOB, pto::ScatterConflict::CONFLICT, T, TIDX, PR, PC, \
                  PIC, (int64_t)RVR, (int64_t)RVC, (int64_t)RTR>(out, src, indices);                                   \
    }                                                                                                                  \
    void Launch_##NAME(THOST *out, THOST *src, TIDX *indices, void *stream)                                            \
    {                                                                                                                  \
        runMSCATTER_##NAME(reinterpret_cast<T *>(out), reinterpret_cast<T *>(src), indices);                           \
    }

#define DEFINE_ELEM2D_CHUNKED(NAME, THOST, T, TIDX, R, C, CR, TS, ATOMIC, OOB, CONFLICT)                              \
    extern "C" __global__ AICORE void runMSCATTER_##NAME(__gm__ T *out, __gm__ T *src, __gm__ TIDX *indices)          \
    {                                                                                                                 \
        runElem2DChunked<pto::ScatterAtomicOp::ATOMIC, pto::ScatterOOB::OOB, pto::ScatterConflict::CONFLICT, T, TIDX, \
                         R, C, CR, TS>(out, src, indices);                                                            \
    }                                                                                                                 \
    void Launch_##NAME(THOST *out, THOST *src, TIDX *indices, void *stream)                                           \
    {                                                                                                                 \
        runMSCATTER_##NAME(reinterpret_cast<T *>(out), reinterpret_cast<T *>(src), indices);                          \
    }

#define DEFINE_ELEM2D_DYN_UB(NAME, THOST, T, TIDX, R, C, TS, DYN_UB_BYTES, ATOMIC, OOB, CONFLICT)                    \
    extern "C" __global__ AICORE void runMSCATTER_##NAME(__gm__ T *out, __gm__ T *src, __gm__ TIDX *indices)         \
    {                                                                                                                \
        runElem2D<pto::ScatterAtomicOp::ATOMIC, pto::ScatterOOB::OOB, pto::ScatterConflict::CONFLICT, T, TIDX, R, C, \
                  TS>(out, src, indices);                                                                            \
    }                                                                                                                \
    void Launch_##NAME(THOST *out, THOST *src, TIDX *indices, void *stream)                                          \
    {                                                                                                                \
        runMSCATTER_##NAME(reinterpret_cast<T *>(out), reinterpret_cast<T *>(src), indices);                         \
    }

DEFINE_ROW(row_float_random_8x32_64rows, float, float, int32_t, 8, 32, 64, None, Undefined, Last)
DEFINE_ROW(row_float_same_8x32_16rows, float, float, int32_t, 8, 32, 16, None, Undefined, Last)
DEFINE_ROW(row_half_random_16x64_64rows, aclFloat16, half, int32_t, 16, 64, 64, None, Undefined, Last)
DEFINE_ROW(row_int32_random_8x16_32rows, int32_t, int32_t, int32_t, 8, 16, 32, None, Undefined, Last)
DEFINE_ROW(row_uint8_random_8x32_32rows, uint8_t, uint8_t, int32_t, 8, 32, 32, None, Undefined, Last)
DEFINE_ROW(row_int16_random_8x16_32rows, int16_t, int16_t, int32_t, 8, 16, 32, None, Undefined, Last)
DEFINE_ROW(row_float_atomicadd_8x32_8rows, float, float, int32_t, 8, 32, 8, Add, Undefined, Default)
DEFINE_ROW(row_float_skip_8x32_8rows, float, float, int32_t, 8, 32, 8, None, Skip, Last)
DEFINE_ROW(row_int32_clamp_8x16_8rows, int32_t, int32_t, int32_t, 8, 16, 8, None, Clamp, Last)
DEFINE_ROW(row_half_wrap_8x32_8rows, aclFloat16, half, int32_t, 8, 32, 8, None, Wrap, Last)

DEFINE_ROW_COLIDX(row_colidx_float_random_8x32_64rows, float, float, int32_t, 8, 32, 64, None, Undefined, Last)
DEFINE_ROW_COLIDX(row_colidx_int32_clamp_8x16_8rows, int32_t, int32_t, int32_t, 8, 16, 8, None, Clamp, Last)
DEFINE_ROW_COLIDX(row_colidx_half_random_16x64_64rows, aclFloat16, half, int32_t, 16, 64, 64, None, Undefined, Last)

DEFINE_ELEM(elem_float_random_64_128size, float, float, int32_t, 64, 128, None, Undefined, Last)
DEFINE_ELEM(elem_float_same_64_8size, float, float, int32_t, 64, 8, None, Undefined, Last)
DEFINE_ELEM(elem_float_seq_32_32size, float, float, int32_t, 32, 32, None, Undefined, Last)
DEFINE_ELEM(elem_half_random_64_128size, aclFloat16, half, int32_t, 64, 128, None, Undefined, Last)
DEFINE_ELEM(elem_int32_random_32_64size, int32_t, int32_t, int32_t, 32, 64, None, Undefined, Last)
DEFINE_ELEM(elem_uint8_random_64_128size, uint8_t, uint8_t, int32_t, 64, 128, None, Undefined, Last)
DEFINE_ELEM(elem_int16_random_32_64size, int16_t, int16_t, int32_t, 32, 64, None, Undefined, Last)
DEFINE_ELEM(elem_float_atomicadd_32_32size, float, float, int32_t, 32, 32, Add, Undefined, Default)
DEFINE_ELEM(elem_int32_atomicadd_skip_32_16size, int32_t, int32_t, int32_t, 32, 16, Add, Skip, Default)
DEFINE_ELEM(elem_float_skip_32_16size, float, float, int32_t, 32, 16, None, Skip, Last)
DEFINE_ELEM(elem_int32_clamp_32_16size, int32_t, int32_t, int32_t, 32, 16, None, Clamp, Last)
DEFINE_ELEM(elem_half_wrap_32_16size, aclFloat16, half, int32_t, 32, 16, None, Wrap, Last)
DEFINE_ELEM(elem_float_default_seq_32_32size, float, float, int32_t, 32, 32, None, Undefined, Default)
DEFINE_ELEM(elem_float_small_16_32size, float, float, int32_t, 16, 32, None, Undefined, Last)
DEFINE_ELEM(elem_int32_atomicmax_random_32_32size, int32_t, int32_t, int32_t, 32, 32, Max, Undefined, Default)
DEFINE_ELEM(elem_float_atomicmin_random_32_32size, float, float, int32_t, 32, 32, Min, Undefined, Default)
DEFINE_ELEM(elem_float_last_same_32_8size, float, float, int32_t, 32, 8, None, Undefined, Last)
DEFINE_ELEM(elem_int32_last_seq_32_32size, int32_t, int32_t, int32_t, 32, 32, None, Undefined, Last)
DEFINE_ELEM(elem_float_clamp_no_dup_32_16size, float, float, int32_t, 32, 16, None, Clamp, Last)
DEFINE_ELEM(elem_uint8_wrap_64_16size, uint8_t, uint8_t, int32_t, 64, 16, None, Wrap, Last)
DEFINE_ELEM(elem_int16_clamp_32_16size, int16_t, int16_t, int32_t, 32, 16, None, Clamp, Last)

DEFINE_ELEM2D(elem2d_float_8x32_random_256size, float, float, int32_t, 8, 32, 256, None, Undefined, Last)
DEFINE_ELEM2D(elem2d_int32_8x16_random_256size, int32_t, int32_t, int32_t, 8, 16, 256, None, Undefined, Last)
DEFINE_ELEM2D(elem2d_half_4x32_random_256size, aclFloat16, half, int32_t, 4, 32, 256, None, Undefined, Last)
DEFINE_ELEM2D(elem2d_int32_unaligned_3x8_64size, int32_t, int32_t, int32_t, 3, 8, 64, None, Undefined, Last)
DEFINE_ELEM2D(elem2d_uint8_unaligned_3x32_256size, uint8_t, uint8_t, int32_t, 3, 32, 256, None, Undefined, Last)
DEFINE_ELEM2D_PAD(elem2d_int32_unaligned_3x3_in_3x8_64size, int32_t, int32_t, int32_t, 3, 3, 3, 8, 64, None, Undefined,
                  Last)
DEFINE_ELEM2D_PAD(elem2d_int32_unaligned_9x9_in_9x16_256size, int32_t, int32_t, int32_t, 9, 9, 9, 16, 256, None,
                  Undefined, Last)
DEFINE_ELEM2D_PAD(elem2d_int32_scalar_1x1_in_1x8_8size, int32_t, int32_t, int32_t, 1, 1, 1, 8, 8, None, Undefined, Last)
DEFINE_ROW_PAD(row_int32_unaligned_3x8_8rows, int32_t, int32_t, int32_t, 3, 8, 8, 8, None, Undefined, Last)
DEFINE_ROW_PAD(row_int32_unaligned_9x16_16rows, int32_t, int32_t, int32_t, 9, 16, 16, 16, None, Undefined, Last)

DEFINE_ELEM2D_CHUNKED(elem2d_float_2048x8_last_256size, float, float, int32_t, 2048, 8, 128, 256, None, Undefined, Last)
DEFINE_ELEM2D_CHUNKED(elem2d_float_2048x8_default_16384size, float, float, int32_t, 2048, 8, 128, 16384, None,
                      Undefined, Default)

DEFINE_ELEM2D_DYN_UB(elem2d_float_2304x8_last_256size, float, float, int32_t, 2304, 8, 256, (2304 * 8 * 8), None,
                     Undefined, Last)
DEFINE_ELEM2D_DYN_UB(elem2d_float_2304x8_default_18432size, float, float, int32_t, 2304, 8, 18432, (2304 * 8 * 8), None,
                     Undefined, Default)
DEFINE_ELEM2D_DYN_UB(elem2d_float_2560x8_last_256size, float, float, int32_t, 2560, 8, 256, (2560 * 8 * 8), None,
                     Undefined, Last)
DEFINE_ELEM2D_DYN_UB(elem2d_float_2560x8_default_20480size, float, float, int32_t, 2560, 8, 20480, (2560 * 8 * 8), None,
                     Undefined, Default)
DEFINE_ELEM2D_DYN_UB(elem2d_float_2816x8_last_256size, float, float, int32_t, 2816, 8, 256, (2816 * 8 * 8), None,
                     Undefined, Last)
DEFINE_ELEM2D_DYN_UB(elem2d_float_2816x8_default_22528size, float, float, int32_t, 2816, 8, 22528, (2816 * 8 * 8), None,
                     Undefined, Default)
DEFINE_ELEM2D_DYN_UB(elem2d_float_3072x8_last_256size, float, float, int32_t, 3072, 8, 256, (3072 * 8 * 8), None,
                     Undefined, Last)
DEFINE_ELEM2D_DYN_UB(elem2d_float_3072x8_default_24576size, float, float, int32_t, 3072, 8, 24576, (3072 * 8 * 8), None,
                     Undefined, Default)
DEFINE_ELEM2D_DYN_UB(elem2d_float_3200x8_last_256size, float, float, int32_t, 3200, 8, 256, (3200 * 8 * 8), None,
                     Undefined, Last)
DEFINE_ELEM2D_DYN_UB(elem2d_float_3200x8_default_25600size, float, float, int32_t, 3200, 8, 25600, (3200 * 8 * 8), None,
                     Undefined, Default)
DEFINE_ELEM2D_DYN_UB(elem2d_float_3456x8_last_256size, float, float, int32_t, 3456, 8, 256, (3456 * 8 * 8), None,
                     Undefined, Last)
DEFINE_ELEM2D_DYN_UB(elem2d_float_3456x8_default_27648size, float, float, int32_t, 3456, 8, 27648, (3456 * 8 * 8), None,
                     Undefined, Default)

DEFINE_ELEM2D_DYN(elem2d_dyn_user_float_1x9_in_1x16_3x10, float, float, int32_t, 1, 16, 1, 9, 3, 10, None, Skip, Last)
DEFINE_ELEM2D_DYN(elem2d_dyn_int32_4x8_in_4x8_64size, int32_t, int32_t, int32_t, 4, 8, 4, 8, 8, 8, None, Undefined,
                  Last)
DEFINE_ELEM2D_DYN(elem2d_dyn_float_3x3_in_3x8_64size, float, float, int32_t, 3, 8, 3, 3, 8, 8, None, Undefined, Last)
DEFINE_ELEM2D_DYN(elem2d_dyn_half_8x16_in_8x16_4x32, aclFloat16, half, int32_t, 8, 16, 8, 16, 4, 32, None, Undefined,
                  Last)
DEFINE_ROW_DYN(row_dyn_int32_3x16_8rows, int32_t, int32_t, int32_t, 3, 16, 8, 3, 16, 8, None, Undefined, Last)
DEFINE_ROW_DYN(row_dyn_half_4x32_16rows, aclFloat16, half, int32_t, 4, 32, 8, 4, 32, 16, None, Undefined, Last)
