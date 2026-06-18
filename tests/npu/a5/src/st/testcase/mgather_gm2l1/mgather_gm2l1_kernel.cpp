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
#include <pto/npu/a5/MGather.hpp>
#include "acl/acl.h"

using namespace pto;

template <typename TileDataDst, typename TileDataSrc>
__tf__ PTO_INTERNAL void tf_copy_cbuf_to_ubuf(typename TileDataDst::TileDType __out__ dst,
                                              typename TileDataSrc::TileDType __in__ src, int vec_core, int block_count,
                                              int block_len, int src_stride, int dst_stride)
{
    copy_cbuf_to_ubuf((__ubuf__ void *)__cce_get_tile_ptr(dst), (__cbuf__ void *)__cce_get_tile_ptr(src), vec_core,
                      block_count, block_len, src_stride, dst_stride);
}

template <typename DstTileData, typename SrcTileData, uint8_t syncID>
AICORE inline void MovL1ToUbuf(DstTileData &dstTile, SrcTileData &srcTile)
{
#if defined(__DAV_CUBE__)
    uint16_t blockCount = 1;
    uint16_t blockLen = DstTileData::Rows * DstTileData::Cols * sizeof(typename SrcTileData::DType) / BLOCK_BYTE_SIZE;
#ifndef __PTO_AUTO__
    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
#endif
    tf_copy_cbuf_to_ubuf<DstTileData, SrcTileData>(dstTile.data(), srcTile.data(), 0, blockCount, blockLen, 0, 0);
    tf_copy_cbuf_to_ubuf<DstTileData, SrcTileData>(dstTile.data(), srcTile.data(), 1, blockCount, blockLen, 0, 0);
#ifndef __PTO_AUTO__
    set_flag(PIPE_MTE1, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_MTE1, PIPE_MTE3, EVENT_ID0);
#endif
    set_intra_block(PIPE_MTE1, syncID);
    set_intra_block(PIPE_MTE1, syncID + 16);
#endif
}

template <pto::GatherOOB Oob, typename T, typename TIdx, uint32_t kRows, uint32_t kCols, uint32_t kTableRows>
inline AICORE void runRowL1(__gm__ T *out, __gm__ T *table, __gm__ TIdx *indices)
{
    using TableShape = pto::Shape<1, 1, 1, kTableRows, kCols>;
    using TableStride = pto::Stride<1, 1, 1, kCols, 1>;
    using IdxShape = pto::Shape<1, 1, 1, 1, kRows>;
    using IdxStride = pto::Stride<1, 1, 1, kRows, 1>;

    GlobalTensor<T, TableShape, TableStride, Layout::ND> tableGlobal(table);
    GlobalTensor<TIdx, IdxShape, IdxStride, Layout::ND> idxGlobal(indices);

    using DstTile = Tile<TileType::Mat, T, kRows, kCols, BLayout::ColMajor, kRows, kCols, SLayout::RowMajor, 512>;
    using TileUBData = Tile<TileType::Vec, T, kRows, kCols, BLayout::RowMajor, -1, -1>;

    DstTile dstTile;
    TASSIGN(dstTile, 0x0);
    TileUBData ubTile(kRows, kCols);
    TASSIGN(ubTile, 0x0);

    using GlobalDataOut =
        GlobalTensor<T, pto::Shape<1, 1, 1, kRows, kCols>,
                     pto::Stride<1 * kRows * kCols, 1 * kRows * kCols, kRows * kCols, kCols, 1>, Layout::ND>;
    GlobalDataOut dstGlobal(out);

    MGATHER<Coalesce::Row, Oob>(dstTile, tableGlobal, idxGlobal);

    constexpr uint8_t syncID = 0;
    MovL1ToUbuf<TileUBData, DstTile, syncID>(ubTile, dstTile);

#if defined(__DAV_VEC__)
    wait_intra_block(PIPE_MTE3, syncID);
    TSTORE(dstGlobal, ubTile);
#endif
    out = dstGlobal.data();
}

template <pto::GatherOOB Oob, typename T, typename TIdx, uint32_t kRows, uint32_t kCols, uint32_t kTableSize>
inline AICORE void runElemL1(__gm__ T *out, __gm__ T *table, __gm__ TIdx *indices, __gm__ T *scratch)
{
    using TableShape = pto::Shape<1, 1, 1, 1, kTableSize>;
    using TableStride = pto::Stride<1, 1, 1, kTableSize, 1>;
    using IdxShape = pto::Shape<1, 1, 1, kRows, kCols>;
    using IdxStride = pto::Stride<1, 1, 1, kCols, 1>;
    using ScratchShape = pto::Shape<1, 1, 1, 1, kRows * kCols>;
    using ScratchStride = pto::Stride<1, 1, 1, kRows * kCols, 1>;

    GlobalTensor<T, TableShape, TableStride, Layout::ND> tableGlobal(table);
    GlobalTensor<TIdx, IdxShape, IdxStride, Layout::ND> idxGlobal(indices);
    GlobalTensor<T, ScratchShape, ScratchStride, Layout::ND> scratchGlobal(scratch);

    using DstTile = Tile<TileType::Mat, T, kRows, kCols, BLayout::ColMajor, kRows, kCols, SLayout::RowMajor, 512>;
    using TileUBData = Tile<TileType::Vec, T, kRows, kCols, BLayout::RowMajor, -1, -1>;

    DstTile dstTile;
    TASSIGN(dstTile, 0x0);
    TileUBData ubTile(kRows, kCols);
    TASSIGN(ubTile, 0x0);

    using GlobalDataOut =
        GlobalTensor<T, pto::Shape<1, 1, 1, kRows, kCols>,
                     pto::Stride<1 * kRows * kCols, 1 * kRows * kCols, kRows * kCols, kCols, 1>, Layout::ND>;
    GlobalDataOut dstGlobal(out);

    MGATHER<Coalesce::Elem, Oob>(dstTile, tableGlobal, idxGlobal, scratchGlobal);

    constexpr uint8_t syncID = 0;
    MovL1ToUbuf<TileUBData, DstTile, syncID>(ubTile, dstTile);

#if defined(__DAV_VEC__)
    wait_intra_block(PIPE_MTE3, syncID);
    TSTORE(dstGlobal, ubTile);
#endif
    out = dstGlobal.data();
}

template <pto::GatherOOB Oob, typename T, typename TIdx, uint32_t kRows, uint32_t kCols, uint32_t kTableSize>
inline AICORE void runElemL1Simt(__gm__ T *out, __gm__ T *table, __gm__ TIdx *indices, __gm__ T *scratch)
{
    using TableShape = pto::Shape<1, 1, 1, 1, kTableSize>;
    using TableStride = pto::Stride<1, 1, 1, kTableSize, 1>;
    using IdxShape = pto::Shape<1, 1, 1, kRows, kCols>;
    using IdxStride = pto::Stride<1, 1, 1, kCols, 1>;
    using ScratchShape = pto::Shape<1, 1, 1, 1, kRows * kCols>;
    using ScratchStride = pto::Stride<1, 1, 1, kRows * kCols, 1>;

    GlobalTensor<T, TableShape, TableStride, Layout::ND> tableGlobal(table);
    GlobalTensor<TIdx, IdxShape, IdxStride, Layout::ND> idxGlobal(indices);
    GlobalTensor<T, ScratchShape, ScratchStride, Layout::ND> scratchGlobal(scratch);

    using DstTile = Tile<TileType::Mat, T, kRows, kCols, BLayout::ColMajor, kRows, kCols, SLayout::RowMajor, 512>;
    using TileUBData = Tile<TileType::Vec, T, kRows, kCols, BLayout::RowMajor, -1, -1>;

    DstTile dstTile;
    TASSIGN(dstTile, 0x0);
    TileUBData ubTile(kRows, kCols);
    TASSIGN(ubTile, 0x0);

    using GlobalDataOut =
        GlobalTensor<T, pto::Shape<1, 1, 1, kRows, kCols>,
                     pto::Stride<1 * kRows * kCols, 1 * kRows * kCols, kRows * kCols, kCols, 1>, Layout::ND>;
    GlobalDataOut dstGlobal(out);

    MGATHER<Coalesce::Elem, Oob, GatherExec::Simt>(dstTile, tableGlobal, idxGlobal, scratchGlobal);

    constexpr uint8_t syncID = 0;
    MovL1ToUbuf<TileUBData, DstTile, syncID>(ubTile, dstTile);

#if defined(__DAV_VEC__)
    wait_intra_block(PIPE_MTE3, syncID);
    TSTORE(dstGlobal, ubTile);
#endif
    out = dstGlobal.data();
}

#define DEFINE_ROW_L1(NAME, THOST, T, TIDX, R, C, TR, OOB)                                                           \
    extern "C" __global__ AICORE void runMGATHER_##NAME(__gm__ T *out, __gm__ T *table, __gm__ TIDX *indices,        \
                                                        __gm__ T *scratch)                                           \
    {                                                                                                                \
        runRowL1<pto::GatherOOB::OOB, T, TIDX, R, C, TR>(out, table, indices);                                       \
    }                                                                                                                \
    void Launch_##NAME(THOST *out, THOST *table, TIDX *indices, THOST *scratch, void *stream)                        \
    {                                                                                                                \
        runMGATHER_##NAME<<<1, nullptr, stream>>>(reinterpret_cast<T *>(out), reinterpret_cast<T *>(table), indices, \
                                                  reinterpret_cast<T *>(scratch));                                   \
    }

#define DEFINE_ELEM_L1(NAME, THOST, T, TIDX, R, C, TS, OOB)                                                          \
    extern "C" __global__ AICORE void runMGATHER_##NAME(__gm__ T *out, __gm__ T *table, __gm__ TIDX *indices,        \
                                                        __gm__ T *scratch)                                           \
    {                                                                                                                \
        runElemL1<pto::GatherOOB::OOB, T, TIDX, R, C, TS>(out, table, indices, scratch);                             \
    }                                                                                                                \
    void Launch_##NAME(THOST *out, THOST *table, TIDX *indices, THOST *scratch, void *stream)                        \
    {                                                                                                                \
        runMGATHER_##NAME<<<1, nullptr, stream>>>(reinterpret_cast<T *>(out), reinterpret_cast<T *>(table), indices, \
                                                  reinterpret_cast<T *>(scratch));                                   \
    }

DEFINE_ROW_L1(row_float_16x16_64rows, float, float, int32_t, 16, 16, 64, Undefined)
DEFINE_ROW_L1(row_half_16x32_64rows, aclFloat16, half, int32_t, 16, 32, 64, Undefined)
DEFINE_ROW_L1(row_bfloat16_16x16_64rows, uint16_t, bfloat16_t, int32_t, 16, 16, 64, Undefined)
DEFINE_ROW_L1(row_int32_16x8_32rows, int32_t, int32_t, int32_t, 16, 8, 32, Undefined)
DEFINE_ROW_L1(row_uint32_16x16_64rows, uint32_t, uint32_t, int32_t, 16, 16, 64, Undefined)
DEFINE_ROW_L1(row_int16_16x16_32rows, int16_t, int16_t, int32_t, 16, 16, 32, Undefined)
DEFINE_ROW_L1(row_uint16_16x32_48rows, uint16_t, uint16_t, int32_t, 16, 32, 48, Undefined)
DEFINE_ROW_L1(row_int8_16x32_64rows, int8_t, int8_t, int32_t, 16, 32, 64, Undefined)
DEFINE_ROW_L1(row_uint8_32x32_64rows, uint8_t, uint8_t, int32_t, 32, 32, 64, Undefined)
DEFINE_ROW_L1(row_float_clamp_16x16_8rows, float, float, int32_t, 16, 16, 8, Clamp)
DEFINE_ROW_L1(row_int32_wrap_16x8_8rows, int32_t, int32_t, int32_t, 16, 8, 8, Wrap)
DEFINE_ROW_L1(row_half_zero_16x16_8rows, aclFloat16, half, int32_t, 16, 16, 8, Zero)

#define DEFINE_ELEM_L1_SIMT(NAME, THOST, T, TIDX, R, C, TS, OOB)                                                     \
    extern "C" __global__ AICORE void runMGATHER_##NAME(__gm__ T *out, __gm__ T *table, __gm__ TIDX *indices,        \
                                                        __gm__ T *scratch)                                           \
    {                                                                                                                \
        runElemL1Simt<pto::GatherOOB::OOB, T, TIDX, R, C, TS>(out, table, indices, scratch);                         \
    }                                                                                                                \
    void Launch_##NAME(THOST *out, THOST *table, TIDX *indices, THOST *scratch, void *stream)                        \
    {                                                                                                                \
        runMGATHER_##NAME<<<1, nullptr, stream>>>(reinterpret_cast<T *>(out), reinterpret_cast<T *>(table), indices, \
                                                  reinterpret_cast<T *>(scratch));                                   \
    }

DEFINE_ELEM_L1(elem_float_16x16_256size, float, float, int32_t, 16, 16, 256, Undefined)
DEFINE_ELEM_L1(elem_half_16x16_256size, aclFloat16, half, int32_t, 16, 16, 256, Undefined)
DEFINE_ELEM_L1(elem_bfloat16_16x16_256size, uint16_t, bfloat16_t, int32_t, 16, 16, 256, Undefined)
DEFINE_ELEM_L1(elem_int32_16x8_128size, int32_t, int32_t, int32_t, 16, 8, 128, Undefined)
DEFINE_ELEM_L1(elem_uint32_16x16_256size, uint32_t, uint32_t, int32_t, 16, 16, 256, Undefined)
DEFINE_ELEM_L1(elem_int16_16x16_256size, int16_t, int16_t, int32_t, 16, 16, 256, Undefined)
DEFINE_ELEM_L1(elem_uint16_16x32_512size, uint16_t, uint16_t, int32_t, 16, 32, 512, Undefined)
DEFINE_ELEM_L1(elem_int8_16x32_512size, int8_t, int8_t, int32_t, 16, 32, 512, Undefined)
DEFINE_ELEM_L1(elem_uint8_32x32_1024size, uint8_t, uint8_t, int32_t, 32, 32, 1024, Undefined)
DEFINE_ELEM_L1(elem_float_clamp_16x16_64size, float, float, int32_t, 16, 16, 64, Clamp)
DEFINE_ELEM_L1(elem_int32_wrap_16x8_32size, int32_t, int32_t, int32_t, 16, 8, 32, Wrap)
DEFINE_ELEM_L1(elem_half_zero_16x16_64size, aclFloat16, half, int32_t, 16, 16, 64, Zero)

DEFINE_ELEM_L1_SIMT(elem_simt_float_16x16_256size, float, float, int32_t, 16, 16, 256, Undefined)
DEFINE_ELEM_L1_SIMT(elem_simt_half_16x16_256size, aclFloat16, half, int32_t, 16, 16, 256, Undefined)
DEFINE_ELEM_L1_SIMT(elem_simt_bfloat16_16x16_256size, uint16_t, bfloat16_t, int32_t, 16, 16, 256, Undefined)
DEFINE_ELEM_L1_SIMT(elem_simt_int32_16x8_128size, int32_t, int32_t, int32_t, 16, 8, 128, Undefined)
DEFINE_ELEM_L1_SIMT(elem_simt_uint32_16x16_256size, uint32_t, uint32_t, int32_t, 16, 16, 256, Undefined)
DEFINE_ELEM_L1_SIMT(elem_simt_int16_16x16_256size, int16_t, int16_t, int32_t, 16, 16, 256, Undefined)
DEFINE_ELEM_L1_SIMT(elem_simt_uint16_16x32_512size, uint16_t, uint16_t, int32_t, 16, 32, 512, Undefined)
DEFINE_ELEM_L1_SIMT(elem_simt_int8_16x32_512size, int8_t, int8_t, int32_t, 16, 32, 512, Undefined)
DEFINE_ELEM_L1_SIMT(elem_simt_uint8_32x32_1024size, uint8_t, uint8_t, int32_t, 32, 32, 1024, Undefined)
DEFINE_ELEM_L1_SIMT(elem_simt_float_clamp_16x16_64size, float, float, int32_t, 16, 16, 64, Clamp)
DEFINE_ELEM_L1_SIMT(elem_simt_int32_wrap_16x8_32size, int32_t, int32_t, int32_t, 16, 8, 32, Wrap)
DEFINE_ELEM_L1_SIMT(elem_simt_half_zero_16x16_64size, aclFloat16, half, int32_t, 16, 16, 64, Zero)
