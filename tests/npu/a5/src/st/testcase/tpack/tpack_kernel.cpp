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
#include <pto/npu/a5/TPack.hpp>
#include "acl/acl.h"
#include <type_traits>

using namespace pto;

#define PTO_CEIL(x, y) ((((x) + (y)-1) / (y)) * (y))

namespace TPackTest {

template <int validRows, int validCols, typename SrcType, typename DstType>
__global__ AICORE void runTPack(__gm__ DstType __out__ *out, __gm__ SrcType __in__ *in)
{
    constexpr int paddedCols_src = PTO_CEIL(validCols, BLOCK_BYTE_SIZE / sizeof(SrcType));
    constexpr int paddedCols_dst = PTO_CEIL(validCols, BLOCK_BYTE_SIZE / sizeof(DstType));

    using SrcGlobal = GlobalTensor<SrcType, Shape<1, 1, 1, validRows, validCols>, pto::Stride<1, 1, 1, validCols, 1>>;
    using DstGlobal = GlobalTensor<DstType, Shape<1, 1, 1, validRows, validCols>, pto::Stride<1, 1, 1, validCols, 1>>;

    using SrcTile = Tile<TileType::Vec, SrcType, validRows, paddedCols_src, BLayout::RowMajor, -1, -1>;
    using DstTile = Tile<TileType::Vec, DstType, validRows, paddedCols_dst, BLayout::RowMajor, -1, -1>;

    SrcTile srcTile(validRows, validCols);
    DstTile dstTile(validRows, validCols);

    SrcGlobal srcGlobal(in);
    DstGlobal dstGlobal(out);

    // Src tile at offset 0x0, dst after src to avoid overlap
    TASSIGN(srcTile, 0x0);
    TASSIGN(dstTile, 0x20000);

    TLOAD(srcTile, srcGlobal);

#ifndef __PTO_AUTO__
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
#endif

    TPACK(dstTile, srcTile);

#ifndef __PTO_AUTO__
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
#endif

    TSTORE(dstGlobal, dstTile);
}

template <int validRows, int validCols, typename SrcType, typename DstType>
void LaunchTPack(DstType *dst, SrcType *src, void *stream)
{
    runTPack<validRows, validCols, SrcType, DstType><<<1, nullptr, stream>>>(dst, src);
}

} // namespace TPackTest

// b32 -> b16 cases
template void TPackTest::LaunchTPack<128, 128, uint32_t, uint16_t>(uint16_t *dst, uint32_t *src, void *stream);
// b32 -> b8 cases
template void TPackTest::LaunchTPack<128, 128, uint32_t, uint8_t>(uint8_t *dst, uint32_t *src, void *stream);
// b16 -> b8 cases
template void TPackTest::LaunchTPack<128, 128, uint16_t, uint8_t>(uint8_t *dst, uint16_t *src, void *stream);
