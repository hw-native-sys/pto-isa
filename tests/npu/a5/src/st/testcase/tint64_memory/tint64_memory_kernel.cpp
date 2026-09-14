/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#include <cstdint>
#include <pto/pto-inst.hpp>
#ifndef __CPU_SIM
#include <acl/acl.h>
#endif
using namespace pto;

template <typename T, int Op, int Cols, int ValidCols>
PTO_INTERNAL void RunBoundary(__gm__ T* out, __gm__ T* input, __gm__ uint32_t* indices)
{
    constexpr bool rowReduce = Op >= 6 && Op <= 8;
    constexpr int outputCols = rowReduce ? 4 : Cols;
    constexpr int outputValid = rowReduce ? 1 : ValidCols;
    using Src = Tile<TileType::Vec, T, 1, Cols, BLayout::RowMajor, 1, ValidCols>;
    using Dst = Tile<TileType::Vec, T, 1, outputCols, BLayout::RowMajor, 1, outputValid>;
    using RawOut = Tile<TileType::Vec, T, 1, outputCols + 64>;
    using Mask = Tile<TileType::Vec, uint8_t, 1, 32>;
    using Idx = Tile<TileType::Vec, uint32_t, 1, (Cols + 7) / 8 * 8, BLayout::RowMajor, 1, ValidCols>;
    using Scalar = Tile<TileType::Vec, T, 4, 1, BLayout::ColMajor, 1, 1>;
    Src src, tmp;
    Dst dst;
    RawOut rawOut;
    Mask mask;
    Idx idx, indexTmp;
    Scalar scalar;
    TASSIGN<((Op == 20 || Op == 21) ? 32768 : 262144 - Cols * sizeof(T))>(src);
    TASSIGN<0>(dst);
    TASSIGN<0>(rawOut);
    TASSIGN<8192>(tmp);
    TASSIGN<16384>(mask);
    TASSIGN<((Op == 20 || Op == 21) ? 262144 - Idx::Numel * sizeof(uint32_t) : 20480)>(idx);
    TASSIGN<24576>(indexTmp);
    TASSIGN<24576>(scalar);
    using SG = GlobalTensor<T, Shape<1, 1, 1, 1, ValidCols>, pto::Stride<1, 1, 1, Cols, 1>>;
    using OG = GlobalTensor<T, Shape<1, 1, 1, 1, outputCols + 64>, pto::Stride<1, 1, 1, outputCols + 64, 1>>;
    using MG = GlobalTensor<uint8_t, Shape<1, 1, 1, 1, 32>, pto::Stride<1, 1, 1, 32, 1>>;
    using IG = GlobalTensor<uint32_t, Shape<1, 1, 1, 1, ValidCols>, pto::Stride<1, 1, 1, (Cols + 7) / 8 * 8, 1>>;
    using ScalarG = GlobalTensor<T, Shape<1, 1, 1, 1, 1>, pto::Stride<1, 1, 1, 1, 1>, Layout::DN>;
    SG sg(input);
    OG og(out);
    MG mg(reinterpret_cast<__gm__ uint8_t*>(out));
    IG ig(indices);
    ScalarG scalarG(input);
    TLOAD(src, sg);
    TLOAD(rawOut, og);
    if constexpr (Op == 15 || Op == 16)
        TLOAD(mask, mg);
    if constexpr (Op >= 19 && Op <= 21)
        TLOAD(idx, ig);
    if constexpr (Op == 13)
        TLOAD(scalar, scalarG);
#ifndef __CPU_SIM
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
#endif
    if constexpr (Op == 0)
        TADD(dst, src, src);
    else if constexpr (Op == 1)
        TADDS(dst, src, T(3));
    else if constexpr (Op == 2)
        TNOT(dst, src);
    else if constexpr (Op == 3)
        TDIV(dst, src, src);
    else if constexpr (Op == 4)
        TDIVS(dst, src, T(3));
    else if constexpr (Op == 5)
        TREM(dst, src, src, tmp);
    else if constexpr (Op == 6)
        TROWMAX(dst, src, tmp);
    else if constexpr (Op == 7)
        TROWMIN(dst, src, tmp);
    else if constexpr (Op == 8)
        TROWSUM(dst, src, tmp);
    else if constexpr (Op == 9)
        TCOLMAX(dst, src);
    else if constexpr (Op == 10)
        TCOLMIN(dst, src);
    else if constexpr (Op == 11)
        TCOLSUM(dst, src);
    else if constexpr (Op == 12)
        TPARTADD(dst, src, src);
    else if constexpr (Op == 13)
        TROWEXPANDADD(dst, src, scalar);
    else if constexpr (Op == 14)
        TCOLEXPANDADD(dst, src, src);
    else if constexpr (Op == 15)
        TSEL(dst, mask, src, src, tmp);
    else if constexpr (Op == 16)
        TSELS(dst, mask, src, tmp, T(3));
    else if constexpr (Op == 17 || Op == 18) {
        using Cmp = Tile<TileType::Vec, uint8_t, 1, 32, BLayout::RowMajor, 1, (ValidCols + 7) / 8>;
        Cmp cmp;
        TASSIGN<0>(cmp);
        if constexpr (Op == 17)
            TCMP(cmp, src, src, CmpMode::EQ);
        else
            TCMPS(cmp, src, T(0), CmpMode::GT);
    } else if constexpr (Op == 19 || Op == 21)
        TSCATTER(dst, src, idx);
    else if constexpr (Op == 20)
        TGATHER(dst, src, idx, indexTmp);
    else if constexpr (Op == 22)
        TCOLEXPAND(dst, src);
#ifndef __CPU_SIM
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
#endif
    TSTORE(og, rawOut);
}

template <typename T, int Op, int Rows, int Cols, int ValidRows, int ValidCols>
PTO_INTERNAL void RunLayout(__gm__ T* out, __gm__ T* input, __gm__ uint32_t* indices)
{
    constexpr auto srcLayout = Op == 3 ? BLayout::ColMajor : BLayout::RowMajor;
    constexpr auto idxLayout = Op == 0 || Op == 2 ? BLayout::ColMajor : BLayout::RowMajor;
    constexpr auto dstLayout = Op == 1 ? BLayout::ColMajor : BLayout::RowMajor;
    using Src = Tile<TileType::Vec, T, Rows, Cols, srcLayout, -1, -1>;
    using Dst = Tile<TileType::Vec, T, Rows, Cols, dstLayout, -1, -1>;
    using Idx = Tile<TileType::Vec, uint32_t, Rows, Cols, idxLayout, -1, -1>;
    using RawSrc = Tile<TileType::Vec, T, 1, Rows * Cols>;
    using RawIdx = Tile<TileType::Vec, uint32_t, 1, Rows * Cols>;
    using RawDst = Tile<TileType::Vec, T, 1, Rows * Cols + 64>;
    Src src(ValidRows, ValidCols);
    Dst dst(ValidRows, ValidCols);
    Idx idx(ValidRows, ValidCols), tmp(ValidRows, ValidCols);
    RawSrc rawSrc;
    RawIdx rawIdx;
    RawDst rawDst;
    TASSIGN<0>(src);
    TASSIGN<0>(rawSrc);
    TASSIGN<8192>(idx);
    TASSIGN<8192>(rawIdx);
    TASSIGN<16384>(tmp);
    TASSIGN<32768>(dst);
    TASSIGN<32768>(rawDst);
    using SG = GlobalTensor<T, Shape<1, 1, 1, 1, Rows * Cols>, pto::Stride<1, 1, 1, Rows * Cols, 1>>;
    using IG = GlobalTensor<uint32_t, Shape<1, 1, 1, 1, Rows * Cols>, pto::Stride<1, 1, 1, Rows * Cols, 1>>;
    using DG = GlobalTensor<T, Shape<1, 1, 1, 1, Rows * Cols + 64>, pto::Stride<1, 1, 1, Rows * Cols + 64, 1>>;
    SG sg(input);
    IG ig(indices);
    DG dg(out);
    TLOAD(rawSrc, sg);
    TLOAD(rawIdx, ig);
    TLOAD(rawDst, dg);
#ifndef __CPU_SIM
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
#endif
    if constexpr (Op < 2)
        TGATHER(dst, src, idx, tmp);
    else
        TSCATTER(dst, src, idx);
#ifndef __CPU_SIM
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
#endif
    TSTORE(dg, rawDst);
}

template <int Offset, int ValidRows, int ValidCols>
PTO_INTERNAL void RunEmpty(__gm__ uint64_t* out, __gm__ uint64_t*, __gm__ uint32_t*)
{
    using Src = Tile<TileType::Vec, int32_t, 4, 16>;
    using Dst = Tile<TileType::Vec, int32_t, 4, 8, BLayout::RowMajor, -1, -1>;
    using Raw = Tile<TileType::Vec, int32_t, 1, 160>;
    using G = GlobalTensor<int32_t, Shape<1, 1, 1, 1, 160>, pto::Stride<1, 1, 1, 160, 1>>;
    Src src;
    Dst dst(ValidRows, ValidCols);
    Raw raw;
    TASSIGN<0>(src);
    TASSIGN<4096>(dst);
    TASSIGN<4096>(raw);
    G g(reinterpret_cast<__gm__ int32_t*>(out));
    TLOAD(raw, g);
#ifndef __CPU_SIM
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
#endif
    TEXTRACT(dst, src, 0, Offset);
#ifndef __CPU_SIM
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
#endif
    TSTORE(g, raw);
}

#ifndef __CPU_SIM
#define PTO_KERNEL(Name, T, ...)                                                                   \
    __global__ AICORE void Kernel_##Name(__gm__ T* out, __gm__ T* input, __gm__ uint32_t* indices) \
    {                                                                                              \
        __VA_ARGS__(out, input, indices);                                                          \
    }                                                                                              \
    void Launch_##Name(T* out, T* input, uint32_t* indices, void* stream)                          \
    {                                                                                              \
        Kernel_##Name<<<1, nullptr, stream>>>(out, input, indices);                                \
    }
#else
#define PTO_KERNEL(Name, T, ...)                                   \
    void Launch_##Name(T* out, T* input, uint32_t* indices, void*) \
    {                                                              \
        NPU_MEMORY_INIT(NPUArch::A5);                              \
        NPU_MEMORY_CLEAR();                                        \
        __VA_ARGS__(out, input, indices);                          \
    }
#endif
#define PTO_BOUND_CASE(Name, T, Op, Cols, ValidCols) PTO_KERNEL(Name, T, RunBoundary<T, Op, Cols, ValidCols>)
#define PTO_LAYOUT_CASE(Name, T, Op, Rows, Cols, ValidRows, ValidCols) \
    PTO_KERNEL(Name, T, RunLayout<T, Op, Rows, Cols, ValidRows, ValidCols>)
#define PTO_EMPTY_CASE(Name, Offset, Rows, Cols) PTO_KERNEL(Name, uint64_t, RunEmpty<Offset, Rows, Cols>)
#include "cases.inc"
