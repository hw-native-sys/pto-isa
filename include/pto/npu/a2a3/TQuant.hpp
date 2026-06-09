/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef TQUANT_HPP
#define TQUANT_HPP

#include "pto/npu/a2a3/TRowExpandMul.hpp"
#include "pto/npu/a2a3/TRowExpandAdd.hpp"
#include "pto/npu/a2a3/TCvt.hpp"
#include "pto/npu/a2a3/TAssign.hpp"
#include <pto/common/type.hpp>

namespace pto {

// Check whether two UB buffers overlap (runtime address comparison).
template <typename TileA, typename TileB>
PTO_INTERNAL bool TQuantBuffersOverlap(TileA &a, TileB &b)
{
#ifndef __PTO_AUTO__
    auto aStart = reinterpret_cast<uintptr_t>(a.data());
    auto aEnd = aStart + TileA::Rows * TileA::RowStride * sizeof(typename TileA::DType);
    auto bStart = reinterpret_cast<uintptr_t>(b.data());
    auto bEnd = bStart + TileB::Rows * TileB::RowStride * sizeof(typename TileB::DType);
    return (aStart < bEnd) && (bStart < aEnd);
#else
    return true;
#endif
}

// s32→fp16 dispatch: uses row-by-row when buffers overlap and there's a tail.
template <int PadColsSrc, typename TileDataCvtF16, typename TileDataCvtS32>
__tf__ PTO_INTERNAL void TQuantCvtS32ToFp16(typename TileDataCvtF16::TileDType __out__ src_f16,
                                            typename TileDataCvtS32::TileDType __in__ src_s32, uint32_t validRow,
                                            uint32_t validCol)
{
    // Row-by-row s32→fp16 conversion for in-place aliased buffers with a tail.
    // Processes each row's head + tail atomically to avoid cross-row data corruption.
    int kValidCols = validCol;
    constexpr int KStaticCols = TileDataCvtS32::Cols;
    constexpr int kS32ElemsPerRepeat = static_cast<int>(REPEAT_BYTE / sizeof(int32_t));    // 64
    constexpr int kS32ElemsPerBlock = static_cast<int>(BLOCK_BYTE_SIZE / sizeof(int32_t)); // 8
    constexpr int kF16ElemsPerBlock = static_cast<int>(BLOCK_BYTE_SIZE / sizeof(half));    // 16
    int kHeadRepeats = kValidCols / kS32ElemsPerRepeat;
    int kTailElems = kValidCols % kS32ElemsPerRepeat;

    __ubuf__ half *fp16Ptr = (__ubuf__ half *)__cce_get_tile_ptr(src_f16);
    __ubuf__ int32_t *s32Ptr = (__ubuf__ int32_t *)__cce_get_tile_ptr(src_s32);

    set_deqscale(static_cast<half>(1.0));
    pipe_barrier(PIPE_V);
    for (uint32_t i = 0; i < validRow; i++) {
        if (kHeadRepeats > 0) {
            vconv_deq(fp16Ptr + i * PadColsSrc, s32Ptr + i * KStaticCols, kHeadRepeats, 1, 1,
                      kS32ElemsPerRepeat / kF16ElemsPerBlock, kS32ElemsPerRepeat / kS32ElemsPerBlock);
        }
        if (kTailElems > 0) {
            SetContinuousMask(kTailElems);
            vconv_deq(fp16Ptr + i * PadColsSrc + kHeadRepeats * kS32ElemsPerRepeat,
                      s32Ptr + i * KStaticCols + kHeadRepeats * kS32ElemsPerRepeat, 1, 1, 1, 1, 1);
            set_vector_mask(-1, -1);
        }
    }
}

template <QuantType quant_type, typename TileDataOut, typename TileDataSrc, typename TileDataPara, typename TileDataTmp>
PTO_INTERNAL void TQUANT_IMPL(TileDataOut &dst, TileDataSrc &src, TileDataPara &scale, TileDataTmp &tmp,
                              TileDataPara *offset = nullptr)
{
    using T = typename TileDataSrc::DType;
    using U = typename TileDataOut::DType;
    static_assert(std::is_same<T, float32_t>::value, "Fix: Input has to be float 32");
    if constexpr (quant_type == QuantType::INT8_SYM) {
        static_assert(std::is_same<U, int8_t>::value, "Fix: Quant INT8 sym: Out data type has to be int8");
    } else if constexpr (quant_type == QuantType::INT8_ASYM) {
        static_assert(std::is_same<U, uint8_t>::value, "Fix: Quant INT8 asym: Out data type has to be uint8");
    }

    constexpr int blockElem = static_cast<int>(BLOCK_BYTE_SIZE / sizeof(half));
    constexpr int PadColsSrc = ((((TileDataSrc::Cols) + (blockElem)-1) / (blockElem)) * (blockElem));
    using TileDataCvtF16 = Tile<TileType::Vec, half, TileDataSrc::Rows, PadColsSrc, BLayout::RowMajor, -1, -1>;
    using TileDataCvtS32 =
        Tile<TileType::Vec, int32_t, TileDataSrc::Rows, TileDataSrc::Cols, BLayout::RowMajor, -1, -1>;

    // scale/offset are per-row scalars (column-major), so TROWEXPANDMUL/ADD take the vbrcb broadcast
    // path and need an explicit scratch tile (one 32B block per row, <= 8KB for up to 256 rows).
    TROWEXPANDMUL_IMPL(src, src, scale, tmp);
    pipe_barrier(PIPE_V);
    if constexpr (quant_type == QuantType::INT8_ASYM) {
        TROWEXPANDADD_IMPL(src, src, *offset, tmp);
        pipe_barrier(PIPE_V);
    }

    TileDataCvtF16 src_f16(src.GetValidRow(), src.GetValidCol());
    TileDataCvtS32 src_s32(src.GetValidRow(), src.GetValidCol());
#ifndef __PTO_AUTO__
    TASSIGN_IMPL(src_f16, reinterpret_cast<uintptr_t>(src.data()));
    TASSIGN_IMPL(src_s32, reinterpret_cast<uintptr_t>(src.data()));
#else
    TRESHAPE_IMPL(src_f16, src);
    TRESHAPE_IMPL(src_s32, src);
#endif

    TCVT_IMPL(src_s32, src, RoundMode::CAST_RINT); // fp32->s32
    pipe_barrier(PIPE_V);

    constexpr int kS32ElemsPerRepeat = static_cast<int>(REPEAT_BYTE / sizeof(int32_t));
    constexpr bool kHasTail = (TileDataCvtS32::Cols % kS32ElemsPerRepeat != 0);
    if constexpr (kHasTail) {
        if (TQuantBuffersOverlap(src_f16, src_s32)) {
            TQuantCvtS32ToFp16<PadColsSrc, TileDataCvtF16, TileDataCvtS32>(
                src_f16.data(), src_s32.data(), src.GetValidRow(), src.GetValidCol()); // s32->fp16
        }
    } else {
        TCVT_IMPL(src_f16, src_s32, RoundMode::CAST_RINT);
    }

    pipe_barrier(PIPE_V);
    TCVT_IMPL(dst, src_f16, RoundMode::CAST_RINT, SaturationMode::ON); // fp16->int8
    pipe_barrier(PIPE_V);
}

template <QuantType quant_type, typename TileDataOut, typename TileDataSrc, typename TileDataPara>
PTO_INTERNAL void TQUANT_IMPL(TileDataOut &dst, TileDataSrc &src, TileDataPara &scale, TileDataPara *offset = nullptr)
{
    using T = typename TileDataSrc::DType;
    using U = typename TileDataOut::DType;
    static_assert(std::is_same<T, float32_t>::value, "Fix: Input has to be float 32");
    if constexpr (quant_type == QuantType::INT8_SYM) {
        static_assert(std::is_same<U, int8_t>::value, "Fix: Quant INT8 sym: Out data type has to be int8");
    } else if constexpr (quant_type == QuantType::INT8_ASYM) {
        static_assert(std::is_same<U, uint8_t>::value, "Fix: Quant INT8 asym: Out data type has to be uint8");
    }

    constexpr int blockElem = static_cast<int>(BLOCK_BYTE_SIZE / sizeof(half));
    constexpr int PadColsSrc = ((((TileDataSrc::Cols) + (blockElem)-1) / (blockElem)) * (blockElem));
    using TileDataCvtF16 = Tile<TileType::Vec, half, TileDataSrc::Rows, PadColsSrc, BLayout::RowMajor, -1, -1>;
    using TileDataCvtS32 =
        Tile<TileType::Vec, int32_t, TileDataSrc::Rows, TileDataSrc::Cols, BLayout::RowMajor, -1, -1>;

    // scale/offset are per-row scalars (column-major), so TROWEXPANDMUL/ADD take the vbrcb broadcast
    // path and need an explicit scratch tile (one 32B block per row, <= 8KB for up to 256 rows).
    TROWEXPANDMUL_IMPL(src, src, scale);
    pipe_barrier(PIPE_V);
    if constexpr (quant_type == QuantType::INT8_ASYM) {
        TROWEXPANDADD_IMPL(src, src, *offset);
        pipe_barrier(PIPE_V);
    }

    TileDataCvtF16 src_f16(src.GetValidRow(), src.GetValidCol());
    TileDataCvtS32 src_s32(src.GetValidRow(), src.GetValidCol());
#ifndef __PTO_AUTO__
    TASSIGN_IMPL(src_f16, reinterpret_cast<uintptr_t>(src.data()));
    TASSIGN_IMPL(src_s32, reinterpret_cast<uintptr_t>(src.data()));
#else
    TRESHAPE_IMPL(src_f16, src);
    TRESHAPE_IMPL(src_s32, src);
#endif

    TCVT_IMPL(src_s32, src, RoundMode::CAST_RINT); // fp32->s32
    pipe_barrier(PIPE_V);

    constexpr int kS32ElemsPerRepeat = static_cast<int>(REPEAT_BYTE / sizeof(int32_t));
    constexpr bool kHasTail = (TileDataCvtS32::Cols % kS32ElemsPerRepeat != 0);
    if constexpr (kHasTail) {
        if (TQuantBuffersOverlap(src_f16, src_s32)) {
            TQuantCvtS32ToFp16<PadColsSrc, TileDataCvtF16, TileDataCvtS32>(
                src_f16.data(), src_s32.data(), src.GetValidRow(), src.GetValidCol()); // s32->fp16
        }
    } else {
        TCVT_IMPL(src_f16, src_s32, RoundMode::CAST_RINT);
    }

    pipe_barrier(PIPE_V);
    TCVT_IMPL(dst, src_f16, RoundMode::CAST_RINT, SaturationMode::ON); // fp16->int8
    pipe_barrier(PIPE_V);
}
} // namespace pto
#endif // TQUANT_HPP
