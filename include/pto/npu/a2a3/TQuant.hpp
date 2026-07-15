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

template <QuantType quant_type, typename TileDataOut, typename TileDataSrc, typename TileDataPara, typename TileDataTmp>
PTO_INTERNAL void TQUANT_IMPL(
    TileDataOut& dst, TileDataSrc& src, TileDataPara& scale, TileDataTmp& tmp, TileDataPara* offset = nullptr)
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

    // tmp is reused afterward for fp32->s32 conversion (A3 does not support in-place tcvt).
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
    TASSIGN_IMPL(src_s32, reinterpret_cast<uintptr_t>(tmp.data()));
#else
    TRESHAPE_IMPL(src_f16, src);
    TRESHAPE_IMPL(src_s32, tmp);
#endif

    TCVT_IMPL(src_s32, src, RoundMode::CAST_RINT); // fp32->s32, dst=tmp src=src
    pipe_barrier(PIPE_V);
    TCVT_IMPL(src_f16, src_s32, RoundMode::CAST_RINT); // s32->fp16, dst=src src=tmp
    pipe_barrier(PIPE_V);
    TCVT_IMPL(dst, src_f16, RoundMode::CAST_RINT, SaturationMode::ON); // fp16->int8, dst=dst src=src
    pipe_barrier(PIPE_V);
}
} // namespace pto
#endif // TQUANT_HPP
