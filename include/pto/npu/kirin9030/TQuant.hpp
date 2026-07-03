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

#include <pto/common/type.hpp>
#include "pto/npu/a5/TReshape.hpp"
#include <type_traits>

namespace pto {

template <typename TileDataOut, typename TileDataSrc, typename TileDataPara>
__tf__ PTO_INTERNAL void TQuant_Int8Sym(typename TileDataOut::TileDType __out__ dst,
                                        typename TileDataSrc::TileDType __in__ src,
                                        typename TileDataPara::TileDType __in__ scale, unsigned validRows,
                                        unsigned validCols)
{
    using T = typename TileDataSrc::DType;  // fp32
    using S = typename TileDataPara::DType; // fp32
    using U = typename TileDataOut::DType;  // int8
    __ubuf__ T *srcPtr = (__ubuf__ T *)__cce_get_tile_ptr(src);
    __ubuf__ U *dstPtr = (__ubuf__ U *)__cce_get_tile_ptr(dst);
    __ubuf__ S *scalePtr = (__ubuf__ S *)__cce_get_tile_ptr(scale);
    uint16_t repeatTimes = CeilDivision(validCols, ELE_CNT_B32);
    __VEC_SCOPE__
    {
        RegTensor<float> v_input, v_scale;
        RegTensor<int32_t> v_s32;
        RegTensor<half> vb16;
        RegTensor<int8_t> v_output_s8;
        for (uint16_t row = 0; row < (uint16_t)validRows; ++row) {
            uint32_t sreg = validCols;
            for (uint16_t idx = 0; idx < repeatTimes; ++idx) {
                MaskReg preg_b32 = CreatePredicate<float>(sreg);
                vlds(v_scale, scalePtr, row, BRC_B32); // broadcast row scaling
                vlds(v_input, srcPtr, ELE_CNT_B32 * idx + row * TileDataSrc::Cols, NORM);
                vmul(v_input, v_input, v_scale, preg_b32, MODE_ZEROING);
                // Round once at fp32 (s32 round-trip) then exact fp32->fp16->s8.
                vcvt(v_s32, v_input, preg_b32, ROUND_R, RS_ENABLE);
                vcvt(v_input, v_s32, preg_b32, ROUND_R);
                vcvt(vb16, v_input, preg_b32, ROUND_R, RS_ENABLE, PART_EVEN);
                vcvt(v_output_s8, vb16, preg_b32, ROUND_R, RS_ENABLE, PART_EVEN);
                vsts(v_output_s8, dstPtr, ELE_CNT_B32 * idx + row * TileDataOut::Cols, PK4_B32, preg_b32);
            }
        }
    }
}

// TQUANT: fp32 -> u8 conversion, Int8Asym
template <typename TileDataOut, typename TileDataSrc, typename TileDataPara>
__tf__ PTO_INTERNAL void TQuant_Int8Asym(typename TileDataOut::TileDType __out__ dst,
                                         typename TileDataSrc::TileDType __in__ src,
                                         typename TileDataPara::TileDType __in__ scale,
                                         typename TileDataPara::TileDType __in__ offset, unsigned validRows,
                                         unsigned validCols)
{
    using T = typename TileDataSrc::DType;  // fp32
    using U = typename TileDataOut::DType;  // uint8
    using S = typename TileDataPara::DType; // fp32
    __ubuf__ T *srcPtr = (__ubuf__ T *)__cce_get_tile_ptr(src);
    __ubuf__ U *dstPtr = (__ubuf__ U *)__cce_get_tile_ptr(dst);
    __ubuf__ S *scalePtr = (__ubuf__ S *)__cce_get_tile_ptr(scale);
    __ubuf__ S *offsetPtr = (__ubuf__ S *)__cce_get_tile_ptr(offset);
    uint16_t repeatTimes = CeilDivision(validCols, ELE_CNT_B32);
    __VEC_SCOPE__
    {
        RegTensor<float> vb32_scale, vb32_input, vb32_offset;
        RegTensor<int32_t> vb32_int;
        RegTensor<half> vb16_output;
        RegTensor<uint8_t> vb8_output;
        for (uint16_t row = 0; row < (uint16_t)validRows; ++row) {
            uint32_t sreg = validCols;
            for (uint16_t idx = 0; idx < repeatTimes; ++idx) {
                MaskReg preg_b32 = CreatePredicate<float>(sreg);
                vlds(vb32_scale, scalePtr, row, BRC_B32);   // broadcast row scaling
                vlds(vb32_offset, offsetPtr, row, BRC_B32); // broadcast row offset
                vlds(vb32_input, srcPtr, ELE_CNT_B32 * idx + row * TileDataSrc::Cols, NORM);
                vmul(vb32_input, vb32_input, vb32_scale, preg_b32, MODE_ZEROING);
                vadd(vb32_input, vb32_input, vb32_offset, preg_b32, MODE_ZEROING);
                // Round once at fp32 (s32 round-trip) then exact fp32->fp16->u8.
                vcvt(vb32_int, vb32_input, preg_b32, ROUND_R, RS_ENABLE);
                vcvt(vb32_input, vb32_int, preg_b32, ROUND_R);
                vcvt(vb16_output, vb32_input, preg_b32, ROUND_R, RS_ENABLE, PART_EVEN);
                vcvt(vb8_output, vb16_output, preg_b32, ROUND_R, RS_ENABLE, PART_EVEN);
                vsts(vb8_output, dstPtr, ELE_CNT_B32 * idx + row * TileDataOut::Cols, PK4_B32, preg_b32);
            }
        }
    }
}

// TQUANT Interface for FP32/FP16/BF16->INT4/8/16
template <QuantType quant_type, typename TileDataOut, typename TileDataSrc, typename TileDataPara>
PTO_INTERNAL void TQUANT_IMPL(TileDataOut &dst, TileDataSrc &src, TileDataPara &scale, TileDataPara *offset = nullptr)
{
    using T = typename TileDataSrc::DType;
    static_assert(std::is_same<T, float32_t>::value, "Fix: Input has to be float 32");

    if constexpr (quant_type == QuantType::INT8_SYM) {
        using U = typename TileDataOut::DType;
        static_assert(std::is_same<U, int8_t>::value, "Fix: Quant INT8 sym: Out data type has to be int8");
        TQuant_Int8Sym<TileDataOut, TileDataSrc, TileDataPara>(dst.data(), src.data(), scale.data(), src.GetValidRow(),
                                                               src.GetValidCol());
    } else if constexpr (quant_type == QuantType::INT8_ASYM) {
        using U = typename TileDataOut::DType;
        static_assert(std::is_same<U, uint8_t>::value, "Fix: Quant INT8 asym: Out data type has to be uint8");
        TQuant_Int8Asym<TileDataOut, TileDataSrc, TileDataPara>(dst.data(), src.data(), scale.data(), offset->data(),
                                                                src.GetValidRow(), src.GetValidCol());
    }
}

template <QuantType quant_type, typename TileDataOut, typename TileDataSrc, typename TileDataExp, typename TileDataMax,
          typename TileDataScaling>
PTO_INTERNAL void TQUANT_IMPL(TileDataOut &dst, TileDataSrc &src, TileDataExp *exp, TileDataMax *max,
                              TileDataScaling *scaling)
{
    static_assert(sizeof(typename TileDataSrc::DType) == 0, "Fix: TQUANT does not support MX data type in Kirin9030.");
}
} // namespace pto
#endif // TQUANT_HPP
