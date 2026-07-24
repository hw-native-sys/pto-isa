/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef TFUSEDMULADDRELU_HPP
#define TFUSEDMULADDRELU_HPP

#include <pto/common/constants.hpp>
#include <pto/common/utils.hpp>
#include <pto/npu/a5/common.hpp>
#include <pto/npu/a5/utils.hpp>
#include <pto/npu/a5/TTernOp.hpp>

namespace pto {

template <typename T>
struct FusedMulAddReluOp {
    PTO_INTERNAL static void TernInstr(
        RegTensor<T>& reg_dst, RegTensor<T>& reg_src0, RegTensor<T>& reg_src1, MaskReg& preg)
    {
        vmadd(reg_dst, reg_src0, reg_src1, preg, MODE_ZEROING);
        vrelu(reg_dst, reg_dst, preg, MODE_ZEROING);
    }
};

template <
    typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1, unsigned ElementsPerRepeat,
    unsigned BlockSizeElem>
__tf__ PTO_INTERNAL OP_NAME(TFUSEDMULADDRELU) OP_TYPE(element_wise) void TFusedMulAddRelu(
    typename TileDataDst::TileDType __out__ dst, typename TileDataSrc0::TileDType __in__ src0,
    typename TileDataSrc1::TileDType __in__ src1, unsigned validRows, unsigned validCols,
    VFImplKind version = VFImplKind::VFIMPL_DEFAULT)
{
    using T = typename TileDataDst::DType;
    __ubuf__ T* dstPtr = (__ubuf__ T*)__cce_get_tile_ptr(dst);
    __ubuf__ T* src0Ptr = (__ubuf__ T*)__cce_get_tile_ptr(src0);
    __ubuf__ T* src1Ptr = (__ubuf__ T*)__cce_get_tile_ptr(src1);

    TernaryInstr<FusedMulAddReluOp<T>, TileDataDst, TileDataSrc0, TileDataSrc1, ElementsPerRepeat, BlockSizeElem>(
        dstPtr, src0Ptr, src1Ptr, validRows, validCols, version);
    return;
}

template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1>
PTO_INTERNAL void TFusedMulAddReluCheck(const TileDataDst& dst, const TileDataSrc0& src0, const TileDataSrc1& src1)
{
    unsigned validRows = dst.GetValidRow();
    unsigned validCols = dst.GetValidCol();
    PTO_ASSERT(
        src0.GetValidRow() == validRows && src0.GetValidCol() == validCols && src1.GetValidRow() == validRows &&
            src1.GetValidCol() == validCols,
        "Fix: TFUSEDMULADDRELU input tile src0 valid shape mismatch with output tile dst shape.");
    using T = typename TileDataDst::DType;
    static_assert(
        TileDataDst::isRowMajor && TileDataSrc0::isRowMajor && TileDataSrc1::isRowMajor,
        "Fix: TFUSEDMULADDRELU only support row major layout.");
    static_assert(
        std::is_same_v<T, typename TileDataSrc0::DType> && std::is_same_v<T, typename TileDataSrc1::DType>,
        "Fix: TFUSEDMULADDRELU the data type of dst must be consistent with of src0 and src1.");
    static_assert(
        std::is_same_v<T, half> || std::is_same_v<T, float16_t> || std::is_same_v<T, float> ||
            std::is_same_v<T, float32_t>,
        "Fix: TFUSEDMULADDRELU has invalid data type.");
}

template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1>
PTO_INTERNAL void TFUSEDMULADDRELU_IMPL(TileDataDst& dst, TileDataSrc0& src0, TileDataSrc1& src1)
{
    using T = typename TileDataDst::DType;
    TFusedMulAddReluCheck<TileDataDst, TileDataSrc0, TileDataSrc1>(dst, src0, src1);
    constexpr unsigned blockSizeElem = BLOCK_BYTE_SIZE / sizeof(T);
    constexpr unsigned elementsPerRepeat = CCE_VL / sizeof(T);

    TFusedMulAddRelu<TileDataDst, TileDataSrc0, TileDataSrc1, elementsPerRepeat, blockSizeElem>(
        dst.data(), src0.data(), src1.data(), dst.GetValidRow(), dst.GetValidCol());
}
} // namespace pto

#endif
