/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef TMOV_COMMON_MEMORY
#define TMOV_COMMON_MEMORY
#include <pto/common/utils.hpp>
#include "pto/common/arch/memory/textract_common.hpp"

template <typename DstTileData, typename SrcTileData>
PTO_INTERNAL void TMovToLeft(DstTileData& dst, SrcTileData& src)
{
    if constexpr (SrcTileData::Rows == 1 && SrcTileData::isRowMajor) {
        pto::TExtractToAVector<DstTileData, SrcTileData>(dst.data(), src.data(), 0, 0, dst.GetValidCol());
    } else if constexpr (DstTileData::SFractal == SrcTileData::SFractal) {
        if constexpr (DstTileData::Compact == pto::CompactMode::Normal) {
            pto::TExtractToACompact<DstTileData, SrcTileData, false>(
                dst.data(), src.data(), 0, 0, dst.GetValidRow(), dst.GetValidCol(), dst.GetKAligned());
        } else {
            pto::TExtractToA<DstTileData, SrcTileData, false>(dst.data(), src.data(), 0, 0);
        }
    } else {
        if constexpr (DstTileData::Compact == pto::CompactMode::Normal || sizeof(typename SrcTileData::DType) == 1) {
            pto::TExtractToACompact<DstTileData, SrcTileData, true>(
                dst.data(), src.data(), 0, 0, dst.GetValidRow(), dst.GetValidCol(), dst.GetKAligned());
        } else {
            pto::TExtractToA<DstTileData, SrcTileData, true>(dst.data(), src.data(), 0, 0);
        }
    }
}

template <typename DstTileData, typename SrcTileData>
PTO_INTERNAL void TMovToRight(DstTileData& dst, SrcTileData& src)
{
    if constexpr (DstTileData::SFractal == SrcTileData::SFractal) {
        if constexpr (DstTileData::Compact == pto::CompactMode::Normal) {
            pto::TExtractToBCompact<DstTileData, SrcTileData, false>(
                dst.data(), src.data(), 0, 0, dst.GetValidRow(), dst.GetValidCol());
        } else {
            pto::TExtractToB<DstTileData, SrcTileData, false>(dst.data(), src.data(), 0, 0);
        }
    } else {
        if constexpr (DstTileData::Compact == pto::CompactMode::Normal || sizeof(typename SrcTileData::DType) == 1) {
            pto::TExtractToBCompact<DstTileData, SrcTileData, true>(
                dst.data(), src.data(), 0, 0, dst.GetValidRow(), dst.GetValidCol());
        } else {
            pto::TExtractToB<DstTileData, SrcTileData, true>(dst.data(), src.data(), 0, 0);
        }
    }
}

template <typename DstTileData, typename SrcTileData>
PTO_INTERNAL void TMOV_CONVTILE_IMPL(DstTileData& dst, SrcTileData& src)
{
    if constexpr (SrcTileData::layout == pto::Layout::FRACTAL_Z) { // C1HWNC0, dst dim4 is c0Size
        pto::TExtractToBConv<DstTileData, SrcTileData>(
            dst.data(), src.data(), src.GetShape(3), dst.GetValidRow(), dst.GetValidCol(), 0, 0);
    }
}

template <typename FpTileData>
__tf__ PTO_INTERNAL void SetFPC(typename FpTileData::TileDType __in__ fp)
{
    __fbuf__ typename FpTileData::DType* dstAddrFp = (__fbuf__ typename FpTileData::DType*)__cce_get_tile_ptr(fp);
    uint64_t deqTensorAddr = ((uint64_t)dstAddrFp >> static_cast<uint64_t>(7))
                             << 8; // fpc[15:8] means Quant_PRE_ADDR, uint of 128(2^7)bytes
    set_fpc(deqTensorAddr);
}

#endif
