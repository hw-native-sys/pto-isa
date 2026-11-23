/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef SPARSE_FLASH_MLA_CUSTOM_TSTORE_HPP_
#define SPARSE_FLASH_MLA_CUSTOM_TSTORE_HPP_

#include "../flash_mla_common.h"

#if defined(__CCE_AICORE__)
namespace flash_mla_bf16::custom {

namespace detail {
// A single-command 2D DMA store with the source's physical row pitch.
// PackedRows views a 1-row scalar buffer as contiguous short bursts
// (the LSE scatter); its row pitch equals the selected burst width.
template <bool PackedRows, typename GlobalData, typename SrcTile>
PTO_INTERNAL void TSTORE_IMPL(GlobalData& dst, SrcTile& src)
{
    static_assert(SrcTile::Loc == TileType::Vec && SrcTile::isRowMajor);
    static_assert(std::is_same_v<typename GlobalData::RawDType, typename SrcTile::DType>);
    static_assert(
        GlobalData::staticShape[0] == 1 && GlobalData::staticShape[1] == 1 && GlobalData::staticShape[2] == 1);
    if constexpr (PackedRows) {
        static_assert(SrcTile::Rows == 1 && std::is_same_v<typename SrcTile::DType, float>);
    }
    const uint32_t rows = dst.GetShape(GlobalTensorDim::DIM_3);
    const uint32_t cols = dst.GetShape(GlobalTensorDim::DIM_4);
    const uint32_t sourcePitch = PackedRows ? cols : SrcTile::Cols;
    pto::TStoreInstr<SrcTile, GlobalData>(
        dst.data(), src.data(), rows, cols * sizeof(typename SrcTile::DType),
        static_cast<uint64_t>(dst.GetStride(GlobalTensorDim::DIM_3)) * sizeof(typename SrcTile::DType),
        sourcePitch * sizeof(typename SrcTile::DType));
}

} // namespace detail

template <bool PackedRows = false, typename GlobalData, typename SrcTile, typename... WaitEvents>
PTO_INST RecordEvent TSTORE(GlobalData& dst, SrcTile& src, WaitEvents&... events)
{
    WaitAllEvents(events...);
    detail::TSTORE_IMPL<PackedRows>(dst, src);
    return {};
}

} // namespace flash_mla_bf16::custom
#endif // __CCE_AICORE__
#endif // SPARSE_FLASH_MLA_CUSTOM_TSTORE_HPP_
