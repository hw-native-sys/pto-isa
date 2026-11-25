/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef SPARSE_FLASH_MLA_CUSTOM_MGATHER_HPP_
#define SPARSE_FLASH_MLA_CUSTOM_MGATHER_HPP_

#include "../flash_mla_common.h"

#if defined(__CCE_AICORE__)
namespace flash_mla_bf16::custom {

namespace detail {
// Paired gather32 DMA: VF-produced offset/control tiles are a descriptor
// contract, including duplicate/single-valid pairs and zero-burst invalids.
// Four descriptors are loaded before issuing their four DMA copies.
template <typename RawTile, typename CacheTensor, typename OffsetTile, typename ControlTile>
PTO_INTERNAL void MGATHER_IMPL(RawTile& dst, CacheTensor& cache, OffsetTile& offsets, ControlTile& controls)
{
    static_assert(RawTile::Loc == TileType::Vec && RawTile::Rows == 1);
    static_assert(std::is_same_v<typename RawTile::DType, uint8_t>);
    static_assert(RawTile::Cols == K_CHUNK * CACHE_RECORD_BYTES);
    static_assert(is_global_data_v<CacheTensor> && std::is_same_v<typename CacheTensor::RawDType, uint8_t>);
    static_assert(OffsetTile::Loc == TileType::Vec && ControlTile::Loc == TileType::Vec);
    static_assert(OffsetTile::Rows == 1 && ControlTile::Rows == 1);
    static_assert(std::is_same_v<typename OffsetTile::DType, uint64_t>);
    static_assert(std::is_same_v<typename ControlTile::DType, uint64_t>);
    static_assert(OffsetTile::Cols == K_CHUNK / 2 && ControlTile::Cols == K_CHUNK / 2);
    constexpr uint64_t COPY_CONFIG = uint64_t{CACHE_RECORD_BYTES} << 25;
#pragma unroll 1
    for (uint32_t pair = 0; pair < K_CHUNK / 2; pair += 4) {
        uint64_t control[4], first[4];
#pragma unroll
        for (uint32_t i = 0; i < 4; ++i) {
            control[i] = controls.data()[pair + i];
            first[i] = offsets.data()[pair + i];
        }
#pragma unroll
        for (uint32_t i = 0; i < 4; ++i) {
            copy_gm_to_ubuf_align_v2(
                dst.data() + (pair + i) * 2 * CACHE_RECORD_BYTES, cache.data() + first[i],
                COPY_CONFIG | ((control[i] >> 58) & 0x20u), control[i] & ((uint64_t{1} << 61) - 1));
        }
    }
}

} // namespace detail

template <typename RawTile, typename CacheTensor, typename OffsetTile, typename ControlTile, typename... WaitEvents>
PTO_INST RecordEvent
MGATHER(RawTile& dst, CacheTensor& cache, OffsetTile& offsets, ControlTile& controls, WaitEvents&... events)
{
    WaitAllEvents(events...);
    detail::MGATHER_IMPL(dst, cache, offsets, controls);
    return {};
}

} // namespace flash_mla_bf16::custom
#endif // __CCE_AICORE__
#endif // SPARSE_FLASH_MLA_CUSTOM_MGATHER_HPP_
