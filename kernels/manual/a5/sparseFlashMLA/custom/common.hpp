/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef SPARSE_FLASH_MLA_CUSTOM_COMMON_HPP_
#define SPARSE_FLASH_MLA_CUSTOM_COMMON_HPP_

#include "../flash_mla_common.h"

namespace flash_mla_bf16 {
// Metadata byte layout is shared by its vector producer and DMA consumers.
constexpr uint32_t RAW_STRIDE = CACHE_RECORD_BYTES;
constexpr uint32_t INDEX_TOKENS = 256;
constexpr uint32_t INDEX_BATCH_TILES = INDEX_TOKENS / N;
constexpr uint32_t INDEX_OFFSET = INDEX_TOKENS * sizeof(uint32_t);
constexpr uint32_t INDEX_PAIR_OFFSET = INDEX_OFFSET + INDEX_TOKENS / 2 * sizeof(uint64_t);
constexpr uint32_t INDEX_ANY_OFFSET = INDEX_PAIR_OFFSET + INDEX_TOKENS / 2 * sizeof(uint64_t);
constexpr uint32_t INDEX_SLOT_STRIDE = (INDEX_ANY_OFFSET + INDEX_BATCH_TILES * sizeof(float) + 255) / 256 * 256;

#if defined(__CCE_AICORE__)
namespace custom {
// NORM row-state loads require readable upper-lane padding in the caller's
// arena; only AIV_M live lanes are consumed or written.
using RowStateTile = Tile<TileType::Vec, float, 1, AIV_M>;
using IndexMaskTile = Tile<TileType::Vec, float, 1, N>;
using SoftMaxScratchTile = Tile<TileType::Vec, float, 1, 128>;

} // namespace custom
#endif // __CCE_AICORE__
} // namespace flash_mla_bf16
#endif // SPARSE_FLASH_MLA_CUSTOM_COMMON_HPP_
