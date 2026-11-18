/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under
the terms and conditions of CANN Open Software License Agreement Version 2.0
(the "License"). Please refer to the License for details. You may not use this
file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON AN "AS
IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A
PARTICULAR PURPOSE. See LICENSE in the root of the software repository for the
full text of the License.
*/

#ifndef __PTO_RESHAPE__
#define __PTO_RESHAPE__

#include "pto/common/pto_tile.hpp"
#include <cstddef>

namespace pto {
template <typename TileDataOut, typename TileDataIn>
PTO_INTERNAL void TRESHAPE_IMPL(TileDataOut& dst, TileDataIn& src)
{
    static_assert(is_tile_data_v<TileDataIn>, "input must be a Tile instance.");
    static_assert(is_tile_data_v<TileDataOut>, "output must be a Tile instance.");

    constexpr TileType Loc = TileDataIn::Loc;
    constexpr TileType NewLoc = TileDataOut::Loc;
    constexpr SLayout SFractal = TileDataIn::SFractal;
    constexpr SLayout NewSFractal = TileDataOut::SFractal;

    // 1. TileType must match
    static_assert(Loc == NewLoc, "TRESHAPE: Source and target TileType must be same.");
    // 2. Compare physical storage, including packed elements and panel padding.
    static_assert(TileDataIn::StorageBytes == TileDataOut::StorageBytes, "TRESHAPE: Total byte size must match.");
    // 3. reshape between non-boxed and boxed tile is not allowed.
    static_assert(
        (SFractal == SLayout::NoneBox && NewSFractal == SLayout::NoneBox) ||
            (SFractal != SLayout::NoneBox && NewSFractal != SLayout::NoneBox),
        "TRESHAPE: Cannot reshape between boxed and non-boxed layouts.");

#ifdef __CPU_SIM
    dst.data() = reinterpret_cast<typename TileDataOut::DType*>(src.data());
#else
    const std::byte* src_bytes = reinterpret_cast<const std::byte*>(src.data());
    std::byte* dst_bytes = reinterpret_cast<std::byte*>(dst.data());

    for (std::size_t i = 0; i < TileDataIn::StorageBytes; ++i) {
        dst_bytes[i] = src_bytes[i];
    }
#endif
}
} // namespace pto

#endif
