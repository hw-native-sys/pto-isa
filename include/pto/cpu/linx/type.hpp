/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_CPU_LINX_TYPE_HPP
#define PTO_CPU_LINX_TYPE_HPP

#include <array>
#include <bit>
#include <cstdint>
#include <cstring>
#include <type_traits>
// The native Linx compiler publishes this same spelling for its BF16 carrier.
#define __bf16 __blkc_bf16
#include <pto/npu/linx/impl/jcore/type.hpp>

static_assert(
    sizeof(unsigned long) == 8 && std::endian::native == std::endian::little,
    "Linx CPU-SIM requires a little-endian LP64 host");

template <int LogicalBytes, typename Scalar = uint32_t>
struct linx_tile_carrier {
    static_assert(LogicalBytes % sizeof(Scalar) == 0);
    static constexpr unsigned kElements = LogicalBytes / sizeof(Scalar);
    using ScalarType = Scalar;
    using RegisterType = std::array<Scalar, kElements>;
    RegisterType Register{};
};

namespace pto::linx_cpu {
template <typename Fragment, int ArrayRows, int ArrayCols, typename Carrier>
Carrier assembleTileArray(const Carrier& source)
{
    // CUBE fragments are already ordered by physical CELLs. Repacking them as
    // logical rows would corrupt M16/M32/N8 storage and borrowed range views.
    if constexpr (!Fragment::isRowMajor || Fragment::isBoxedLayout)
        return source;
    Carrier result{};
    const auto* in = reinterpret_cast<const unsigned char*>(&source);
    auto* out = reinterpret_cast<unsigned char*>(&result);
    // TileArray slots are compact fragments; the resulting RowMajor parent
    // concatenates columns within each row, rather than concatenating buffers.
    constexpr int width = Fragment::CubeElementBits;
    for (int ar = 0; ar < ArrayRows; ++ar)
        for (int ac = 0; ac < ArrayCols; ++ac)
            for (int r = 0; r < Fragment::Rows; ++r)
                for (int c = 0; c < Fragment::Cols * width; ++c) {
                    const auto src =
                        (ar * ArrayCols + ac) * Fragment::LogicalTileBytes * 8 + r * Fragment::Cols * width + c;
                    const auto dst = ((ar * Fragment::Rows + r) * ArrayCols + ac) * Fragment::Cols * width + c;
                    out[dst / 8] |= ((in[src / 8] >> (src % 8)) & 1U) << (dst % 8);
                }
    return result;
}
} // namespace pto::linx_cpu
#endif
