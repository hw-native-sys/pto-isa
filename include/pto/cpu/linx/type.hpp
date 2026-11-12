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
#endif
