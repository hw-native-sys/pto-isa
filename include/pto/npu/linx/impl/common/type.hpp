/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

// Integrated from LinxISA/Linx-TileOP-API; see ../../README.md for provenance.

#ifndef PTO_LINX_VENDOR_INCLUDE_TYPE_H
#define PTO_LINX_VENDOR_INCLUDE_TYPE_H

template <typename E>
struct type_traits {
    static constexpr int bits = sizeof(E) * 8;
};

#ifdef __linx
#include <pto/npu/linx/impl/jcore/type.hpp>
#elif defined(__cpu_sim__)
// The Linx CPU simulator uses the same architectural type codes as the Linx
// compiler. Pull in the host storage fallback together with those codes so
// common Tile traits stay identical between NPU and CPU-sim builds.
#include <pto/npu/linx/impl/jcore/type.hpp>
#else
#error "__linx or __cpu_sim__ must be defined"
#endif

#endif
