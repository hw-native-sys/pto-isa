/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software; you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_NPU_LINX_INSTR_HPP
#define PTO_NPU_LINX_INSTR_HPP

#include <pto/npu/linx/target.hpp>
#include <pto/npu/linx/impl/common/pto_tileop.hpp>

// Linx TileOP keeps the implementation functions in the global namespace.
// Re-export the common operation names from namespace pto so existing PTO
// call sites can keep using `pto::TADD`, `pto::TMATMUL`, and similar APIs.
namespace pto {
using ::TADD;
using ::TAND;
using ::TCMP;
using ::TCVT;
using ::TDIV;
using ::TEXP;
using ::TGEMV;
using ::TLOAD;
using ::TMATMUL;
using ::TMUL;
using ::TSTORE;
using ::TTRANS;
using ::TSEL;
using ::TSQRT;
using ::TSUB;
using ::TMAX;
using ::TMIN;
} // namespace pto

#endif // PTO_NPU_LINX_INSTR_HPP
