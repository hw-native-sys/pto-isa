/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software; you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_CPU_LINX_INSTR_HPP
#define PTO_CPU_LINX_INSTR_HPP

#include <pto/cpu/linx/target.hpp>
#include <pto/npu/linx/impl/common/pto_tileop.hpp>

template <typename... Args>
inline void TADD(Args&&... args) {
  TADD_Impl(static_cast<Args&&>(args)...);
}

template <typename... Args>
inline void TMUL(Args&&... args) {
  TMUL_Impl(static_cast<Args&&>(args)...);
}

template <typename... Args>
inline void TSUB(Args&&... args) {
  TSUB_Impl(static_cast<Args&&>(args)...);
}

template <typename... Args>
inline void TCMP(Args&&... args) {
  TCMP_Impl(static_cast<Args&&>(args)...);
}

template <typename... Args>
inline void TSEL(Args&&... args) {
  TSELECT_Impl(static_cast<Args&&>(args)...);
}

template <typename... Args>
inline void TTRANS(Args&&... args) {
  TTRANS_Impl(static_cast<Args&&>(args)...);
}

template <typename... Args>
inline void TEXP(Args&&... args) {
  TEXP_Impl(static_cast<Args&&>(args)...);
}

template <typename... Args>
inline void TSQRT(Args&&... args) {
  TSQRT_Impl(static_cast<Args&&>(args)...);
}

template <typename... Args>
inline void TMAX(Args&&... args) {
  TMAX_Impl(static_cast<Args&&>(args)...);
}

template <typename... Args>
inline void TMIN(Args&&... args) {
  TMIN_Impl(static_cast<Args&&>(args)...);
}

template <typename... Args>
inline void TDIV(Args&&... args) {
  TDIV_Impl(static_cast<Args&&>(args)...);
}

namespace pto {
using ::TADD;
using ::TCMP;
using ::TDIV;
using ::TEXP;
using ::TLOAD;
using ::TMUL;
using ::TSTORE;
using ::TTRANS;
using ::TSEL;
using ::TSQRT;
using ::TSUB;
using ::TMAX;
using ::TMIN;
} // namespace pto

#endif // PTO_CPU_LINX_INSTR_HPP
