/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
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

namespace pto::linx {
template <typename... Args>
PTO_INTERNAL void TADD_IMPL(Args&... args)
{
    ::LINX_TADD(args...);
}
template <typename... Args>
PTO_INTERNAL void TABS_IMPL(Args&... args)
{
    ::LINX_TABS(args...);
}
template <typename... Args>
PTO_INTERNAL void TAND_IMPL(Args&... args)
{
    ::LINX_TAND(args...);
}
template <typename... Args>
PTO_INTERNAL void TOR_IMPL(Args&... args)
{
    ::LINX_TOR(args...);
}
template <typename... Args>
PTO_INTERNAL void TSUB_IMPL(Args&... args)
{
    ::LINX_TSUB(args...);
}
template <typename... Args>
PTO_INTERNAL void TMUL_IMPL(Args&... args)
{
    ::LINX_TMUL(args...);
}
template <auto PrecisionType, typename... Args>
PTO_INTERNAL void TDIV_IMPL(Args&... args)
{
    static_assert(PrecisionType == DivAlgorithm::DEFAULT, "Linx supports only the default math algorithm");
    ::LINX_TDIV(args...);
}
template <typename... Args>
PTO_INTERNAL void TMAX_IMPL(Args&... args)
{
    ::LINX_TMAX(args...);
}
template <typename... Args>
PTO_INTERNAL void TMIN_IMPL(Args&... args)
{
    ::LINX_TMIN(args...);
}
template <auto PrecisionType, typename... Args>
PTO_INTERNAL void TSQRT_IMPL(Args&... args)
{
    static_assert(PrecisionType == SqrtAlgorithm::DEFAULT, "Linx supports only the default math algorithm");
    ::LINX_TSQRT(args...);
}
template <auto PrecisionType, typename... Args>
PTO_INTERNAL void TEXP_IMPL(Args&... args)
{
    static_assert(PrecisionType == ExpAlgorithm::DEFAULT, "Linx supports only the default math algorithm");
    ::LINX_TEXP(args...);
}
// Native TLSU uses the row stride but transfers contiguous columns. A dynamic
// column stride cannot establish this contract, even when its runtime value is 1.
template <typename T, typename G>
PTO_INTERNAL void TLOAD_IMPL(T& dst, G& src)
{
    static_assert(G::staticStride[4] == 1, "Linx device TLOAD requires a static GM column stride of 1");
    ::LINX_TLOAD(dst, src);
}
template <typename G, typename T>
PTO_INTERNAL void TSTORE_IMPL(G& dst, T& src)
{
    static_assert(G::staticStride[4] == 1, "Linx device TSTORE requires a static GM column stride of 1");
    ::LINX_TSTORE(dst, src);
}
template <typename... Args>
PTO_INTERNAL void TMATMUL_IMPL(Args&... args)
{
    ::LINX_TMATMUL(args...);
}
template <typename D, typename A, typename B>
PTO_INTERNAL void TCMP_IMPL(D& dst, A& src0, B& src1, CmpMode mode)
{
    switch (mode) {
        case CmpMode::EQ:
            ::LINX_TCMP<CmpMode::EQ>(dst, src0, src1);
            return;
        case CmpMode::NE:
            ::LINX_TCMP<CmpMode::NE>(dst, src0, src1);
            return;
        case CmpMode::LT:
            ::LINX_TCMP<CmpMode::LT>(dst, src0, src1);
            return;
        case CmpMode::GT:
            ::LINX_TCMP<CmpMode::GT>(dst, src0, src1);
            return;
        case CmpMode::LE:
            ::LINX_TCMP<CmpMode::LE>(dst, src0, src1);
            return;
        case CmpMode::GE:
            ::LINX_TCMP<CmpMode::GE>(dst, src0, src1);
            return;
    }
    __builtin_trap();
}

template <typename T, typename M, typename Tmp>
PTO_INTERNAL void TSEL_IMPL(T& dst, M& mask, T& src0, T& src1, Tmp& tmp)
{
    // Preserve both sources when dst aliases either one. Native TSEL takes
    // its false input from the prior destination register.
    (void)tmp;
    T result = src1;
    ::LINX_TSEL(result, mask, src0);
    dst = result;
}

template <typename D, typename S, typename Tmp>
PTO_INTERNAL void TTRANS_IMPL(D& dst, S& src, Tmp& tmp)
{
    (void)tmp;
    ::LINX_TTRANS(dst, src);
}
} // namespace pto::linx
#include <pto/npu/linx/public_instr.hpp>
#endif
