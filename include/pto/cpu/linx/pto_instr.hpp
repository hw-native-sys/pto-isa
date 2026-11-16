/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
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

namespace pto::linx {
// The imported simulator predates Linx CUBE packing. Reject layouts it cannot
// index faithfully rather than treating them as a zero-stride column layout.
template <typename T>
constexpr void validateCpuTile()
{
    static_assert(
        !T::IsCubeLayout && !T::isBoxedLayout, "Linx CPU-sim currently supports unboxed RowMajor/ColMajor tiles");
    static_assert(T::ValidRow > 0 && T::ValidCol > 0, "Linx CPU-sim currently requires static valid shapes");
    static_assert(T::ValidRow <= T::Rows && T::ValidCol <= T::Cols, "Linx CPU-sim valid shape exceeds physical shape");
}

template <typename T>
constexpr size_t tileOffset(size_t row, size_t col)
{
    validateCpuTile<T>();
    return row * T::RowStride + col * T::ColStride;
}
template <typename... Args>
PTO_INTERNAL void TADD_IMPL(Args&... args)
{
    (validateCpuTile<Args>(), ...);
    ::TADD_Impl(args...);
}
template <typename... Args>
PTO_INTERNAL void TABS_IMPL(Args&... args)
{
    (validateCpuTile<Args>(), ...);
    ::TABS_Impl(args...);
}
template <typename... Args>
PTO_INTERNAL void TAND_IMPL(Args&... args)
{
    (validateCpuTile<Args>(), ...);
    ::TAND_Impl(args...);
}
template <typename... Args>
PTO_INTERNAL void TOR_IMPL(Args&... args)
{
    (validateCpuTile<Args>(), ...);
    ::TOR_Impl(args...);
}
template <typename... Args>
PTO_INTERNAL void TSUB_IMPL(Args&... args)
{
    (validateCpuTile<Args>(), ...);
    ::TSUB_Impl(args...);
}
template <typename... Args>
PTO_INTERNAL void TMUL_IMPL(Args&... args)
{
    (validateCpuTile<Args>(), ...);
    ::TMUL_Impl(args...);
}
template <auto PrecisionType, typename... Args>
PTO_INTERNAL void TDIV_IMPL(Args&... args)
{
    static_assert(PrecisionType == DivAlgorithm::DEFAULT, "Linx supports only the default math algorithm");
    (validateCpuTile<Args>(), ...);
    ::TDIV_Impl(args...);
}
template <typename... Args>
PTO_INTERNAL void TMAX_IMPL(Args&... args)
{
    (validateCpuTile<Args>(), ...);
    ::TMAX_Impl(args...);
}
template <typename... Args>
PTO_INTERNAL void TMIN_IMPL(Args&... args)
{
    (validateCpuTile<Args>(), ...);
    ::TMIN_Impl(args...);
}
template <auto PrecisionType, typename... Args>
PTO_INTERNAL void TSQRT_IMPL(Args&... args)
{
    static_assert(PrecisionType == SqrtAlgorithm::DEFAULT, "Linx supports only the default math algorithm");
    (validateCpuTile<Args>(), ...);
    ::TSQRT_Impl(args...);
}
template <auto PrecisionType, typename... Args>
PTO_INTERNAL void TEXP_IMPL(Args&... args)
{
    static_assert(PrecisionType == ExpAlgorithm::DEFAULT, "Linx supports only the default math algorithm");
    (validateCpuTile<Args>(), ...);
    ::TEXP_Impl(args...);
}
template <typename T, typename G>
PTO_INTERNAL void TLOAD_IMPL(T& dst, G& src)
{
    validateCpuTile<T>();
    static_assert(std::is_same_v<typename T::DType, typename G::DType>);
    PTO_CPU_ASSERT(src.GetShape(0) == 1 && src.GetShape(1) == 1 && src.GetShape(2) == 1);
    PTO_CPU_ASSERT(src.GetShape(3) >= T::ValidRow && src.GetShape(4) >= T::ValidCol);
    for (size_t r = 0; r < T::ValidRow; ++r)
        for (size_t c = 0; c < T::ValidCol; ++c)
            dst.data()[tileOffset<T>(r, c)] = src.data()[r * src.GetStride(3) + c * src.GetStride(4)];
}

template <typename G, typename T>
PTO_INTERNAL void TSTORE_IMPL(G& dst, T& src)
{
    validateCpuTile<T>();
    static_assert(std::is_same_v<typename T::DType, typename G::DType>);
    PTO_CPU_ASSERT(dst.GetShape(0) == 1 && dst.GetShape(1) == 1 && dst.GetShape(2) == 1);
    PTO_CPU_ASSERT(dst.GetShape(3) >= T::ValidRow && dst.GetShape(4) >= T::ValidCol);
    for (size_t r = 0; r < T::ValidRow; ++r)
        for (size_t c = 0; c < T::ValidCol; ++c)
            dst.data()[r * dst.GetStride(3) + c * dst.GetStride(4)] = src.data()[tileOffset<T>(r, c)];
}

template <typename D, typename A, typename B>
PTO_INTERNAL void TCMP_IMPL(D& dst, A& src0, B& src1, CmpMode mode)
{
    static_assert(D::ValidRow == A::ValidRow && A::ValidRow == B::ValidRow);
    static_assert(D::ValidCol == A::ValidCol && A::ValidCol == B::ValidCol);
    PTO_CPU_ASSERT(is_valid_cmp_mode(mode));
    for (size_t r = 0; r < D::ValidRow; ++r) {
        for (size_t c = 0; c < D::ValidCol; ++c) {
            auto a = src0.data()[tileOffset<A>(r, c)];
            auto b = src1.data()[tileOffset<B>(r, c)];
            bool value = false;
            switch (mode) {
                case CmpMode::EQ:
                    value = a == b;
                    break;
                case CmpMode::NE:
                    value = a != b;
                    break;
                case CmpMode::LT:
                    value = a < b;
                    break;
                case CmpMode::GT:
                    value = a > b;
                    break;
                case CmpMode::LE:
                    value = a <= b;
                    break;
                case CmpMode::GE:
                    value = a >= b;
                    break;
            }
            dst.data()[tileOffset<D>(r, c)] = value;
        }
    }
}

template <typename T, typename M, typename Tmp>
PTO_INTERNAL void TSEL_IMPL(T& dst, M& mask, T& src0, T& src1, Tmp& tmp)
{
    (void)tmp;
    static_assert(M::ValidRow == T::ValidRow && M::ValidCol == T::ValidCol);
    for (size_t r = 0; r < T::ValidRow; ++r) {
        for (size_t c = 0; c < T::ValidCol; ++c) {
            auto index = tileOffset<T>(r, c);
            auto predicate = mask.data()[tileOffset<M>(r, c)];
            PTO_CPU_ASSERT(predicate == 0 || predicate == 1);
            dst.data()[index] = predicate ? src0.data()[index] : src1.data()[index];
        }
    }
}

template <typename D, typename S, typename Tmp>
PTO_INTERNAL void TTRANS_IMPL(D& dst, S& src, Tmp& tmp)
{
    (void)tmp;
    static_assert(D::ValidRow == S::ValidCol && D::ValidCol == S::ValidRow);
    S saved = src;
    for (size_t r = 0; r < D::ValidRow; ++r)
        for (size_t c = 0; c < D::ValidCol; ++c)
            dst.data()[tileOffset<D>(r, c)] = saved.data()[tileOffset<S>(c, r)];
}

template <typename C, typename A, typename B>
PTO_INTERNAL void TMATMUL_IMPL(C& dst, A& src0, B& src1)
{
    static_assert(A::ValidCol == B::ValidRow && C::ValidRow == A::ValidRow && C::ValidCol == B::ValidCol);
    C result;
    for (size_t r = 0; r < C::ValidRow; ++r) {
        for (size_t c = 0; c < C::ValidCol; ++c) {
            typename C::DType sum = 0;
            for (size_t k = 0; k < A::ValidCol; ++k)
                sum += static_cast<typename C::DType>(src0.data()[tileOffset<A>(r, k)]) *
                       static_cast<typename C::DType>(src1.data()[tileOffset<B>(k, c)]);
            result.data()[tileOffset<C>(r, c)] = sum;
        }
    }
    for (size_t r = 0; r < C::ValidRow; ++r)
        for (size_t c = 0; c < C::ValidCol; ++c)
            dst.data()[tileOffset<C>(r, c)] = result.data()[tileOffset<C>(r, c)];
}
} // namespace pto::linx
#include <pto/npu/linx/public_instr.hpp>
#endif
