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

#ifndef PTO_LINX_VENDOR_PTO_TILE_HPP
#define PTO_LINX_VENDOR_PTO_TILE_HPP

#include "pto/npu/linx/impl/common/layout.hpp"
#include <pto/npu/linx/impl/common/type.hpp>
#include <cstdint>
#include <pto/common/comparison.hpp>
#include <type_traits>

namespace pto {

template <typename Element>
inline constexpr int tile_element_bits_v =
    type_traits<Element>::TypeCode == __type_fp4_e2m1x2 || type_traits<Element>::TypeCode == __type_fp4_e1m2x2 ||
            type_traits<Element>::TypeCode == __type_fp4_hif4x2 || type_traits<Element>::TypeCode == __type_int4x2 ||
            type_traits<Element>::TypeCode == __type_uint4x2 ?
        4 :
    (type_traits<Element>::TypeCode == __type_fp6_e3m2 || type_traits<Element>::TypeCode == __type_fp5_e2m3) ?
        8 :
        type_traits<Element>::bits;

// PTO-ISA GMOVTypeLegal. Keep this based on architectural TypeCode values,
// not C++ sizeof: several packed formats use an 8-bit carrier but have their
// own legal GMOV encodings.
constexpr bool is_gmov_type_code(int type_code)
{
    switch (type_code) {
        case __type_fp32:
        case __type_tf32:
        case __type_hf32:
        case __type_fp16:
        case __type_bf16:
        case __type_hif8:
        case __type_fp8_e4m3:
        case __type_fp8_e5m2:
        case __type_fp6_e3m2:
        case __type_fp5_e2m3:
        case __type_fp4_e2m1x2:
        case __type_fp4_e1m2x2:
        case __type_fp8_e8m0:
        case __type_fp4_hif4x2:
        case __type_fp8_e6m2:
        case __type_int32:
        case __type_int16:
        case __type_int8:
        case __type_int4x2:
        case __type_uint32:
        case __type_uint16:
        case __type_uint8:
        case __type_uint4x2:
            return true;
        default:
            return false;
    }
}

/// Compile-time validity check for the six ISA comparison modes. Rejects any
/// out-of-range value that a bogus static_cast would otherwise smuggle into
/// the B.DATR CMode field.
constexpr bool is_valid_cmp_mode(CmpMode Mode)
{
    switch (Mode) {
        case CmpMode::EQ:
        case CmpMode::NE:
        case CmpMode::LT:
        case CmpMode::GT:
        case CmpMode::LE:
        case CmpMode::GE:
            return true;
    }
    return false;
}

// Public PTO enum values are stable across targets. Linx encodes GT/LE in
// the opposite order, so lower by semantic mode instead of casting blindly.
constexpr unsigned cmp_mode_code(CmpMode mode)
{
    switch (mode) {
        case CmpMode::EQ:
            return 0;
        case CmpMode::NE:
            return 1;
        case CmpMode::LT:
            return 2;
        case CmpMode::GT:
            return 3;
        case CmpMode::LE:
            return 4;
        case CmpMode::GE:
            return 5;
    }
    __builtin_trap();
}

// Rounding modes for TQUANT/TDEQUANT. The values are PTO bundle RMode
// encodings; encoding zero selects the operation default, which is RNE here.
enum class RoundMode : uint8_t {
    RNE = 0, ///< round to nearest even / operation default
    RTZ = 2, ///< round toward zero
    RTM = 3, ///< round toward minus infinity
    RTP = 4, ///< round toward plus infinity
    RNA = 5, ///< round to nearest, ties away
    RTO = 6, ///< round to odd
    RHB = 7, ///< reciprocal-half bias

    // Compatibility aliases for the former direction-based names.
    RNONE = RNE,
    RDN = RTM,
    RUP = RTP,
};

constexpr bool is_valid_round_mode(RoundMode Mode)
{
    switch (Mode) {
        case RoundMode::RNE:
        case RoundMode::RTZ:
        case RoundMode::RTM:
        case RoundMode::RTP:
        case RoundMode::RNA:
        case RoundMode::RTO:
        case RoundMode::RHB:
            return true;
    }
    return false;
}

/// Padding Value : keep SAME with asm encoding
enum class PadValue {
    Zero = 0,
    Max = 1,
    Min = 2,
    Null = 3,
};

enum class FixpPreQuantMode : uint8_t {
    None = 0,
    F322F16 = 1,
    VREQS8Pre = 2,
    REQS8Pre = 3,
    VDEQF16 = 4,
    DEQF16 = 5,
    VSHIFTS322S16 = 12,
    SHIFTS322S16 = 13,
    F322BF16 = 16,
    QF322S4Pre = 17,
    VQF322S4Pre = 18,
    QF322S16Pre = 19,
    VQF322S16Pre = 20,
    VQF322S8Pre = 23,
    QF322S8Pre = 24,
    QF322HIF8Pre = 25,
    QF322FP8Pre = 26,
    QF322F32Pre = 27,
    VQF322HIF8Pre = 28,
    QF322F16Pre = 32,
    VQF322F16Pre = 33,
    QF322BF16Pre = 34,
    QS322BF16Pre = 35,
    VQF322BF16Pre = 36,
    VQF322FP8Pre = 37,
    VQF322F32Pre = 38,
    VQS322BF16Pre = 39,
};

enum class FixpReluMode : uint8_t {
    None = 0,
    Relu = 1,
    LRelu = 2,
    PRelu = 3,
};

// CUBE InternalAcc control (PTO-ISA 0.58.6, spec#236): encoded into the
// B.DATR PadValueOrByteId[1:0] field of Matrix/CUBE bundles only.
//   0 = None (final D output, no hint)  [default]
//   1 = RawAccumulator (CCTRL[0]: raw accumulator-type D; forbids final
//       post-process except legal CScale)
//   2 = InternalAccHint (CCTRL[1]: ACC-only transparent cache-use/prefetch
//       hint on explicit C)
//   3 = RawAccumulator | InternalAccHint
enum CubeControl : uint8_t {
    CubeCtrlNone = 0,
    CubeCtrlRawAccumulator = 1,
    CubeCtrlInternalAccHint = 2,
    CubeCtrlRawAccAndHint = 3,
};

struct FixpAttr {
    FixpPreQuantMode PreQuant = FixpPreQuantMode::None;
    FixpReluMode Relu = FixpReluMode::None;
    uint8_t GroupNCode = 0;
    bool RowMaxEn = false;
    bool GroupMaxEn = false;
    bool RowMaxInit = false;
    bool MaxAbsEn = false;
    bool TransA = false;
    bool TransB = false;
    bool CScaleEn = false;
    CubeControl CubeCtrl = CubeCtrlNone;

    // CUBE InternalAcc control setters (PTO-ISA 0.58.6, spec#236). CCTRL[0]
    // (raw accumulator D) forbids final post-process; CCTRL[1] (ACC-only
    // transparent cache hint) is validated against the operation kind at the
    // wrapper layer.
    // Chainable: preserves all other attributes, unlike the parameter-free
    // static constructors (f16/bf16/keep_acc) which start from defaults.
    constexpr FixpAttr with_cube_ctrl(CubeControl Ctrl) const
    {
        FixpAttr Attr = *this;
        Attr.CubeCtrl = Ctrl;
        return Attr;
    }
    static constexpr FixpAttr keep_acc(FixpReluMode ReluMode = FixpReluMode::None)
    {
        FixpAttr Attr;
        Attr.Relu = ReluMode;
        return Attr;
    }

    static constexpr FixpAttr f16(FixpReluMode ReluMode = FixpReluMode::None)
    {
        FixpAttr Attr;
        Attr.PreQuant = FixpPreQuantMode::F322F16;
        Attr.Relu = ReluMode;
        return Attr;
    }

    static constexpr FixpAttr bf16(FixpReluMode ReluMode = FixpReluMode::None)
    {
        FixpAttr Attr;
        Attr.PreQuant = FixpPreQuantMode::F322BF16;
        Attr.Relu = ReluMode;
        return Attr;
    }

    constexpr FixpAttr transpose_a(bool Enable = true) const
    {
        FixpAttr Attr = *this;
        Attr.TransA = Enable;
        return Attr;
    }

    constexpr FixpAttr transpose_b(bool Enable = true) const
    {
        FixpAttr Attr = *this;
        Attr.TransB = Enable;
        return Attr;
    }

    constexpr FixpAttr cscale_enable(bool Enable = true) const
    {
        FixpAttr Attr = *this;
        Attr.CScaleEn = Enable;
        return Attr;
    }

    constexpr uint32_t encoding() const
    {
        return (static_cast<uint32_t>(PreQuant) << 26) | (static_cast<uint32_t>(Relu) << 23) |
               (static_cast<uint32_t>(GroupNCode) << 19) | (static_cast<uint32_t>(RowMaxEn) << 18) |
               (static_cast<uint32_t>(GroupMaxEn) << 17) | (static_cast<uint32_t>(RowMaxInit) << 16) |
               (static_cast<uint32_t>(MaxAbsEn) << 15) | (static_cast<uint32_t>(TransB) << 8) |
               (static_cast<uint32_t>(TransA) << 7) | (static_cast<uint32_t>(CScaleEn) << 9) | 0x2023;
    }

    constexpr bool operator==(const FixpAttr&) const = default;
};

// PTO-ISA #260 (TileCarrierWidthCompatible): a comparison/select source
// may back data of a different type than the operation type when element
// widths are equal and neither side is a packed 4-bit type. The packed
// containers (FP4X2/S4X2/U4X2) carry two elements per 8-bit lane and are
// excluded from reinterpretation.
constexpr int type_traits_code_bits(int Code)
{
    switch (Code) {
        case __type_fp64:
        case __type_int64:
        case __type_uint64:
            return 64;
        case __type_fp32:
        case __type_tf32:
        case __type_hf32:
        case __type_int32:
        case __type_uint32:
            return 32;
        case __type_fp16:
        case __type_bf16:
        case __type_int16:
        case __type_uint16:
            return 16;
        case __type_hif8:
        case __type_fp8_e4m3:
        case __type_fp8_e5m2:
        case __type_int8:
        case __type_uint8:
        case __type_fp8_e8m0:
        case __type_fp8_e6m2:
        case __type_rcpe6m2:
            return 8;
        default:
            return -1;
    }
}

constexpr bool tile_carrier_width_compatible(int StoredCode, int OperationCode)
{
    if (StoredCode == OperationCode)
        return true;
    constexpr int Packed4BitCodes[] = {
        __type_fp4_e2m1x2, __type_fp4_e1m2x2, __type_fp4_hif4x2, __type_int4x2, __type_uint4x2};
    for (int Code : Packed4BitCodes) {
        if (StoredCode == Code || OperationCode == Code)
            return false;
    }
    return type_traits_code_bits(StoredCode) == type_traits_code_bits(OperationCode);
}

// PTO-ISA v0.58 (pto-spec#322) TCVT datatype-pair legality for the derived
// RCPE6M2 type. RCPE6M2 (code 21) reuses the E6M2 code space as a source-only
// reciprocal type: it has no destination encoding, and the only legal TCVT
// destinations for an RCPE6M2 source are FP16 and BF16. All other pairs are
// left permissive here so existing conversions (including E6M2, which is not
// narrowed) remain unaffected.
constexpr bool is_legal_tcvt_datatype_pair(int SrcCode, int DstCode)
{
    if (DstCode == __type_rcpe6m2)
        return false; // RCPE6M2 has no destination encoding.
    if (SrcCode == __type_rcpe6m2)
        return DstCode == __type_fp16 || DstCode == __type_bf16;
    return true;
}

// PTO-SPEC #388: explicit nonzero TCVT rounding modes round same-type
// floating operands to an integer-valued result. RTO is not applicable to
// this specialization. Compare operation TypeCodes, rather than C++ or
// backing-carrier types, so reinterpret_tile views follow the ISA rule.
constexpr bool is_tcvt_same_type_round_integral_fp(int SrcCode, int DstCode)
{
    return SrcCode == DstCode &&
           (SrcCode == __type_fp16 || SrcCode == __type_bf16 || SrcCode == __type_fp32 || SrcCode == __type_fp64);
}

constexpr bool is_valid_fixp_pre_quant(FixpPreQuantMode Mode)
{
    switch (Mode) {
        case FixpPreQuantMode::None:
        case FixpPreQuantMode::F322F16:
        case FixpPreQuantMode::VREQS8Pre:
        case FixpPreQuantMode::REQS8Pre:
        case FixpPreQuantMode::VDEQF16:
        case FixpPreQuantMode::DEQF16:
        case FixpPreQuantMode::VSHIFTS322S16:
        case FixpPreQuantMode::SHIFTS322S16:
        case FixpPreQuantMode::F322BF16:
        case FixpPreQuantMode::QF322S4Pre:
        case FixpPreQuantMode::VQF322S4Pre:
        case FixpPreQuantMode::QF322S16Pre:
        case FixpPreQuantMode::VQF322S16Pre:
        case FixpPreQuantMode::VQF322S8Pre:
        case FixpPreQuantMode::QF322S8Pre:
        case FixpPreQuantMode::QF322HIF8Pre:
        case FixpPreQuantMode::QF322FP8Pre:
        case FixpPreQuantMode::QF322F32Pre:
        case FixpPreQuantMode::VQF322HIF8Pre:
        case FixpPreQuantMode::QF322F16Pre:
        case FixpPreQuantMode::VQF322F16Pre:
        case FixpPreQuantMode::QF322BF16Pre:
        case FixpPreQuantMode::QS322BF16Pre:
        case FixpPreQuantMode::VQF322BF16Pre:
        case FixpPreQuantMode::VQF322FP8Pre:
        case FixpPreQuantMode::VQF322F32Pre:
        case FixpPreQuantMode::VQS322BF16Pre:
            return true;
    }
    return false;
}

constexpr bool is_valid_fixp_attr(FixpAttr Attr)
{
    if (!is_valid_fixp_pre_quant(Attr.PreQuant) ||
        static_cast<uint8_t>(Attr.Relu) > static_cast<uint8_t>(FixpReluMode::PRelu) || Attr.GroupNCode > 9)
        return false;
    if (!Attr.RowMaxEn && Attr.RowMaxInit)
        return false;
    if (Attr.GroupMaxEn != (Attr.GroupNCode != 0))
        return false;
    if (!Attr.RowMaxEn && !Attr.GroupMaxEn && Attr.MaxAbsEn)
        return false;
    return true;
}

constexpr bool is_basic_fixp_attr(FixpAttr Attr)
{
    const bool ParameterFreePreQuant = Attr.PreQuant == FixpPreQuantMode::None ||
                                       Attr.PreQuant == FixpPreQuantMode::F322F16 ||
                                       Attr.PreQuant == FixpPreQuantMode::F322BF16;
    const bool ParameterFreeRelu = Attr.Relu == FixpReluMode::None || Attr.Relu == FixpReluMode::Relu;
    return ParameterFreePreQuant && ParameterFreeRelu && !Attr.RowMaxEn && !Attr.GroupMaxEn && !Attr.RowMaxInit &&
           !Attr.MaxAbsEn;
}

constexpr bool is_parameter_free_fixp_pre_quant(FixpPreQuantMode Mode)
{
    return Mode == FixpPreQuantMode::None || Mode == FixpPreQuantMode::F322F16 || Mode == FixpPreQuantMode::F322BF16;
}

constexpr bool is_scalar_fixp_pre_quant(FixpPreQuantMode Mode)
{
    switch (Mode) {
        case FixpPreQuantMode::REQS8Pre:
        case FixpPreQuantMode::DEQF16:
        case FixpPreQuantMode::SHIFTS322S16:
        case FixpPreQuantMode::QF322S4Pre:
        case FixpPreQuantMode::QF322S16Pre:
        case FixpPreQuantMode::QF322S8Pre:
        case FixpPreQuantMode::QF322HIF8Pre:
        case FixpPreQuantMode::QF322FP8Pre:
        case FixpPreQuantMode::QF322F32Pre:
        case FixpPreQuantMode::QF322F16Pre:
        case FixpPreQuantMode::QF322BF16Pre:
        case FixpPreQuantMode::QS322BF16Pre:
            return true;
        default:
            return false;
    }
}

constexpr bool is_vector_fixp_pre_quant(FixpPreQuantMode Mode)
{
    switch (Mode) {
        case FixpPreQuantMode::VREQS8Pre:
        case FixpPreQuantMode::VDEQF16:
        case FixpPreQuantMode::VSHIFTS322S16:
        case FixpPreQuantMode::VQF322S4Pre:
        case FixpPreQuantMode::VQF322S16Pre:
        case FixpPreQuantMode::VQF322S8Pre:
        case FixpPreQuantMode::VQF322HIF8Pre:
        case FixpPreQuantMode::VQF322F16Pre:
        case FixpPreQuantMode::VQF322BF16Pre:
        case FixpPreQuantMode::VQF322FP8Pre:
        case FixpPreQuantMode::VQF322F32Pre:
        case FixpPreQuantMode::VQS322BF16Pre:
            return true;
        default:
            return false;
    }
}

enum class MatrixNumericClass : uint8_t {
    Unsupported,
    Floating,
    Signed,
    Unsigned,
};

constexpr MatrixNumericClass matrix_numeric_class(int TypeCode)
{
    switch (TypeCode) {
        case __type_fp32:
        case __type_tf32:
        case __type_hf32:
        case __type_fp16:
        case __type_bf16:
        case __type_hif8:
        case __type_fp8_e4m3:
        case __type_fp8_e5m2:
        case __type_fp6_e3m2:
        case __type_fp5_e2m3:
        case __type_fp4_e2m1x2:
        case __type_fp4_e1m2x2:
            return MatrixNumericClass::Floating;
        case __type_int16:
        case __type_int8:
        case __type_int4x2:
            return MatrixNumericClass::Signed;
        case __type_uint16:
        case __type_uint8:
        case __type_uint4x2:
            return MatrixNumericClass::Unsigned;
        default:
            return MatrixNumericClass::Unsupported;
    }
}

constexpr bool matrix_mx_input_supported(int TypeCode)
{
    return TypeCode == __type_fp16 || TypeCode == __type_bf16 || TypeCode == __type_fp8_e4m3 ||
           TypeCode == __type_fp8_e5m2 || TypeCode == __type_fp4_e2m1x2 || TypeCode == __type_fp4_e1m2x2 ||
           TypeCode == __type_fp4_hif4x2;
}

constexpr bool matrix_mx_input_needs_scale(int TypeCode)
{
    return TypeCode == __type_fp8_e4m3 || TypeCode == __type_fp8_e5m2 || TypeCode == __type_fp4_e2m1x2 ||
           TypeCode == __type_fp4_e1m2x2 || TypeCode == __type_fp4_hif4x2;
}

// Each Matrix-MX primary selects its scale contract independently.  HiF4X2
// uses a raw U32 scale for every 64 logical K elements; the other scaled MX
// carriers use E8M0 for every 32 logical K elements.
constexpr int matrix_mx_scale_carrier_type(int TypeCode)
{
    return TypeCode == __type_fp4_hif4x2 ? __type_uint32 : __type_fp8_e8m0;
}

constexpr int matrix_mx_scale_group_size(int TypeCode) { return TypeCode == __type_fp4_hif4x2 ? 64 : 32; }

constexpr int matrix_accumulator_type_code(int InputTypeCode)
{
    switch (matrix_numeric_class(InputTypeCode)) {
        case MatrixNumericClass::Floating:
            return __type_fp32;
        case MatrixNumericClass::Signed:
            return __type_int32;
        case MatrixNumericClass::Unsigned:
            return __type_uint32;
        default:
            return -1;
    }
}

// PTO #346: RowMax/GroupMax reduce the final post-processed D value.  Their
// reduction carrier is therefore limited to the effective floating D types;
// accumulator integer types and low-precision encoded types are not legal
// reduction destinations.
constexpr bool matrix_final_d_reduction_type_legal(int TypeCode)
{
    return TypeCode == __type_fp32 || TypeCode == __type_fp16 || TypeCode == __type_bf16;
}
constexpr bool matrix_mode_uses_s32_accumulator(FixpPreQuantMode Mode)
{
    switch (Mode) {
        case FixpPreQuantMode::VREQS8Pre:
        case FixpPreQuantMode::REQS8Pre:
        case FixpPreQuantMode::VDEQF16:
        case FixpPreQuantMode::DEQF16:
        case FixpPreQuantMode::VSHIFTS322S16:
        case FixpPreQuantMode::SHIFTS322S16:
        case FixpPreQuantMode::QF322S4Pre:
        case FixpPreQuantMode::VQF322S4Pre:
        case FixpPreQuantMode::QF322S16Pre:
        case FixpPreQuantMode::VQF322S16Pre:
        case FixpPreQuantMode::QS322BF16Pre:
        case FixpPreQuantMode::VQS322BF16Pre:
            return true;
        default:
            return false;
    }
}

template <typename Left, typename Right, bool MX = false>
constexpr bool matrix_input_pair_legal()
{
    constexpr int LeftCode = type_traits<typename Left::DType>::TypeCode;
    constexpr int RightCode = type_traits<typename Right::DType>::TypeCode;
    if constexpr (MX)
        return matrix_mx_input_supported(LeftCode) && matrix_mx_input_supported(RightCode);
    constexpr MatrixNumericClass LeftClass = matrix_numeric_class(LeftCode);
    return LeftClass != MatrixNumericClass::Unsupported && LeftClass == matrix_numeric_class(RightCode);
}

template <FixpAttr Attr, typename Left, typename Right, bool MX = false>
constexpr bool matrix_accumulator_mode_legal()
{
    if constexpr (!matrix_input_pair_legal<Left, Right, MX>())
        return false;
    constexpr int AccCode =
        MX ? __type_fp32 : matrix_accumulator_type_code(type_traits<typename Left::DType>::TypeCode);
    if constexpr (Attr.PreQuant == FixpPreQuantMode::None)
        return AccCode == __type_fp32 || AccCode == __type_int32 || AccCode == __type_uint32;
    if constexpr (matrix_mode_uses_s32_accumulator(Attr.PreQuant))
        return AccCode == __type_int32;
    return AccCode == __type_fp32;
}

template <FixpAttr Attr, typename DType>
constexpr bool is_fixp_output_type();

template <FixpAttr Attr, typename Left, typename Right, typename Output, bool MX = false>
constexpr bool matrix_output_type_legal()
{
    if constexpr (!matrix_accumulator_mode_legal<Attr, Left, Right, MX>())
        return false;
    constexpr int OutputCode = type_traits<typename Output::DType>::TypeCode;
    if constexpr (Attr.PreQuant == FixpPreQuantMode::None) {
        constexpr int AccCode =
            MX ? __type_fp32 : matrix_accumulator_type_code(type_traits<typename Left::DType>::TypeCode);
        return OutputCode == AccCode;
    }
    return is_fixp_output_type<Attr, typename Output::DType>();
}

template <typename Left, typename Right, typename Accumulator, bool MX = false>
constexpr bool matrix_accumulator_type_legal()
{
    if constexpr (!matrix_input_pair_legal<Left, Right, MX>())
        return false;
    constexpr int Expected =
        MX ? __type_fp32 : matrix_accumulator_type_code(type_traits<typename Left::DType>::TypeCode);
    return type_traits<typename Accumulator::DType>::TypeCode == Expected;
}

template <FixpAttr Attr, typename DType>
constexpr bool is_fixp_output_type()
{
    constexpr int TypeCode = type_traits<DType>::TypeCode;
    if constexpr (Attr.PreQuant == FixpPreQuantMode::None)
        // PreQuant=None preserves the derived FP32/S32/U32 AccType.
        return TypeCode == __type_fp32 || TypeCode == __type_int32 || TypeCode == __type_uint32;
    if constexpr (
        Attr.PreQuant == FixpPreQuantMode::F322F16 || Attr.PreQuant == FixpPreQuantMode::VDEQF16 ||
        Attr.PreQuant == FixpPreQuantMode::DEQF16 || Attr.PreQuant == FixpPreQuantMode::QF322F16Pre ||
        Attr.PreQuant == FixpPreQuantMode::VQF322F16Pre)
        return TypeCode == __type_fp16;
    if constexpr (
        Attr.PreQuant == FixpPreQuantMode::F322BF16 || Attr.PreQuant == FixpPreQuantMode::QF322BF16Pre ||
        Attr.PreQuant == FixpPreQuantMode::QS322BF16Pre || Attr.PreQuant == FixpPreQuantMode::VQF322BF16Pre ||
        Attr.PreQuant == FixpPreQuantMode::VQS322BF16Pre)
        return TypeCode == __type_bf16;
    if constexpr (
        Attr.PreQuant == FixpPreQuantMode::VREQS8Pre || Attr.PreQuant == FixpPreQuantMode::REQS8Pre ||
        Attr.PreQuant == FixpPreQuantMode::VQF322S8Pre || Attr.PreQuant == FixpPreQuantMode::QF322S8Pre)
        return TypeCode == __type_int8;
    if constexpr (
        Attr.PreQuant == FixpPreQuantMode::VSHIFTS322S16 || Attr.PreQuant == FixpPreQuantMode::SHIFTS322S16 ||
        Attr.PreQuant == FixpPreQuantMode::QF322S16Pre || Attr.PreQuant == FixpPreQuantMode::VQF322S16Pre)
        return TypeCode == __type_int16;
    if constexpr (Attr.PreQuant == FixpPreQuantMode::QF322S4Pre || Attr.PreQuant == FixpPreQuantMode::VQF322S4Pre)
        return TypeCode == __type_int4x2;
    if constexpr (Attr.PreQuant == FixpPreQuantMode::QF322HIF8Pre || Attr.PreQuant == FixpPreQuantMode::VQF322HIF8Pre)
        return TypeCode == __type_hif8;
    if constexpr (Attr.PreQuant == FixpPreQuantMode::QF322FP8Pre || Attr.PreQuant == FixpPreQuantMode::VQF322FP8Pre)
        // Generic FP8 output is normalized to E4M3 only (no E5M2 alias) per
        // PTO 0.58 (handoff 3327).
        return TypeCode == __type_fp8_e4m3;
    if constexpr (Attr.PreQuant == FixpPreQuantMode::QF322F32Pre || Attr.PreQuant == FixpPreQuantMode::VQF322F32Pre)
        return TypeCode == __type_fp32;
    return false;
}

template <FixpAttr Attr, typename DType>
constexpr bool is_basic_fixp_output_type()
{
    return is_fixp_output_type<Attr, DType>();
}

/// Layout for GlobalTensor
/// ND: lower tow dimensions are arranged in RowMajor order;
/// DN: lower tow dimensions are arranged in ColMajor order;
/// NZ: lower tow dimensions are arranged in RowMajor order;
///     third and fourth dimensions are arranged in ColMajor order.
/// SCALE:
/// MAX:
enum class Layout {
    ND,
    DN,
    NZ,
    SCALE,
    MAX,
};

constexpr int DYNAMIC = -1;

template <int N1 = DYNAMIC, int N2 = DYNAMIC, int N3 = DYNAMIC, int N4 = DYNAMIC, int N5 = DYNAMIC>
struct Shape {
    static constexpr int staticShape[5] = {N1, N2, N3, N4, N5};

    inline Shape()
    {
        static_assert(
            (N1 == DYNAMIC) + (N2 == DYNAMIC) + (N3 == DYNAMIC) + (N4 == DYNAMIC) + (N5 == DYNAMIC) == 0,
            "0-parameter constructors is only applicable to Stride with 0 dynamic dimension.");
    }

    inline Shape(int n)
    {
        static_assert(
            (N1 == DYNAMIC) + (N2 == DYNAMIC) + (N3 == DYNAMIC) + (N4 == DYNAMIC) + (N5 == DYNAMIC) == 1,
            "1-parameter constructors is only applicable to Stride with 1 dynamic dimension.");
        if constexpr (N1 == DYNAMIC)
            shape[0] = n;
        if constexpr (N2 == DYNAMIC)
            shape[1] = n;
        if constexpr (N3 == DYNAMIC)
            shape[2] = n;
        if constexpr (N4 == DYNAMIC)
            shape[3] = n;
        if constexpr (N5 == DYNAMIC)
            shape[4] = n;
    }

    inline Shape(int n1, int n2)
    {
        static_assert(
            (N1 == DYNAMIC) + (N2 == DYNAMIC) + (N3 == DYNAMIC) + (N4 == DYNAMIC) + (N5 == DYNAMIC) == 2,
            "2-parameter constructors is only applicable to Stride with 2 dynamic dimension.");

        int idx = 0;
        const int vals[] = {n1, n2};
        if constexpr (N1 == DYNAMIC)
            shape[0] = vals[idx++];
        if constexpr (N2 == DYNAMIC)
            shape[1] = vals[idx++];
        if constexpr (N3 == DYNAMIC)
            shape[2] = vals[idx++];
        if constexpr (N4 == DYNAMIC)
            shape[3] = vals[idx++];
        if constexpr (N5 == DYNAMIC)
            shape[4] = vals[idx++];
    }

    inline Shape(int n1, int n2, int n3)
    {
        static_assert(
            (N1 == DYNAMIC) + (N2 == DYNAMIC) + (N3 == DYNAMIC) + (N4 == DYNAMIC) + (N5 == DYNAMIC) == 3,
            "3-parameter constructors is only applicable to Stride with 3 dynamic dimension.");
        int idx = 0;
        const int vals[] = {n1, n2, n3};
        if constexpr (N1 == DYNAMIC)
            shape[0] = vals[idx++];
        if constexpr (N2 == DYNAMIC)
            shape[1] = vals[idx++];
        if constexpr (N3 == DYNAMIC)
            shape[2] = vals[idx++];
        if constexpr (N4 == DYNAMIC)
            shape[3] = vals[idx++];
        if constexpr (N5 == DYNAMIC)
            shape[4] = vals[idx++];
    }

    inline Shape(int n1, int n2, int n3, int n4)
    {
        static_assert(
            (N1 == DYNAMIC) + (N2 == DYNAMIC) + (N3 == DYNAMIC) + (N4 == DYNAMIC) + (N5 == DYNAMIC) == 4,
            "4-parameter constructors is only applicable to Stride with 4 dynamic dimension.");
        int idx = 0;
        const int vals[] = {n1, n2, n3, n4};
        if constexpr (N1 == DYNAMIC)
            shape[0] = vals[idx++];
        if constexpr (N2 == DYNAMIC)
            shape[1] = vals[idx++];
        if constexpr (N3 == DYNAMIC)
            shape[2] = vals[idx++];
        if constexpr (N4 == DYNAMIC)
            shape[3] = vals[idx++];
        if constexpr (N5 == DYNAMIC)
            shape[4] = vals[idx++];
    }

    inline Shape(int n1, int n2, int n3, int n4, int n5)
    {
        static_assert(
            (N1 == DYNAMIC) + (N2 == DYNAMIC) + (N3 == DYNAMIC) + (N4 == DYNAMIC) + (N5 == DYNAMIC) == 5,
            "5-parameter constructors is only applicable to Stride with 5 dynamic dimension.");
        if constexpr (N1 == DYNAMIC)
            shape[0] = n1;
        if constexpr (N2 == DYNAMIC)
            shape[1] = n2;
        if constexpr (N3 == DYNAMIC)
            shape[2] = n3;
        if constexpr (N4 == DYNAMIC)
            shape[3] = n4;
        if constexpr (N5 == DYNAMIC)
            shape[4] = n5;
    }

public:
    int shape[5] = {1};
};

template <int SN1 = DYNAMIC, int SN2 = DYNAMIC, int SN3 = DYNAMIC, int SN4 = DYNAMIC, int SN5 = DYNAMIC>
struct Stride {
    static constexpr int staticStride[5] = {SN1, SN2, SN3, SN4, SN5};

    inline Stride() {}

    inline Stride(int n)
    {
        static_assert(
            (SN1 == DYNAMIC) + (SN2 == DYNAMIC) + (SN3 == DYNAMIC) + (SN4 == DYNAMIC) + (SN5 == DYNAMIC) == 1,
            "1-parameter constructors is only applicable to Stride with 1 dynamic dimension.");

        if constexpr (SN1 == DYNAMIC)
            stride[0] = n;
        if constexpr (SN2 == DYNAMIC)
            stride[1] = n;
        if constexpr (SN3 == DYNAMIC)
            stride[2] = n;
        if constexpr (SN4 == DYNAMIC)
            stride[3] = n;
        if constexpr (SN5 == DYNAMIC)
            stride[4] = n;
    }

    inline Stride(int n1, int n2)
    {
        static_assert(
            (SN1 == DYNAMIC) + (SN2 == DYNAMIC) + (SN3 == DYNAMIC) + (SN4 == DYNAMIC) + (SN5 == DYNAMIC) == 2,
            "2-parameter constructors is only applicable to Stride with 2 dynamic dimension.");
        int idx = 0;
        const int vals[] = {n1, n2};
        if constexpr (SN1 == DYNAMIC)
            stride[0] = vals[idx++];
        if constexpr (SN2 == DYNAMIC)
            stride[1] = vals[idx++];
        if constexpr (SN3 == DYNAMIC)
            stride[2] = vals[idx++];
        if constexpr (SN4 == DYNAMIC)
            stride[3] = vals[idx++];
        if constexpr (SN5 == DYNAMIC)
            stride[4] = vals[idx++];
    }

    inline Stride(int n1, int n2, int n3)
    {
        static_assert(
            (SN1 == DYNAMIC) + (SN2 == DYNAMIC) + (SN3 == DYNAMIC) + (SN4 == DYNAMIC) + (SN5 == DYNAMIC) == 3,
            "3-parameter constructors is only applicable to Stride with 3 dynamic dimension.");
        int idx = 0;
        const int vals[] = {n1, n2, n3};
        if constexpr (SN1 == DYNAMIC)
            stride[0] = vals[idx++];
        if constexpr (SN2 == DYNAMIC)
            stride[1] = vals[idx++];
        if constexpr (SN3 == DYNAMIC)
            stride[2] = vals[idx++];
        if constexpr (SN4 == DYNAMIC)
            stride[3] = vals[idx++];
        if constexpr (SN5 == DYNAMIC)
            stride[4] = vals[idx++];
    }

    inline Stride(int n1, int n2, int n3, int n4)
    {
        static_assert(
            (SN1 == DYNAMIC) + (SN2 == DYNAMIC) + (SN3 == DYNAMIC) + (SN4 == DYNAMIC) + (SN5 == DYNAMIC) == 4,
            "4-parameter constructors is only applicable to Stride with 4 dynamic dimension.");
        int idx = 0;
        const int vals[] = {n1, n2, n3, n4};
        if constexpr (SN1 == DYNAMIC)
            stride[0] = vals[idx++];
        if constexpr (SN2 == DYNAMIC)
            stride[1] = vals[idx++];
        if constexpr (SN3 == DYNAMIC)
            stride[2] = vals[idx++];
        if constexpr (SN4 == DYNAMIC)
            stride[3] = vals[idx++];
        if constexpr (SN5 == DYNAMIC)
            stride[4] = vals[idx++];
    }

    inline Stride(int n1, int n2, int n3, int n4, int n5)
    {
        static_assert(
            (SN1 == DYNAMIC) + (SN2 == DYNAMIC) + (SN3 == DYNAMIC) + (SN4 == DYNAMIC) + (SN5 == DYNAMIC) == 5,
            "5-parameter constructors is only applicable to Stride with 5 dynamic dimension.");
        if constexpr (SN1 == DYNAMIC)
            stride[0] = n1;
        if constexpr (SN2 == DYNAMIC)
            stride[1] = n2;
        if constexpr (SN3 == DYNAMIC)
            stride[2] = n3;
        if constexpr (SN4 == DYNAMIC)
            stride[3] = n4;
        if constexpr (SN5 == DYNAMIC)
            stride[4] = n5;
    }

public:
    int stride[5] = {1};
};

template <typename Element_, typename Shape_, typename Stride_, Layout Layout_ = Layout::ND>
struct GlobalTensor {
    using Shape = Shape_;
    using Stride = Stride_;
    using DType = Element_;
    static constexpr Layout layout = Layout_;

    static const Shape defaultShape;
    static const Stride defaultStride;

    static constexpr int staticShape[5] = {
        Shape::staticShape[0], Shape::staticShape[1], Shape::staticShape[2], Shape::staticShape[3],
        Shape::staticShape[4]};
    static constexpr int staticStride[5] = {
        Stride::staticStride[0], Stride::staticStride[1], Stride::staticStride[2], Stride::staticStride[3],
        Stride::staticStride[4]};
    static constexpr bool isRowMajor = Layout_ == Layout::ND;
    static constexpr int RowStride = staticStride[3];
    static constexpr int ColStride = staticStride[4];

    inline GlobalTensor() = default;

    inline GlobalTensor(DType* data, const Shape& shape = defaultShape, const Stride& stride = defaultStride)
    {
        data_ = data;

        if constexpr (staticShape[0] == DYNAMIC)
            shape_.shape[0] = shape.shape[0];
        if constexpr (staticShape[1] == DYNAMIC)
            shape_.shape[1] = shape.shape[1];
        if constexpr (staticShape[2] == DYNAMIC)
            shape_.shape[2] = shape.shape[2];
        if constexpr (staticShape[3] == DYNAMIC)
            shape_.shape[3] = shape.shape[3];
        if constexpr (staticShape[4] == DYNAMIC)
            shape_.shape[4] = shape.shape[4];

        if constexpr (staticStride[0] == DYNAMIC)
            stride_.stride[0] = stride.stride[0];
        if constexpr (staticStride[1] == DYNAMIC)
            stride_.stride[1] = stride.stride[1];
        if constexpr (staticStride[2] == DYNAMIC)
            stride_.stride[2] = stride.stride[2];
        if constexpr (staticStride[3] == DYNAMIC)
            stride_.stride[3] = stride.stride[3];
        if constexpr (staticStride[4] == DYNAMIC)
            stride_.stride[4] = stride.stride[4];
    }

    // Source descriptors are also commonly constructed from read-only buffers.
    // Keep the descriptor representation mutable because the same type is used
    // by store operations, while accepting const pointers at the API boundary.
    inline GlobalTensor(const DType* data, const Shape& shape = defaultShape, const Stride& stride = defaultStride)
        : GlobalTensor(const_cast<DType*>(data), shape, stride)
    {}

    inline int GetShape(const int dim) const
    {
        switch (dim) {
            case 0:
                return GetShapeSize<staticShape[0]>(dim);
            case 1:
                return GetShapeSize<staticShape[1]>(dim);
            case 2:
                return GetShapeSize<staticShape[2]>(dim);
            case 3:
                return GetShapeSize<staticShape[3]>(dim);
            case 4:
                return GetShapeSize<staticShape[4]>(dim);
            default:
                return -1;
        }
    }

    inline int GetStride(const int dim) const
    {
        switch (dim) {
            case 0:
                return GetStrideSize<staticStride[0]>(dim);
            case 1:
                return GetStrideSize<staticStride[1]>(dim);
            case 2:
                return GetStrideSize<staticStride[2]>(dim);
            case 3:
                return GetStrideSize<staticStride[3]>(dim);
            case 4:
                return GetStrideSize<staticStride[4]>(dim);
            default:
                return -1;
        }
    }

    inline size_t GetStrideBytes(const int dim) const
    {
        const size_t elements = static_cast<size_t>(GetStride(dim));
        return (elements * tile_element_bits_v<DType> + 7) / 8;
    }

    DType* data() { return data_; }
    const DType* data() const { return data_; }

private:
    template <int StaticShape>
    inline int GetShapeSize(const int dim) const
    {
        if constexpr (StaticShape == DYNAMIC)
            return shape_.shape[dim];
        else
            return StaticShape;
    }

    template <int StaticStride>
    inline int GetStrideSize(const int dim) const
    {
        if constexpr (StaticStride == DYNAMIC)
            return stride_.stride[dim];
        else
            return StaticStride;
    }

    DType* data_;
    Shape shape_ = defaultShape;
    Stride stride_ = defaultStride;
};

template <typename Element_, typename Shape_, typename Stride_, Layout Layout_>
const typename GlobalTensor<Element_, Shape_, Stride_, Layout_>::Shape
    GlobalTensor<Element_, Shape_, Stride_, Layout_>::defaultShape{};

template <typename Element_, typename Shape_, typename Stride_, Layout Layout_>
const typename GlobalTensor<Element_, Shape_, Stride_, Layout_>::Stride
    GlobalTensor<Element_, Shape_, Stride_, Layout_>::defaultStride{};

template <
    Location Loc_, typename Element_, const int Rows_, const int Cols_, const BLayout BFractal_ = BLayout::RowMajor,
    const int RowValid_ = Rows_, const int ColValid_ = Cols_, const SLayout SFractal_ = SLayout::NoneBox,
    const int SFractalSize_ = 512, const PadValue PadVal_ = PadValue::Null,
    const CompactMode Compact_ = CompactMode::Null, const bool ElementProfile_ = false>
struct Tile {
public:
    using DType = Element_;

    static constexpr int getInnerRow()
    {
        if constexpr (SFractalSize_ == 1024) { // output/acc
            static_assert(type_traits<DType>::bits == 32, "Size of datatype != 4");
            return 16;
        } else {
            return isBoxedLayout ? (isInnerRowMajor ? 16 : byteSize * 8 / type_traits<DType>::bits) : 1;
        }
    }

    static constexpr int getInnerCol()
    {
        if constexpr (SFractalSize_ == 1024) { // output/acc
            static_assert(type_traits<DType>::bits == 32, "Size of datatype != 4");
            return 16;
        } else {
            return isBoxedLayout ? (isInnerRowMajor ? byteSize * 8 / type_traits<DType>::bits : 16) : 1;
        }
    }

    static constexpr Location Loc = Loc_;
    static constexpr int Rows = Rows_;
    static constexpr int Cols = Cols_;
    static constexpr bool IsCubeLayout =
        BFractal_ == BLayout::CubeM16 || BFractal_ == BLayout::CubeM32 || BFractal_ == BLayout::CubeN8;
    static constexpr int RowStride = BFractal_ == BLayout::RowMajor ? Cols : BFractal_ == BLayout::ColMajor ? 1 : 0;
    static constexpr int ColStride = BFractal_ == BLayout::RowMajor ? 1 : BFractal_ == BLayout::ColMajor ? Rows : 0;

    static constexpr int CubeCellBytes = 128;
    static constexpr int CubeElementBits = tile_element_bits_v<DType>;
    static constexpr bool IsCubeM32B64 = BFractal_ == BLayout::CubeM32 && CubeElementBits == 64;
    static constexpr int CubeCellRows = [] {
        if constexpr (BFractal_ == BLayout::CubeM16)
            return 16;
        if constexpr (BFractal_ == BLayout::CubeM32)
            return 32;
        if constexpr (BFractal_ == BLayout::CubeN8)
            return CubeElementBits == 64 ? 2 :
                   CubeElementBits == 32 ? 4 :
                   CubeElementBits == 16 ? 8 :
                   CubeElementBits == 8  ? 16 :
                                           32;
        return 0;
    }();
    static constexpr int CubeCellCols = [] {
        if constexpr (BFractal_ == BLayout::CubeN8)
            return 8;
        if constexpr (BFractal_ == BLayout::CubeM16)
            return 128 * 8 / (16 * CubeElementBits);
        if constexpr (BFractal_ == BLayout::CubeM32)
            return IsCubeM32B64 ? 1 : 128 * 8 / (32 * CubeElementBits);
        return 0;
    }();
    // Minimum CELL envelope required by the declared logical shape.  M16/M32
    // descriptors do not use this envelope as their final physical geometry:
    // their allocated SizeCode is a power-of-two capacity and the architectural
    // physical column count must account for that complete capacity.
    static constexpr int CubeMinimumStorageRows = [] {
        if constexpr (BFractal_ == BLayout::CubeN8)
            return ((Rows + CubeCellRows - 1) / CubeCellRows) * CubeCellRows;
        if constexpr (IsCubeLayout)
            return CubeCellRows;
        return Rows;
    }();
    static constexpr int CubeMinimumStorageCols =
        IsCubeLayout ? ((Cols + CubeCellCols - 1) / CubeCellCols) * CubeCellCols : Cols;
    static constexpr int CubeRequiredCellCount =
        IsCubeM32B64 ? 2 * Cols :
        IsCubeLayout ? (CubeMinimumStorageRows / CubeCellRows) * (CubeMinimumStorageCols / CubeCellCols) :
                       0;
    static constexpr int CubeRequiredBytes = IsCubeLayout ? CubeRequiredCellCount * CubeCellBytes : 0;
    static constexpr int CubeLoadLayout = BFractal_ == BLayout::CubeM32 ? LayoutCvtEnum::ND2M32 :
                                          BFractal_ == BLayout::CubeM16 ? LayoutCvtEnum::ND2M16 :
                                          BFractal_ == BLayout::CubeN8  ? LayoutCvtEnum::ND2N8 :
                                                                          -1;
    static constexpr int CubeStoreLayout = BFractal_ == BLayout::CubeM32 ? LayoutCvtEnum::M322ND :
                                           BFractal_ == BLayout::CubeM16 ? LayoutCvtEnum::M162ND :
                                           BFractal_ == BLayout::CubeN8  ? LayoutCvtEnum::N82ND :
                                                                           -1;
    static constexpr int round_capacity(int bytes)
    {
        int capacity = 128;
        while (capacity < bytes)
            capacity *= 2;
        return capacity;
    }
    static constexpr int StorageBytes =
        round_capacity(IsCubeLayout ? CubeRequiredBytes : (Rows * Cols * tile_element_bits_v<DType> + 7) / 8);
    // PTO #399 exact-capacity CUBE geometry.  For M16/M32, every byte selected
    // by SizeCode belongs to the physical descriptor, including capacity tail
    // beyond the minimum logical CELL envelope.  N8 retains its valid-derived
    // KxN geometry and may have unused carrier capacity.
    static constexpr int CubePhysicalRows = [] {
        if constexpr (BFractal_ == BLayout::CubeM16)
            return 16;
        if constexpr (BFractal_ == BLayout::CubeM32)
            return 32;
        if constexpr (BFractal_ == BLayout::CubeN8)
            return CubeMinimumStorageRows;
        return Rows;
    }();
    static constexpr int CubePhysicalCols = [] {
        if constexpr (BFractal_ == BLayout::CubeM16 || BFractal_ == BLayout::CubeM32)
            return (StorageBytes * 8) / (CubePhysicalRows * CubeElementBits);
        if constexpr (BFractal_ == BLayout::CubeN8)
            return CubeMinimumStorageCols;
        return Cols;
    }();
    static constexpr int CubeStorageRows = CubePhysicalRows;
    static constexpr int CubeStorageCols = CubePhysicalCols;
    static constexpr int PhysicalRows = CubePhysicalRows;
    static constexpr int PhysicalCols = CubePhysicalCols;
    static constexpr int CubeCellCount =
        IsCubeLayout ? (BFractal_ == BLayout::CubeM16 || BFractal_ == BLayout::CubeM32 ? StorageBytes / CubeCellBytes :
                                                                                         CubeRequiredCellCount) :
                       0;
    static constexpr int CubeStorageIndex(int row, int column)
    {
        if constexpr (IsCubeM32B64)
            return column * 32 + row;
        const int cell_elements = CubeCellRows * CubeCellCols;
        if constexpr (BFractal_ == BLayout::CubeN8) {
            const int k_repeat = CubeMinimumStorageRows / CubeCellRows;
            const int cell_k = row / CubeCellRows;
            const int cell_n = column / CubeCellCols;
            const int inner_row = row % CubeCellRows;
            const int inner_column = column % CubeCellCols;
            const int local = inner_column * CubeCellRows + inner_row;
            return (cell_n * k_repeat + cell_k) * cell_elements + local;
        }
        int mapped_column = column % CubeCellCols;
        if constexpr (BFractal_ == BLayout::CubeM16) {
            if constexpr (CubeElementBits == 4) {
                if (mapped_column >= 4 && mapped_column < 8)
                    mapped_column += 4;
                else if (mapped_column >= 8 && mapped_column < 12)
                    mapped_column -= 4;
            }
        }
        const int cell_index = column / CubeCellCols;
        const int local = row * CubeCellCols + mapped_column;
        return cell_index * cell_elements + local;
    }
    static constexpr int CubePhysicalCellsPerLogicalColumn = IsCubeM32B64 ? 2 : 1;
    static constexpr int CubePhysicalCellIndex(int column, bool high_plane)
    {
        return IsCubeM32B64 ? 2 * column + (high_plane ? 1 : 0) : column;
    }
    static constexpr int CubePhysicalWordIndex(int row) { return row; }
    static constexpr int CubePhysicalPlaneByteOffset(int row, int column, bool high_plane)
    {
        return CubePhysicalCellIndex(column, high_plane) * CubeCellBytes + CubePhysicalWordIndex(row) * 4;
    }
    static constexpr uint32_t CubePhysicalPlaneWord(uint64_t value, bool high_plane)
    {
        return high_plane ? static_cast<uint32_t>(value >> 32) : static_cast<uint32_t>(value);
    }
    static constexpr bool CubePhysicalCellRangeComplete(int offset_cells, int cell_count)
    {
        return !IsCubeM32B64 || (offset_cells >= 0 && cell_count > 0 && offset_cells % 2 == 0 && cell_count % 2 == 0);
    }

    static constexpr int kBytes = (Rows_ * Cols_ * tile_element_bits_v<DType> + 7) / 8;
    // static_assert(kBytes % 512 == 0, "Tile size must be 512 bytes aligned");
    // static_assert(((kBytes / 512 - 1) & (kBytes / 512)) == 0, "Tile size must by (512 * 2 ^ n) Bytes");

    static constexpr int ValidRow = RowValid_;
    static constexpr int ValidCol = ColValid_;
    // Location::Shared is a storage-class marker for SharedTile<LocalTile>; a
    // bare Tile<Shared,...> would wrongly allocate a Local TileDType data_ payload
    // and expose data(), which a Shared operand must never own. Route Shared
    // storage through SharedTile<> instead.
    static_assert(
        Loc_ != Location::Shared, "Use SharedTile<LocalTile> for Location::Shared; Tile<> is for "
                                  "local register-backed storage only.");
    static_assert(Rows > 0 && ValidRow <= Rows && Cols > 0 && ValidCol <= Cols, "Invalid Tile Layout.");
    static_assert(
        !IsCubeLayout || (CubeElementBits == 4 || CubeElementBits == 8 || CubeElementBits == 16 ||
                          CubeElementBits == 32 || (BFractal_ == BLayout::CubeM16 && CubeElementBits == 64) ||
                          (BFractal_ == BLayout::CubeM32 && CubeElementBits == 64) ||
                          (BFractal_ == BLayout::CubeN8 && CubeElementBits == 64)),
        "CUBE CELL layouts support only 4/8/16/32-bit widths, "
        "64-bit CUBE_M32, or the existing 64-bit CUBE_N8 form");
    static_assert(BFractal_ != BLayout::CubeM16 || CubeElementBits != 64, "64-bit CUBE_M16 layouts are not supported");
    static_assert(!IsCubeM32B64 || Rows == 32, "64-bit CUBE_M32 layouts require exactly 32 physical rows");
    static_assert(BFractal_ != BLayout::CubeM16 || Rows <= 16, "CUBE_M16 supports at most 16 logical rows");
    static_assert(BFractal_ != BLayout::CubeM32 || Rows <= 32, "CUBE_M32 supports at most 32 logical rows");
    static_assert(
        !IsCubeLayout || CubeRequiredBytes <= StorageBytes, "CUBE CELL envelope exceeds its allocated capacity");
    static_assert(
        BFractal_ != BLayout::CubeM16 ||
            (CubePhysicalRows == 16 && CubePhysicalCols * CubePhysicalRows * CubeElementBits == StorageBytes * 8),
        "CUBE_M16 physical geometry must exactly cover SizeCode capacity");
    static_assert(
        BFractal_ != BLayout::CubeM32 ||
            (CubePhysicalRows == 32 && CubePhysicalCols * CubePhysicalRows * CubeElementBits == StorageBytes * 8),
        "CUBE_M32 physical geometry must exactly cover SizeCode capacity");

    static constexpr BLayout BFractal = BFractal_;
    static constexpr SLayout SFractal = SFractal_;
    static constexpr int Numel = Rows * Cols;
    static constexpr bool isRowMajor = BFractal_ == BLayout::RowMajor;

    static constexpr int SFractalSize = SFractalSize_;
    static constexpr PadValue PadVal = PadVal_;
    static constexpr CompactMode Compact = Compact_;
    static constexpr bool IsElementProfile = ElementProfile_;
    static constexpr bool IsElement32Type =
        std::is_same_v<DType, uint32_t> || std::is_same_v<DType, int32_t> || std::is_same_v<DType, float>;
    static constexpr bool IsElement16Type = std::is_same_v<DType, __half> || std::is_same_v<DType, uint16_t>;
    static constexpr bool IsElement64Type = std::is_same_v<DType, int64_t> || std::is_same_v<DType, uint64_t>;
    static constexpr bool IsElementProfileShape =
        (std::is_same_v<DType, float> && (Cols == 1 || Cols == 4 || Cols == 64 || Cols == 512)) ||
        ((std::is_same_v<DType, uint32_t> || std::is_same_v<DType, int32_t>) &&
         (Cols == 1 || Cols == 2 || Cols == 4 || Cols == 8 || Cols == 16 || Cols == 32 || Cols == 64)) ||
        (IsElement16Type && (Cols == 2 || Cols == 4 || Cols == 16 || Cols == 64)) ||
        (IsElement64Type && (Cols == 1 || Cols == 2 || Cols == 4 || Cols == 8 || Cols == 16 || Cols == 32));
    static_assert(
        !IsElementProfile || (Loc_ == Location::Vec && BFractal_ == BLayout::CubeM32 && Rows == 32 &&
                              IsElementProfileShape && ValidRow == Rows && ValidCol == Cols &&
                              SFractal_ == SLayout::NoneBox && SFractalSize_ == 512 && PadVal_ == PadValue::Null &&
                              Compact_ == CompactMode::Null && (IsElement32Type || IsElement16Type || IsElement64Type)),
        "element Tile profile has an unsupported dtype or M32 shape");
    static constexpr int LogicalTileBytes = StorageBytes;
    static constexpr int TilesizeCode = LogicalTileBytes == 128    ? __tilesize_128B :
                                        LogicalTileBytes == 256    ? __tilesize_256B :
                                        LogicalTileBytes == 512    ? __tilesize_512B :
                                        LogicalTileBytes == 1024   ? __tilesize_1KB :
                                        LogicalTileBytes == 2048   ? __tilesize_2KB :
                                        LogicalTileBytes == 4096   ? __tilesize_4KB :
                                        LogicalTileBytes == 8192   ? __tilesize_8KB :
                                        LogicalTileBytes == 16384  ? __tilesize_16KB :
                                        LogicalTileBytes == 32768  ? __tilesize_32KB :
                                        LogicalTileBytes == 65536  ? __tilesize_64KB :
                                        LogicalTileBytes == 131072 ? __tilesize_128KB :
                                        LogicalTileBytes == 262144 ? __tilesize_256KB :
                                                                     __tilesize_unknown;
    static constexpr bool IsValidActiveSize = TilesizeCode >= __tilesize_128B && TilesizeCode <= __tilesize_256KB;

    // constructor for static shape
    Tile() {};
    template <int RowMask = ValidRow, int ColMask = ValidCol>
    Tile(std::enable_if_t<(RowMask > 0) && (ColMask > 0), DType> val);

    // constructor for both dimensions are runtime variables
    template <int RowMask = ValidRow, int ColMask = ValidCol>
    Tile(
        std::enable_if_t<RowMask == -1 && ColMask == -1, size_t> VR,
        std::enable_if_t<RowMask == -1 && ColMask == -1, size_t> VC);
    template <int RowMask = ValidRow, int ColMask = ValidCol>
    Tile(
        DType val, std::enable_if_t<RowMask == -1 && ColMask == -1, size_t> VR,
        std::enable_if_t<RowMask == -1 && ColMask == -1, size_t> VC);

    // constructor for row dimension is runtime variables
    template <int RowMask = ValidRow, int ColMask = ValidCol>
    Tile(std::enable_if_t<(RowMask == -1) && (ColMask > 0), size_t> VR);
    template <int RowMask = ValidRow, int ColMask = ValidCol>
    Tile(DType val, std::enable_if_t<(RowMask == -1) && (ColMask > 0), size_t> VR);

    // constructor for col dimension is runtime variables
    template <int RowMask = ValidRow, int ColMask = ValidCol>
    Tile(std::enable_if_t<(RowMask > 0) && (ColMask == -1), size_t> VC);
    template <int RowMask = ValidRow, int ColMask = ValidCol>
    Tile(DType val, std::enable_if_t<(RowMask > 0) && (ColMask == -1), size_t> VC);

    static constexpr int byteSize = 32;
    static constexpr bool isBoxedLayout = (SFractal != SLayout::NoneBox);
    static constexpr bool isInnerRowMajor = (SFractal == SLayout::RowMajor);
    static constexpr bool isInnerColMajor = (SFractal == SLayout::ColMajor);

    static constexpr int InnerRows = getInnerRow();
    static constexpr int InnerCols = getInnerCol();

    static constexpr int InnerNumel = InnerRows * InnerCols;

    static_assert(Rows % InnerRows == 0, "Layout rows must be divisible by inner box rows");
    static_assert(Cols % InnerCols == 0, "Layout cols must be divisible by inner box cols");

    static_assert(
        IsCubeLayout || SFractal_ == SLayout::NoneBox ||
            (SFractal_ != SLayout::NoneBox) && (Rows % InnerRows == 0 && Cols % InnerCols == 0),
        "Boxed layouts require Rows/Cols to be integer multiples of their inner box.");

    static_assert(SFractalSize_ == 512 || SFractalSize_ == 1024, "SFractalSize_ illegal");

#ifdef __linx
    using TileCarrierScalar = std::conditional_t<IsElementProfile, DType, uint32_t>;
    using TileDType = linx_tile_carrier<LogicalTileBytes, TileCarrierScalar>;
    using TileRegisterType = typename TileDType::RegisterType;
#else
    using TileDType = DType[StorageBytes * 8 / type_traits<DType>::bits];
#endif

#ifdef __linx
    __attribute__((always_inline)) TileRegisterType& data() { return data_.Register; }
    __attribute__((always_inline)) const TileRegisterType& data() const { return data_.Register; }
#else
    TileDType& data() { return data_; }
    const TileDType& data() const { return data_; }
#endif

    // record dynamic shape info
    int RowMaskInternal;
    int ColMaskInternal;

    template <int RowMask = ValidRow>
    static constexpr std::enable_if_t<(RowMask > 0), int> GetValidRow()
    {
        return RowMask;
    }

    template <int RowMask = ValidRow>
    std::enable_if_t<RowMask == -1, int> GetValidRow() const
    {
        return RowMaskInternal;
    }

    template <int ColMask = ValidCol>
    static constexpr std::enable_if_t<(ColMask > 0), int> GetValidCol()
    {
        return ColMask;
    }

    template <int ColMask = ValidCol>
    std::enable_if_t<ColMask == -1, int> GetValidCol() const
    {
        return ColMaskInternal;
    }

    void assignData(TileDType data) { data_ = data; }

    TileDType data_;
};

template <typename Element_, const int Rows_, const int Cols_, const int RowValid_ = Rows_, const int ColValid_ = Cols_>
using TileLeft =
    Tile<Location::Left, Element_, Rows_, Cols_, BLayout::ColMajor, RowValid_, ColValid_, SLayout::RowMajor, 512>;

template <typename Element_, const int Rows_, const int Cols_, const int RowValid_ = Rows_, const int ColValid_ = Cols_>
using TileRight =
    Tile<Location::Right, Element_, Rows_, Cols_, BLayout::RowMajor, RowValid_, ColValid_, SLayout::ColMajor, 512>;

template <typename Element_, const int Rows_, const int Cols_, const int RowValid_ = Rows_, const int ColValid_ = Cols_>
using CubeTileM16 = Tile<Location::Left, Element_, Rows_, Cols_, BLayout::CubeM16, RowValid_, ColValid_>;

template <typename Element_, const int Rows_, const int Cols_, const int RowValid_ = Rows_, const int ColValid_ = Cols_>
using CubeTileM32 = Tile<Location::Left, Element_, Rows_, Cols_, BLayout::CubeM32, RowValid_, ColValid_>;

// Vector operands may use the same persistent CUBE cell layouts as matrix
// operands.  The layout conversion selector is determined solely by
// BFractal (ND2M16/ND2M32 for loads and M162ND/M322ND for stores), so these
// aliases intentionally share the Tile implementation with CubeTileM16/M32.
template <typename Element_, const int Rows_, const int Cols_, const int RowValid_ = Rows_, const int ColValid_ = Cols_>
using VecTileM16 = Tile<Location::Vec, Element_, Rows_, Cols_, BLayout::CubeM16, RowValid_, ColValid_>;

template <typename Element_, const int Rows_, const int Cols_, const int RowValid_ = Rows_, const int ColValid_ = Cols_>
using VecTileM32 = Tile<Location::Vec, Element_, Rows_, Cols_, BLayout::CubeM32, RowValid_, ColValid_>;

// Logical element tiles keep the CUBE_M32 carrier private to the API. Existing
// 32-bit profiles retain their 32-element CELL parts. Dense F16/U16 profiles
// use complete 64-element 32x2 CELL parts; matching U32/S32 logical64 tiles
// provide index and predicate-compatible element carriers.
namespace detail {
template <typename Element_, int Elements_>
struct element_tile_profile {
    static constexpr bool IsInteger32 = std::is_same_v<Element_, uint32_t> || std::is_same_v<Element_, int32_t>;
    static constexpr bool IsFloat32 = std::is_same_v<Element_, float>;
    static constexpr bool IsDense16 = std::is_same_v<Element_, __half> || std::is_same_v<Element_, uint16_t>;
    static constexpr bool IsInteger64 = std::is_same_v<Element_, uint64_t> || std::is_same_v<Element_, int64_t>;
    static_assert(
        IsInteger32 || IsFloat32 || IsDense16 || IsInteger64,
        "ElementTile supports U64, S64, U32, S32, F32, F16, or U16 elements");
    static constexpr bool IsSupportedShape =
        (IsFloat32 && (Elements_ == 32 || Elements_ == 128 || Elements_ == 2048 || Elements_ == 16384)) ||
        (IsInteger32 && (Elements_ == 32 || Elements_ == 64 || Elements_ == 128 || Elements_ == 256 ||
                         Elements_ == 512 || Elements_ == 1024 || Elements_ == 2048)) ||
        (IsDense16 && (Elements_ == 64 || Elements_ == 128 || Elements_ == 512 || Elements_ == 2048)) ||
        (IsInteger64 && (Elements_ == 32 || Elements_ == 64 || Elements_ == 128 || Elements_ == 256 ||
                         Elements_ == 512 || Elements_ == 1024));
    static_assert(IsSupportedShape, "ElementTile dtype/count combination is not supported");
    using type = Tile<
        Location::Vec, Element_, 32, Elements_ / 32, BLayout::CubeM32, 32, Elements_ / 32, SLayout::NoneBox, 512,
        PadValue::Null, CompactMode::Null, true>;
};
} // namespace detail

template <typename Element_, int Elements_>
using ElementTile = typename detail::element_tile_profile<Element_, Elements_>::type;

template <typename T>
inline constexpr bool is_element_tile_v = [] {
    using TileType = std::remove_cvref_t<T>;
    if constexpr (requires { TileType::IsElementProfile; })
        return TileType::IsElementProfile;
    return false;
}();

template <typename Element_, const int Rows_, const int Cols_, const int RowValid_ = Rows_, const int ColValid_ = Cols_>
using CubeTileN8 = Tile<Location::Right, Element_, Rows_, Cols_, BLayout::CubeN8, RowValid_, ColValid_>;

template <typename Element_, const int Rows_, const int Cols_, const int RowValid_ = Rows_, const int ColValid_ = Cols_>
using CubeAccumulatorM16 = Tile<Location::Acc, Element_, Rows_, Cols_, BLayout::CubeM16, RowValid_, ColValid_>;

template <typename Element_, const int Rows_, const int Cols_, const int RowValid_ = Rows_, const int ColValid_ = Cols_>
using CubeAccumulatorM32 = Tile<Location::Acc, Element_, Rows_, Cols_, BLayout::CubeM32, RowValid_, ColValid_>;

// PTO-ISA #339: Matrix Bias is a logical 1xN Local CUBE_N8 operand. The
// physical row count is the N8 CELL height for the element width.
template <typename Element_, const int Cols_, const int ColValid_ = Cols_>
using CubeBias = Tile<
    Location::Bias, Element_,
    type_traits<Element_>::bits == 64 ? 2 :
    type_traits<Element_>::bits == 32 ? 4 :
    type_traits<Element_>::bits == 16 ? 8 :
    type_traits<Element_>::bits == 8  ? 16 :
                                        32,
    Cols_, BLayout::CubeN8, 1, ColValid_>;

// Cooperative Shared matrix primaries are published as ordinary RowMajor
// rectangles. Their Left/Right role controls CUBE operand ordering; CELL
// layout is materialized only for Local matrix storage and destinations.
template <typename Element_, const int Rows_, const int Cols_, const int RowValid_ = Rows_, const int ColValid_ = Cols_>
using SharedMatrixLeft = Tile<Location::Left, Element_, Rows_, Cols_, BLayout::RowMajor, RowValid_, ColValid_>;

template <typename Element_, const int Rows_, const int Cols_, const int RowValid_ = Rows_, const int ColValid_ = Cols_>
using SharedMatrixRight = Tile<Location::Right, Element_, Rows_, Cols_, BLayout::RowMajor, RowValid_, ColValid_>;

template <int Rows, int Cols, bool RowMajor>
struct stride_selector;

template <>
struct stride_selector<DYNAMIC, DYNAMIC, true> {
    using type = Stride<1, 1, DYNAMIC, DYNAMIC, DYNAMIC>;
};

template <int Cols>
struct stride_selector<DYNAMIC, Cols, true> {
    using type = Stride<1, 1, -1, Cols, 1>;
};

template <int Cols>
struct stride_selector<DYNAMIC, Cols, false> {
    using type = Stride<1, 1, -1, 1, -1>;
};

template <int Rows>
struct stride_selector<Rows, DYNAMIC, true> {
    using type = Stride<1, 1, -1, -1, 1>;
};
template <int Rows>
struct stride_selector<Rows, DYNAMIC, false> {
    using type = Stride<1, 1, -1, 1, Rows>;
};

template <int Rows, int Cols, bool RowMajor>
struct stride_selector {
    using type = std::conditional_t<RowMajor, Stride<1, 1, Rows * Cols, Cols, 1>, Stride<1, 1, Rows * Cols, 1, Rows>>;
};

template <typename Element_, typename MLayout_>
struct global_tensor {
public:
    using DType = Element_;
    using MLayout = MLayout_;

    static constexpr int Numel = get_numel<MLayout>;
    static constexpr int Rows = num_rows<MLayout>;
    static constexpr int Cols = num_cols<MLayout>;
    static constexpr int RowStride = row_stride<MLayout>;
    static constexpr int ColStride = col_stride<MLayout>;
    static constexpr LayoutEnum kType = layout_type<MLayout>;

    static constexpr bool isRowMajor = kType == LayoutEnum::kRowMajor;

    using shape_t = Shape<1, 1, 1, 1, 1>;
    // Static MatrixLayout can describe a submatrix with a larger parent pitch.
    // Preserve its declared logical strides rather than replacing them with
    // dense Rows/Cols. Dynamic layouts keep the existing runtime constructors.
    using stride_t = std::conditional_t<
        Rows != DYNAMIC && Cols != DYNAMIC, Stride<1, 1, Rows * Cols, RowStride, ColStride>,
        typename stride_selector<Rows, Cols, isRowMajor>::type>;
    static constexpr Layout layout_t = isRowMajor ? Layout::ND : Layout::DN;

    using Impl = GlobalTensor<DType, shape_t, stride_t, layout_t>;

    static constexpr int staticShape[5] = {
        Impl::staticShape[0], Impl::staticShape[1], Impl::staticShape[2], Impl::staticShape[3], Impl::staticShape[4]};

    static constexpr int staticStride[5] = {
        Impl::staticStride[0], Impl::staticStride[1], Impl::staticStride[2], Impl::staticStride[3],
        Impl::staticStride[4]};

    template <typename T = void, typename = std::enable_if_t<(Rows != DYNAMIC && Cols != DYNAMIC), T>>
    global_tensor(DType* data) : impl_(data), layout_(MLayout{})
    {}

    global_tensor(const DType* data) : impl_(const_cast<DType*>(data)), layout_(MLayout{}) {}

    template <
        typename T = void,
        typename = std::enable_if_t<(Rows == DYNAMIC && Cols != DYNAMIC) || (Rows != DYNAMIC && Cols == DYNAMIC), T>>
    global_tensor(DType* data, int stride)
    {
        if constexpr (
            (Rows == DYNAMIC && Cols != DYNAMIC && isRowMajor) || (Rows != DYNAMIC && Cols == DYNAMIC && !isRowMajor)) {
            impl_ = Impl(data, shape_t{}, stride_t(stride));
            layout_ = MLayout{};
        } else {
            impl_ = Impl(data, shape_t{}, stride_t(stride, stride));
            layout_ = MLayout{};
        }
    }

    template <typename T = void, typename = std::enable_if_t<(Rows == DYNAMIC && Cols == DYNAMIC), T>>
    global_tensor(DType* data, int dynamicRow, int dynamicCol)
        : impl_(data, shape_t{}, stride_t(dynamicRow * dynamicCol, dynamicCol, dynamicRow)), layout_(MLayout{})
    {}

    const DType* data() const { return impl_.data(); }
    DType* data() { return impl_.data(); }

    int GetShape(int dim) const { return impl_.GetShape(dim); }
    int GetStride(int dim) const { return impl_.GetStride(dim); }
    size_t GetStrideBytes(int dim) const { return impl_.GetStrideBytes(dim); }

private:
    Impl impl_;
    MLayout layout_;
};

// Packed parameter words consumed by BSTART.TIMG2COL.  The bit layout is
// defined by PTO-BSTART-TIMG2COL-PARAMS-001; keeping the words opaque here
// prevents the API from duplicating the architectural field packing.
struct TIMG2COLParams {
    uint64_t param0 = 0;
    uint64_t param1 = 0;
    uint64_t param2 = 0;
};

// PTO-BSTART-TIMG2COL-PARAMS-001.  Keep TIMG2COLParams ABI-compatible as
// three opaque words, but expose the normative decode here so wrappers can
// reject malformed runtime parameters before emitting the bundle.
struct TIMG2COLDecodedParams {
    uint32_t input_h;
    uint32_t input_w;
    uint32_t cin;
    uint32_t kernel_h;
    uint32_t kernel_w;
    uint32_t pad_top;
    uint32_t pad_left;
    uint32_t pad_bottom;
    uint32_t pad_right;
    uint32_t dilation_h;
    uint32_t dilation_w;
    uint32_t conv_stride_h;
    uint32_t conv_stride_w;
    uint32_t param_version;
    uint32_t extension_class;
    uint64_t row_start;
    uint64_t col_start;
};

constexpr TIMG2COLDecodedParams decode_timg2col_params(TIMG2COLParams p)
{
    return {
        static_cast<uint32_t>(p.param0 & 0xffff),
        static_cast<uint32_t>((p.param0 >> 16) & 0xffff),
        static_cast<uint32_t>((p.param0 >> 32) & 0xffff),
        static_cast<uint32_t>((p.param0 >> 48) & 0xff),
        static_cast<uint32_t>((p.param0 >> 56) & 0xff),
        static_cast<uint32_t>(p.param1 & 0xff),
        static_cast<uint32_t>((p.param1 >> 8) & 0xff),
        static_cast<uint32_t>((p.param1 >> 16) & 0xff),
        static_cast<uint32_t>((p.param1 >> 24) & 0xff),
        static_cast<uint32_t>((p.param1 >> 32) & 0x1f),
        static_cast<uint32_t>((p.param1 >> 37) & 0x1f),
        static_cast<uint32_t>((p.param1 >> 42) & 0x3f),
        static_cast<uint32_t>((p.param1 >> 48) & 0x3f),
        static_cast<uint32_t>((p.param1 >> 55) & 0xf),
        static_cast<uint32_t>((p.param1 >> 59) & 0xf),
        p.param2 & 0xffffffffULL,
        p.param2 >> 32};
}

constexpr bool is_timg2col_params_base_extension_legal(TIMG2COLParams p)
{
    return ((p.param1 >> 54) & 1) == 0 && ((p.param1 >> 63) & 1) == 0 && ((p.param1 >> 55) & 0xff) == 0;
}

constexpr bool is_timg2col_params_base_legal(TIMG2COLParams p)
{
    const auto d = decode_timg2col_params(p);
    return d.input_h != 0 && d.input_w != 0 && d.cin != 0 && d.kernel_h != 0 && d.kernel_w != 0 && d.dilation_h != 0 &&
           d.dilation_w != 0 && d.conv_stride_h != 0 && d.conv_stride_w != 0 && d.param_version == 0 &&
           d.extension_class == 0 && is_timg2col_params_base_extension_legal(p);
}

// Packed operands for the PTO weight-mode TLOAD contract.  ShapeWord packs
// Cin[15:0], Cout[31:16], KernelH[39:32], KernelW[47:40], with [63:48]
// reserved and zero.  StartWord packs NStart[31:0] and KStart[63:32].
// KStart and the requested K window are validated by the ISA operation;
// callers must use C0-aligned values for the selected data type.
struct WeightTLOADParams {
    uint64_t shape_word = 0;
    uint64_t start_word = 0;
};

constexpr WeightTLOADParams make_weight_tload_params(
    uint16_t cin, uint16_t cout, uint8_t kernel_h, uint8_t kernel_w, uint32_t n_start = 0, uint32_t k_start = 0)
{
    return {
        static_cast<uint64_t>(cin) | (static_cast<uint64_t>(cout) << 16) | (static_cast<uint64_t>(kernel_h) << 32) |
            (static_cast<uint64_t>(kernel_w) << 40),
        static_cast<uint64_t>(n_start) | (static_cast<uint64_t>(k_start) << 32)};
}

// Shared-generation metadata required by the PTO weight-mode publication
// protocol.  The bit positions intentionally mirror
// BundleWeightTLOADGenerationMetadata: Layout[4:0], DataType[9:5],
// ValidK[25:10], ValidN[41:26], TotalK[57:42], SizeCode[61:58].
struct WeightTLOADGenerationMetadata {
    uint8_t layout;
    uint8_t data_type;
    uint16_t valid_k;
    uint16_t valid_n;
    uint16_t total_k;
    uint8_t size_code;
};

constexpr uint64_t encode_weight_tload_generation_metadata(WeightTLOADGenerationMetadata metadata)
{
    return (static_cast<uint64_t>(metadata.layout) & 0x1f) | ((static_cast<uint64_t>(metadata.data_type) & 0x1f) << 5) |
           ((static_cast<uint64_t>(metadata.valid_k) & 0xffff) << 10) |
           ((static_cast<uint64_t>(metadata.valid_n) & 0xffff) << 26) |
           ((static_cast<uint64_t>(metadata.total_k) & 0xffff) << 42) |
           ((static_cast<uint64_t>(metadata.size_code) & 0xf) << 58);
}

template <typename T>
struct is_global : std::false_type {};
template <typename T>
struct is_tile : std::false_type {
    static constexpr SLayout layout_enum = SLayout::NoneBox;
};

template <typename Element_, typename Shape_, typename Stride_>
struct is_global<GlobalTensor<Element_, Shape_, Stride_>> : std::true_type {};

template <typename Element_, typename Shape_, typename Stride_, Layout Layout_>
struct is_global<GlobalTensor<Element_, Shape_, Stride_, Layout_>> : std::true_type {};

template <typename Element_, typename Layout_>
struct is_global<global_tensor<Element_, Layout_>> : std::true_type {};

template <
    Location Loc_, typename Element_, const int Rows_, const int Cols_, const BLayout BFractal_, const int RowValid_,
    const int ColValid_, const SLayout SFractal_, const int SFractalSize_, const PadValue PadVal_,
    const CompactMode Compact_, const bool ElementProfile_>
struct is_tile<Tile<
    Loc_, Element_, Rows_, Cols_, BFractal_, RowValid_, ColValid_, SFractal_, SFractalSize_, PadVal_, Compact_,
    ElementProfile_>> : std::true_type {
    static constexpr SLayout layout_enum = SFractal_;
};

template <typename T>
constexpr bool is_boxed_tile = is_tile<T>::value && (is_tile<T>::layout_enum != SLayout::NoneBox);

template <typename tile_shape>
struct is_Nz_layout {
    static constexpr bool value = !tile_shape::isRowMajor && tile_shape::isBoxedLayout && tile_shape::isInnerRowMajor;
};

template <typename tile_shape>
struct is_Zn_layout {
    static constexpr bool value = tile_shape::isRowMajor && tile_shape::isBoxedLayout && tile_shape::isInnerColMajor;
};

template <typename T>
concept is_global_data_v = is_global<T>::value;

template <typename T>
concept is_tile_data_v = is_tile<T>::value;

template <typename T>
concept is_boxed_data_v = is_boxed_tile<T>;

// v5 Shared storage-class wrapper. SharedTile<LocalTile> is public C++ sugar
// that changes a matrix operand's storage class (Local -> Shared) so the
// compiler lowers it via a B.IOS binder instead of a B.IOT source stream.
// It preserves the wrapped Local Tile's role, shape, dtype and layout exactly
// (per the LinxISA v0.58 Shared semantics) and owns no Local TileDType payload.
//
// The compiler allocates the opaque handle to S0..S255. It must not be
// materialized as an ordinary integer or passed through a normal GPR ABI.
template <typename LocalTile>
class SharedTile {
    static_assert(is_tile<LocalTile>::value, "SharedTile<LocalTile>: LocalTile must be an ordinary Tile");
    static_assert(LocalTile::Loc != Location::Shared, "SharedTile cannot wrap a SharedTile (nesting not allowed)");

public:
    using LocalTileType = LocalTile;
    using DType = typename LocalTile::DType;

    // Storage-class marker; role, shape, dtype and layout are forwarded from the
    // wrapped Local Tile so a Shared Right is indistinguishable from its Local
    // counterpart except for storage/lowering.
    static constexpr Location Loc = Location::Shared;
    static constexpr Location Role = LocalTile::Loc;
    static constexpr int Rows = LocalTile::Rows;
    static constexpr int Cols = LocalTile::Cols;
    static constexpr int RowStride = LocalTile::RowStride;
    static constexpr int ColStride = LocalTile::ColStride;
    static constexpr int ValidRow = LocalTile::ValidRow;
    static constexpr int ValidCol = LocalTile::ValidCol;
    static constexpr BLayout BFractal = LocalTile::BFractal;
    static constexpr SLayout SFractal = LocalTile::SFractal;
    static constexpr int SFractalSize = LocalTile::SFractalSize;
    static constexpr PadValue PadVal = LocalTile::PadVal;
    static constexpr CompactMode Compact = LocalTile::Compact;
    static constexpr bool IsCubeLayout = LocalTile::IsCubeLayout;
    static constexpr int PhysicalRows = LocalTile::PhysicalRows;
    static constexpr int PhysicalCols = LocalTile::PhysicalCols;
    using TileDType = typename LocalTile::TileDType;
    static constexpr int LogicalTileBytes = LocalTile::LogicalTileBytes;
    static constexpr int TilesizeCode = LocalTile::TilesizeCode;
    static constexpr bool IsValidActiveSize = LocalTile::IsValidActiveSize;
    static constexpr bool isRowMajor = LocalTile::isRowMajor;
    static constexpr bool isBoxedLayout = LocalTile::isBoxedLayout;
    static constexpr bool isInnerRowMajor = LocalTile::isInnerRowMajor;
    static constexpr bool isInnerColMajor = LocalTile::isInnerColMajor;
    static constexpr int InnerRows = LocalTile::InnerRows;
    static constexpr int InnerCols = LocalTile::InnerCols;
    static constexpr int InnerNumel = LocalTile::InnerNumel;
    static constexpr int Numel = LocalTile::Numel;
    static constexpr int byteSize = LocalTile::byteSize;
    static constexpr int kBytes = LocalTile::kBytes;

    SharedTile()
        : RowMaskInternal(ValidRow == DYNAMIC ? 0 : ValidRow),
          ColMaskInternal(ValidCol == DYNAMIC ? 0 : ValidCol),
          PendingRowMaskInternal(0),
          PendingColMaskInternal(0)
    {}

    explicit SharedTile(const LocalTile& local)
        : RowMaskInternal(local.GetValidRow()),
          ColMaskInternal(local.GetValidCol()),
          PendingRowMaskInternal(0),
          PendingColMaskInternal(0)
    {}

    int GetValidRow() const { return RowMaskInternal; }
    int GetValidCol() const { return ColMaskInternal; }

    template <typename TileLike>
    void SetValidShape(const TileLike& local)
    {
        RowMaskInternal = local.GetValidRow();
        ColMaskInternal = local.GetValidCol();
    }

    void BeginAssembledValidShape(int rows, int columns)
    {
        PendingRowMaskInternal = rows;
        PendingColMaskInternal = columns;
    }

    void MergeAssembledValidShape(int rows, int columns)
    {
        if (rows > PendingRowMaskInternal)
            PendingRowMaskInternal = rows;
        if (columns > PendingColMaskInternal)
            PendingColMaskInternal = columns;
    }

    void PublishAssembledValidShape()
    {
        RowMaskInternal = PendingRowMaskInternal;
        ColMaskInternal = PendingColMaskInternal;
    }

    unsigned long& handle_ref() { return Handle; }
    unsigned long handle() const { return Handle; }

    // No data() accessor: a Shared operand must never be passed to an ordinary
    // tile-register inline-asm operand. Shared-aware operations use handle() with
    // the compiler's dedicated `S` register constraint.

private:
    unsigned long Handle;
    int RowMaskInternal;
    int ColMaskInternal;
    int PendingRowMaskInternal;
    int PendingColMaskInternal;
};

// v5 SharedTile traits. is_tile_data_v stays scoped to ordinary Local Tiles
// so TADD/TCVT/TSTORE-Local etc. cannot accidentally accept a Shared operand;
// APIs that explicitly support Shared use is_shared_tile_v / tile_role_v.
template <typename T>
struct is_shared_tile : std::false_type {};
template <typename LocalTile>
struct is_shared_tile<SharedTile<LocalTile>> : std::true_type {};

template <typename T>
concept is_shared_tile_v = is_shared_tile<T>::value;

// A Shared slot is an architectural name, not a C++ value.  This carrier is
// deliberately empty so a Shared producer cannot lower its result through a
// normal i64 object, stack slot, or C++ ABI.  Operations accepting a slot
// print S<Slot_> directly in their B.IOS operands.
template <unsigned Slot_, typename LocalTile>
class SharedTileSlot {
    static_assert(Slot_ < 64, "SharedTileSlot must name S0..S63");
    static_assert(is_tile<LocalTile>::value, "SharedTileSlot<Slot, LocalTile>: LocalTile must be a Tile");
    static_assert(LocalTile::Loc != Location::Shared, "SharedTileSlot cannot wrap a SharedTile");

public:
    using LocalTileType = LocalTile;
    using DType = typename LocalTile::DType;
    static constexpr unsigned Slot = Slot_;
    static constexpr Location Loc = Location::Shared;
    static constexpr Location Role = LocalTile::Loc;
    static constexpr int Rows = LocalTile::Rows;
    static constexpr int Cols = LocalTile::Cols;
    static constexpr int RowStride = LocalTile::RowStride;
    static constexpr int ColStride = LocalTile::ColStride;
    static constexpr int ValidRow = LocalTile::ValidRow;
    static constexpr int ValidCol = LocalTile::ValidCol;
    static constexpr BLayout BFractal = LocalTile::BFractal;
    static constexpr SLayout SFractal = LocalTile::SFractal;
    static constexpr int SFractalSize = LocalTile::SFractalSize;
    static constexpr PadValue PadVal = LocalTile::PadVal;
    static constexpr CompactMode Compact = LocalTile::Compact;
    static constexpr bool IsCubeLayout = LocalTile::IsCubeLayout;
    static constexpr int PhysicalRows = LocalTile::PhysicalRows;
    static constexpr int PhysicalCols = LocalTile::PhysicalCols;
    using TileDType = typename LocalTile::TileDType;
    static constexpr int LogicalTileBytes = LocalTile::LogicalTileBytes;
    static constexpr int TilesizeCode = LocalTile::TilesizeCode;
    static constexpr bool IsValidActiveSize = LocalTile::IsValidActiveSize;
    static constexpr bool isRowMajor = LocalTile::isRowMajor;
    static constexpr bool isBoxedLayout = LocalTile::isBoxedLayout;
    static constexpr bool isInnerRowMajor = LocalTile::isInnerRowMajor;
    static constexpr bool isInnerColMajor = LocalTile::isInnerColMajor;
    static constexpr int InnerRows = LocalTile::InnerRows;
    static constexpr int InnerCols = LocalTile::InnerCols;
    static constexpr int InnerNumel = LocalTile::InnerNumel;
    static constexpr int Numel = LocalTile::Numel;
    static constexpr int byteSize = LocalTile::byteSize;
    static constexpr int kBytes = LocalTile::kBytes;
};

template <typename T>
struct is_shared_tile_slot : std::false_type {};
template <unsigned Slot_, typename LocalTile>
struct is_shared_tile_slot<SharedTileSlot<Slot_, LocalTile>> : std::true_type {};

template <unsigned Slot_, typename LocalTile>
struct is_shared_tile<SharedTileSlot<Slot_, LocalTile>> : std::true_type {};

template <typename T>
concept is_shared_tile_slot_v = is_shared_tile_slot<T>::value;

// Zero-instruction Shared matrix-role view.  A Shared handle carries storage
// and payload identity; the Left/Right role is metadata consumed by the cube
// operand selector.  This view deliberately aliases the source handle instead
// of creating another Shared allocation or issuing a move/publication.
template <Location Role_, typename SourceSharedTile>
class SharedTileRoleView {
    static_assert(Role_ == Location::Left || Role_ == Location::Right, "SharedTileRoleView role must be Left or Right");
    static_assert(is_shared_tile<SourceSharedTile>::value, "SharedTileRoleView source must be a SharedTile");
    static_assert(
        SourceSharedTile::Role == Location::Left || SourceSharedTile::Role == Location::Right,
        "SharedTileRoleView source must be a Shared matrix");
    static_assert(SourceSharedTile::Role != Role_, "SharedTileRoleView must change the matrix role");
    static_assert(
        SourceSharedTile::BFractal == BLayout::RowMajor && SourceSharedTile::SFractal == SLayout::NoneBox,
        "SharedTileRoleView requires an ordinary RowMajor Shared "
        "matrix layout");

public:
    using Source = SourceSharedTile;
    using LocalTileType = typename SourceSharedTile::LocalTileType;
    using DType = typename SourceSharedTile::DType;
    using TileDType = typename SourceSharedTile::TileDType;

    static constexpr Location Loc = Location::Shared;
    static constexpr Location Role = Role_;
    static constexpr int Rows = SourceSharedTile::Rows;
    static constexpr int Cols = SourceSharedTile::Cols;
    static constexpr int RowStride = SourceSharedTile::RowStride;
    static constexpr int ColStride = SourceSharedTile::ColStride;
    static constexpr int ValidRow = SourceSharedTile::ValidRow;
    static constexpr int ValidCol = SourceSharedTile::ValidCol;
    static constexpr BLayout BFractal = SourceSharedTile::BFractal;
    static constexpr SLayout SFractal = SourceSharedTile::SFractal;
    static constexpr int SFractalSize = SourceSharedTile::SFractalSize;
    static constexpr PadValue PadVal = SourceSharedTile::PadVal;
    static constexpr CompactMode Compact = SourceSharedTile::Compact;
    static constexpr bool IsCubeLayout = SourceSharedTile::IsCubeLayout;
    static constexpr int PhysicalRows = SourceSharedTile::PhysicalRows;
    static constexpr int PhysicalCols = SourceSharedTile::PhysicalCols;
    static constexpr int LogicalTileBytes = SourceSharedTile::LogicalTileBytes;
    static constexpr int TilesizeCode = SourceSharedTile::TilesizeCode;
    static constexpr bool IsValidActiveSize = SourceSharedTile::IsValidActiveSize;
    static constexpr bool isRowMajor = SourceSharedTile::isRowMajor;
    static constexpr bool isBoxedLayout = SourceSharedTile::isBoxedLayout;
    static constexpr bool isInnerRowMajor = SourceSharedTile::isInnerRowMajor;
    static constexpr bool isInnerColMajor = SourceSharedTile::isInnerColMajor;
    static constexpr int InnerRows = SourceSharedTile::InnerRows;
    static constexpr int InnerCols = SourceSharedTile::InnerCols;
    static constexpr int InnerNumel = SourceSharedTile::InnerNumel;
    static constexpr int Numel = SourceSharedTile::Numel;
    static constexpr int byteSize = SourceSharedTile::byteSize;
    static constexpr int kBytes = SourceSharedTile::kBytes;

    explicit constexpr SharedTileRoleView(SourceSharedTile& source) : SourceValue(source) {}

    int GetValidRow() const { return SourceValue.GetValidRow(); }
    int GetValidCol() const { return SourceValue.GetValidCol(); }
    unsigned long handle() const { return SourceValue.handle(); }
    unsigned long& handle_ref() { return SourceValue.handle_ref(); }

private:
    SourceSharedTile& SourceValue;
};

template <Location Role_, typename SourceSharedTile>
struct is_shared_tile<SharedTileRoleView<Role_, SourceSharedTile>> : std::true_type {};

// Reinterpret the matrix role of an existing Shared handle.  This is not
// reinterpret_tile: dtype and the physical storage remain unchanged, and no
// instruction is emitted.  The named lvalue requirement prevents a view from
// outliving its Shared handle source.
template <Location Role_, is_shared_tile_v SourceSharedTile>
inline auto reinterpret_shared_tile(SourceSharedTile& source)
{
    static_assert(
        Role_ == Location::Left || Role_ == Location::Right,
        "reinterpret_shared_tile target role must be Left or Right");
    return SharedTileRoleView<Role_, SourceSharedTile>(source);
}

// An ordinary (non-Shared) Tile that can own a Local register payload.
template <typename T>
concept is_local_tile_v = is_tile<T>::value && T::Loc != Location::Shared;

// TMATMUL matrix operands may live in either Local or Shared storage. Their
// matrix role remains Left/Right; only the instruction operand transport
// changes (Local B.IOT versus Shared B.IOS binder).
template <typename T>
concept is_local_or_shared_left =
    (is_tile<T>::value && T::Loc == Location::Left) || (is_shared_tile<T>::value && T::Role == Location::Left);

template <typename T>
concept is_local_or_shared_right =
    (is_tile<T>::value && T::Loc == Location::Right) || (is_shared_tile<T>::value && T::Role == Location::Right);

// MX scale operands may be any tile in either storage class: Local (Vec or
// Left/Right) matching the plain TMatmulAllOptions usage, or Shared when the
// paired matrix/vector is Shared (handoff Sec 1.5: MX Shared pair is Shared
// B/ScaleB, or all four Shared). Storage pairing is resolved by the emitter's
// A/B dispatch; a mismatched storage fails at the asm constraint (Tr vs Sr),
// so the concept only accepts "some tile", not a specific role or storage.
template <typename T>
concept is_any_tile_data_v = is_tile<T>::value || is_shared_tile<T>::value;

// Matrix role of a tile operand, transparently unwrapping SharedTile. A plain
// Tile's role IS its Loc; a SharedTile preserves the wrapped Local Tile's
// role in ::Role. Dispatch via a helper so a plain Tile is never required to
// define ::Role (the ternary would ill-form otherwise).
namespace detail {
template <typename T, bool IsShared>
struct tile_role_impl {
    static constexpr Location value = T::Loc;
};
template <typename T>
struct tile_role_impl<T, true> {
    static constexpr Location value = T::Role;
};
} // namespace detail
template <typename T>
inline constexpr Location tile_role_v = detail::tile_role_impl<T, is_shared_tile<T>::value>::value;

// detail: Shared handle access for the compiler builtin path. The handle is an
// opaque integer carrying the architectural Shared ID; it must never be forgeable
// by ordinary integers, and must never enter a Tile-vector OverloadType list.
namespace detail {
template <typename LocalTile>
constexpr unsigned long shared_handle(const SharedTile<LocalTile>& tile)
{
    return tile.handle();
}
} // namespace detail

namespace fixp {

struct NoOperand {};

constexpr FixpAttr with_relu(FixpAttr Attr, FixpReluMode Mode)
{
    Attr.Relu = Mode;
    return Attr;
}

constexpr FixpAttr with_row_max(FixpAttr Attr, bool Init)
{
    Attr.RowMaxEn = true;
    Attr.RowMaxInit = Init;
    return Attr;
}

constexpr FixpAttr with_group_max(FixpAttr Attr, uint8_t GroupNCode)
{
    Attr.GroupMaxEn = true;
    Attr.GroupNCode = GroupNCode;
    return Attr;
}

constexpr FixpAttr with_max_abs(FixpAttr Attr)
{
    Attr.MaxAbsEn = true;
    return Attr;
}

constexpr int group_n_from_code(uint8_t Code)
{
    constexpr int Values[] = {0, 8, 16, 32, 48, 64, 80, 96, 112, 128};
    return Code <= 9 ? Values[Code] : 0;
}

template <int GroupN>
constexpr uint8_t group_n_code()
{
    static_assert(
        GroupN == 8 || GroupN == 16 || GroupN == 32 || GroupN == 48 || GroupN == 64 || GroupN == 80 || GroupN == 96 ||
            GroupN == 112 || GroupN == 128,
        "FPATR config GroupN must be 8, 16, 32, 48, 64, 80, 96, "
        "112 or 128");
    if constexpr (GroupN == 8)
        return 1;
    if constexpr (GroupN == 16)
        return 2;
    if constexpr (GroupN == 32)
        return 3;
    if constexpr (GroupN == 48)
        return 4;
    if constexpr (GroupN == 64)
        return 5;
    if constexpr (GroupN == 80)
        return 6;
    if constexpr (GroupN == 96)
        return 7;
    if constexpr (GroupN == 112)
        return 8;
    return 9;
}

template <
    FixpAttr Attr_, typename QuantTile_ = NoOperand, typename ReluTile_ = NoOperand, typename RowMaxIn_ = NoOperand,
    typename RowMaxOut_ = NoOperand, typename GroupMaxOut_ = NoOperand, typename CScaleTile_ = NoOperand>
struct Options {
    static constexpr FixpAttr Attr = Attr_;
    using QuantTile = QuantTile_;
    using ReluTile = ReluTile_;
    using RowMaxIn = RowMaxIn_;
    using RowMaxOut = RowMaxOut_;
    using GroupMaxOut = GroupMaxOut_;
    using CScaleTile = CScaleTile_;

    uint64_t QuantDescriptor = 0;
    uint64_t LReluDescriptor = 0;
    QuantTile* Quant = nullptr;
    ReluTile* Relu = nullptr;
    RowMaxIn* RowIn = nullptr;
    RowMaxOut* RowOut = nullptr;
    GroupMaxOut* GroupOut = nullptr;
    CScaleTile* CScale = nullptr;

    constexpr Options() = default;

    constexpr Options(
        uint64_t QuantDescriptor, uint64_t LReluDescriptor, QuantTile* Quant, ReluTile* Relu, RowMaxIn* RowIn,
        RowMaxOut* RowOut, GroupMaxOut* GroupOut, CScaleTile* CScale = nullptr)
        : QuantDescriptor(QuantDescriptor),
          LReluDescriptor(LReluDescriptor),
          Quant(Quant),
          Relu(Relu),
          RowIn(RowIn),
          RowOut(RowOut),
          GroupOut(GroupOut),
          CScale(CScale)
    {}

    template <bool Enable = true>
    constexpr auto transpose_a() const
    {
        constexpr FixpAttr NewAttr = Attr.transpose_a(Enable);
        return Options<NewAttr, QuantTile, ReluTile, RowMaxIn, RowMaxOut, GroupMaxOut, CScaleTile>(
            QuantDescriptor, LReluDescriptor, Quant, Relu, RowIn, RowOut, GroupOut, CScale);
    }

    template <bool Enable = true>
    constexpr auto transpose_b() const
    {
        constexpr FixpAttr NewAttr = Attr.transpose_b(Enable);
        return Options<NewAttr, QuantTile, ReluTile, RowMaxIn, RowMaxOut, GroupMaxOut, CScaleTile>(
            QuantDescriptor, LReluDescriptor, Quant, Relu, RowIn, RowOut, GroupOut, CScale);
    }

    template <bool Enable = true>
    constexpr auto cscale_enable() const
    {
        constexpr FixpAttr NewAttr = Attr.cscale_enable(Enable);
        return Options<NewAttr, QuantTile, ReluTile, RowMaxIn, RowMaxOut, GroupMaxOut, CScaleTile>(
            QuantDescriptor, LReluDescriptor, Quant, Relu, RowIn, RowOut, GroupOut, CScale);
    }

    constexpr auto relu() const
    {
        static_assert(Attr.Relu == FixpReluMode::None, "FPATR config ReLU mode was already selected");
        constexpr FixpAttr NewAttr = with_relu(Attr, FixpReluMode::Relu);
        return Options<NewAttr, QuantTile, ReluTile, RowMaxIn, RowMaxOut, GroupMaxOut, CScaleTile>(
            QuantDescriptor, LReluDescriptor, Quant, Relu, RowIn, RowOut, GroupOut, CScale);
    }

    constexpr auto lrelu(uint64_t Descriptor) const
    {
        static_assert(Attr.Relu == FixpReluMode::None, "FPATR config ReLU mode was already selected");
        constexpr FixpAttr NewAttr = with_relu(Attr, FixpReluMode::LRelu);
        return Options<NewAttr, QuantTile, ReluTile, RowMaxIn, RowMaxOut, GroupMaxOut, CScaleTile>(
            QuantDescriptor, Descriptor, Quant, Relu, RowIn, RowOut, GroupOut, CScale);
    }

    template <is_local_tile_v Tile>
    constexpr auto prelu(Tile& Parameter) const
    {
        static_assert(Attr.Relu == FixpReluMode::None, "FPATR config ReLU mode was already selected");
        static_assert(std::is_same_v<ReluTile, NoOperand>, "FPATR config PReLU Tile was already supplied");
        constexpr FixpAttr NewAttr = with_relu(Attr, FixpReluMode::PRelu);
        return Options<NewAttr, QuantTile, Tile, RowMaxIn, RowMaxOut, GroupMaxOut, CScaleTile>(
            QuantDescriptor, LReluDescriptor, Quant, &Parameter, RowIn, RowOut, GroupOut, CScale);
    }

    template <is_local_tile_v Tile>
    constexpr auto row_max(Tile& Output) const
    {
        static_assert(!Attr.RowMaxEn, "FPATR config RowMax was already enabled");
        static_assert(
            std::is_same_v<RowMaxIn, NoOperand> && std::is_same_v<RowMaxOut, NoOperand>,
            "FPATR config RowMax operands were already supplied");
        constexpr FixpAttr NewAttr = with_row_max(Attr, false);
        return Options<NewAttr, QuantTile, ReluTile, NoOperand, Tile, GroupMaxOut, CScaleTile>(
            QuantDescriptor, LReluDescriptor, Quant, Relu, nullptr, &Output, GroupOut, CScale);
    }

    template <is_local_tile_v InputTile, is_local_tile_v OutputTile>
    constexpr auto row_max(InputTile& Input, OutputTile& Output) const
    {
        static_assert(!Attr.RowMaxEn, "FPATR config RowMax was already enabled");
        static_assert(
            std::is_same_v<RowMaxIn, NoOperand> && std::is_same_v<RowMaxOut, NoOperand>,
            "FPATR config RowMax operands were already supplied");
        constexpr FixpAttr NewAttr = with_row_max(Attr, true);
        return Options<NewAttr, QuantTile, ReluTile, InputTile, OutputTile, GroupMaxOut, CScaleTile>(
            QuantDescriptor, LReluDescriptor, Quant, Relu, &Input, &Output, GroupOut, CScale);
    }

    template <int GroupN, is_local_tile_v Tile>
    constexpr auto group_max(Tile& Output) const
    {
        static_assert(!Attr.GroupMaxEn, "FPATR config GroupMax was already enabled");
        static_assert(std::is_same_v<GroupMaxOut, NoOperand>, "FPATR config GroupMax output was already supplied");
        constexpr FixpAttr NewAttr = with_group_max(Attr, group_n_code<GroupN>());
        return Options<NewAttr, QuantTile, ReluTile, RowMaxIn, RowMaxOut, Tile, CScaleTile>(
            QuantDescriptor, LReluDescriptor, Quant, Relu, RowIn, RowOut, &Output, CScale);
    }

    constexpr auto max_abs() const
    {
        static_assert(Attr.RowMaxEn || Attr.GroupMaxEn, "FPATR config max_abs requires RowMax or GroupMax");
        constexpr FixpAttr NewAttr = with_max_abs(Attr);
        return Options<NewAttr, QuantTile, ReluTile, RowMaxIn, RowMaxOut, GroupMaxOut, CScaleTile>(
            QuantDescriptor, LReluDescriptor, Quant, Relu, RowIn, RowOut, GroupOut, CScale);
    }

    template <is_local_tile_v Tile>
    constexpr auto cscale(Tile& Scale) const
    {
        static_assert(std::is_same_v<CScaleTile, NoOperand>, "FPATR CScale tile was already supplied");
        constexpr FixpAttr NewAttr = Attr.cscale_enable();
        return Options<NewAttr, QuantTile, ReluTile, RowMaxIn, RowMaxOut, GroupMaxOut, Tile>(
            QuantDescriptor, LReluDescriptor, Quant, Relu, RowIn, RowOut, GroupOut, &Scale);
    }

    // CUBE InternalAcc controls (issue: PTO-ISA spec#236). These only change
    // the B.DATR CCTRL bits; operands and all other attributes are carried.
    constexpr auto raw_acc() const
    {
        static_assert(
            Attr.CubeCtrl == CubeCtrlNone || Attr.CubeCtrl == CubeCtrlRawAccAndHint,
            "CCTRL raw-accumulator bit was already configured");
        constexpr FixpAttr NewAttr =
            Attr.with_cube_ctrl(static_cast<CubeControl>(Attr.CubeCtrl | CubeCtrlRawAccumulator));
        return Options<NewAttr, QuantTile, ReluTile, RowMaxIn, RowMaxOut, GroupMaxOut, CScaleTile>(
            QuantDescriptor, LReluDescriptor, Quant, Relu, RowIn, RowOut, GroupOut, CScale);
    }
    constexpr auto acc_hint() const
    {
        static_assert(
            Attr.CubeCtrl == CubeCtrlNone || Attr.CubeCtrl == CubeCtrlRawAccumulator,
            "CCTRL InternalAcc hint bit was already configured");
        constexpr FixpAttr NewAttr =
            Attr.with_cube_ctrl(static_cast<CubeControl>(Attr.CubeCtrl | CubeCtrlInternalAccHint));
        return Options<NewAttr, QuantTile, ReluTile, RowMaxIn, RowMaxOut, GroupMaxOut, CScaleTile>(
            QuantDescriptor, LReluDescriptor, Quant, Relu, RowIn, RowOut, GroupOut, CScale);
    }
};

template <FixpPreQuantMode Mode = FixpPreQuantMode::None>
constexpr auto convert()
{
    static_assert(is_parameter_free_fixp_pre_quant(Mode), "fixp::convert accepts only parameter-free PreQuant modes");
    constexpr FixpAttr Attr = [] {
        FixpAttr Value;
        Value.PreQuant = Mode;
        return Value;
    }();
    return Options<Attr>{};
}

template <FixpPreQuantMode Mode>
constexpr auto scalar(uint64_t QuantDescriptor)
{
    static_assert(is_scalar_fixp_pre_quant(Mode), "fixp::scalar requires a scalar-parameter PreQuant mode");
    constexpr FixpAttr Attr = [] {
        FixpAttr Value;
        Value.PreQuant = Mode;
        return Value;
    }();
    return Options<Attr>(QuantDescriptor, 0, nullptr, nullptr, nullptr, nullptr, nullptr);
}

template <FixpPreQuantMode Mode, is_local_tile_v Tile>
constexpr auto vector(Tile& QuantParameter)
{
    static_assert(is_vector_fixp_pre_quant(Mode), "fixp::vector requires a vector-parameter PreQuant mode");
    constexpr FixpAttr Attr = [] {
        FixpAttr Value;
        Value.PreQuant = Mode;
        return Value;
    }();
    return Options<Attr, Tile>(0, 0, &QuantParameter, nullptr, nullptr, nullptr, nullptr);
}

constexpr auto keep_acc() { return convert<FixpPreQuantMode::None>(); }
constexpr auto f16() { return convert<FixpPreQuantMode::F322F16>(); }
constexpr auto bf16() { return convert<FixpPreQuantMode::F322BF16>(); }
constexpr auto s8(uint64_t QuantDescriptor) { return scalar<FixpPreQuantMode::QF322S8Pre>(QuantDescriptor); }

template <is_local_tile_v Tile>
constexpr auto s8(Tile& QuantParameter)
{
    return vector<FixpPreQuantMode::VQF322S8Pre>(QuantParameter);
}

template <typename T>
struct is_options : std::false_type {};
template <
    FixpAttr Attr, typename QuantTile, typename ReluTile, typename RowMaxIn, typename RowMaxOut, typename GroupMaxOut,
    typename CScaleTile>
struct is_options<Options<Attr, QuantTile, ReluTile, RowMaxIn, RowMaxOut, GroupMaxOut, CScaleTile>> : std::true_type {};

template <typename T>
concept is_options_v = is_options<std::remove_cv_t<T>>::value;

} // namespace fixp

template <typename shape>
int index(int i, int j)
{
    if constexpr (is_global_data_v<shape>) {
        return i * shape::RowStride + j * shape::ColStride;
    } else if constexpr (is_tile_data_v<shape>) {
        if constexpr (shape::IsCubeLayout) {
            return shape::CubeStorageIndex(i, j);
        } else if constexpr (is_boxed_data_v<shape>) {
            int sub_tile_i = i / shape::InnerRows;
            int sub_tile_j = j / shape::InnerCols;
            int idx_i = i % shape::InnerRows;
            int idx_j = j % shape::InnerCols;
            if constexpr (is_Nz_layout<shape>::value) {
                return sub_tile_j * shape::Rows * shape::InnerCols + sub_tile_i * shape::InnerNumel +
                       idx_i * shape::InnerCols + idx_j;
            } else if constexpr (is_Zn_layout<shape>::value) {
                return sub_tile_i * shape::Cols * shape::InnerRows + sub_tile_j * shape::InnerNumel + idx_i +
                       idx_j * shape::InnerRows;
            } else {
                static_assert((is_Nz_layout<shape>::value) || (is_Zn_layout<shape>::value), "illegal layout");
            }
        } else {
            return i * shape::RowStride + j * shape::ColStride;
        }
    }
}

template <typename tile_shape>
const char* get_layout_str()
{
    if constexpr (!tile_shape::isBoxedLayout) {
        if constexpr (tile_shape::isRowMajor)
            return "RowMajor";
        return "ColMajor";
    }
    if constexpr (is_Nz_layout<tile_shape>::value)
        return "NzLayout";
    if constexpr (is_Zn_layout<tile_shape>::value)
        return "ZnLayout";
    return "Other";
}

template <typename tile_shape>
void print_tile_info()
{
#ifndef __linx
    std::cout << "Tile Rows Number: " << tile_shape::Rows << std::endl;
    std::cout << "Tile Columns Number: " << tile_shape::Cols << std::endl;
    std::cout << "Tile Active Rows Number: " << tile_shape::ValidRow << std::endl;
    std::cout << "Tile Active Columns Number: " << tile_shape::ValidCol << std::endl;
    if constexpr (tile_shape::isBoxedLayout) {
        std::cout << "Tile Fractal Inner Rows Number: " << tile_shape::InnerRows << std::endl;
        std::cout << "Tile Fractal Inner Columns Number: " << tile_shape::InnerCols << std::endl;
    }
    std::cout << "Tile Size: " << tile_shape::Numel << std::endl;
    std::cout << "Tile Layout: " << get_layout_str<tile_shape>() << std::endl;
    std::cout << "Tile Data Dump: " << std::endl;
#else
    (void)sizeof(tile_shape);
#endif
}

} // namespace pto

//===--- Tile datatype reinterpret view (PTO v0.58) ---===//
// reinterpret_tile<NewDType>(src): zero-instruction datatype reinterpret.
// The underlying Tile register/storage bit pattern is unchanged; only the
// static DType (and downstream ISA datatype encoding) is re-interpreted by
// the view. First phase: Local Tile -> Local view, equal-bit-width only,
// layout/shape/valid/Location preserved, no TCVT, no payload copy.
namespace pto {

// Whether T has a PTO TypeCode (has a type_traits specialization).
template <typename T, typename = void>
struct has_ptotype_traits : std::false_type {};
template <typename T>
struct has_ptotype_traits<T, std::void_t<decltype(type_traits<T>::TypeCode)>> : std::true_type {};

template <typename T>
inline constexpr bool is_supported_dtype_v =
    has_ptotype_traits<std::remove_cv_t<T>>::value && (type_traits<std::remove_cv_t<T>>::bits > 0);

// Zero-instruction datatype view over an existing Local Tile. Storage carrier
// (TileDType) and data() forward to the source; DType and all shape/role
// statics are re-declared so downstream ops see NewDType while binding the
// source's exact Tile register.
template <typename NewDType, typename SourceTile>
class ReinterpretedTileView {
public:
    using DType = NewDType;
    using Source = SourceTile;
    // Same storage carrier as the source Tile, so its physical bytes and
    // tile_type_traits<...TileDType>::TilesizeCode are unchanged by the
    // reinterpret.
    using TileDType = typename SourceTile::TileDType;

    static constexpr Location Loc = SourceTile::Loc;
    static constexpr int Rows = SourceTile::Rows;
    static constexpr int Cols = SourceTile::Cols;
    static constexpr int RowStride = SourceTile::RowStride;
    static constexpr int ColStride = SourceTile::ColStride;
    static constexpr int ValidRow = SourceTile::ValidRow;
    static constexpr int ValidCol = SourceTile::ValidCol;
    static constexpr BLayout BFractal = SourceTile::BFractal;
    static constexpr SLayout SFractal = SourceTile::SFractal;
    static constexpr int SFractalSize = SourceTile::SFractalSize;
    static constexpr PadValue PadVal = SourceTile::PadVal;
    static constexpr CompactMode Compact = SourceTile::Compact;
    static constexpr bool isRowMajor = SourceTile::isRowMajor;
    static constexpr bool isBoxedLayout = SourceTile::isBoxedLayout;
    static constexpr bool isInnerRowMajor = SourceTile::isInnerRowMajor;
    static constexpr bool isInnerColMajor = SourceTile::isInnerColMajor;
    static constexpr int InnerRows = SourceTile::InnerRows;
    static constexpr int InnerCols = SourceTile::InnerCols;
    static constexpr int InnerNumel = SourceTile::InnerNumel;
    static constexpr int Numel = SourceTile::Numel;
    static constexpr int byteSize = SourceTile::byteSize;
    // Physical storage identity: the view occupies exactly the source bytes.
    static constexpr int kBytes = SourceTile::kBytes;
    static constexpr int StorageBytes = SourceTile::StorageBytes;
    static constexpr int LogicalTileBytes = SourceTile::LogicalTileBytes;
    static constexpr int TilesizeCode = SourceTile::TilesizeCode;
    static constexpr bool IsValidActiveSize = SourceTile::IsValidActiveSize;
    static constexpr bool IsCubeLayout = SourceTile::IsCubeLayout;
    static constexpr int PhysicalRows = SourceTile::PhysicalRows;
    static constexpr int PhysicalCols = SourceTile::PhysicalCols;

    explicit constexpr ReinterpretedTileView(SourceTile& Source) : SourceValue(Source) {}

    // Same register carrier as the source (no copy). Only const access is
    // exposed for const sources; the non-const path keeps the same carrier.
    decltype(auto) data() { return SourceValue.data(); }
    decltype(auto) data() const { return SourceValue.data(); }

    template <int RowMask = ValidRow>
    static constexpr std::enable_if_t<(RowMask > 0), int> GetValidRow()
    {
        return SourceTile::template GetValidRow<RowMask>();
    }
    template <int RowMask = ValidRow>
    std::enable_if_t<RowMask == -1, int> GetValidRow() const
    {
        return SourceValue.GetValidRow();
    }
    template <int ColMask = ValidCol>
    static constexpr std::enable_if_t<(ColMask > 0), int> GetValidCol()
    {
        return SourceTile::template GetValidCol<ColMask>();
    }
    template <int ColMask = ValidCol>
    std::enable_if_t<ColMask == -1, int> GetValidCol() const
    {
        return SourceValue.GetValidCol();
    }

private:
    SourceTile& SourceValue;
};

// PTO 0.58.7 TEXPDIF legality. A reinterpret view exposes the operation dtype
// while retaining its Source::DType as the physical backing carrier.
template <typename Source, typename Destination>
constexpr bool texpdif_type_pair_legal_v =
    (std::is_same_v<Source, __half> && (std::is_same_v<Destination, __half> || std::is_same_v<Destination, __fp32>)) ||
    (std::is_same_v<Source, __bf16> && (std::is_same_v<Destination, __bf16> || std::is_same_v<Destination, __fp32>)) ||
    (std::is_same_v<Source, __fp32> && std::is_same_v<Destination, __fp32>);

template <typename TileT>
struct texpdif_backing_dtype {
    using type = typename TileT::DType;
};
template <typename NewDType, typename SourceTile>
struct texpdif_backing_dtype<ReinterpretedTileView<NewDType, SourceTile>> {
    using type = typename SourceTile::DType;
};
template <typename TileT>
using texpdif_backing_dtype_t = typename texpdif_backing_dtype<TileT>::type;

template <typename D, typename A, typename B>
constexpr void validate_texpdif_operands()
{
    static_assert(
        D::Loc != Location::Shared && A::Loc != Location::Shared && B::Loc != Location::Shared,
        "TEXPDIF requires Local operands; Shared tiles are illegal");
    static_assert(
        A::BFractal == B::BFractal && D::BFractal == A::BFractal, "TEXPDIF source and destination layouts must match");
    static_assert(
        A::BFractal == BLayout::RowMajor || A::BFractal == BLayout::CubeM16 || A::BFractal == BLayout::CubeM32,
        "TEXPDIF supports RowMajor, CUBE_M16, and CUBE_M32 only");
    static_assert(
        A::Rows == B::Rows && A::Cols == B::Cols && D::Rows == A::Rows && D::Cols == A::Cols &&
            A::ValidRow == B::ValidRow && A::ValidCol == B::ValidCol && D::ValidRow == A::ValidRow &&
            D::ValidCol == A::ValidCol,
        "TEXPDIF source and destination logical shapes must match");
    static_assert(
        std::is_same_v<typename A::DType, typename B::DType>, "TEXPDIF sources must select the same operation dtype");
    static_assert(
        texpdif_type_pair_legal_v<typename A::DType, typename D::DType>,
        "TEXPDIF supports only FP16/BF16/FP32 source pairs and FP32 widening");
    static_assert(
        type_traits<texpdif_backing_dtype_t<A>>::bits == type_traits<typename A::DType>::bits &&
            type_traits<texpdif_backing_dtype_t<B>>::bits == type_traits<typename B::DType>::bits,
        "TEXPDIF source backing carriers must be non-packed and equal-width");
}

// A ReinterpretedTileView is a Local tile-shaped operand (not Shared).
template <typename NewDType, typename SourceTile>
struct is_tile<ReinterpretedTileView<NewDType, SourceTile>> : std::true_type {
    static constexpr SLayout layout_enum = SourceTile::SFractal;
};

// ---- Range modifiers (PTO-ISA 0.58.4, ADR-0098) --------------------------
//
// B.SUBVIEW / B.ASSEMBLE attach to the immediately preceding B.IOT/B.IOS
// binder and carry a range descriptor (RegSrc + uimm11 offset + parent size).
// These wrappers model source (Subview) and destination (Assemble) range
// carriers without creating a second Tile register namespace: they forward
// the parent's shape/dtype/storage and expose the modifier runtime fields.
namespace range {

// RegSrc=24 is an internal marker for a runtime base register selected by
// the compiler. It is not an encodable ISA register selector.
inline constexpr unsigned AutoRegSrc = 24;
inline constexpr std::size_t RangeAddressUnitBytes = 128;

constexpr bool is_valid_parent_size_code(unsigned code)
{
    return code <= 12; // 0 is legal on non-INIT modifiers; 13..15 reserved
}

// PTO 0.58.6 TIMG2COL accepts the scalar feature-map types below.  Keep this
// predicate separate from GMOV and matrix predicates: TIMG2COL has its own
// architectural data-type domain and, in particular, does not accept packed
// MX carriers merely because they have a one-byte storage representation.
constexpr bool is_timg2col_type_code(int type_code)
{
    switch (type_code) {
        case __type_fp32:
        case __type_tf32:
        case __type_hf32:
        case __type_fp16:
        case __type_bf16:
        case __type_hif8:
        case __type_fp8_e4m3:
        case __type_fp8_e5m2:
        case __type_fp8_e8m0:
        case __type_int32:
        case __type_int16:
        case __type_int8:
        case __type_uint32:
        case __type_uint16:
        case __type_uint8:
            return true;
        default:
            return false;
    }
}

constexpr bool is_timg2col_single_pe_mask(unsigned mask) { return mask == 1 || mask == 2 || mask == 4 || mask == 8; }
constexpr bool is_valid_subview_size_code(unsigned code) { return code >= 1 && code <= 12; }
constexpr bool is_valid_uimm11(unsigned u) { return u <= 2047; }
constexpr unsigned subview_size_code_for_bytes(std::size_t bytes)
{
    switch (bytes) {
        case 128:
            return __tilesize_128B;
        case 256:
            return __tilesize_256B;
        case 512:
            return __tilesize_512B;
        case 1024:
            return __tilesize_1KB;
        case 2048:
            return __tilesize_2KB;
        case 4096:
            return __tilesize_4KB;
        case 8192:
            return __tilesize_8KB;
        case 16384:
            return __tilesize_16KB;
        case 32768:
            return __tilesize_32KB;
        case 65536:
            return __tilesize_64KB;
        case 131072:
            return __tilesize_128KB;
        case 262144:
            return __tilesize_256KB;
        default:
            return 0;
    }
}
constexpr std::size_t subview_bytes_for_size_code(unsigned code)
{
    return is_valid_subview_size_code(code) ? (static_cast<std::size_t>(128) << (code - 1)) : 0;
}

// PTO-ISA BundleRangeSubviewLegal: B.SUBVIEW is defined for an assigned
// Local or Shared tile using persistent CUBE CELL storage. The Local form is
// selected by layout; Location::Vec is a valid Local CUBE carrier. SharedTile
// wraps a Local Tile, so its subview parent is the wrapped LocalTile; ordinary
// Tiles are the parent themselves.
template <typename Parent, typename = void>
struct subview_parent_tile {
    using type = Parent;
};

template <typename LocalTile>
struct subview_parent_tile<SharedTile<LocalTile>> {
    using type = LocalTile;
};

template <typename Parent>
using subview_parent_tile_t = typename subview_parent_tile<Parent>::type;

template <typename Parent, typename = void>
struct is_legal_subview_parent : std::false_type {};

// The Local and Shared B.SUBVIEW contracts require a persistent CUBE layout.
// SharedTile wraps a Local Tile, so checking the wrapped tile's layout covers
// both carrier kinds.
template <typename Parent>
struct is_legal_subview_parent<Parent, std::void_t<decltype(subview_parent_tile_t<Parent>::IsCubeLayout)>>
    : std::bool_constant<subview_parent_tile_t<Parent>::IsCubeLayout> {};

template <typename Parent>
inline constexpr bool is_legal_subview_parent_v = is_legal_subview_parent<Parent>::value;

/// Source-side range carrier. Forwards every tile-shaped static member of
/// Parent so it can be bound as a Local operand; the B.SUBVIEW line is
/// emitted after the source binder by the consuming operation.
///
/// uimm11 Offset is a compile-time count of 128-byte range units. High-level
/// factories either use the zero register or mark RegSrc as AutoRegSrc so
/// inline asm can allocate a GPR for the runtime base-unit value. Explicit
/// RegSrc 0..23 remains available to the low-level carrier and *_reg helpers
/// for fixed ABI and encoding tests.
template <typename Parent, unsigned SubviewSizeCode_, unsigned OffsetUnits_ = 0, unsigned RegSrc_ = 2>
class Subview {
    static_assert(
        is_legal_subview_parent_v<Parent>, "B.SUBVIEW parent must use an assigned Local or Shared CUBE "
                                           "tile layout");
    static_assert(
        is_valid_subview_size_code(SubviewSizeCode_), "B.SUBVIEW SubviewSizeCode must be 1..12 (128B..256KB per PE)");
    static_assert(is_valid_uimm11(OffsetUnits_), "B.SUBVIEW uimm11 offset must be 0..2047");
    static_assert(
        subview_bytes_for_size_code(SubviewSizeCode_) <= Parent::LogicalTileBytes,
        "B.SUBVIEW length cannot exceed the parent Tile capacity");
    static_assert(
        subview_parent_tile_t<Parent>::CubePhysicalCellRangeComplete(
            OffsetUnits_, subview_bytes_for_size_code(SubviewSizeCode_) / RangeAddressUnitBytes),
        "B.SUBVIEW CUBE_M32 64-bit range must cover complete CELL pairs");
    static_assert(
        RegSrc_ <= 23 || RegSrc_ == AutoRegSrc,
        "B.SUBVIEW RegSrc must be an absolute GPR selector 0..23 or AutoRegSrc");

public:
    using DType = typename Parent::DType;
    using ParentTile = Parent;
    using TileDType = typename Parent::TileDType;

    static constexpr unsigned SubviewSizeCode = SubviewSizeCode_;
    static constexpr unsigned OffsetUnits = OffsetUnits_;
    static constexpr unsigned Offset = OffsetUnits;
    static constexpr unsigned RegSrc = RegSrc_;
    static constexpr Location Loc = Parent::Loc;
    static constexpr int Rows = Parent::Rows;
    static constexpr int Cols = Parent::Cols;
    static constexpr int RowStride = Parent::RowStride;
    static constexpr int ColStride = Parent::ColStride;
    static constexpr int ValidRow = Parent::ValidRow;
    static constexpr int ValidCol = Parent::ValidCol;
    static constexpr BLayout BFractal = Parent::BFractal;
    static constexpr SLayout SFractal = Parent::SFractal;
    static constexpr int SFractalSize = Parent::SFractalSize;
    static constexpr bool isRowMajor = Parent::isRowMajor;
    static constexpr bool isBoxedLayout = Parent::isBoxedLayout;
    static constexpr bool isInnerRowMajor = Parent::isInnerRowMajor;
    static constexpr bool isInnerColMajor = Parent::isInnerColMajor;
    static constexpr int InnerRows = Parent::InnerRows;
    static constexpr int InnerCols = Parent::InnerCols;
    static constexpr int Numel = Parent::Numel;
    static constexpr bool IsCubeLayout = Parent::IsCubeLayout;
    static constexpr int PhysicalRows = Parent::PhysicalRows;
    static constexpr int PhysicalCols = Parent::PhysicalCols;
    static constexpr int LogicalTileBytes = Parent::LogicalTileBytes;
    static constexpr int TilesizeCode = Parent::TilesizeCode;
    static constexpr bool IsValidActiveSize = Parent::IsValidActiveSize;

    Subview(Parent& parent, uintptr_t range_base = 0) : ParentValue(parent), RangeBaseValue(range_base) {}

    // Local sources bind through data(); Shared sources bind through their
    // opaque S-register handle and are emitted as B.IOS by the consumer.
    decltype(auto) data()
        requires(!is_shared_tile_v<Parent>)
    {
        return ParentValue.data();
    }
    decltype(auto) data() const
        requires(!is_shared_tile_v<Parent>)
    {
        return ParentValue.data();
    }
    unsigned long handle()
        requires(is_shared_tile_v<Parent>)
    {
        return ParentValue.handle();
    }
    unsigned long handle() const
        requires(is_shared_tile_v<Parent>)
    {
        return ParentValue.handle();
    }

    int GetValidRow() const { return ParentValue.GetValidRow(); }
    int GetValidCol() const { return ParentValue.GetValidCol(); }
    uintptr_t GetRangeBase() const { return RangeBaseValue; }

private:
    Parent& ParentValue;
    uintptr_t RangeBaseValue;
};

/// Destination-side range carrier. Capable of the multi-PE Shared
/// destination requirement (operation enforces it); carries INIT/LAST and
/// ParentSizeCode. INIT/LAST/OffsetUnits/RegSrc are compile-time constants for
/// the same inline-asm "i" constraint reason as Subview: the range
/// descriptor is a static part of the destination binder contract. The
/// runtime base and compile-time offset are both counts of 128-byte units.
/// High-level factories hide RegSrc; explicit selectors remain available to
/// the low-level carrier and *_reg helpers for ABI and encoding tests.
template <
    typename Parent, unsigned ParentSizeCode_, bool INIT_ = true, bool LAST_ = false, unsigned OffsetUnits_ = 0,
    unsigned RegSrc_ = 2, unsigned WriterSizeCode_ = 0>
class Assemble {
    static_assert(
        is_valid_parent_size_code(ParentSizeCode_), "B.ASSEMBLE ParentSizeCode must be 0..12; 13..15 reserved");
    static_assert(
        !((INIT_ == false) && (ParentSizeCode_ != 0)), "B.ASSEMBLE: non-INIT modifier requires ParentSizeCode=0");
    static_assert(
        !((INIT_ != false) && (ParentSizeCode_ == 0)), "B.ASSEMBLE: INIT modifier requires ParentSizeCode 1..12");
    static_assert(is_valid_uimm11(OffsetUnits_), "B.ASSEMBLE uimm11 offset must be 0..2047");
    static_assert(
        !INIT_ || subview_bytes_for_size_code(ParentSizeCode_) <= Parent::LogicalTileBytes,
        "B.ASSEMBLE length cannot exceed the parent Tile capacity");
    static constexpr unsigned EffectiveWriterSizeCode =
        WriterSizeCode_ != 0 ? WriterSizeCode_ : (INIT_ ? ParentSizeCode_ : Parent::TilesizeCode);
    static_assert(
        subview_parent_tile_t<Parent>::CubePhysicalCellRangeComplete(
            OffsetUnits_, subview_bytes_for_size_code(EffectiveWriterSizeCode) / RangeAddressUnitBytes),
        "B.ASSEMBLE CUBE_M32 64-bit range must cover complete CELL pairs");
    static_assert(
        RegSrc_ <= 23 || RegSrc_ == AutoRegSrc,
        "B.ASSEMBLE RegSrc must be an absolute GPR selector 0..23 or AutoRegSrc");

public:
    using DType = typename Parent::DType;
    using ParentTile = Parent;
    using TileDType = typename Parent::TileDType;

    static constexpr unsigned ParentSizeCode = ParentSizeCode_;
    static constexpr bool INIT = INIT_;
    static constexpr bool LAST = LAST_;
    static constexpr unsigned OffsetUnits = OffsetUnits_;
    static constexpr unsigned Offset = OffsetUnits;
    static constexpr unsigned RegSrc = RegSrc_;
    // PTO-ISA #265: the B.ASSEMBLE field-5 writer extent for this carrier.
    // The factories derive it from the declared fragment length; 0 means the
    // legacy fallback (derive from the parent carrier capacity).
    static constexpr unsigned WriterSizeCode = WriterSizeCode_;
    static constexpr Location Loc = Parent::Loc;
    static constexpr int Rows = Parent::Rows;
    static constexpr int Cols = Parent::Cols;
    static constexpr int RowStride = Parent::RowStride;
    static constexpr int ColStride = Parent::ColStride;
    static constexpr int ValidRow = Parent::ValidRow;
    static constexpr int ValidCol = Parent::ValidCol;
    static constexpr BLayout BFractal = Parent::BFractal;
    static constexpr SLayout SFractal = Parent::SFractal;
    static constexpr int SFractalSize = Parent::SFractalSize;
    static constexpr bool isRowMajor = Parent::isRowMajor;
    static constexpr bool isBoxedLayout = Parent::isBoxedLayout;
    static constexpr bool isInnerRowMajor = Parent::isInnerRowMajor;
    static constexpr bool isInnerColMajor = Parent::isInnerColMajor;
    static constexpr int InnerRows = Parent::InnerRows;
    static constexpr int InnerCols = Parent::InnerCols;
    static constexpr int Numel = Parent::Numel;
    static constexpr bool IsCubeLayout = Parent::IsCubeLayout;
    static constexpr int PhysicalRows = Parent::PhysicalRows;
    static constexpr int PhysicalCols = Parent::PhysicalCols;
    static constexpr int LogicalTileBytes = Parent::LogicalTileBytes;
    static constexpr int TilesizeCode = Parent::TilesizeCode;
    static constexpr bool IsValidActiveSize = Parent::IsValidActiveSize;

    Assemble(Parent& parent, uintptr_t range_base = 0) : ParentValue(parent), RangeBaseValue(range_base) {}

    // Local binds through data(); SharedTile binds through handle() (B.IOS).
    decltype(auto) data()
        requires(!is_shared_tile_v<Parent>)
    {
        return ParentValue.data();
    }
    decltype(auto) data() const
        requires(!is_shared_tile_v<Parent>)
    {
        return ParentValue.data();
    }
    unsigned long handle()
        requires(is_shared_tile_v<Parent>)
    {
        return ParentValue.handle();
    }
    unsigned long handle() const
        requires(is_shared_tile_v<Parent>)
    {
        return ParentValue.handle();
    }
    unsigned long& handle_ref()
        requires(is_shared_tile_v<Parent>)
    {
        return ParentValue.handle_ref();
    }

    int GetValidRow() const { return ParentValue.GetValidRow(); }
    int GetValidCol() const { return ParentValue.GetValidCol(); }
    uintptr_t GetRangeBase() const { return RangeBaseValue; }

    void UpdateAssembledValidShape(int rows, int columns)
        requires(is_shared_tile_v<Parent>)
    {
        if constexpr (INIT)
            ParentValue.BeginAssembledValidShape(rows, columns);
        else
            ParentValue.MergeAssembledValidShape(rows, columns);
        if constexpr (LAST)
            ParentValue.PublishAssembledValidShape();
    }

private:
    Parent& ParentValue;
    uintptr_t RangeBaseValue;
};

// Ergonomic range factories. Both the length and the offset are expressed
// in 128-byte range units (the uimm11 granularity), so a call site counts
// fragments and positions in the same unit. Length 0 derives the parent's
// full capacity.
template <std::size_t LengthUnits_ = 0, unsigned OffsetUnits_ = 0, typename Parent>
    requires(is_legal_subview_parent_v<Parent>)
auto subview(Parent& parent)
    -> Subview<
        Parent, subview_size_code_for_bytes(LengthUnits_ == 0 ? Parent::LogicalTileBytes : LengthUnits_ * 128),
        OffsetUnits_, 0>
{
    constexpr std::size_t LengthBytes = LengthUnits_ == 0 ? Parent::LogicalTileBytes : LengthUnits_ * 128;
    static_assert(LengthBytes <= Parent::LogicalTileBytes, "B.SUBVIEW length cannot exceed the parent Tile capacity");
    static_assert(
        subview_size_code_for_bytes(LengthBytes) != 0, "B.SUBVIEW length must be a power-of-two multiple of 128 B "
                                                       "up to 256 KiB");
    return {parent, 0};
}

template <std::size_t LengthUnits_ = 0, unsigned OffsetUnits_ = 0, typename Parent>
    requires(is_legal_subview_parent_v<Parent>)
auto subview(Parent& parent, uintptr_t range_base_units)
    -> Subview<
        Parent, subview_size_code_for_bytes(LengthUnits_ == 0 ? Parent::LogicalTileBytes : LengthUnits_ * 128),
        OffsetUnits_, AutoRegSrc>
{
    constexpr std::size_t LengthBytes = LengthUnits_ == 0 ? Parent::LogicalTileBytes : LengthUnits_ * 128;
    static_assert(LengthBytes <= Parent::LogicalTileBytes, "B.SUBVIEW length cannot exceed the parent Tile capacity");
    static_assert(
        subview_size_code_for_bytes(LengthBytes) != 0, "B.SUBVIEW length must be a power-of-two multiple of 128 B "
                                                       "up to 256 KiB");
    return {parent, range_base_units};
}

template <unsigned SubviewSizeCode_, typename Parent>
    requires(is_legal_subview_parent_v<Parent>)
auto subview_sized(Parent& parent) -> Subview<Parent, SubviewSizeCode_, 0, 0>
{
    return {parent, 0};
}

template <unsigned SubviewSizeCode_, typename Parent>
    requires(is_legal_subview_parent_v<Parent>)
auto subview_sized(Parent& parent, uintptr_t range_base) -> Subview<Parent, SubviewSizeCode_, 0, AutoRegSrc>
{
    return {parent, range_base};
}

/// Resolves the fragment length in bytes from 128-byte units (0 = the
/// parent's full capacity) and validates it against the B.ASSEMBLE size
/// code table.
template <std::size_t LengthUnits_, typename Parent>
constexpr std::size_t assemble_length_bytes()
{
    constexpr std::size_t LengthBytes = LengthUnits_ == 0 ? Parent::LogicalTileBytes : LengthUnits_ * 128;
    static_assert(LengthBytes <= Parent::LogicalTileBytes, "B.ASSEMBLE length cannot exceed the parent Tile capacity");
    static_assert(
        subview_size_code_for_bytes(LengthBytes) != 0, "B.ASSEMBLE length must be a power-of-two multiple of 128 B "
                                                       "up to 256 KiB");
    return LengthBytes;
}

// Length and offset are both 128-byte units: LengthUnits_ counts the
// fragment size and OffsetUnits_ the fragment position, in the same
// granularity as the encoded uimm11.
#define PTO_DEFINE_ASSEMBLE_FACTORY(Name, Init, Last)                                                                  \
    template <std::size_t LengthUnits_ = 0, unsigned OffsetUnits_ = 0, typename Parent>                                \
    auto Name(Parent& parent)                                                                                          \
        -> Assemble<                                                                                                   \
            Parent, Init ? subview_size_code_for_bytes(assemble_length_bytes<LengthUnits_, Parent>()) : 0, Init, Last, \
            OffsetUnits_, 0, subview_size_code_for_bytes(assemble_length_bytes<LengthUnits_, Parent>())>               \
    {                                                                                                                  \
        constexpr std::size_t CheckedLength = assemble_length_bytes<LengthUnits_, Parent>();                           \
        (void)CheckedLength;                                                                                           \
        return {parent, 0};                                                                                            \
    }                                                                                                                  \
                                                                                                                       \
    template <std::size_t LengthUnits_ = 0, unsigned OffsetUnits_ = 0, typename Parent>                                \
    auto Name(Parent& parent, uintptr_t range_base_units)                                                              \
        -> Assemble<                                                                                                   \
            Parent, Init ? subview_size_code_for_bytes(assemble_length_bytes<LengthUnits_, Parent>()) : 0, Init, Last, \
            OffsetUnits_, AutoRegSrc, subview_size_code_for_bytes(assemble_length_bytes<LengthUnits_, Parent>())>      \
    {                                                                                                                  \
        constexpr std::size_t CheckedLength = assemble_length_bytes<LengthUnits_, Parent>();                           \
        (void)CheckedLength;                                                                                           \
        return {parent, range_base_units};                                                                             \
    }

PTO_DEFINE_ASSEMBLE_FACTORY(assemble, true, false)
PTO_DEFINE_ASSEMBLE_FACTORY(assemble_init_last, true, true)
PTO_DEFINE_ASSEMBLE_FACTORY(assemble_middle, false, false)
PTO_DEFINE_ASSEMBLE_FACTORY(assemble_last, false, true)

#undef PTO_DEFINE_ASSEMBLE_FACTORY

// Compatibility aliases for callers using the earlier *_at spelling.
template <unsigned Offset_, typename Parent>
    requires(is_legal_subview_parent_v<Parent>)
auto subview_at(Parent& parent) -> decltype(subview<0, Offset_>(parent))
{
    return subview<0, Offset_>(parent);
}

template <unsigned Offset_, typename Parent>
    requires(is_legal_subview_parent_v<Parent>)
auto subview_at(Parent& parent, uintptr_t range_base) -> decltype(subview<0, Offset_>(parent, range_base))
{
    return subview<0, Offset_>(parent, range_base);
}

template <unsigned SubviewSizeCode_, unsigned Offset_, typename Parent>
    requires(is_legal_subview_parent_v<Parent>)
auto subview_sized_at(Parent& parent) -> Subview<Parent, SubviewSizeCode_, Offset_, 0>
{
    return {parent, 0};
}

template <unsigned SubviewSizeCode_, unsigned Offset_, typename Parent>
    requires(is_legal_subview_parent_v<Parent>)
auto subview_sized_at(Parent& parent, uintptr_t range_base) -> Subview<Parent, SubviewSizeCode_, Offset_, AutoRegSrc>
{
    return {parent, range_base};
}

template <unsigned Offset_, unsigned RegSrc_, typename Parent>
    requires(is_legal_subview_parent_v<Parent>)
auto subview_at_reg(Parent& parent, uintptr_t range_base = 0) -> Subview<Parent, Parent::TilesizeCode, Offset_, RegSrc_>
{
    return {parent, range_base};
}

template <unsigned SubviewSizeCode_, unsigned Offset_, unsigned RegSrc_, typename Parent>
    requires(is_legal_subview_parent_v<Parent>)
auto subview_sized_at_reg(Parent& parent, uintptr_t range_base = 0)
    -> Subview<Parent, SubviewSizeCode_, Offset_, RegSrc_>
{
    return {parent, range_base};
}

template <unsigned OffsetUnits_, typename Parent>
auto assemble_at(Parent& parent) -> decltype(assemble<0, OffsetUnits_>(parent))
{
    return assemble<0, OffsetUnits_>(parent);
}

template <unsigned OffsetUnits_, typename Parent>
auto assemble_at(Parent& parent, uintptr_t range_base_units)
    -> decltype(assemble<0, OffsetUnits_>(parent, range_base_units))
{
    return assemble<0, OffsetUnits_>(parent, range_base_units);
}

#define PTO_DEFINE_ASSEMBLE_AT_ALIAS(Name)                                  \
    template <unsigned OffsetUnits_, typename Parent>                       \
    auto Name##_at(Parent& parent)->decltype(Name<0, OffsetUnits_>(parent)) \
    {                                                                       \
        return Name<0, OffsetUnits_>(parent);                               \
    }                                                                       \
                                                                            \
    template <unsigned OffsetUnits_, typename Parent>                       \
    auto Name##_at(Parent& parent, uintptr_t range_base_units)              \
        ->decltype(Name<0, OffsetUnits_>(parent, range_base_units))         \
    {                                                                       \
        return Name<0, OffsetUnits_>(parent, range_base_units);             \
    }

PTO_DEFINE_ASSEMBLE_AT_ALIAS(assemble_init_last)
PTO_DEFINE_ASSEMBLE_AT_ALIAS(assemble_middle)
PTO_DEFINE_ASSEMBLE_AT_ALIAS(assemble_last)

#undef PTO_DEFINE_ASSEMBLE_AT_ALIAS

template <unsigned OffsetUnits_, unsigned RegSrc_, typename Parent>
auto assemble_at_reg(Parent& parent, uintptr_t range_base_units = 0)
    -> Assemble<Parent, Parent::TilesizeCode, true, false, OffsetUnits_, RegSrc_>
{
    return {parent, range_base_units};
}

template <unsigned OffsetUnits_, unsigned RegSrc_, typename Parent>
auto assemble_init_last_at_reg(Parent& parent, uintptr_t range_base_units = 0)
    -> Assemble<Parent, Parent::TilesizeCode, true, true, OffsetUnits_, RegSrc_>
{
    return {parent, range_base_units};
}

template <unsigned OffsetUnits_, unsigned RegSrc_, typename Parent>
auto assemble_middle_at_reg(Parent& parent, uintptr_t range_base_units = 0)
    -> Assemble<Parent, 0, false, false, OffsetUnits_, RegSrc_>
{
    return {parent, range_base_units};
}

template <unsigned OffsetUnits_, unsigned RegSrc_, typename Parent>
auto assemble_last_at_reg(Parent& parent, uintptr_t range_base_units = 0)
    -> Assemble<Parent, 0, false, true, OffsetUnits_, RegSrc_>
{
    return {parent, range_base_units};
}

} // namespace range

// Range wrappers count as Local tile-shaped operands for binding purposes.
template <typename Parent, unsigned SC, auto... Rest>
struct is_tile<range::Subview<Parent, SC, Rest...>> : std::true_type {
    static constexpr SLayout layout_enum = Parent::SFractal;
};
template <typename Parent, unsigned SC, auto... Rest>
struct is_tile<range::Assemble<Parent, SC, Rest...>> : std::true_type {
    static constexpr SLayout layout_enum = Parent::SFractal;
};

// Range-modifier traits: a Subview is a source-side range carrier and an
// Assemble is a destination-side range carrier.
template <typename T>
struct is_subview : std::false_type {};
template <typename Parent, unsigned SC, auto... Rest>
struct is_subview<range::Subview<Parent, SC, Rest...>> : std::true_type {};
template <typename T>
struct is_assemble : std::false_type {};
template <typename Parent, unsigned SC, auto... Rest>
struct is_assemble<range::Assemble<Parent, SC, Rest...>> : std::true_type {};

template <typename T>
concept is_subview_v = is_subview<T>::value;
template <typename T>
concept is_assemble_v = is_assemble<T>::value;
template <typename T>
concept is_range_v = is_subview<T>::value || is_assemble<T>::value;

// Equal bit-width is required: the reinterpret must not change the number of
// logical elements, physical bytes or TileSizeCode.
template <typename SourceTile, typename NewDType>
constexpr bool reinterpret_tile_equal_width_v =
    type_traits<typename SourceTile::DType>::bits == type_traits<NewDType>::bits;

// The source must be an ordinary Local Tile (Shared is out of scope for the
// first phase; a Shared view would need the Sr binder contract).
template <typename SourceTile>
constexpr bool reinterpret_tile_source_is_local_v = is_tile<SourceTile>::value && SourceTile::Loc != Location::Shared;

// The new dtype must be encodable in the source's layout. We accept any
// equal-width dtype whose type_traits exists; boxed/fractal layouts are
// preserved unchanged because rows/cols/inner box are untouched, so the
// existing Tile layout static_asserts remain valid for the same dimensions.
// A NewDType with no PTO TypeCode is rejected by is_supported_dtype_v.
template <typename SourceTile, typename NewDType>
constexpr bool reinterpret_tile_layout_legal_v =
    reinterpret_tile_equal_width_v<SourceTile, NewDType> && is_supported_dtype_v<NewDType>;

// Physical storage preservation: same bytes, same TilesizeCode, same carrier.
template <typename SourceTile, typename NewDType>
constexpr bool reinterpret_tile_storage_compatible_v = reinterpret_tile_equal_width_v<SourceTile, NewDType>;

// The view must not dangle: it holds a reference, so it must not be bound to
// a temporary Tile. reinterpret_tile takes SourceTile& (non-const), which
// already rejects rvalues; the const overload takes const SourceTile&, which
// also rejects prvalue temporaries (they bind to const& only via materialized
// temporaries -- rejected by requiring a named lvalue at the call site).

/// Zero-instruction datatype reinterpret over a Local Tile.
template <typename NewDType, is_tile_data_v SourceTile>
inline auto reinterpret_tile(SourceTile& Source)
{
    using OldDType = typename SourceTile::DType;
    static_assert(is_supported_dtype_v<NewDType>, "reinterpret_tile target dtype has no PTO TypeCode");
    static_assert(
        reinterpret_tile_source_is_local_v<SourceTile>, "reinterpret_tile first phase supports Local Tiles only"
                                                        " (Shared requires a separate Shared view)");
    static_assert(
        reinterpret_tile_layout_legal_v<SourceTile, NewDType>, "reinterpret_tile requires equal-bit-width dtypes "
                                                               "compatible with the source layout");
    static_assert(
        reinterpret_tile_storage_compatible_v<SourceTile, NewDType>,
        "reinterpret_tile must preserve the source Tile storage");
    return ReinterpretedTileView<NewDType, SourceTile>(Source);
}

} // namespace pto

#endif
