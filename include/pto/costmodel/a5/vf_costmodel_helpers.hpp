/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_MOCKER_A5_VF_COSTMODEL_HELPERS_HPP
#define PTO_MOCKER_A5_VF_COSTMODEL_HELPERS_HPP

#include <algorithm>
#include <array>
#include <cstdint>
#include <string_view>
#include <utility>

#include <pto/costmodel/a5/formula_costmodel/formula_backend_vf.hpp>
#include <pto/costmodel/lightweight_costmodel_types.hpp>

namespace pto::mocker::lightweight::a5 {

inline constexpr uint64_t kEvenModulo = 2;
inline constexpr uint64_t kElementsPerRepeatB32 = 64;
inline constexpr uint64_t kElementsPerRepeatB16 = 128;
inline constexpr uint64_t kElementsPerRepeatB8 = 256;
inline constexpr uint64_t kSort32ElementsPerRepeat = 32;

struct ScatterIndexFitParams {
    double baseCycles;
    double loopLogSlope;
    double tileSizeSlope;
    double initRepeatSquareSlope;
    double rowInitInteractionSlope;
};

struct ContinuousPiecewiseFitParams {
    double initialSlope;
    double bias;
    double largeSlope;
    double breakpoint;
};

// TSCATTER initializes the complete physical destination tile before executing
// its row/repeat loop.  The benchmark initializes every index to zero, so the
// fitted interaction term also represents the resulting write conflicts.
inline constexpr ScatterIndexFitParams kScatterIndexB16Fp16Fit = {
    42.73689546, 1.31912775, 1.2301575, 0.25136473, -0.11392576};
inline constexpr ContinuousPiecewiseFitParams kScatterIndexB16Fp16Row1Fit = {
    10.03889047, 64.35634627, 13.8580264, 16.0};

inline double EvalContinuousPiecewise(double loopCount, const ContinuousPiecewiseFitParams& fit)
{
    return fit.initialSlope * std::min(loopCount, fit.breakpoint) + fit.bias +
           fit.largeSlope * std::max(loopCount - fit.breakpoint, 0.0);
}

inline uint64_t CeilDivU64(uint64_t x, uint64_t y) { return y == 0 ? 0 : (x + y - 1) / y; }

inline constexpr std::array kOpcodeKeys = {
    std::pair{PtoOpcode::TADD, std::string_view("TADD")},
    std::pair{PtoOpcode::TSUB, std::string_view("TSUB")},
    std::pair{PtoOpcode::TMUL, std::string_view("TMUL")},
    std::pair{PtoOpcode::TDIV, std::string_view("TDIV")},
    std::pair{PtoOpcode::TRECIP, std::string_view("TRECIP")},
    std::pair{PtoOpcode::TADDS, std::string_view("TADDS")},
    std::pair{PtoOpcode::TSUBS, std::string_view("TSUBS")},
    std::pair{PtoOpcode::TMULS, std::string_view("TMULS")},
    std::pair{PtoOpcode::TDIVS, std::string_view("TDIVS")},
    std::pair{PtoOpcode::TMINS, std::string_view("TMINS")},
    std::pair{PtoOpcode::TMAXS, std::string_view("TMAXS")},
    std::pair{PtoOpcode::TABS, std::string_view("TABS")},
    std::pair{PtoOpcode::TNEG, std::string_view("TNEG")},
    std::pair{PtoOpcode::TEXP, std::string_view("TEXP")},
    std::pair{PtoOpcode::TSQRT, std::string_view("TSQRT")},
    std::pair{PtoOpcode::TRSQRT, std::string_view("TRSQRT")},
    std::pair{PtoOpcode::TLOG, std::string_view("TLOG")},
    std::pair{PtoOpcode::TRELU, std::string_view("TRELU")},
    std::pair{PtoOpcode::TLRELU, std::string_view("TLRELU")},
    std::pair{PtoOpcode::TNOT, std::string_view("TNOT")},
    std::pair{PtoOpcode::TCVT, std::string_view("TCVT")},
    std::pair{PtoOpcode::TSEL, std::string_view("TSEL")},
    std::pair{PtoOpcode::TMAX, std::string_view("TMAX")},
    std::pair{PtoOpcode::TMIN, std::string_view("TMIN")},
    std::pair{PtoOpcode::TAND, std::string_view("TAND")},
    std::pair{PtoOpcode::TOR, std::string_view("TOR")},
    std::pair{PtoOpcode::TANDS, std::string_view("TANDS")},
    std::pair{PtoOpcode::TORS, std::string_view("TORS")},
    std::pair{PtoOpcode::TXOR, std::string_view("TXOR")},
    std::pair{PtoOpcode::TXORS, std::string_view("TXORS")},
    std::pair{PtoOpcode::TFMOD, std::string_view("TFMOD")},
    std::pair{PtoOpcode::TFMODS, std::string_view("TFMODS")},
    std::pair{PtoOpcode::TREM, std::string_view("TREM")},
    std::pair{PtoOpcode::TREMS, std::string_view("TREMS")},
    std::pair{PtoOpcode::TPOW, std::string_view("TPOW")},
    std::pair{PtoOpcode::TPOWS, std::string_view("TPOWS")},
    std::pair{PtoOpcode::TMADD, std::string_view("TMADD")},
    std::pair{PtoOpcode::TMULA, std::string_view("TMULA")},
    std::pair{PtoOpcode::TPRELU, std::string_view("TPRELU")},
    std::pair{PtoOpcode::TPARTADD, std::string_view("TPARTADD")},
    std::pair{PtoOpcode::TPARTMUL, std::string_view("TPARTMUL")},
    std::pair{PtoOpcode::TPARTMAX, std::string_view("TPARTMAX")},
    std::pair{PtoOpcode::TPARTMIN, std::string_view("TPARTMIN")},
    std::pair{PtoOpcode::TAXPY, std::string_view("TAXPY")},
    std::pair{PtoOpcode::TCMP, std::string_view("TCMP")},
    std::pair{PtoOpcode::TCMPS, std::string_view("TCMPS")},
    std::pair{PtoOpcode::TROWARGMAX, std::string_view("TROWARGMAX")},
    std::pair{PtoOpcode::TROWARGMIN, std::string_view("TROWARGMIN")},
    std::pair{PtoOpcode::TCOLARGMAX, std::string_view("TCOLARGMAX")},
    std::pair{PtoOpcode::TCOLARGMIN, std::string_view("TCOLARGMIN")},
    std::pair{PtoOpcode::TROWSUM, std::string_view("TROWSUM")},
    std::pair{PtoOpcode::TROWMAX, std::string_view("TROWMAX")},
    std::pair{PtoOpcode::TROWMIN, std::string_view("TROWMIN")},
    std::pair{PtoOpcode::TROWPROD, std::string_view("TROWPROD")},
    std::pair{PtoOpcode::TCOLSUM, std::string_view("TCOLSUM")},
    std::pair{PtoOpcode::TCOLMAX, std::string_view("TCOLMAX")},
    std::pair{PtoOpcode::TCOLMIN, std::string_view("TCOLMIN")},
    std::pair{PtoOpcode::TCOLPROD, std::string_view("TCOLPROD")},
    std::pair{PtoOpcode::TROWEXPAND, std::string_view("TROWEXPAND")},
    std::pair{PtoOpcode::TROWEXPANDADD, std::string_view("TROWEXPANDADD")},
    std::pair{PtoOpcode::TROWEXPANDSUB, std::string_view("TROWEXPANDSUB")},
    std::pair{PtoOpcode::TROWEXPANDMUL, std::string_view("TROWEXPANDMUL")},
    std::pair{PtoOpcode::TROWEXPANDDIV, std::string_view("TROWEXPANDDIV")},
    std::pair{PtoOpcode::TROWEXPANDMAX, std::string_view("TROWEXPANDMAX")},
    std::pair{PtoOpcode::TROWEXPANDMIN, std::string_view("TROWEXPANDMIN")},
    std::pair{PtoOpcode::TROWEXPANDEXPDIF, std::string_view("TROWEXPANDEXPDIF")},
    std::pair{PtoOpcode::TCOLEXPAND, std::string_view("TCOLEXPAND")},
    std::pair{PtoOpcode::TCOLEXPANDADD, std::string_view("TCOLEXPANDADD")},
    std::pair{PtoOpcode::TCOLEXPANDSUB, std::string_view("TCOLEXPANDSUB")},
    std::pair{PtoOpcode::TCOLEXPANDMUL, std::string_view("TCOLEXPANDMUL")},
    std::pair{PtoOpcode::TCOLEXPANDDIV, std::string_view("TCOLEXPANDDIV")},
    std::pair{PtoOpcode::TCOLEXPANDMAX, std::string_view("TCOLEXPANDMAX")},
    std::pair{PtoOpcode::TCOLEXPANDMIN, std::string_view("TCOLEXPANDMIN")},
    std::pair{PtoOpcode::TCOLEXPANDEXPDIF, std::string_view("TCOLEXPANDEXPDIF")},
    std::pair{PtoOpcode::TEXPANDS, std::string_view("TEXPANDS")},
    std::pair{PtoOpcode::TSELS, std::string_view("TSELS")},
    std::pair{PtoOpcode::TCONCAT, std::string_view("TCONCAT")},
    std::pair{PtoOpcode::TGATHER, std::string_view("TGATHER")},
    std::pair{PtoOpcode::TGATHERB, std::string_view("TGATHERB")},
    std::pair{PtoOpcode::TSCATTER, std::string_view("TSCATTER")},
    std::pair{PtoOpcode::TMOV, std::string_view("TMOV")},
    std::pair{PtoOpcode::TSORT32, std::string_view("TSORT32")},
    std::pair{PtoOpcode::TMRGSORT, std::string_view("TMRGSORT")},
    std::pair{PtoOpcode::TFILLPAD, std::string_view("TFILLPAD")},
    std::pair{PtoOpcode::TFILLPAD_EXPAND, std::string_view("TFILLPAD_EXPAND")},
    std::pair{PtoOpcode::TTRI, std::string_view("TTRI")},
    std::pair{PtoOpcode::TDEQUANT, std::string_view("TDEQUANT")},
};

inline std::string_view PtoOpcodeToKey(PtoOpcode op)
{
    for (const auto& [opcode, key] : kOpcodeKeys) {
        if (opcode == op) {
            return key;
        }
    }
    return "";
}

inline std::string_view DTypeToKey(DType dtype)
{
    switch (dtype) {
        case DType::Float:
            return "fp32";
        case DType::Half:
            return "fp16";
        case DType::Float8E4M3:
            return "fp8_e4m3";
        case DType::Float8E5M2:
            return "fp8_e5m2";
        case DType::HFloat8:
            return "hif8";
        case DType::Float4E1M2:
            return "fp4_e1m2";
        case DType::Float4E2M1:
            return "fp4_e2m1";
        case DType::Int8:
            return "int8";
        case DType::Int16:
            return "int16";
        case DType::Int32:
            return "int32";
        case DType::Uint8:
            return "uint8";
        case DType::Uint16:
            return "uint16";
        case DType::Uint32:
            return "uint32";
        case DType::BFloat16:
            return "bf16";
        default:
            return "";
    }
}

inline bool TryGetA5ElementsPerRepeatByDTypeKey(std::string_view dtype_key, uint64_t& elements_per_repeat)
{
    if (dtype_key == "fp4_e1m2" || dtype_key == "fp4_e2m1") {
        elements_per_repeat = kElementsPerRepeatB8;
        return true;
    }
    if (dtype_key == "fp16" || dtype_key == "bf16") {
        elements_per_repeat = kElementsPerRepeatB16;
        return true;
    }
    if (dtype_key == "fp32" || dtype_key == "int32" || dtype_key == "uint32" || dtype_key == "fp8_e4m3" ||
        dtype_key == "fp8_e5m2" || dtype_key == "hif8") {
        elements_per_repeat = kElementsPerRepeatB32;
        return true;
    }
    if (dtype_key == "int8" || dtype_key == "uint8") {
        elements_per_repeat = kElementsPerRepeatB8;
        return true;
    }
    if (dtype_key == "int16" || dtype_key == "uint16") {
        elements_per_repeat = kElementsPerRepeatB16;
        return true;
    }
    return false;
}

inline bool TryGetA5ElementsPerRepeat(DType dtype, uint64_t& elements_per_repeat)
{
    return TryGetA5ElementsPerRepeatByDTypeKey(DTypeToKey(dtype), elements_per_repeat);
}

inline bool TryGetA5CvtElementsPerRepeat(
    std::string_view src_dtype, std::string_view dst_dtype, fit::ShapePath shape_path, uint64_t& elements_per_repeat)
{
    // FP16-to-I32 expands each B16 source into one B32 lane, like FP16-to-FP32.
    if (((src_dtype == "fp16" || src_dtype == "bf16") && dst_dtype == "fp32") ||
        (src_dtype == "fp16" && dst_dtype == "int32")) {
        elements_per_repeat = kElementsPerRepeatB32;
        return true;
    }
    if (shape_path == fit::ShapePath::Path1D) {
        return TryGetA5ElementsPerRepeatByDTypeKey(src_dtype, elements_per_repeat);
    }
    if ((src_dtype == "fp32" && (dst_dtype == "fp16" || dst_dtype == "bf16" || dst_dtype == "fp8_e4m3" ||
                                 dst_dtype == "fp8_e5m2" || dst_dtype == "hif8")) ||
        ((src_dtype == "fp16" || src_dtype == "bf16") &&
         (dst_dtype == "fp16" || dst_dtype == "fp4_e1m2" || dst_dtype == "fp4_e2m1" || dst_dtype == "hif8"))) {
        elements_per_repeat = kElementsPerRepeatB16;
        return true;
    }
    return TryGetA5ElementsPerRepeatByDTypeKey(src_dtype, elements_per_repeat);
}

inline std::string_view TselOpParams(DType dtype, fit::ShapePath shape_path, uint64_t inner_iter)
{
    if (dtype == DType::Float || dtype == DType::Int32 || dtype == DType::Uint32) {
        if (shape_path == fit::ShapePath::Path1D) {
            return (inner_iter % kEvenModulo == 0) ? "TSel_b32_even" : "TSel_b32_odd";
        }
        return "TSel_b32";
    }
    if (dtype == DType::Half || dtype == DType::BFloat16 || dtype == DType::Int16 || dtype == DType::Uint16 ||
        dtype == DType::Int8 || dtype == DType::Uint8) {
        return "TSel_b16_8";
    }
    return "";
}

inline std::string_view DefaultOpParams(PtoOpcode op)
{
    switch (op) {
        case PtoOpcode::TRECIP:
            return "default";
        default:
            return "";
    }
}

inline std::string_view IndexOpParams(DType index_dtype)
{
    switch (index_dtype) {
        case DType::Int16:
        case DType::Uint16:
            return "index_b16";
        case DType::Int32:
        case DType::Uint32:
            return "index_b32";
        default:
            return "";
    }
}

inline std::string_view CvtOpParams(RoundMode round_mode, SaturationMode saturation_mode)
{
    if (round_mode == RoundMode::CAST_NONE && saturation_mode == SaturationMode::ON) {
        return "CAST_NONE_SAT_ON_normal";
    }
    if (round_mode == RoundMode::CAST_RINT && saturation_mode == SaturationMode::ON) {
        return "CAST_RINT_SAT_ON_normal";
    }
    if (round_mode == RoundMode::CAST_TRUNC && saturation_mode == SaturationMode::OFF) {
        return "CAST_TRUNC_SAT_OFF_normal";
    }
    return "";
}

inline bool IsA5ScalarVfOp(PtoOpcode op)
{
    switch (op) {
        case PtoOpcode::TADDS:
        case PtoOpcode::TSUBS:
        case PtoOpcode::TMULS:
        case PtoOpcode::TDIVS:
        case PtoOpcode::TMINS:
        case PtoOpcode::TMAXS:
        case PtoOpcode::TNEG:
        case PtoOpcode::TANDS:
        case PtoOpcode::TORS:
        case PtoOpcode::TXORS:
        case PtoOpcode::TFMODS:
        case PtoOpcode::TREMS:
        case PtoOpcode::TPOWS:
            return true;
        default:
            return false;
    }
}

inline bool IsA5PathIndependentVfOp(PtoOpcode op)
{
    switch (op) {
        case PtoOpcode::TCMPS:
        case PtoOpcode::TCOLARGMAX:
        case PtoOpcode::TCOLARGMIN:
        case PtoOpcode::TGATHER:
        case PtoOpcode::TMOV:
        case PtoOpcode::TROWARGMAX:
        case PtoOpcode::TSCATTER:
        case PtoOpcode::TSORT32:
        case PtoOpcode::TTRI:
            return true;
        default:
            return false;
    }
}

inline fit::ShapePath ResolveShapePath(
    PtoOpcode op, int64_t rows, int64_t cols, int64_t valid_cols, A5ShapePathHint shape_path_hint)
{
    if (shape_path_hint == A5ShapePathHint::Path1D) {
        return fit::ShapePath::Path1D;
    }
    if (shape_path_hint == A5ShapePathHint::Path2D) {
        return fit::ShapePath::Path2D;
    }
    if (op == PtoOpcode::TDEQUANT) {
        return rows == 1 ? fit::ShapePath::Path1D : fit::ShapePath::Path2D;
    }
    if (IsA5ScalarVfOp(op)) {
        return valid_cols == cols ? fit::ShapePath::Path1D : fit::ShapePath::Path2D;
    }
    if (rows == 1 || valid_cols == cols) {
        return fit::ShapePath::Path1D;
    }
    return fit::ShapePath::Path2D;
}

inline fit::ShapePath ResolveShapePath(PtoOpcode op, int64_t rows, int64_t cols, int64_t valid_cols)
{
    if (IsA5ScalarVfOp(op)) {
        return valid_cols == cols ? fit::ShapePath::Path1D : fit::ShapePath::Path2D;
    }
    if (rows == 1 || valid_cols == cols) {
        return fit::ShapePath::Path1D;
    }
    return fit::ShapePath::Path2D;
}

inline std::string_view ResolveDefaultVfImplKind(PtoOpcode op, fit::ShapePath shape_path)
{
    if (op == PtoOpcode::TDEQUANT) {
        return "POST_UPDATE";
    }
    if (op == PtoOpcode::TCVT) {
        return (shape_path == fit::ShapePath::Path1D) ? "NO_POST_UPDATE" : "POST_UPDATE";
    }
    if (op == PtoOpcode::TSEL) {
        return (shape_path == fit::ShapePath::Path1D) ? "NO_POST_UPDATE" : "DEFAULT";
    }
    return (shape_path == fit::ShapePath::Path1D) ? "POST_UPDATE" : "NO_POST_UPDATE";
}

inline std::string_view ResolveVfImplKind(PtoOpcode op, VFImplKind vf_impl_kind, fit::ShapePath shape_path)
{
    if (op == PtoOpcode::TCVT) {
        // The caller resolves dimensionality from layout plus the explicit
        // VFIMPL_2D_NO_POST_UPDATE override in implTCVT. Preserve the actual
        // predicate strategy instead of aliasing an unmeasured implementation.
        if (shape_path == fit::ShapePath::Path1D || vf_impl_kind == VFImplKind::VFIMPL_1D_NO_POST_UPDATE ||
            vf_impl_kind == VFImplKind::VFIMPL_2D_NO_POST_UPDATE) {
            return "NO_POST_UPDATE";
        }
        return "POST_UPDATE";
    }
    switch (vf_impl_kind) {
        case VFImplKind::VFIMPL_1D_NO_POST_UPDATE:
            return "NO_POST_UPDATE";
        case VFImplKind::VFIMPL_2D_NO_POST_UPDATE:
            return "NO_POST_UPDATE";
        case VFImplKind::VFIMPL_1D_POST_UPDATE:
            return "POST_UPDATE";
        case VFImplKind::VFIMPL_2D_POST_UPDATE:
            return "POST_UPDATE";
        case VFImplKind::VFIMPL_DEFAULT:
        default:
            return ResolveDefaultVfImplKind(op, shape_path);
    }
}

} // namespace pto::mocker::lightweight::a5

#endif
