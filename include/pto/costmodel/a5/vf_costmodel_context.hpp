/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_MOCKER_A5_VF_COSTMODEL_CONTEXT_HPP
#define PTO_MOCKER_A5_VF_COSTMODEL_CONTEXT_HPP

#include <pto/costmodel/a5/vf_costmodel_helpers.hpp>

namespace pto::mocker::lightweight::a5 {

struct VfCurveContext {
    int64_t valid_rows_i = 0;
    int64_t valid_cols_i = 0;
    std::string_view op_key;
    fit::ShapePath shape_path = fit::ShapePath::Path1D;
    std::string_view src_dtype;
    std::string_view dst_dtype = "none";
    std::string_view op_params;
    uint64_t elements_per_repeat = 0;
};

inline bool TryInitVfCurveContext(const CostModelInput& input, VfCurveContext& context)
{
    context.valid_rows_i = input.valid_rows > 0 ? input.valid_rows : input.rows;
    context.valid_cols_i = input.valid_cols > 0 ? input.valid_cols : input.cols;
    if (context.valid_rows_i <= 0 || context.valid_cols_i <= 0) {
        return false;
    }

    context.op_key = PtoOpcodeToKey(input.op);
    if (context.op_key.empty()) {
        return false;
    }
    context.shape_path = ResolveShapePath(input.op, input.rows, input.cols, context.valid_cols_i, input.a5_shape_path);
    // implTCVT explicitly selects castData_2D_NoPostUpdate for this mode,
    // even for an otherwise flattenable layout. Do not substitute a measured
    // POST_UPDATE curve when this distinct implementation has no fitted row.
    if (input.op == PtoOpcode::TCVT && input.vf_impl_kind == VFImplKind::VFIMPL_2D_NO_POST_UPDATE) {
        context.shape_path = fit::ShapePath::Path2D;
    }
    context.src_dtype = DTypeToKey(input.dtype);
    context.op_params = input.a5_op_params;
    return !context.src_dtype.empty();
}

inline std::string_view PatternOpParams(MaskPattern pattern, bool columnAxis)
{
    switch (pattern) {
        case MaskPattern::P0101:
            return columnAxis ? "pattern_P0101_col" : "pattern_P0101_row";
        case MaskPattern::P1010:
            return columnAxis ? "pattern_P1010_col" : "pattern_P1010_row";
        case MaskPattern::P0001:
            return columnAxis ? "pattern_P0001_col" : "pattern_P0001_row";
        case MaskPattern::P0010:
            return columnAxis ? "pattern_P0010_col" : "pattern_P0010_row";
        case MaskPattern::P0100:
            return columnAxis ? "pattern_P0100_col" : "pattern_P0100_row";
        case MaskPattern::P1000:
            return columnAxis ? "pattern_P1000_col" : "pattern_P1000_row";
        case MaskPattern::P1111:
            return columnAxis ? "pattern_P1111_col" : "pattern_P1111_row";
        default:
            return {};
    }
}

inline bool IsDenseFp16Fp32CastGeometry(const CostModelInput& input, const VfCurveContext& context)
{
    constexpr int64_t kMinMeasuredRows = 8;
    constexpr int64_t kMaxMeasuredRows = 128;
    constexpr int64_t kMeasuredRowAlignment = 8;
    constexpr int64_t kMinMeasuredCols = 32;
    constexpr int64_t kMaxMeasuredCols = 256;
    constexpr int64_t kMeasuredColAlignment = 16;
    return input.rows == context.valid_rows_i && input.cols == context.valid_cols_i && input.dst_rows == input.rows &&
           input.dst_cols == input.cols && (input.dst_valid_rows == 0 || input.dst_valid_rows == input.rows) &&
           (input.dst_valid_cols == 0 || input.dst_valid_cols == input.cols) && input.rows >= kMinMeasuredRows &&
           input.rows <= kMaxMeasuredRows && input.rows % kMeasuredRowAlignment == 0 &&
           input.cols >= kMinMeasuredCols && input.cols <= kMaxMeasuredCols && input.cols % kMeasuredColAlignment == 0;
}

inline bool IsDenseFp16Fp32Cast(const CostModelInput& input, const VfCurveContext& context)
{
    return input.op == PtoOpcode::TCVT && input.dtype == DType::Half && input.dst_dtype == DType::Float &&
           input.round_mode == RoundMode::CAST_NONE && input.saturation_mode == SaturationMode::ON &&
           input.vf_impl_kind == VFImplKind::VFIMPL_DEFAULT && context.shape_path == fit::ShapePath::Path2D &&
           context.op_params == "CAST_RINT_SAT_ON_normal" && input.layout == MemLayout::ND &&
           IsDenseFp16Fp32CastGeometry(input, context);
}

inline void ResolveCastVfOpParams(const CostModelInput& input, VfCurveContext& context)
{
    context.dst_dtype = DTypeToKey(input.dst_dtype);
    if (input.op == PtoOpcode::TCVT && context.op_params.empty()) {
        context.op_params = (input.round_mode == RoundMode::CAST_NONE && input.saturation_mode == SaturationMode::ON) ?
                                std::string_view("CAST_RINT_SAT_ON_normal") :
                                CvtOpParams(input.round_mode, input.saturation_mode);
    }
    // This conversion's dense 2D profile calibrates the per-row and inner
    // loop costs independently; padded layouts retain their own lookup.
    if (input.op == PtoOpcode::TCVT && input.dtype == DType::Half && input.dst_dtype == DType::Int32 &&
        context.shape_path == fit::ShapePath::Path2D && input.rows == context.valid_rows_i &&
        input.cols == context.valid_cols_i && context.op_params == "CAST_RINT_SAT_ON_normal") {
        context.op_params = "CAST_RINT_SAT_ON_dense";
    }
    // Runtime-dynamic cast16to32: independently measured row and inner-loop
    // contributions. Preserve legacy curves for static/1D, padded, unknown
    // destination geometry, explicit VF modes and inputs outside this domain.
    if (IsDenseFp16Fp32Cast(input, context)) {
        context.op_params = "CAST_NONE_dense_fp16_fp32";
    }
}

inline bool TryResolveVfOpParams(const CostModelInput& input, VfCurveContext& context)
{
    if ((input.op == PtoOpcode::TGATHER || input.op == PtoOpcode::TSCATTER) && input.has_mask_pattern) {
        if (input.has_dtype2) {
            return false;
        }
        const bool columnAxis = input.op == PtoOpcode::TGATHER ? input.gather_axis == GatherAxis::GATHER_COL :
                                                                 input.scatter_axis == ScatterAxis::SCATTER_COL;
        // Explicit semantic inputs take precedence over legacy free-form lookup keys.
        context.op_params = PatternOpParams(input.mask_pattern, columnAxis);
        return !context.op_params.empty();
    }
    if (input.op == PtoOpcode::TCVT || input.op == PtoOpcode::TDEQUANT) {
        ResolveCastVfOpParams(input, context);
    } else if (
        (input.op == PtoOpcode::TGATHER || input.op == PtoOpcode::TSCATTER) && input.has_dtype2 &&
        context.op_params.empty()) {
        context.op_params = IndexOpParams(input.dtype2);
    } else if (
        input.op == PtoOpcode::TSORT32 && context.valid_cols_i < static_cast<int64_t>(kSort32ElementsPerRepeat) &&
        context.op_params.empty()) {
        context.op_params = "no_tmp_subrepeat";
    } else if (context.op_params.empty()) {
        context.op_params = DefaultOpParams(input.op);
    }
    return !context.dst_dtype.empty();
}

inline bool TryResolveDenseRowTselParams(const CostModelInput& input, uint64_t inner_iter, VfCurveContext& context)
{
    constexpr int64_t kSelectB16MaxDenseRowCols = 28032;
    constexpr int64_t kSelectB32MaxDenseRowCols = 15096;
    constexpr int64_t kSelectB32MaxOddRowCols = 320;
    const bool dense = input.rows == context.valid_rows_i && input.cols == context.valid_cols_i;
    const bool paired = inner_iter % kEvenModulo == 0;
    if (dense && context.shape_path == fit::ShapePath::Path1D && input.rows == 1) {
        if (input.dtype == DType::Uint16 && input.cols <= kSelectB16MaxDenseRowCols) {
            context.op_params = "TSel_b16_dense_row";
            return true;
        }
        if (input.dtype == DType::Uint32 &&
            input.cols <= (paired ? kSelectB32MaxDenseRowCols : kSelectB32MaxOddRowCols)) {
            context.op_params = paired ? "TSel_b32_dense_row_even" : "TSel_b32_dense_row_odd";
            return true;
        }
    }
    return false;
}

inline bool TryResolveTselOpParams(const CostModelInput& input, uint64_t inner_iter, VfCurveContext& context)
{
    if (input.op != PtoOpcode::TSEL) {
        return true;
    }
    if (context.op_params.empty()) {
        // B32 pair and remainder schedules have separate measured domains.
        if (TryResolveDenseRowTselParams(input, inner_iter, context)) {
            return true;
        }
        constexpr int64_t kSelectB32MaxDenseRows = 64;
        constexpr int64_t kSelectB32MaxDenseCols = 384;
        constexpr int64_t kSelectB32MaxSingleVectorRows = 471;
        constexpr int64_t kSelectB32MaxSinglePairRows = 123;
        const bool dense = input.rows == context.valid_rows_i && input.cols == context.valid_cols_i;
        const bool paired = inner_iter % kEvenModulo == 0;
        const bool measuredDenseRows = (input.rows <= kSelectB32MaxDenseRows && input.cols <= kSelectB32MaxDenseCols) ||
                                       (inner_iter == 1 && input.rows <= kSelectB32MaxSingleVectorRows) ||
                                       (inner_iter == kEvenModulo && input.rows <= kSelectB32MaxSinglePairRows);
        if (dense && input.dtype == DType::Uint32 && context.shape_path == fit::ShapePath::Path2D &&
            measuredDenseRows) {
            context.op_params = paired ? "TSel_b32_dense_even" : "TSel_b32_dense_odd";
            return true;
        }
        if (input.dtype == DType::Uint16 && context.shape_path == fit::ShapePath::Path2D &&
            input.rows == context.valid_rows_i && input.cols == context.valid_cols_i) {
            context.op_params = "TSel_b16_8_dense";
            return true;
        }
        context.op_params = TselOpParams(input.dtype, context.shape_path, inner_iter);
    }
    return !context.op_params.empty();
}

inline void ResolveSingleRepeatVfOpParams(
    const CostModelInput& input, VfCurveContext& context, int64_t validRows, int64_t validCols, uint64_t inner_iter)
{
    constexpr uint64_t kSingleRow = 1;
    constexpr uint64_t kSingleInnerIteration = 1;
    const bool hasB16SingleRepeatProfile =
        ((input.dtype == DType::Int16 || input.dtype == DType::Uint16) &&
         (input.op == PtoOpcode::TAND || input.op == PtoOpcode::TOR || input.op == PtoOpcode::TNOT)) ||
        (input.dtype == DType::Uint16 && (input.op == PtoOpcode::TANDS || input.op == PtoOpcode::TORS ||
                                          input.op == PtoOpcode::TXOR || input.op == PtoOpcode::TXORS));
    if (hasB16SingleRepeatProfile && context.shape_path == fit::ShapePath::Path1D &&
        validRows == static_cast<int64_t>(kSingleRow) && input.rows == validRows && input.cols == validCols &&
        inner_iter == kSingleInnerIteration && context.op_params.empty()) {
        context.op_params = "single_row_single_repeat";
    }
}

inline void ResolveSingleRowVfOpParams(const CostModelInput& input, uint64_t outer_iter, VfCurveContext& context)
{
    constexpr uint64_t kSingleRow = 1;
    if (context.shape_path == fit::ShapePath::Path1D && outer_iter == kSingleRow) {
        if (context.src_dtype == "fp16" &&
            (input.op == PtoOpcode::TADD || input.op == PtoOpcode::TMUL || input.op == PtoOpcode::TABS ||
             input.op == PtoOpcode::TMAX || input.op == PtoOpcode::TMIN) &&
            context.op_params.empty()) {
            context.op_params = "row1";
        } else if (input.op == PtoOpcode::TGATHER && context.op_params == "index_b32") {
            context.op_params = "index_b32_row1";
        } else if (
            (input.op == PtoOpcode::TCOLARGMAX || input.op == PtoOpcode::TCOLARGMIN) &&
            (context.src_dtype == "fp32" || context.src_dtype == "int32")) {
            context.op_params = "row1";
        }
    }
}

inline void ResolveCompareVfOpParams(
    const CostModelInput& input, VfCurveContext& context, int64_t validRows, int64_t validCols)
{
    constexpr uint64_t kSingleRow = 1;
    // Scalar b32 comparisons pack two vector predicates per inner iteration.
    // The calibrated single-iteration curve depends on rows, not vector repeats.
    constexpr std::array kSingleCompareDTypes = {
        std::string_view("bf16"),   std::string_view("fp32"), std::string_view("int32"), std::string_view("uint32"),
        std::string_view("uint16"), std::string_view("int8"), std::string_view("uint8")};
    const bool hasSingleCompareFit =
        std::find(kSingleCompareDTypes.begin(), kSingleCompareDTypes.end(), context.src_dtype) !=
        kSingleCompareDTypes.end();
    const uint64_t compareElementsPerIteration = std::max(context.elements_per_repeat, kElementsPerRepeatB16);
    if (input.op == PtoOpcode::TCMPS && hasSingleCompareFit && context.shape_path == fit::ShapePath::Path2D &&
        context.op_params.empty() && validRows > static_cast<int64_t>(kSingleRow) && input.rows == validRows &&
        input.cols == validCols && validCols > 0 && static_cast<uint64_t>(validCols) <= compareElementsPerIteration) {
        context.op_params = "single_compare_iteration";
    }
    if (input.op == PtoOpcode::TCMPS && validCols > static_cast<int64_t>(kElementsPerRepeatB16) &&
        context.op_params.empty()) {
        context.op_params = "multi_inner";
    }
}

inline void ResolveSingleInnerCastVfOpParams(const CostModelInput& input, uint64_t inner_iter, VfCurveContext& context)
{
    constexpr uint64_t kSingleInnerIteration = 1;
    const bool hasSingleInnerTcvtFit =
        (context.src_dtype == "bf16" && (context.dst_dtype == "fp4_e1m2" || context.dst_dtype == "fp4_e2m1")) ||
        (context.src_dtype == "fp32" &&
         (context.dst_dtype == "fp8_e4m3" || context.dst_dtype == "fp8_e5m2" || context.dst_dtype == "hif8"));
    if (input.op == PtoOpcode::TCVT && hasSingleInnerTcvtFit && context.shape_path == fit::ShapePath::Path2D &&
        inner_iter == kSingleInnerIteration && context.op_params == "CAST_RINT_SAT_ON_normal") {
        context.op_params = "CAST_RINT_SAT_ON_inner1";
    }
}

inline void ResolveShapeSpecificOpParams(
    const CostModelInput& input, uint64_t outer_iter, uint64_t inner_iter, VfCurveContext& context)
{
    const int64_t validRows = input.valid_rows > 0 ? input.valid_rows : input.rows;
    const int64_t validCols = input.valid_cols > 0 ? input.valid_cols : input.cols;
    ResolveSingleRepeatVfOpParams(input, context, validRows, validCols, inner_iter);
    ResolveSingleRowVfOpParams(input, outer_iter, context);
    if (context.src_dtype == "fp16" && context.shape_path == fit::ShapePath::Path2D && context.op_params.empty() &&
        input.rows == validRows && input.cols == validCols &&
        (input.op == PtoOpcode::TABS || input.op == PtoOpcode::TNEG)) {
        context.op_params = "dense_2d";
    }
    ResolveCompareVfOpParams(input, context, validRows, validCols);
    ResolveSingleInnerCastVfOpParams(input, inner_iter, context);
}

} // namespace pto::mocker::lightweight::a5

#endif
