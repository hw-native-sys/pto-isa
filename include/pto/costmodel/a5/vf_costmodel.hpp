/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_MOCKER_A5_VF_COSTMODEL_HPP
#define PTO_MOCKER_A5_VF_COSTMODEL_HPP

#include <algorithm>
#include <array>
#include <cstdint>
#include <string_view>
#include <utility>

#include <pto/costmodel/a5/formula_costmodel/formula_backend_vf.hpp>
#include <pto/costmodel/a5/vf_costmodel_helpers.hpp>

namespace pto::mocker::lightweight::a5 {

inline bool TryEstimateA5MetadataCycles(const CostModelInput& input, uint64_t& cycles);
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
        context.dst_dtype = DTypeToKey(input.dst_dtype);
        if (input.op == PtoOpcode::TCVT && context.op_params.empty()) {
            context.op_params =
                (input.round_mode == RoundMode::CAST_NONE && input.saturation_mode == SaturationMode::ON) ?
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
        if (input.op == PtoOpcode::TCVT && input.dtype == DType::Half && input.dst_dtype == DType::Float &&
            input.round_mode == RoundMode::CAST_NONE && input.saturation_mode == SaturationMode::ON &&
            input.vf_impl_kind == VFImplKind::VFIMPL_DEFAULT && context.shape_path == fit::ShapePath::Path2D &&
            context.op_params == "CAST_RINT_SAT_ON_normal" && input.layout == MemLayout::ND &&
            input.rows == context.valid_rows_i && input.cols == context.valid_cols_i && input.dst_rows == input.rows &&
            input.dst_cols == input.cols && (input.dst_valid_rows == 0 || input.dst_valid_rows == input.rows) &&
            (input.dst_valid_cols == 0 || input.dst_valid_cols == input.cols) && input.rows >= 8 && input.rows <= 128 &&
            input.rows % 8 == 0 && input.cols >= 32 && input.cols <= 256 && input.cols % 16 == 0) {
            context.op_params = "CAST_NONE_dense_fp16_fp32";
        }
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

inline bool TryResolveElementsPerRepeat(const CostModelInput& input, VfCurveContext& context)
{
    // Dense Vec-to-Vec TMOV always retains the row loop, even without padding.
    // Bounds describe the measured domain; multi-repeat rows were sampled only
    // at short row counts. Do not extrapolate their Cartesian product.
    constexpr int64_t kMoveMaxBytes = 128 * 1024;
    constexpr int64_t kMoveMaxRows = 4092;
    constexpr int64_t kMoveMultiRepeatMaxRows = 48;
    constexpr int64_t kMoveMaxRowRepeats = 4;
    constexpr int64_t kMoveAlignmentBytes = 32;
    constexpr int64_t kMoveTailPartitions = 4;
    const bool moveB16 = input.dtype == DType::BFloat16 || input.dtype == DType::Int16 || input.dtype == DType::Uint16;
    const bool moveB32 = input.dtype == DType::Float || input.dtype == DType::Int32 || input.dtype == DType::Uint32;
    const int64_t moveBytes = moveB16 ? sizeof(uint16_t) : sizeof(uint32_t);
    const int64_t moveWidth = moveB16 ? kElementsPerRepeatB16 : kElementsPerRepeatB32;
    const int64_t moveRows = context.valid_rows_i;
    const int64_t moveCols = context.valid_cols_i;
    const bool moveRow = moveRows == 1 && (moveCols <= moveWidth || moveCols % (moveWidth / kMoveTailPartitions) == 0);
    const bool moveGrid =
        moveRows > 1 && moveRows <= kMoveMaxRows &&
        (moveCols <= moveWidth || (moveRows <= kMoveMultiRepeatMaxRows && moveCols <= kMoveMaxRowRepeats * moveWidth));
    if (input.op == PtoOpcode::TMOV && (moveB16 || moveB32) && context.op_params.empty() && input.rows == moveRows &&
        input.cols == moveCols && moveCols > 0 && moveCols % (kMoveAlignmentBytes / moveBytes) == 0 &&
        moveRows <= kMoveMaxBytes / moveBytes / moveCols && (moveRow || moveGrid) &&
        (input.dst_rows <= 0 || input.dst_rows == moveRows) && (input.dst_cols <= 0 || input.dst_cols == moveCols) &&
        (input.dst_valid_rows <= 0 || input.dst_valid_rows == moveRows) &&
        (input.dst_valid_cols <= 0 || input.dst_valid_cols == moveCols)) {
        context.elements_per_repeat = static_cast<uint64_t>(moveWidth);
        context.shape_path = moveRows == 1 ? fit::ShapePath::Path1D : fit::ShapePath::Path2D;
        context.op_params = "dense_vec_move";
        return true;
    }
    // TCmps B32 packs a pair of predicates per iteration; B8/B16 use one vector.
    // These limits describe the dense calibration domain, not instruction costs.
    constexpr int64_t kCompareMinCols = 32;
    constexpr int64_t kCompare2DMaxCols = 384;
    constexpr int64_t kCompareB8MaxRows = 362;
    constexpr int64_t kCompareB16MaxRows = 295;
    constexpr int64_t kCompareB32MaxRows = 229;
    constexpr int64_t kCompareB8MaxElements = 130944;
    constexpr int64_t kCompareB16MaxElements = 87296;
    constexpr int64_t kCompareB32MaxElements = 51296;
    constexpr int64_t kCompareB32MaxRowCols = 49152;
    const bool compareB8 = input.dtype == DType::Int8 || input.dtype == DType::Uint8;
    const bool compareB16 = input.dtype == DType::BFloat16 || input.dtype == DType::Uint16;
    const bool compareB32 = input.dtype == DType::Float || input.dtype == DType::Int32 || input.dtype == DType::Uint32;
    const int64_t compareWidth = compareB8 ? kElementsPerRepeatB8 : kElementsPerRepeatB16;
    const int64_t compareMaxRows =
        compareB8 ? kCompareB8MaxRows : (compareB16 ? kCompareB16MaxRows : kCompareB32MaxRows);
    const int64_t compareMaxElements =
        compareB8 ? kCompareB8MaxElements : (compareB16 ? kCompareB16MaxElements : kCompareB32MaxElements);
    const bool compareRow = context.shape_path == fit::ShapePath::Path1D && context.valid_rows_i == 1 &&
                            context.valid_cols_i <= (compareB32 ? kCompareB32MaxRowCols : compareMaxElements);
    const bool compareGrid = context.shape_path == fit::ShapePath::Path2D && context.valid_rows_i > 1 &&
                             context.valid_rows_i <= compareMaxRows && context.valid_cols_i > compareWidth &&
                             context.valid_cols_i <= kCompare2DMaxCols &&
                             context.valid_rows_i <= compareMaxElements / context.valid_cols_i;
    if (input.op == PtoOpcode::TCMPS && (compareB8 || compareB16 || compareB32) && input.rows == context.valid_rows_i &&
        input.cols == context.valid_cols_i && context.valid_cols_i >= kCompareMinCols && context.op_params.empty() &&
        (compareRow || compareGrid)) {
        context.elements_per_repeat = static_cast<uint64_t>(compareWidth);
        context.op_params = "dense_packed_compare";
        return true;
    }
    if (input.op == PtoOpcode::TCMP && (input.dtype == DType::Float || input.dtype == DType::Int32) &&
        context.shape_path == fit::ShapePath::Path2D && input.rows == context.valid_rows_i &&
        input.cols == context.valid_cols_i && context.op_params.empty()) {
        // TCmp_32B combines two B32 predicates in each inner iteration.
        constexpr uint64_t kPredicatesPerCompareIteration = 2;
        context.elements_per_repeat = kPredicatesPerCompareIteration * kElementsPerRepeatB32;
        context.op_params = "packed_predicate_dense";
        return true;
    }
    if (input.op == PtoOpcode::TSORT32) {
        context.elements_per_repeat = kSort32ElementsPerRepeat;
        return true;
    }
    if ((input.op == PtoOpcode::TGATHER || input.op == PtoOpcode::TSCATTER) && input.has_dtype2) {
        return TryGetA5ElementsPerRepeat(input.dtype2, context.elements_per_repeat);
    }
    if (input.op == PtoOpcode::TCVT) {
        return TryGetA5CvtElementsPerRepeat(
            context.src_dtype, context.dst_dtype, context.shape_path, context.elements_per_repeat);
    }
    if (input.op == PtoOpcode::TDEQUANT) {
        return TryGetA5ElementsPerRepeat(input.dst_dtype, context.elements_per_repeat);
    }
    return TryGetA5ElementsPerRepeat(input.dtype, context.elements_per_repeat);
}

inline void ResolveLoopCounts(
    const VfCurveContext& context, uint64_t& loop_count, uint64_t& outer_iter, uint64_t& inner_iter,
    uint64_t& tail_count, uint64_t& row_inner_iter)
{
    const uint64_t valid_rows = static_cast<uint64_t>(context.valid_rows_i);
    const uint64_t valid_cols = static_cast<uint64_t>(context.valid_cols_i);
    row_inner_iter = CeilDivU64(valid_cols, context.elements_per_repeat);
    if (context.shape_path == fit::ShapePath::Path1D) {
        inner_iter = CeilDivU64(valid_rows * valid_cols, context.elements_per_repeat);
        outer_iter = valid_rows;
        loop_count = inner_iter;
        tail_count = (valid_rows * valid_cols) % context.elements_per_repeat;
        return;
    }
    inner_iter = CeilDivU64(valid_cols, context.elements_per_repeat);
    outer_iter = valid_rows;
    loop_count = outer_iter * inner_iter;
    tail_count = valid_cols % context.elements_per_repeat;
}

// The legacy overload reports one outer iteration for flattened 1D tiles.
inline void ResolveLoopCounts(
    const VfCurveContext& context, uint64_t& loop_count, uint64_t& outer_iter, uint64_t& inner_iter,
    uint64_t& tail_count)
{
    uint64_t row_inner_iter = 0;
    ResolveLoopCounts(context, loop_count, outer_iter, inner_iter, tail_count, row_inner_iter);
    if (context.shape_path == fit::ShapePath::Path1D) {
        outer_iter = 1;
    }
}

inline bool TryResolveTselOpParams(const CostModelInput& input, uint64_t inner_iter, VfCurveContext& context)
{
    if (input.op != PtoOpcode::TSEL) {
        return true;
    }
    if (context.op_params.empty()) {
        // B32 selection processes vector pairs, then traverses the rows again
        // for an odd remainder. Keep the two measured schedules separate.
        constexpr int64_t kSelectB16MaxDenseRowCols = 28032;
        constexpr int64_t kSelectB32MaxDenseRowCols = 15096;
        constexpr int64_t kSelectB32MaxOddRowCols = 320;
        constexpr int64_t kSelectB32MaxDenseRows = 64;
        constexpr int64_t kSelectB32MaxDenseCols = 384;
        constexpr int64_t kSelectB32MaxSingleVectorRows = 471;
        constexpr int64_t kSelectB32MaxSinglePairRows = 123;
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

inline void ResolveShapeSpecificOpParams(
    const CostModelInput& input, uint64_t outer_iter, uint64_t inner_iter, VfCurveContext& context)
{
    constexpr uint64_t kSingleRow = 1;
    constexpr uint64_t kSingleInnerIteration = 1;
    const int64_t validRows = input.valid_rows > 0 ? input.valid_rows : input.rows;
    const int64_t validCols = input.valid_cols > 0 ? input.valid_cols : input.cols;
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
    if (context.src_dtype == "fp16" && context.shape_path == fit::ShapePath::Path2D && context.op_params.empty() &&
        input.rows == validRows && input.cols == validCols &&
        (input.op == PtoOpcode::TABS || input.op == PtoOpcode::TNEG)) {
        context.op_params = "dense_2d";
    }
    // Scalar b32 comparisons pack two vector predicates per inner iteration.
    // The calibrated single-iteration curve depends on rows, not vector repeats.
    const bool hasSingleCompareFit = context.src_dtype == "bf16" || context.src_dtype == "fp32" ||
                                     context.src_dtype == "int32" || context.src_dtype == "uint32" ||
                                     context.src_dtype == "uint16" || context.src_dtype == "int8" ||
                                     context.src_dtype == "uint8";
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
    const bool hasSingleInnerTcvtFit =
        (context.src_dtype == "bf16" && (context.dst_dtype == "fp4_e1m2" || context.dst_dtype == "fp4_e2m1")) ||
        (context.src_dtype == "fp32" &&
         (context.dst_dtype == "fp8_e4m3" || context.dst_dtype == "fp8_e5m2" || context.dst_dtype == "hif8"));
    if (input.op == PtoOpcode::TCVT && hasSingleInnerTcvtFit && context.shape_path == fit::ShapePath::Path2D &&
        inner_iter == kSingleInnerIteration && context.op_params == "CAST_RINT_SAT_ON_normal") {
        context.op_params = "CAST_RINT_SAT_ON_inner1";
    }
}

inline bool BuildVfCurveKeyAndLoopCount(
    const CostModelInput& input, fit::VfCurveKey& key, uint64_t& loop_count, uint64_t& outer_iter, uint64_t& inner_iter,
    uint64_t& tail_count, uint64_t& row_inner_iter)
{
    VfCurveContext context;
    if (!TryInitVfCurveContext(input, context) || !TryResolveVfOpParams(input, context) ||
        !TryResolveElementsPerRepeat(input, context) || context.elements_per_repeat == 0) {
        return false;
    }

    ResolveLoopCounts(context, loop_count, outer_iter, inner_iter, tail_count, row_inner_iter);
    ResolveShapeSpecificOpParams(input, outer_iter, inner_iter, context);
    const auto tail_kind = tail_count == 0 ? fit::TailKind::Full : fit::TailKind::Tail;
    if (!TryResolveTselOpParams(input, inner_iter, context)) {
        return false;
    }

    key = fit::VfCurveKey{
        context.op_key,
        context.src_dtype,
        context.dst_dtype,
        context.shape_path,
        ResolveVfImplKind(input.op, input.vf_impl_kind, context.shape_path),
        tail_kind,
        context.op_params,
    };
    // Short dense single-row profiles are calibrated separately from flattened
    // multi-row and padded paths. Keep existing keys outside the measured range
    // and preserve explicit semantic parameters (mask, conversion mode, etc.).
    constexpr uint64_t kCalibratedRowRepeats = 8;
    // Additional dense uint16 scalar profiles validate the same linear curve
    // over a wider range. These are measured-domain limits, not cycle constants.
    constexpr int64_t kCalibratedAndXorScalarCols = 43632;
    constexpr int64_t kCalibratedOrScalarCols = 65456;
    constexpr int64_t kCalibratedExpandB16Cols = 130928;
    constexpr int64_t kCalibratedExpandB32Cols = 65536;
    const bool calibratedExpandRow =
        input.op == PtoOpcode::TEXPANDS &&
        ((input.dtype == DType::Int16 && context.valid_cols_i <= kCalibratedExpandB16Cols) ||
         ((input.dtype == DType::Float || input.dtype == DType::Int32) &&
          context.valid_cols_i <= kCalibratedExpandB32Cols));
    const bool calibratedScalarRow = input.dtype == DType::Uint16 &&
                                     (((input.op == PtoOpcode::TANDS || input.op == PtoOpcode::TXORS) &&
                                       context.valid_cols_i <= kCalibratedAndXorScalarCols) ||
                                      (input.op == PtoOpcode::TORS && context.valid_cols_i <= kCalibratedOrScalarCols));
    const bool defaultRowParams = key.op_params.empty() || key.op_params == "single_row_single_repeat" ||
                                  key.op_params == "multi_inner" || key.op_params == "TSel_b16_8";
    constexpr int64_t kCalibratedAndOrCols = 43648;
    constexpr int64_t kCalibratedXorCols = 32768;
    constexpr int64_t kCalibratedNotCols = 65536;
    // Wide unary profiles cover aligned rows and half-vector tails. Do not
    // extrapolate the measured tail throughput to other residual widths.
    const bool calibratedNotRow =
        input.op == PtoOpcode::TNOT && (input.dtype == DType::Int16 || input.dtype == DType::Uint16) &&
        context.valid_cols_i <= kCalibratedNotCols && (tail_count == 0 || tail_count == kElementsPerRepeatB16 / 2);
    const bool calibratedBinaryRow =
        input.dtype == DType::Uint16 && (((input.op == PtoOpcode::TAND || input.op == PtoOpcode::TOR) &&
                                          context.valid_cols_i <= kCalibratedAndOrCols) ||
                                         (input.op == PtoOpcode::TXOR && context.valid_cols_i <= kCalibratedXorCols));
    // Wide dense-row profiles capture the measured short/long loop throughput
    // transition without changing the established short-row calibration.
    if (key.shape_path == fit::ShapePath::Path1D && context.valid_rows_i == 1 && input.rows == 1 &&
        input.cols == context.valid_cols_i && loop_count > kCalibratedRowRepeats &&
        (calibratedBinaryRow || calibratedNotRow) && defaultRowParams && input.a5_op_params.empty()) {
        key.op_params = "dense_row1_wide";
    }
    if (key.shape_path == fit::ShapePath::Path1D && context.valid_rows_i == 1 && input.rows == 1 &&
        input.cols == context.valid_cols_i &&
        (loop_count <= kCalibratedRowRepeats || calibratedScalarRow || calibratedExpandRow) && defaultRowParams &&
        input.a5_op_params.empty()) {
        auto rowKey = key;
        rowKey.op_params = "dense_row1";
        const fit::VfFormulaParam* rowParam = nullptr;
        if (fit::TryLookupVfFormulaParam(rowKey, rowParam)) {
            key = rowKey;
        }
    }
    return true;
}

// Use current curve keys while preserving the legacy outer-loop convention.
inline bool BuildVfCurveKeyAndLoopCount(
    const CostModelInput& input, fit::VfCurveKey& key, uint64_t& loop_count, uint64_t& outer_iter, uint64_t& inner_iter,
    uint64_t& tail_count)
{
    uint64_t row_inner_iter = 0;
    if (!BuildVfCurveKeyAndLoopCount(input, key, loop_count, outer_iter, inner_iter, tail_count, row_inner_iter)) {
        return false;
    }
    if (key.shape_path == fit::ShapePath::Path1D) {
        outer_iter = 1;
    }
    return true;
}

inline bool TryEstimateA5ScatterPatternCycles(const CostModelInput& input, uint64_t& cycles)
{
    uint64_t elementsPerRepeat = 0;
    if (!TryGetA5ElementsPerRepeat(input.dtype, elementsPerRepeat) ||
        (elementsPerRepeat != kElementsPerRepeatB16 && elementsPerRepeat != kElementsPerRepeatB32) ||
        input.has_dtype2 || input.scatter_axis != ScatterAxis::SCATTER_ROW ||
        (input.mask_pattern != MaskPattern::P0001 && input.mask_pattern != MaskPattern::P0101)) {
        return false;
    }
    const int64_t validRows = input.valid_rows > 0 ? input.valid_rows : input.rows;
    const int64_t validCols = input.valid_cols > 0 ? input.valid_cols : input.cols;
    // Unlike the effective output shape, the destination allocation cannot be
    // inferred from the mask: the implementation clears padding as well.
    if (validRows <= 0 || validCols <= 0 || input.dst_rows <= 0 || input.dst_cols <= 0) {
        return false;
    }
    constexpr int64_t kAlternatingExpansion = 2;
    constexpr int64_t kQuarterExpansion = 4;
    const int64_t expansion = input.mask_pattern == MaskPattern::P0001 ? kQuarterExpansion : kAlternatingExpansion;
    // The additional dtypes have hardware evidence for dense destinations only.
    // Keep the independently calibrated fp16 physical-padding path unchanged.
    if (input.dtype != DType::Half && (input.dst_rows != validRows || input.dst_cols != validCols * expansion)) {
        return false;
    }
    if (validRows > input.rows || validCols > input.cols || input.dst_rows < validRows ||
        input.dst_cols < validCols * expansion || (input.dst_valid_rows > 0 && input.dst_valid_rows != validRows) ||
        (input.dst_valid_cols > 0 && input.dst_valid_cols != validCols * expansion)) {
        return false;
    }
    fit::VfCurveKey key{
        "TSCATTER",
        DTypeToKey(input.dtype),
        "none",
        fit::ShapePath::Path1D,
        "POST_UPDATE",
        validCols % elementsPerRepeat == 0 ? fit::TailKind::Full : fit::TailKind::Tail,
        PatternOpParams(input.mask_pattern, false)};
    const fit::VfFormulaParam* param = nullptr;
    if (!fit::TryLookupVfFormulaParam(key, param)) {
        return false;
    }
    const auto rows = static_cast<uint64_t>(validRows);
    const auto rowRepeats = CeilDivU64(static_cast<uint64_t>(validCols), elementsPerRepeat);
    const auto initRepeats =
        CeilDivU64(static_cast<uint64_t>(input.dst_rows) * static_cast<uint64_t>(input.dst_cols), elementsPerRepeat);
    cycles = fit::EvalVfFormula(*param, rows * rowRepeats, rows, rowRepeats, 0, rowRepeats, initRepeats);
    if (input.mask_pattern == MaskPattern::P0101) {
        // The two fitted throughput limits cover repeat-dominated and row-loop-
        // dominated shapes without a shape-specific threshold or lookup table.
        key.op_params = "pattern_P0101_row_loop";
        if (!fit::TryLookupVfFormulaParam(key, param)) {
            return false;
        }
        cycles = std::max(
            cycles, fit::EvalVfFormula(*param, rows * rowRepeats, rows, rowRepeats, 0, rowRepeats, initRepeats));
    }
    return cycles > 0;
}

inline bool TryEstimateA5ConcatCycles(const CostModelInput& input, uint64_t& cycles)
{
    const int64_t rows = input.valid_rows > 0 ? input.valid_rows : input.rows;
    const int64_t cols = input.valid_cols > 0 ? input.valid_cols : input.cols;
    if (!input.concat_dense_sources || rows <= 0 || rows != input.rows || cols != input.cols ||
        input.concat_src0_cols <= 0 || input.concat_src0_cols != input.concat_src1_cols ||
        input.concat_src0_cols > cols || input.concat_src1_cols != cols - input.concat_src0_cols) {
        return false;
    }
    uint64_t elementsPerRepeat = 0;
    if (!TryGetA5ElementsPerRepeat(input.dtype, elementsPerRepeat)) {
        return false;
    }
    // Copy and scatter each traverse their own source. Equal-width sources
    // allow the calibrated repeat coefficient to represent both loops.
    const auto sourceCols = static_cast<uint64_t>(input.concat_src0_cols);
    const auto repeats = CeilDivU64(sourceCols, elementsPerRepeat);
    const fit::VfCurveKey key{
        "TCONCAT",
        DTypeToKey(input.dtype),
        "none",
        fit::ShapePath::Path2D,
        "NO_POST_UPDATE",
        sourceCols % elementsPerRepeat == 0 ? fit::TailKind::Full : fit::TailKind::Tail,
        "concat_equal_dense"};
    const fit::VfFormulaParam* param = nullptr;
    if (!fit::TryLookupVfFormulaParam(key, param)) {
        return false;
    }
    cycles = fit::EvalVfFormula(
        *param, static_cast<uint64_t>(rows) * repeats, static_cast<uint64_t>(rows), repeats, 0, repeats);
    return cycles > 0;
}

inline bool TryEstimateA5VfCycles(const CostModelInput& input, uint64_t& cycles)
{
    if (input.metadata.kind != MetadataGeometry::Kind::Missing) {
        return TryEstimateA5MetadataCycles(input, cycles);
    }
    if (input.op == PtoOpcode::SET_QUANT_VECTOR) {
        // Public tag validates Scaling geometry and explicit aligned address;
        // uint64 tile data need not be representable by the generic dtype enum.
        if (input.a5_op_params != "quant_vector_scaling_u64_128") {
            return false;
        }
        const fit::VfCurveKey key{"SET_QUANT_VECTOR",     "uint64",         "none",
                                  fit::ShapePath::Path2D, "NO_POST_UPDATE", fit::TailKind::Full,
                                  input.a5_op_params};
        const fit::VfFormulaParam* param = nullptr;
        if (!fit::TryLookupVfFormulaParam(key, param)) {
            return false;
        }
        // Measured FIX register-write increment, not address packing or initialization.
        cycles = fit::EvalVfFormula(*param, 1, 1, 1, 0, 1);
        return cycles > 0;
    }
    if (input.op == PtoOpcode::SET_QUANT_SCALAR) {
        // No tile operand: the public wrapper validates OutType and scale in this tag.
        if (input.a5_op_params != "quant_scalar_half_positive") {
            return false;
        }
        const fit::VfCurveKey key{"SET_QUANT_SCALAR",     "fp16",           "none",
                                  fit::ShapePath::Path2D, "NO_POST_UPDATE", fit::TailKind::Full,
                                  input.a5_op_params};
        const fit::VfFormulaParam* param = nullptr;
        if (!fit::TryLookupVfFormulaParam(key, param)) {
            return false;
        }
        // FIX register-write increment; excludes scalar packing and end-to-end API latency.
        cycles = fit::EvalVfFormula(*param, 1, 1, 1, 0, 1);
        return cycles > 0;
    }
    if (input.op == PtoOpcode::SETFMATRIX || input.op == PtoOpcode::SET_IMG2COL_RPT ||
        input.op == PtoOpcode::SET_IMG2COL_PADDING) {
        if (input.a5_op_params != "image_config_manual_half" || input.dtype != DType::Half) {
            return false;
        }
        const char* op = input.op == PtoOpcode::SETFMATRIX      ? "SETFMATRIX" :
                         input.op == PtoOpcode::SET_IMG2COL_RPT ? "SET_IMG2COL_RPT" :
                                                                  "SET_IMG2COL_PADDING";
        const fit::VfCurveKey key{
            op, "fp16", "none", fit::ShapePath::Path2D, "NO_POST_UPDATE", fit::TailKind::Full, input.a5_op_params};
        const fit::VfFormulaParam* param = nullptr;
        if (!fit::TryLookupVfFormulaParam(key, param)) {
            return false;
        }
        // Measured MTE1 register-configuration increment, not scalar packing or total API latency.
        cycles = fit::EvalVfFormula(*param, 1, 1, 1, 0, 1);
        return cycles > 0;
    }
    if (input.op == PtoOpcode::TIMG2COL) {
        if (input.a5_op_params != "image_left_manual_1x1" || input.dtype != DType::Half || input.rows < 16 ||
            input.rows > 96 || input.rows % 16 != 0 || input.cols < 16 || input.cols > 128 || input.cols % 16 != 0 ||
            input.valid_rows != input.rows || input.valid_cols != input.cols) {
            return false;
        }
        const fit::VfCurveKey key{
            "TIMG2COL",        "fp16", "none", fit::ShapePath::Path2D, "NO_POST_UPDATE", fit::TailKind::Full,
            input.a5_op_params};
        const fit::VfFormulaParam* param = nullptr;
        if (!fit::TryLookupVfFormulaParam(key, param)) {
            return false;
        }
        // AIC MTE1 service cycles, excluding the fixed initialization cost.
        cycles = fit::EvalVfFormula(*param, input.rows * input.cols, input.rows, input.cols, 0, 1);
        return cycles > 0;
    }
    if ((input.op == PtoOpcode::TEXTRACT || input.op == PtoOpcode::TINSERT) &&
        input.a5_op_params == "fp_acc_vec_offset64_relu") {
        const bool insert = input.op == PtoOpcode::TINSERT;
        const auto rows = input.rows - 1;
        const auto cols = input.cols - 96;
        if (input.dtype != DType::Half || rows < 16 || rows > 64 || rows % 8 != 0 || cols < 32 || cols > 256 ||
            cols % 16 != 0 || input.valid_rows != (insert ? input.rows : rows) ||
            input.valid_cols != (insert ? input.cols : cols)) {
            return false;
        }
        const fit::VfCurveKey key{
            insert ? "TINSERT_FP" : "TEXTRACT_FP",
            "fp32",
            "fp16",
            fit::ShapePath::Path2D,
            "POST_UPDATE",
            fit::TailKind::Full,
            input.a5_op_params};
        const fit::VfFormulaParam* param = nullptr;
        if (!fit::TryLookupVfFormulaParam(key, param)) {
            return false;
        }
        // Empirical FIX service granularity, validated only in the bounded domain above.
        uint64_t chunks = 1;
        const auto required = CeilDivU64(static_cast<uint64_t>(cols), 64);
        while (chunks < required) {
            chunks *= 2;
        }
        cycles = fit::EvalVfFormula(*param, static_cast<uint64_t>(rows) * chunks, rows, chunks, 0, chunks);
        return cycles > 0;
    }
    if (input.op == PtoOpcode::TEXTRACT || input.op == PtoOpcode::TINSERT) {
        constexpr int64_t kMeasuredOffsetBytes = 32;
        constexpr int64_t kMeasuredOffsetRows = 1;
        constexpr int64_t kMaxMeasuredRows = 16;
        constexpr int64_t kMinMeasuredCols = 32;
        constexpr int64_t kMaxMeasuredCols = 1024;
        constexpr int64_t kMeasuredColAlignment = 32;
        if (input.a5_op_params != "dynamic_slice_offset1_32B" ||
            (input.dtype != DType::Half && input.dtype != DType::Float && input.dtype != DType::Int32)) {
            return false;
        }
        const int64_t elementBytes = input.dtype == DType::Half ? sizeof(uint16_t) : sizeof(uint32_t);
        // Record dimensions describe the destination. TINSERT's source is
        // the smaller copied window, certified by the public-wrapper tag.
        const bool insert = input.op == PtoOpcode::TINSERT;
        const int64_t rows = input.rows - (insert ? kMeasuredOffsetRows : 0);
        const int64_t cols = input.cols - (insert ? kMeasuredOffsetBytes / elementBytes : 0);
        if (rows < 1 || rows > kMaxMeasuredRows || cols < kMinMeasuredCols || cols > kMaxMeasuredCols ||
            cols % kMeasuredColAlignment != 0) {
            return false;
        }
        const fit::VfCurveKey key{
            insert ? "TINSERT" : "TEXTRACT",
            DTypeToKey(input.dtype),
            "none",
            fit::ShapePath::Path2D,
            "POST_UPDATE",
            fit::TailKind::Full,
            input.a5_op_params};
        const fit::VfFormulaParam* param = nullptr;
        if (!fit::TryLookupVfFormulaParam(key, param)) {
            return false;
        }
        const auto width = input.dtype == DType::Half ? kElementsPerRepeatB16 : kElementsPerRepeatB32;
        const auto inner = CeilDivU64(static_cast<uint64_t>(cols), width);
        cycles = fit::EvalVfFormula(*param, rows * inner, rows, inner, 0, inner);
        return cycles > 0;
    }
    if (input.op == PtoOpcode::THISTOGRAM) {
        constexpr int64_t kMaxMeasuredRows = 16;
        constexpr int64_t kMinMeasuredCols = 32;
        constexpr int64_t kMaxMeasuredCols = 1024;
        constexpr int64_t kMeasuredColAlignment = 32;
        constexpr uint64_t kHistogramElementsPerIteration = 256;
        const auto rows = input.valid_rows > 0 ? input.valid_rows : input.rows;
        const auto cols = input.valid_cols > 0 ? input.valid_cols : input.cols;
        if (input.a5_op_params != "u32_byte3_padded256" || input.dtype != DType::Uint32 || rows < 1 ||
            rows > kMaxMeasuredRows || rows != input.rows || cols < kMinMeasuredCols || cols > kMaxMeasuredCols ||
            cols % kMeasuredColAlignment != 0 ||
            input.cols != CeilDivU64(static_cast<uint64_t>(cols), kHistogramElementsPerIteration) *
                              kHistogramElementsPerIteration) {
            return false;
        }
        const fit::VfCurveKey key{"THISTOGRAM",        "uint32",          "none", fit::ShapePath::Path2D, "POST_UPDATE",
                                  fit::TailKind::Full, input.a5_op_params};
        const fit::VfFormulaParam* param = nullptr;
        if (!fit::TryLookupVfFormulaParam(key, param)) {
            return false;
        }
        const auto inner = CeilDivU64(static_cast<uint64_t>(cols), kHistogramElementsPerIteration);
        cycles = fit::EvalVfFormula(*param, rows * inner, rows, inner, 0, inner);
        return cycles > 0;
    }
    if (input.op == PtoOpcode::TQUANT) {
        constexpr int64_t kMaxMeasuredRows = 16;
        constexpr int64_t kMinMeasuredCols = 32;
        constexpr int64_t kMaxMeasuredCols = 1024;
        constexpr int64_t kMeasuredColAlignment = 32;
        const bool symmetric = input.a5_op_params == "dense_fp32_int8_sym" && input.dtype == DType::Int8;
        const bool asymmetric = input.a5_op_params == "dense_fp32_int8_asym" && input.dtype == DType::Uint8;
        if ((!symmetric && !asymmetric) || input.rows < 1 || input.rows > kMaxMeasuredRows ||
            input.cols < kMinMeasuredCols || input.cols > kMaxMeasuredCols || input.cols % kMeasuredColAlignment != 0 ||
            (input.valid_rows > 0 && input.valid_rows != input.rows) ||
            (input.valid_cols > 0 && input.valid_cols != input.cols)) {
            return false;
        }
        const fit::VfCurveKey key{"TQUANT",
                                  "fp32",
                                  DTypeToKey(input.dtype),
                                  fit::ShapePath::Path2D,
                                  "POST_UPDATE",
                                  fit::TailKind::Full,
                                  input.a5_op_params};
        const fit::VfFormulaParam* param = nullptr;
        if (!fit::TryLookupVfFormulaParam(key, param)) {
            return false;
        }
        // Quantization operates on fp32 source lanes, not the byte output lanes.
        const auto inner = CeilDivU64(static_cast<uint64_t>(input.cols), kElementsPerRepeatB32);
        cycles = fit::EvalVfFormula(*param, input.rows * inner, input.rows, inner, 0, inner);
        return cycles > 0;
    }
    if (input.op == PtoOpcode::TRANDOM) {
        constexpr int64_t kMaxMeasuredRows = 16;
        constexpr int64_t kMinMeasuredCols = 32;
        constexpr int64_t kMaxMeasuredCols = 1024;
        constexpr int64_t kMeasuredColAlignment = 32;
        // Each Philox inner iteration produces four 64-element registers.
        constexpr uint64_t kRandomElementsPerIteration = 256;
        if ((input.a5_op_params != "dense_rounds7" && input.a5_op_params != "dense_rounds10") || input.rows < 1 ||
            input.rows > kMaxMeasuredRows || input.cols < kMinMeasuredCols || input.cols > kMaxMeasuredCols ||
            input.cols % kMeasuredColAlignment != 0 || (input.valid_rows > 0 && input.valid_rows != input.rows) ||
            (input.valid_cols > 0 && input.valid_cols != input.cols)) {
            return false;
        }
        const fit::VfCurveKey key{
            "TRANDOM",           DTypeToKey(input.dtype), "none", fit::ShapePath::Path2D, "POST_UPDATE",
            fit::TailKind::Full, input.a5_op_params};
        const fit::VfFormulaParam* param = nullptr;
        if (!fit::TryLookupVfFormulaParam(key, param)) {
            return false;
        }
        const auto inner = CeilDivU64(static_cast<uint64_t>(input.cols), kRandomElementsPerIteration);
        cycles = fit::EvalVfFormula(*param, input.rows * inner, input.rows, inner, 0, inner);
        return cycles > 0;
    }
    if (input.op == PtoOpcode::TFILLPAD_INPLACE) {
        // Only the measured suffix-fill path is represented: no valid-element
        // copy and no additional bottom rows. Other geometries need separate fits.
        constexpr int64_t kMaxMeasuredRows = 16;
        constexpr int64_t kMinMeasuredCols = 32;
        constexpr int64_t kMaxMeasuredCols = 256;
        constexpr int64_t kMeasuredColAlignment = 32;
        if (input.a5_op_params != "suffix32B_no_bottom" || input.rows < 1 || input.rows > kMaxMeasuredRows ||
            input.cols < kMinMeasuredCols || input.cols > kMaxMeasuredCols || input.cols % kMeasuredColAlignment != 0 ||
            input.valid_rows != std::max<int64_t>(input.rows - 1, 1)) {
            return false;
        }
        const fit::VfCurveKey key{"TFILLPAD_INPLACE",     DTypeToKey(input.dtype), "none",
                                  fit::ShapePath::Path2D, "POST_UPDATE",           fit::TailKind::Full,
                                  "suffix32B_no_bottom"};
        const fit::VfFormulaParam* param = nullptr;
        if (!fit::TryLookupVfFormulaParam(key, param)) {
            return false;
        }
        cycles = fit::EvalVfFormula(*param, input.valid_rows, 0, 0, 0, 0);
        return cycles > 0;
    }
    if (input.op == PtoOpcode::TCONCAT && (input.concat_src0_cols != 0 || input.concat_src1_cols != 0)) {
        return TryEstimateA5ConcatCycles(input, cycles);
    }
    if (input.op == PtoOpcode::TSCATTER && input.has_mask_pattern && input.mask_pattern == MaskPattern::P1111 &&
        !input.has_dtype2) {
        CostModelInput moveInput = input;
        moveInput.op = PtoOpcode::TMOV;
        moveInput.has_mask_pattern = false;
        moveInput.a5_op_params = {};
        return TryEstimateA5VfCycles(moveInput, cycles);
    }
    if (input.op == PtoOpcode::TSCATTER && input.has_mask_pattern) {
        return TryEstimateA5ScatterPatternCycles(input, cycles);
    }
    if (input.op == PtoOpcode::TSCATTER && input.dtype == DType::Half && input.has_dtype2 &&
        (input.dtype2 == DType::Int16 || input.dtype2 == DType::Uint16)) {
        const int64_t validRows = input.valid_rows > 0 ? input.valid_rows : input.rows;
        const int64_t validCols = input.valid_cols > 0 ? input.valid_cols : input.cols;
        const int64_t physicalRows = input.rows > 0 ? input.rows : validRows;
        const int64_t physicalCols = input.cols > 0 ? input.cols : validCols;
        if (validRows <= 0 || validCols <= 0 || physicalRows <= 0 || physicalCols <= 0) {
            return false;
        }
        constexpr int64_t kSingleRow = 1;
        if (validRows == kSingleRow) {
            const double loopCount =
                static_cast<double>(CeilDivU64(static_cast<uint64_t>(validCols), kElementsPerRepeatB16));
            cycles = fit::RoundToCycles(EvalContinuousPiecewise(loopCount, kScatterIndexB16Fp16Row1Fit));
            return cycles > 0;
        }
        const double rows = static_cast<double>(validRows);
        const double loopCount =
            rows * static_cast<double>(CeilDivU64(static_cast<uint64_t>(validCols), kElementsPerRepeatB16));
        const double physicalElements = static_cast<double>(physicalRows) * static_cast<double>(physicalCols);
        const double initRepeats = static_cast<double>(CeilDivU64(
            static_cast<uint64_t>(physicalRows) * static_cast<uint64_t>(physicalCols), kElementsPerRepeatB16));
        const auto& fit = kScatterIndexB16Fp16Fit;
        const double value = fit.baseCycles + fit.loopLogSlope * loopCount * std::log2(std::max(rows, 1.0)) +
                             fit.tileSizeSlope * std::sqrt(physicalElements) +
                             fit.initRepeatSquareSlope * initRepeats * initRepeats +
                             fit.rowInitInteractionSlope * rows * initRepeats;
        cycles = fit::RoundToCycles(value);
        return cycles > 0;
    }

    fit::VfCurveKey key{};
    uint64_t loop_count = 0;
    uint64_t outer_iter = 0;
    uint64_t inner_iter = 0;
    uint64_t tail_count = 0;
    uint64_t row_inner_iter = 0;
    if (!BuildVfCurveKeyAndLoopCount(input, key, loop_count, outer_iter, inner_iter, tail_count, row_inner_iter)) {
        return false;
    }

    const fit::VfFormulaParam* param = nullptr;
    if (!fit::TryLookupVfFormulaParam(key, param)) {
        if (key.shape_path != fit::ShapePath::Path2D || !IsA5PathIndependentVfOp(input.op)) {
            return false;
        }
        // These PTO implementations use their own row/repeat loops instead of
        // selecting a different VF body for 1D and 2D tiles.  Keep the 2D loop
        // features calculated above, but use the canonical formula key that
        // was fitted before shape-path information reached the costmodel.
        key.shape_path = fit::ShapePath::Path1D;
        key.vf_impl_kind = "POST_UPDATE";
        if (!fit::TryLookupVfFormulaParam(key, param)) {
            return false;
        }
    }

    cycles = fit::EvalVfFormula(*param, loop_count, outer_iter, inner_iter, tail_count, row_inner_iter);
    return true;
}

} // namespace pto::mocker::lightweight::a5

#include <pto/costmodel/a5/metadata_costmodel.hpp>

#endif
