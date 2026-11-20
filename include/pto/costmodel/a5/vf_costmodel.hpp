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
    const bool defaultRowParams = key.op_params.empty() || key.op_params == "single_row_single_repeat" ||
                                  key.op_params == "multi_inner" || key.op_params == "TSel_b16_8";
    if (key.shape_path == fit::ShapePath::Path1D && context.valid_rows_i == 1 && input.rows == 1 &&
        input.cols == context.valid_cols_i && loop_count <= kCalibratedRowRepeats && defaultRowParams &&
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

#endif
