/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_MOCKER_A5_VF_COSTMODEL_GEOMETRY_HPP
#define PTO_MOCKER_A5_VF_COSTMODEL_GEOMETRY_HPP

#include <pto/costmodel/a5/vf_costmodel_context.hpp>

namespace pto::mocker::lightweight::a5 {

inline bool ValidScatterPatternVfGeometry(const CostModelInput& input, int64_t validRows, int64_t validCols)
{
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
    return true;
}

inline bool MatchesDenseMoveVfDestination(const CostModelInput& input, int64_t rows, int64_t cols)
{
    return (input.dst_rows <= 0 || input.dst_rows == rows) && (input.dst_cols <= 0 || input.dst_cols == cols) &&
           (input.dst_valid_rows <= 0 || input.dst_valid_rows == rows) &&
           (input.dst_valid_cols <= 0 || input.dst_valid_cols == cols);
}

inline bool IsDenseMoveVfGeometry(
    const CostModelInput& input, const VfCurveContext& context, int64_t moveWidth, int64_t moveBytes)
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
    const int64_t moveRows = context.valid_rows_i;
    const int64_t moveCols = context.valid_cols_i;
    const bool moveRow = moveRows == 1 && (moveCols <= moveWidth || moveCols % (moveWidth / kMoveTailPartitions) == 0);
    const bool moveGrid =
        moveRows > 1 && moveRows <= kMoveMaxRows &&
        (moveCols <= moveWidth || (moveRows <= kMoveMultiRepeatMaxRows && moveCols <= kMoveMaxRowRepeats * moveWidth));
    return input.rows == moveRows && input.cols == moveCols && moveCols > 0 &&
           moveCols % (kMoveAlignmentBytes / moveBytes) == 0 && moveRows <= kMoveMaxBytes / moveBytes / moveCols &&
           (moveRow || moveGrid) && MatchesDenseMoveVfDestination(input, moveRows, moveCols);
}

inline bool TryResolveDenseMoveElementsPerRepeat(const CostModelInput& input, VfCurveContext& context)
{
    const bool moveB16 = input.dtype == DType::BFloat16 || input.dtype == DType::Int16 || input.dtype == DType::Uint16;
    const bool moveB32 = input.dtype == DType::Float || input.dtype == DType::Int32 || input.dtype == DType::Uint32;
    const int64_t moveBytes = moveB16 ? sizeof(uint16_t) : sizeof(uint32_t);
    const int64_t moveWidth = moveB16 ? kElementsPerRepeatB16 : kElementsPerRepeatB32;
    if (input.op == PtoOpcode::TMOV && (moveB16 || moveB32) && context.op_params.empty() &&
        IsDenseMoveVfGeometry(input, context, moveWidth, moveBytes)) {
        context.elements_per_repeat = static_cast<uint64_t>(moveWidth);
        context.shape_path = context.valid_rows_i == 1 ? fit::ShapePath::Path1D : fit::ShapePath::Path2D;
        context.op_params = "dense_vec_move";
        return true;
    }
    return false;
}

struct DenseCompareVfLimits {
    int64_t width;
    int64_t maxRows;
    int64_t maxElements;
    int64_t maxRowCols;
};

inline bool TryGetDenseCompareVfLimits(DType dtype, DenseCompareVfLimits& limits)
{
    constexpr int64_t kCompareB8MaxRows = 362;
    constexpr int64_t kCompareB16MaxRows = 295;
    constexpr int64_t kCompareB32MaxRows = 229;
    constexpr int64_t kCompareB8MaxElements = 130944;
    constexpr int64_t kCompareB16MaxElements = 87296;
    constexpr int64_t kCompareB32MaxElements = 51296;
    constexpr int64_t kCompareB32MaxRowCols = 49152;
    constexpr DenseCompareVfLimits kByteLimits{
        kElementsPerRepeatB8, kCompareB8MaxRows, kCompareB8MaxElements, kCompareB8MaxElements};
    constexpr DenseCompareVfLimits kHalfwordLimits{
        kElementsPerRepeatB16, kCompareB16MaxRows, kCompareB16MaxElements, kCompareB16MaxElements};
    constexpr DenseCompareVfLimits kWordLimits{
        kElementsPerRepeatB16, kCompareB32MaxRows, kCompareB32MaxElements, kCompareB32MaxRowCols};
    // TCmps B32 packs two predicates; its repeat width is therefore B16-sized.
    constexpr std::array kMeasuredLimits = {
        std::pair{DType::Int8, kByteLimits},         std::pair{DType::Uint8, kByteLimits},
        std::pair{DType::BFloat16, kHalfwordLimits}, std::pair{DType::Uint16, kHalfwordLimits},
        std::pair{DType::Float, kWordLimits},        std::pair{DType::Int32, kWordLimits},
        std::pair{DType::Uint32, kWordLimits}};
    for (const auto& [type, measured] : kMeasuredLimits) {
        if (dtype == type) {
            limits = measured;
            return true;
        }
    }
    return false;
}

inline bool IsDenseCompareVfGeometry(const VfCurveContext& context, const DenseCompareVfLimits& limits)
{
    constexpr int64_t kCompare2DMaxCols = 384;
    const bool row = context.shape_path == fit::ShapePath::Path1D && context.valid_rows_i == 1 &&
                     context.valid_cols_i <= limits.maxRowCols;
    const bool grid = context.shape_path == fit::ShapePath::Path2D && context.valid_rows_i > 1 &&
                      context.valid_rows_i <= limits.maxRows && context.valid_cols_i > limits.width &&
                      context.valid_cols_i <= kCompare2DMaxCols &&
                      context.valid_rows_i <= limits.maxElements / context.valid_cols_i;
    return row || grid;
}

inline bool TryResolveDenseCompareElementsPerRepeat(const CostModelInput& input, VfCurveContext& context)
{
    constexpr int64_t kCompareMinCols = 32;
    DenseCompareVfLimits limits{};
    if (input.op != PtoOpcode::TCMPS || !TryGetDenseCompareVfLimits(input.dtype, limits) ||
        input.rows != context.valid_rows_i || input.cols != context.valid_cols_i ||
        context.valid_cols_i < kCompareMinCols || !context.op_params.empty() ||
        !IsDenseCompareVfGeometry(context, limits)) {
        return false;
    }
    context.elements_per_repeat = static_cast<uint64_t>(limits.width);
    context.op_params = "dense_packed_compare";
    return true;
}

inline bool TryResolveElementsPerRepeat(const CostModelInput& input, VfCurveContext& context)
{
    if (TryResolveDenseMoveElementsPerRepeat(input, context) ||
        TryResolveDenseCompareElementsPerRepeat(input, context)) {
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

inline bool HasExtendedDenseRowVfFit(const CostModelInput& input, const VfCurveContext& context)
{
    constexpr int64_t kCalibratedExpandB16Cols = 130928;
    constexpr int64_t kCalibratedExpandB32Cols = 65536;
    const bool expand = input.op == PtoOpcode::TEXPANDS &&
                        ((input.dtype == DType::Int16 && context.valid_cols_i <= kCalibratedExpandB16Cols) ||
                         ((input.dtype == DType::Float || input.dtype == DType::Int32) &&
                          context.valid_cols_i <= kCalibratedExpandB32Cols));
    constexpr int64_t kCalibratedAndXorScalarCols = 43632;
    constexpr int64_t kCalibratedOrScalarCols = 65456;
    const bool scalar = input.dtype == DType::Uint16 &&
                        (((input.op == PtoOpcode::TANDS || input.op == PtoOpcode::TXORS) &&
                          context.valid_cols_i <= kCalibratedAndXorScalarCols) ||
                         (input.op == PtoOpcode::TORS && context.valid_cols_i <= kCalibratedOrScalarCols));
    return expand || scalar;
}

inline bool HasWideDenseRowVfFit(const CostModelInput& input, const VfCurveContext& context, uint64_t tail_count)
{
    constexpr int64_t kCalibratedNotCols = 65536;
    const bool unary = input.op == PtoOpcode::TNOT && (input.dtype == DType::Int16 || input.dtype == DType::Uint16) &&
                       context.valid_cols_i <= kCalibratedNotCols &&
                       (tail_count == 0 || tail_count == kElementsPerRepeatB16 / kEvenModulo);
    constexpr int64_t kCalibratedAndOrCols = 43648;
    constexpr int64_t kCalibratedXorCols = 32768;
    const bool binary =
        input.dtype == DType::Uint16 && (((input.op == PtoOpcode::TAND || input.op == PtoOpcode::TOR) &&
                                          context.valid_cols_i <= kCalibratedAndOrCols) ||
                                         (input.op == PtoOpcode::TXOR && context.valid_cols_i <= kCalibratedXorCols));
    return unary || binary;
}

inline void ResolveDenseRowVfCurveKey(
    const CostModelInput& input, const VfCurveContext& context, uint64_t loop_count, uint64_t tail_count,
    fit::VfCurveKey& key)
{
    // Short dense single-row profiles are calibrated separately from flattened
    // multi-row and padded paths. Keep existing keys outside the measured range
    // and preserve explicit semantic parameters (mask, conversion mode, etc.).
    constexpr uint64_t kCalibratedRowRepeats = 8;
    // Additional dense uint16 scalar profiles validate the same linear curve
    // over a wider range. These are measured-domain limits, not cycle constants.
    const bool calibratedExtendedRow = HasExtendedDenseRowVfFit(input, context);
    const bool defaultRowParams = key.op_params.empty() || key.op_params == "single_row_single_repeat" ||
                                  key.op_params == "multi_inner" || key.op_params == "TSel_b16_8";
    // Wide unary profiles cover aligned rows and half-vector tails. Do not
    // extrapolate the measured tail throughput to other residual widths.
    const bool calibratedWideRow = HasWideDenseRowVfFit(input, context, tail_count);
    const bool denseRow = key.shape_path == fit::ShapePath::Path1D && context.valid_rows_i == 1 && input.rows == 1 &&
                          input.cols == context.valid_cols_i;
    // Wide dense-row profiles capture the measured short/long loop throughput
    // transition without changing the established short-row calibration.
    if (denseRow && loop_count > kCalibratedRowRepeats && calibratedWideRow && defaultRowParams &&
        input.a5_op_params.empty()) {
        key.op_params = "dense_row1_wide";
    }
    if (denseRow && (loop_count <= kCalibratedRowRepeats || calibratedExtendedRow) && defaultRowParams &&
        input.a5_op_params.empty()) {
        auto rowKey = key;
        rowKey.op_params = "dense_row1";
        const fit::VfFormulaParam* rowParam = nullptr;
        if (fit::TryLookupVfFormulaParam(rowKey, rowParam)) {
            key = rowKey;
        }
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
    ResolveDenseRowVfCurveKey(input, context, loop_count, tail_count, key);
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

} // namespace pto::mocker::lightweight::a5

#endif
