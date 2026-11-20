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

#include <pto/costmodel/a5/vf_costmodel_geometry.hpp>

namespace pto::mocker::lightweight::a5 {

inline bool TryEstimateA5MetadataCycles(const CostModelInput& input, uint64_t& cycles);

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
    if (!ValidScatterPatternVfGeometry(input, validRows, validCols)) {
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

inline bool TryEstimateA5VfFormulaCycles(const CostModelInput& input, uint64_t& cycles)
{
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

inline bool TryEstimateA5ScatterIndexCycles(const CostModelInput& input, uint64_t& cycles)
{
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
    const double initRepeats = static_cast<double>(
        CeilDivU64(static_cast<uint64_t>(physicalRows) * static_cast<uint64_t>(physicalCols), kElementsPerRepeatB16));
    const auto& fit = kScatterIndexB16Fp16Fit;
    const double value = fit.baseCycles + fit.loopLogSlope * loopCount * std::log2(std::max(rows, 1.0)) +
                         fit.tileSizeSlope * std::sqrt(physicalElements) +
                         fit.initRepeatSquareSlope * initRepeats * initRepeats +
                         fit.rowInitInteractionSlope * rows * initRepeats;
    cycles = fit::RoundToCycles(value);
    return cycles > 0;
}

inline bool TryEstimateA5ScatterCycles(const CostModelInput& input, uint64_t& cycles)
{
    if (input.has_mask_pattern && input.mask_pattern == MaskPattern::P1111 && !input.has_dtype2) {
        CostModelInput moveInput = input;
        moveInput.op = PtoOpcode::TMOV;
        moveInput.has_mask_pattern = false;
        moveInput.a5_op_params = {};
        return TryEstimateA5VfFormulaCycles(moveInput, cycles);
    }
    if (input.has_mask_pattern) {
        return TryEstimateA5ScatterPatternCycles(input, cycles);
    }
    if (input.dtype == DType::Half && input.has_dtype2 &&
        (input.dtype2 == DType::Int16 || input.dtype2 == DType::Uint16)) {
        return TryEstimateA5ScatterIndexCycles(input, cycles);
    }
    return TryEstimateA5VfFormulaCycles(input, cycles);
}

inline bool TryEstimateA5SetQuantCycles(const CostModelInput& input, uint64_t& cycles)
{
    // Public tags certify either aligned uint64 Scaling geometry or a positive
    // fp16 scalar scale. Both paths measure the FIX register-write increment.
    const bool vector = input.op == PtoOpcode::SET_QUANT_VECTOR;
    const auto expectedParams = vector ? "quant_vector_scaling_u64_128" : "quant_scalar_half_positive";
    if (input.a5_op_params != expectedParams) {
        return false;
    }
    const fit::VfCurveKey key{
        vector ? "SET_QUANT_VECTOR" : "SET_QUANT_SCALAR",
        vector ? "uint64" : "fp16",
        "none",
        fit::ShapePath::Path2D,
        "NO_POST_UPDATE",
        fit::TailKind::Full,
        input.a5_op_params};
    const fit::VfFormulaParam* param = nullptr;
    if (!fit::TryLookupVfFormulaParam(key, param)) {
        return false;
    }
    // Excludes scalar/address packing, initialization and end-to-end latency.
    cycles = fit::EvalVfFormula(*param, 1, 1, 1, 0, 1);
    return cycles > 0;
}

inline bool TryEstimateA5ImageConfigCycles(const CostModelInput& input, uint64_t& cycles)
{
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

inline bool TryEstimateA5ImageToColCycles(const CostModelInput& input, uint64_t& cycles)
{
    constexpr int64_t kImageMinDimension = 16;
    constexpr int64_t kImageMaxRows = 96;
    constexpr int64_t kImageMaxCols = 128;
    constexpr int64_t kImageDimensionAlignment = 16;
    if (input.a5_op_params != "image_left_manual_1x1" || input.dtype != DType::Half ||
        input.rows < kImageMinDimension || input.rows > kImageMaxRows || input.rows % kImageDimensionAlignment != 0 ||
        input.cols < kImageMinDimension || input.cols > kImageMaxCols || input.cols % kImageDimensionAlignment != 0 ||
        input.valid_rows != input.rows || input.valid_cols != input.cols) {
        return false;
    }
    const fit::VfCurveKey key{
        "TIMG2COL", "fp16", "none", fit::ShapePath::Path2D, "NO_POST_UPDATE", fit::TailKind::Full, input.a5_op_params};
    const fit::VfFormulaParam* param = nullptr;
    if (!fit::TryLookupVfFormulaParam(key, param)) {
        return false;
    }
    // AIC MTE1 service cycles, excluding the fixed initialization cost.
    cycles = fit::EvalVfFormula(*param, input.rows * input.cols, input.rows, input.cols, 0, 1);
    return cycles > 0;
}

inline bool TryEstimateA5FpSliceCycles(const CostModelInput& input, uint64_t& cycles)
{
    constexpr int64_t kSliceOffsetRows = 1;
    constexpr int64_t kSliceOffsetCols = 96;
    constexpr int64_t kMinMeasuredRows = 16;
    constexpr int64_t kMaxMeasuredRows = 64;
    constexpr int64_t kMeasuredRowAlignment = 8;
    constexpr int64_t kMinMeasuredCols = 32;
    constexpr int64_t kMaxMeasuredCols = 256;
    constexpr int64_t kMeasuredColAlignment = 16;
    constexpr uint64_t kFixElementsPerChunk = 64;
    constexpr uint64_t kChunkGrowthFactor = 2;
    const bool insert = input.op == PtoOpcode::TINSERT;
    const auto rows = input.rows - kSliceOffsetRows;
    const auto cols = input.cols - kSliceOffsetCols;
    if (input.dtype != DType::Half || rows < kMinMeasuredRows || rows > kMaxMeasuredRows ||
        rows % kMeasuredRowAlignment != 0 || cols < kMinMeasuredCols || cols > kMaxMeasuredCols ||
        cols % kMeasuredColAlignment != 0 || input.valid_rows != (insert ? input.rows : rows) ||
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
    const auto required = CeilDivU64(static_cast<uint64_t>(cols), kFixElementsPerChunk);
    while (chunks < required) {
        chunks *= kChunkGrowthFactor;
    }
    cycles = fit::EvalVfFormula(*param, static_cast<uint64_t>(rows) * chunks, rows, chunks, 0, chunks);
    return cycles > 0;
}

inline bool TryEstimateA5SliceCycles(const CostModelInput& input, uint64_t& cycles)
{
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

inline bool TryEstimateA5HistogramCycles(const CostModelInput& input, uint64_t& cycles)
{
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
        input.cols !=
            CeilDivU64(static_cast<uint64_t>(cols), kHistogramElementsPerIteration) * kHistogramElementsPerIteration) {
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

inline bool TryEstimateA5QuantCycles(const CostModelInput& input, uint64_t& cycles)
{
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
    const fit::VfCurveKey key{
        "TQUANT",          "fp32", DTypeToKey(input.dtype), fit::ShapePath::Path2D, "POST_UPDATE", fit::TailKind::Full,
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

inline bool TryEstimateA5RandomCycles(const CostModelInput& input, uint64_t& cycles)
{
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

inline bool TryEstimateA5FillPadInplaceCycles(const CostModelInput& input, uint64_t& cycles)
{
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

inline bool TryEstimateA5VfCycles(const CostModelInput& input, uint64_t& cycles)
{
    if (input.metadata.kind != MetadataGeometry::Kind::Missing) {
        return TryEstimateA5MetadataCycles(input, cycles);
    }
    switch (input.op) {
        case PtoOpcode::SET_QUANT_VECTOR:
        case PtoOpcode::SET_QUANT_SCALAR:
            return TryEstimateA5SetQuantCycles(input, cycles);
        case PtoOpcode::SETFMATRIX:
        case PtoOpcode::SET_IMG2COL_RPT:
        case PtoOpcode::SET_IMG2COL_PADDING:
            return TryEstimateA5ImageConfigCycles(input, cycles);
        case PtoOpcode::TIMG2COL:
            return TryEstimateA5ImageToColCycles(input, cycles);
        case PtoOpcode::TEXTRACT:
        case PtoOpcode::TINSERT:
            return input.a5_op_params == "fp_acc_vec_offset64_relu" ? TryEstimateA5FpSliceCycles(input, cycles) :
                                                                      TryEstimateA5SliceCycles(input, cycles);
        case PtoOpcode::THISTOGRAM:
            return TryEstimateA5HistogramCycles(input, cycles);
        case PtoOpcode::TQUANT:
            return TryEstimateA5QuantCycles(input, cycles);
        case PtoOpcode::TRANDOM:
            return TryEstimateA5RandomCycles(input, cycles);
        case PtoOpcode::TFILLPAD_INPLACE:
            return TryEstimateA5FillPadInplaceCycles(input, cycles);
        case PtoOpcode::TSCATTER:
            return TryEstimateA5ScatterCycles(input, cycles);
        case PtoOpcode::TCONCAT:
            if (input.concat_src0_cols != 0 || input.concat_src1_cols != 0) {
                return TryEstimateA5ConcatCycles(input, cycles);
            }
            break;
        default:
            break;
    }
    return TryEstimateA5VfFormulaCycles(input, cycles);
}

} // namespace pto::mocker::lightweight::a5

#include <pto/costmodel/a5/metadata_costmodel.hpp>

#endif
