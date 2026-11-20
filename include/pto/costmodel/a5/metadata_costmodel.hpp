/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/
#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <string>

namespace pto::mocker::lightweight::a5 {

inline constexpr uint64_t kMetadataUbCapacityBytes = 262144;
inline constexpr int64_t kMetadataVectorBytes = 256;
inline constexpr int64_t kMetadataBlockBytes = 32;
inline constexpr size_t kMetadataFeatureCount = 6;

inline bool ValidObservableOperand(const OperandGeometry& g)
{
    return g.rows > 0 && g.cols > 0 && g.valid_rows >= 0 && g.valid_cols >= 0 && g.valid_rows <= g.rows &&
           g.valid_cols <= g.cols && g.row_stride == g.cols;
}

inline bool ValidMetadataStorage(const OperandGeometry& g, uint64_t bytes)
{
    if (!ValidObservableOperand(g) || !g.has_address || g.address % kMetadataVectorBytes != 0 ||
        g.address >= kMetadataUbCapacityBytes || uint64_t(g.rows) > kMetadataUbCapacityBytes / bytes ||
        uint64_t(g.cols) > kMetadataUbCapacityBytes / bytes / uint64_t(g.rows)) {
        return false;
    }
    return uint64_t(g.rows) * uint64_t(g.cols) * bytes <= kMetadataUbCapacityBytes - g.address;
}

inline bool MetadataStorageDisjoint(const OperandGeometry& a, const OperandGeometry& b, uint64_t bytes)
{
    return a.address + uint64_t(a.rows) * uint64_t(a.cols) * bytes <= b.address ||
           b.address + uint64_t(b.rows) * uint64_t(b.cols) * bytes <= a.address;
}

inline bool BuildMetadataPaddingFeatures(
    const CostModelInput& input, int64_t bytes, bool padding, std::array<double, kMetadataFeatureCount>& x,
    std::string_view& family)
{
    constexpr int64_t kMaxPaddingRows = 256;
    constexpr int64_t kMinPaddingRowBytes = 64;
    constexpr int64_t kMaxPaddingRowBytes = 2048;
    const auto& d = input.metadata.dst;
    const auto& a = input.metadata.src0;
    const int64_t r = d.rows, c = d.cols, width = kMetadataVectorBytes / bytes;
    const auto repeats = [width](int64_t n) { return (n + width - 1) / width; };
    if (r > kMaxPaddingRows || c * bytes < kMinPaddingRowBytes || c * bytes > kMaxPaddingRowBytes ||
        a.valid_rows <= 0 || a.valid_cols <= 0 || (padding && (a.rows != r || a.cols != c)) ||
        (!padding && (a.rows > r || a.cols > c))) {
        return false;
    }
    x = {
        double(a.valid_rows * repeats(a.valid_cols)),
        double(a.valid_rows * repeats(c - a.valid_cols)),
        double(repeats((r - a.valid_rows) * c)),
        double(a.valid_rows),
        1.0,
        0.0};
    family = "metadata_padding_v2";
    return true;
}

inline bool ValidMetadataPartialOperands(const CostModelInput& input, uint64_t bytes)
{
    const auto& g = input.metadata;
    const auto& d = g.dst;
    const auto& a = g.src0;
    const auto& b = g.src1;
    const bool part = input.op == PtoOpcode::TPARTADD || input.op == PtoOpcode::TPARTMUL ||
                      input.op == PtoOpcode::TPARTMAX || input.op == PtoOpcode::TPARTMIN;
    return part && g.kind == MetadataGeometry::Kind::Partial && ValidMetadataStorage(b, bytes) &&
           MetadataStorageDisjoint(d, b, bytes) && MetadataStorageDisjoint(a, b, bytes) && a.rows == d.rows &&
           a.cols == d.cols && b.rows == d.rows && b.cols == d.cols && b.valid_rows <= d.rows && b.valid_cols <= d.cols;
}

inline bool ValidMetadataPartialCopyDomain(const OperandGeometry& d, int64_t bytes)
{
    constexpr int64_t kMaxCopyRows = 64;
    constexpr int64_t kMinCopyRowBytes = 64;
    constexpr int64_t kMaxCopyRowBytes = 1024;
    return d.rows <= kMaxCopyRows && d.cols * bytes >= kMinCopyRowBytes && d.cols * bytes <= kMaxCopyRowBytes;
}

inline bool BuildMetadataPartialFeatures(
    const CostModelInput& input, int64_t bytes, std::array<double, kMetadataFeatureCount>& x, std::string_view& family)
{
    constexpr int64_t kMaxPartialRows = 2728;
    constexpr int64_t kMaxPartialTileBytes = 87296;
    const auto& d = input.metadata.dst;
    const auto& a = input.metadata.src0;
    const auto& b = input.metadata.src1;
    const int64_t r = d.rows, c = d.cols, width = kMetadataVectorBytes / bytes;
    const auto repeats = [width](int64_t n) { return (n + width - 1) / width; };
    const int64_t q = repeats(c);
    if (!ValidMetadataPartialOperands(input, bytes))
        return false;
    const bool fullA = a.valid_rows == r && a.valid_cols == c;
    const bool fullB = b.valid_rows == r && b.valid_cols == c;
    if (r > kMaxPartialRows || r * c * bytes > kMaxPartialTileBytes || (!fullA && !fullB))
        return false;
    if (fullA && fullB) {
        if (!ValidMetadataPartialCopyDomain(d, bytes))
            return false;
        x = {double(r * q), double(r), 1.0, 0.0, 0.0, 0.0};
        family = "metadata_partial_equal_v1";
        return true;
    }
    if (a.valid_rows == 0 || a.valid_cols == 0 || b.valid_rows == 0 || b.valid_cols == 0) {
        // Only the row-empty path was measured, not arbitrary empty geometry.
        const auto& empty = fullA ? b : a;
        if (empty.valid_rows != 0 || empty.valid_cols != c || !ValidMetadataPartialCopyDomain(d, bytes))
            return false;
        x = {double(r * q), double(r), 1.0, 0.0, 0.0, 0.0};
        family = "metadata_partial_empty_v1";
        return true;
    }
    const int64_t overlapRows = std::min(a.valid_rows, b.valid_rows);
    const int64_t overlapRepeats = overlapRows * repeats(std::min(a.valid_cols, b.valid_cols));
    x = {
        double(overlapRepeats), double(r * q - overlapRepeats), double(overlapRows), double(r - overlapRows), 1.0, 0.0};
    family = "metadata_partial_v2";
    return true;
}

inline bool BuildMetadataFeatures(const CostModelInput& input, std::array<double, 6>& x, std::string_view& family)
{
    const auto& g = input.metadata;
    const auto& d = g.dst;
    const auto& a = g.src0;
    if (!g.supported_layout || !g.runtime_valid_shape || !ValidObservableOperand(d) || d.valid_rows != d.rows ||
        d.valid_cols != d.cols || input.vf_impl_kind != VFImplKind::VFIMPL_DEFAULT) {
        return false;
    }
    uint64_t dtypeBytes = 0;
    if (!TryGetDTypeSizeBytes(input.dtype, dtypeBytes))
        return false;
    const int64_t bytes = static_cast<int64_t>(dtypeBytes);
    if (!ValidMetadataStorage(d, dtypeBytes) || d.cols * bytes % kMetadataBlockBytes != 0)
        return false;
    x = {};
    if (!ValidMetadataStorage(a, dtypeBytes) || !MetadataStorageDisjoint(d, a, dtypeBytes) ||
        a.cols * bytes % kMetadataBlockBytes != 0 || a.valid_rows > d.rows || a.valid_cols > d.cols) {
        return false;
    }
    const bool padding = input.op == PtoOpcode::TFILLPAD && g.kind == MetadataGeometry::Kind::Padding;
    const bool expand = input.op == PtoOpcode::TFILLPAD_EXPAND && g.kind == MetadataGeometry::Kind::PaddingExpand;
    if (padding || expand)
        return BuildMetadataPaddingFeatures(input, bytes, padding, x, family);
    return BuildMetadataPartialFeatures(input, bytes, x, family);
}

inline bool EvaluateMetadataCurve(
    const CostModelInput& input, const std::array<double, 6>& x, std::string_view family, uint64_t& cycles)
{
    const auto op = PtoOpcodeToKey(input.op);
    if (op.empty())
        return false;
    const fit::VfFormulaParam *first = nullptr, *second = nullptr;
    const std::string key0 = std::string(family) + "_0";
    const std::string key1 = std::string(family) + "_1";
    const fit::VfCurveKey k0{
        op, DTypeToKey(input.dtype), "none", fit::ShapePath::Path2D, "POST_UPDATE", fit::TailKind::Full, key0};
    auto k1 = k0;
    k1.op_params = key1;
    if (!fit::TryLookupVfFormulaParam(k0, first) || !fit::TryLookupVfFormulaParam(k1, second))
        return false;
    const double value = first->param0 * x[0] + first->param1 * x[1] + first->param2 * x[2] + first->param3 * x[3] +
                         second->param0 * x[4] + second->param1 * x[5];
    cycles = fit::RoundToCycles(value);
    return cycles > 0;
}

inline bool ValidMetadataComparisonOperands(const MetadataGeometry& g)
{
    constexpr int64_t kMaxComparisonRows = 64;
    constexpr int64_t kMinComparisonCols = 32;
    constexpr int64_t kMaxComparisonCols = 256;
    constexpr int64_t kComparisonColAlignment = 32;
    const auto& s = g.src0;
    const auto& d = g.dst;
    return (g.data_dtype == DType::Float || g.data_dtype == DType::Int32) &&
           ValidMetadataStorage(s, sizeof(uint32_t)) && ValidMetadataStorage(d, sizeof(uint8_t)) &&
           s.valid_rows == s.rows && s.valid_cols == s.cols && s.rows <= kMaxComparisonRows &&
           s.cols >= kMinComparisonCols && s.cols <= kMaxComparisonCols && s.cols % kComparisonColAlignment == 0 &&
           d.rows == s.rows && d.valid_rows == s.rows && d.cols == s.cols && d.valid_cols == s.cols;
}

inline bool ValidMetadataComparisonReadSpan(const OperandGeometry& operand, const OperandGeometry& dst)
{
    constexpr uint64_t kPackedComparisonElements = 128;
    constexpr uint64_t kPackedComparisonReadBytes = 512;
    // B32 loads two full vectors before packing the predicate bits.
    const uint64_t span = uint64_t(operand.rows - 1) * uint64_t(operand.cols) * sizeof(uint32_t) +
                          (uint64_t(operand.cols) + kPackedComparisonElements - 1) / kPackedComparisonElements *
                              kPackedComparisonReadBytes;
    return span <= kMetadataUbCapacityBytes - operand.address &&
           (dst.address + uint64_t(dst.rows) * uint64_t(dst.cols) <= operand.address ||
            operand.address + span <= dst.address);
}

inline bool ValidMetadataComparisonRhs(const CostModelInput& input)
{
    const auto& g = input.metadata;
    if (input.op == PtoOpcode::TCMPS) {
        // Only the immediate-2 fixture is reused. Tile-rhs is a different
        // broadcast path and must not borrow this scalar curve.
        constexpr double kCalibratedComparisonScalar = 2.0;
        return g.scalar_rhs && g.scalar_value == kCalibratedComparisonScalar;
    }
    const auto& s = g.src0;
    const auto& b = g.src1;
    return !g.scalar_rhs && ValidMetadataStorage(b, sizeof(uint32_t)) && b.rows == s.rows && b.cols == s.cols &&
           b.valid_rows == s.rows && b.valid_cols == s.cols && ValidMetadataComparisonReadSpan(b, g.dst) &&
           MetadataStorageDisjoint(s, b, sizeof(uint32_t));
}

inline bool TryEstimateA5MetadataComparisonCycles(const CostModelInput& input, uint64_t& cycles)
{
    constexpr int kEqualComparisonMode = 0;
    cycles = 0;
    const auto& g = input.metadata;
    const auto& s = g.src0;
    // The pre-hoist EQ calibration uses constant-mode calls. RuntimeDispatch
    // needs its own independently validated upstream-native curves. In
    // particular, never substitute the withdrawn post-hoist coefficients.
    if (g.comparison_codegen != ComparisonCodegen::ConstantFolded || g.comparison_mode != kEqualComparisonMode ||
        (input.op != PtoOpcode::TCMP && input.op != PtoOpcode::TCMPS) || !g.supported_layout ||
        !g.runtime_valid_shape || input.vf_impl_kind != VFImplKind::VFIMPL_DEFAULT ||
        !ValidMetadataComparisonOperands(g) || !ValidMetadataComparisonReadSpan(s, g.dst) ||
        !ValidMetadataComparisonRhs(input))
        return false;
    // The old fixture allocates a full-width uint8 mask, even though only
    // packed predicate bits are written. Do not extrapolate to a different
    // destination stride. Reuse its pre-hoist constant-EQ calibration. Clearing the
    // tag avoids recursion; no mutable context is read during estimation.
    CostModelInput normalized{};
    normalized.op = input.op;
    normalized.arch = CostModelArch::A5;
    normalized.dtype = g.data_dtype;
    normalized.rows = normalized.valid_rows = s.rows;
    normalized.cols = normalized.valid_cols = s.cols;
    normalized.a5_shape_path = s.rows == 1 ? A5ShapePathHint::Path1D : A5ShapePathHint::Path2D;
    return TryEstimateA5VfCycles(normalized, cycles);
}

inline bool TryEstimateA5MetadataTriangleCycles(const CostModelInput& input, uint64_t& cycles)
{
    const auto& g = input.metadata;
    const auto& d = g.dst;
    uint64_t bytes = 0;
    // Wider/taller extents were checked against the frozen model without
    // refitting. Half retains its narrower, independently measured domain.
    constexpr int64_t kHalfMaxRows = 512;
    constexpr int64_t kOtherMaxRows = 4095;
    constexpr int64_t kHalfMaxRowBytes = 16384;
    constexpr int64_t kOtherMaxRowBytes = 261888;
    const int64_t maxRows = input.dtype == DType::Half ? kHalfMaxRows : kOtherMaxRows;
    const int64_t maxRowBytes = input.dtype == DType::Half ? kHalfMaxRowBytes : kOtherMaxRowBytes;
    if (input.op != PtoOpcode::TTRI || !g.supported_layout || !g.runtime_valid_shape ||
        input.vf_impl_kind != VFImplKind::VFIMPL_DEFAULT || !TryGetDTypeSizeBytes(input.dtype, bytes) ||
        !ValidMetadataStorage(d, bytes) || d.valid_rows != d.rows || d.valid_cols != d.cols || d.rows > maxRows ||
        d.cols * bytes % kMetadataBlockBytes != 0 || d.cols * bytes > maxRowBytes)
        return false;
    const fit::VfFormulaParam *first = nullptr, *second = nullptr;
    const fit::VfCurveKey k0{
        "TTRI",
        DTypeToKey(input.dtype),
        "none",
        fit::ShapePath::Path2D,
        "POST_UPDATE",
        fit::TailKind::Full,
        "metadata_triangle_pipeline_v4_0"};
    auto k1 = k0;
    k1.op_params = "metadata_triangle_pipeline_v4_1";
    if (!fit::TryLookupVfFormulaParam(k0, first) || !fit::TryLookupVfFormulaParam(k1, second))
        return false;
    const int64_t diagonal = std::clamp(g.diagonal, -d.rows, d.cols);
    const int64_t start = g.upper ? (diagonal > 0 ? 0 : 1 - diagonal) : std::max(int64_t(0), -diagonal);
    const int64_t active = std::max(int64_t(0), d.rows - start);
    const double q = double((d.cols + (kMetadataVectorBytes / bytes) - 1) / (kMetadataVectorBytes / bytes));
    // Both source VF passes traverse every column repeat of their rows,
    // including repeats whose store predicate is empty. The continuous
    // first-pass knee is empirical, not a claimed hardware constant.
    const double value = d.rows * (first->param0 * q + first->param1 * std::min(q, second->param3) + first->param2) +
                         active * (first->param3 * q + second->param0) + second->param1 +
                         second->param2 * (active == 0);
    cycles = fit::RoundToCycles(value);
    return cycles > 0;
}

inline bool SameFullMetadataStorage(const OperandGeometry& operand, const OperandGeometry& dst, uint64_t bytes)
{
    return ValidMetadataStorage(operand, bytes) && operand.rows == dst.rows && operand.cols == dst.cols &&
           operand.valid_rows == operand.rows && operand.valid_cols == operand.cols;
}

inline bool ValidMetadataSelectOperands(const MetadataGeometry& g, uint64_t bytes)
{
    constexpr int64_t kPackedMaskRowBytes = 32;
    return SameFullMetadataStorage(g.dst, g.dst, bytes) && SameFullMetadataStorage(g.src0, g.dst, bytes) &&
           SameFullMetadataStorage(g.src1, g.dst, bytes) && ValidMetadataStorage(g.mask, sizeof(uint8_t)) &&
           g.mask.rows == g.dst.rows && g.mask.valid_rows == g.mask.rows && g.mask.cols == kPackedMaskRowBytes &&
           g.mask.valid_cols == g.mask.cols && MetadataStorageDisjoint(g.dst, g.src0, bytes) &&
           MetadataStorageDisjoint(g.dst, g.src1, bytes) && MetadataStorageDisjoint(g.src0, g.src1, bytes);
}

inline bool TryEstimateA5MetadataSelectCycles(const CostModelInput& input, uint64_t& cycles)
{
    const auto& g = input.metadata;
    const uint64_t bytes = input.dtype == DType::Uint16 ? sizeof(uint16_t) :
                           input.dtype == DType::Uint32 ? sizeof(uint32_t) :
                                                          0;
    constexpr int64_t kMaxSelectRows = 12;
    constexpr int64_t kMinSelectCols = 32;
    constexpr int64_t kMaxSelectCols = 96;
    constexpr int64_t kSelectColAlignment = 32;
    if (input.op != PtoOpcode::TSEL || !bytes || !g.supported_layout || !g.runtime_valid_shape ||
        input.vf_impl_kind != VFImplKind::VFIMPL_DEFAULT || g.dst.rows < 1 || g.dst.rows > kMaxSelectRows ||
        g.dst.cols < kMinSelectCols || g.dst.cols > kMaxSelectCols || g.dst.cols % kSelectColAlignment != 0)
        return false;
    if (!ValidMetadataSelectOperands(g, bytes))
        return false;
    for (const auto* operand : {&g.dst, &g.src0, &g.src1}) {
        const auto end = operand->address + uint64_t(operand->rows) * uint64_t(operand->cols) * bytes;
        const auto maskEnd = g.mask.address + uint64_t(g.mask.rows) * uint64_t(g.mask.cols);
        if (!(end <= g.mask.address || maskEnd <= operand->address))
            return false;
    }
    // Unchanged vector coefficients, independently checked against 12
    // fully specified packed-mask inputs (max error 1.77%). This predicts
    // vector work, not input-content-dependent or end-to-end latency.
    auto legacy = input;
    legacy.metadata = {};
    return TryEstimateA5VfCycles(legacy, cycles);
}

inline bool TryEstimateA5MetadataCycles(const CostModelInput& input, uint64_t& cycles)
{
    if (input.metadata.kind == MetadataGeometry::Kind::Comparison)
        return TryEstimateA5MetadataComparisonCycles(input, cycles);
    if (input.metadata.kind == MetadataGeometry::Kind::Triangle)
        return TryEstimateA5MetadataTriangleCycles(input, cycles);
    if (input.metadata.kind == MetadataGeometry::Kind::Select)
        return TryEstimateA5MetadataSelectCycles(input, cycles);
    // Preserve the established full-equal fp16 contract. Its historical curve
    // is not evidence for partially overlapping or empty-source execution.
    if (input.dtype == DType::Half && input.metadata.kind == MetadataGeometry::Kind::Partial) {
        const auto& g = input.metadata;
        constexpr uint64_t kHalfBytes = sizeof(uint16_t);
        if (!g.supported_layout || !SameFullMetadataStorage(g.dst, g.dst, kHalfBytes) ||
            !SameFullMetadataStorage(g.src0, g.dst, kHalfBytes) ||
            !SameFullMetadataStorage(g.src1, g.dst, kHalfBytes) ||
            !MetadataStorageDisjoint(g.dst, g.src0, kHalfBytes) ||
            !MetadataStorageDisjoint(g.dst, g.src1, kHalfBytes) || !MetadataStorageDisjoint(g.src0, g.src1, kHalfBytes))
            return false;
        auto legacy = input;
        legacy.metadata = {};
        return TryEstimateA5VfCycles(legacy, cycles);
    }
    std::array<double, kMetadataFeatureCount> x{};
    std::string_view family;
    if (!BuildMetadataFeatures(input, x, family))
        return false;
    // Two ordinary CSV entries store one identifiable six-feature model. The
    // suffix distinguishes coefficient blocks, not individual input cases.
    return EvaluateMetadataCurve(input, x, family, cycles);
}

} // namespace pto::mocker::lightweight::a5
