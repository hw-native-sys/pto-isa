/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/
#ifndef PTO_MOCKER_A5_TRANSPOSE_COSTMODEL_HPP
#define PTO_MOCKER_A5_TRANSPOSE_COSTMODEL_HPP

#include <string>
#include <pto/costmodel/a5/formula_costmodel/formula_backend_vf.hpp>

namespace pto::mocker::lightweight::a5 {

struct TransposeFeatures {
    bool scatter = false;
    int64_t outer_iterations = 0;
    int64_t repeats_per_iteration = 0;
    int64_t full_chunks_per_iteration = 0;
    int64_t tail_elements = 0;
    int64_t indexed_stride_bytes = 0;
};

inline bool BuildTransposeFeatures(const CostModelInput& input, TransposeFeatures& features)
{
    constexpr int64_t kVectorBytes = 256;
    constexpr int64_t kBlockBytes = 32;
    const auto& g = input.transpose;
    const int64_t bytes = input.dtype == DType::Half ? sizeof(int16_t) : sizeof(int32_t);
    if (input.op != PtoOpcode::TTRANS || (input.dtype != DType::Half && input.dtype != DType::Float) ||
        !g.ordinary_row_major || g.src_rows <= 0 || g.src_cols <= 0 || g.src_valid_rows <= 0 || g.src_valid_cols <= 0 ||
        g.src_valid_rows > g.src_rows || g.src_valid_cols > g.src_cols || g.dst_rows <= 0 || g.dst_cols <= 0 ||
        g.dst_valid_rows > g.dst_rows || g.dst_valid_cols > g.dst_cols || g.dst_valid_rows != g.src_valid_cols ||
        g.dst_valid_cols != g.src_valid_rows || g.src_row_stride < g.src_cols || g.dst_row_stride < g.dst_cols ||
        g.src_cols % (kBlockBytes / bytes) != 0 || g.dst_cols % (kBlockBytes / bytes) != 0) {
        return false;
    }
    features.scatter = g.src_rows < g.src_cols;
    const int64_t inner = features.scatter ? g.src_valid_cols : g.src_valid_rows;
    const int64_t width = kVectorBytes / bytes;
    features.outer_iterations = features.scatter ? g.src_valid_rows : g.src_valid_cols;
    features.repeats_per_iteration = (inner - 1) / width + 1;
    // The hardware implementation always emits the last chunk through its
    // tail block, including when that block has a full mask.
    features.full_chunks_per_iteration = features.repeats_per_iteration - 1;
    features.tail_elements = (inner - 1) % width + 1;
    features.indexed_stride_bytes = (features.scatter ? g.dst_row_stride : g.src_row_stride) * bytes;
    return true;
}

inline bool TryEstimateA5TransposeCycles(const CostModelInput& input, uint64_t& cycles)
{
    TransposeFeatures features;
    if (!BuildTransposeFeatures(input, features) || !features.scatter)
        return false;
    const auto& g = input.transpose;
    // Existing measurements identify column curves at fixed destination
    // strides, not a general stride-to-bank-conflict relationship.
    constexpr int64_t kSmallCalibratedRows = 16;
    constexpr int64_t kLargeCalibratedRows = 32;
    constexpr int64_t kSmallMinCols = 32;
    constexpr int64_t kSmallMaxCols = 128;
    constexpr int64_t kLargeMinCols = 64;
    constexpr int64_t kLargeMaxCols = 192;
    if (g.src_rows != g.src_valid_rows || g.src_cols != g.src_valid_cols || g.dst_rows != g.dst_valid_rows ||
        g.dst_cols != g.dst_valid_cols || g.src_row_stride != g.src_cols || g.dst_row_stride != g.dst_cols)
        return false;
    const bool small = g.src_rows == kSmallCalibratedRows;
    if (!small && g.src_rows != kLargeCalibratedRows)
        return false;
    if (g.src_cols < (small ? kSmallMinCols : kLargeMinCols) || g.src_cols > (small ? kSmallMaxCols : kLargeMaxCols))
        return false;
    const std::string branch = "scatter_stride_bytes_" + std::to_string(features.indexed_stride_bytes);
    const fit::VfCurveKey key{
        PtoOpcodeToString(input.op),
        input.dtype == DType::Half ? "fp16" : "fp32",
        "none",
        fit::ShapePath::Path1D,
        "NO_POST_UPDATE",
        fit::TailKind::Full,
        branch};
    const fit::VfFormulaParam* param = nullptr;
    if (!fit::TryLookupVfFormulaParam(key, param))
        return false;
    cycles = fit::EvalVfFormula(*param, g.src_valid_cols, 1, 1, 0, 1);
    return true;
}
} // namespace pto::mocker::lightweight::a5
#endif
