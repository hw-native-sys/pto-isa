/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/
#ifndef PTO_MOCKER_A5_SEQUENCE_COSTMODEL_HPP
#define PTO_MOCKER_A5_SEQUENCE_COSTMODEL_HPP

#include <pto/costmodel/a5/formula_costmodel/formula_backend_vf.hpp>

namespace pto::mocker::lightweight::a5 {

// Bounds describe the calibration domain, not hardware limits.
inline constexpr int64_t kSequenceMaxRows = 16;
inline constexpr int64_t kShiftMinCols = 32;
inline constexpr int64_t kShiftMaxCols = 512;
inline constexpr int64_t kShiftMultiRowMaxCols = 384;
inline constexpr int64_t kPartialArgMultiRowMaxCols = 256;
inline constexpr int64_t kCiMaxCols = 1024;
inline constexpr int64_t kInterleaveHalfMaxCols = 768;
inline constexpr int64_t kInterleaveFloatMaxCols = 280;
inline constexpr int64_t kSequenceBlockBytes = 32;
inline constexpr int64_t kSequenceVectorBytes = 256;
inline constexpr int64_t kHalfRowDivisor = 2;

inline bool LookupSequenceFit(const CostModelInput& input, std::string_view branch, const fit::VfFormulaParam*& param)
{
    const char* dtype = input.dtype == DType::Half  ? "fp16" :
                        input.dtype == DType::Float ? "fp32" :
                        input.dtype == DType::Int16 ? "int16" :
                                                      "int32";
    const fit::VfCurveKey key{PtoOpcodeToString(input.op), dtype, "none", fit::ShapePath::Path1D, "NO_POST_UPDATE",
                              fit::TailKind::Full,         branch};
    return fit::TryLookupVfFormulaParam(key, param);
}

inline bool TryEstimateA5SequenceCycles(const CostModelInput& input, uint64_t& cycles)
{
    const bool shift = input.op == PtoOpcode::TSHL || input.op == PtoOpcode::TSHR || input.op == PtoOpcode::TSHLS ||
                       input.op == PtoOpcode::TSHRS;
    const bool part = input.op == PtoOpcode::TPARTARGMAX || input.op == PtoOpcode::TPARTARGMIN;
    const bool interleave = input.op == PtoOpcode::TINTERLEAVE || input.op == PtoOpcode::TDEINTERLEAVE;
    const bool ci = input.op == PtoOpcode::TCI;
    if (!shift && !part && !interleave && !ci)
        return false;
    const int64_t rows = input.valid_rows > 0 ? input.valid_rows : input.rows;
    const int64_t cols = input.valid_cols > 0 ? input.valid_cols : input.cols;
    if (rows <= 0 || cols <= 0 || rows != input.rows || cols != input.cols || rows > kSequenceMaxRows)
        return false;
    if ((shift || ci) && input.dtype != DType::Int16 && input.dtype != DType::Int32)
        return false;
    if ((part || interleave) && input.dtype != DType::Half && input.dtype != DType::Float)
        return false;
    const int64_t bytes =
        (input.dtype == DType::Half || input.dtype == DType::Int16) ? sizeof(int16_t) : sizeof(int32_t);
    if (cols % (kSequenceBlockBytes / bytes) != 0)
        return false;
    const uint64_t repeat =
        static_cast<uint64_t>((cols + kSequenceVectorBytes / bytes - 1) / (kSequenceVectorBytes / bytes));
    const fit::VfFormulaParam* param = nullptr;
    if (ci) {
        if (rows != 1 || cols < kShiftMinCols || cols > kCiMaxCols ||
            (input.a5_op_params != "ci_scalar" && input.a5_op_params != "ci_vector"))
            return false;
        if (!LookupSequenceFit(input, input.a5_op_params, param))
            return false;
        cycles = fit::EvalVfFormula(*param, input.a5_op_params == "ci_scalar" ? cols : repeat, 1, 1, 0, 1);
        return true;
    }
    if (input.a5_op_params != (part ? "dense_index32" : "dense"))
        return false;
    if (shift || part) {
        if (cols < kShiftMinCols || cols > kShiftMaxCols)
            return false;
        if (rows > 1 && cols > (part ? kPartialArgMultiRowMaxCols : kShiftMultiRowMaxCols))
            return false;
        // Only physical single rows or the measured dynamic-valid 2D path.
        if (rows > 1 && input.a5_shape_path != A5ShapePathHint::Path2D)
            return false;
        // The fp32 partial-arg two-row startup has not met the accuracy target.
        if (part && input.dtype == DType::Float && rows == 2)
            return false;
        if (!LookupSequenceFit(input, rows == 1 ? "sequence_row1" : "sequence_rows", param))
            return false;
        cycles = fit::EvalVfFormula(*param, repeat, rows, repeat, 0, repeat);
        if (shift && rows > 1) {
            const fit::VfFormulaParam* single = nullptr;
            if (!LookupSequenceFit(input, "sequence_row1", single))
                return false;
            // Short row loops retain the single-row latency floor. Coefficients
            // are shared with the independently measured single-row curve.
            const double floor = single->param0 * static_cast<double>(rows * repeat) + single->param1;
            cycles = std::max(cycles, fit::RoundToCycles(floor));
        }
        return true;
    }
    const int64_t maxCols = bytes == sizeof(int16_t) ? kInterleaveHalfMaxCols : kInterleaveFloatMaxCols;
    if (cols > maxCols || cols % kHalfRowDivisor != 0)
        return false;
    const bool aligned = (cols * bytes / kHalfRowDivisor) % kSequenceBlockBytes == 0;
    if (!aligned && input.op == PtoOpcode::TDEINTERLEAVE)
        return false;
    if (!LookupSequenceFit(input, aligned ? "interleave_aligned" : "interleave_unaligned", param))
        return false;
    const uint64_t halfRepeat =
        (cols / kHalfRowDivisor + kSequenceVectorBytes / bytes - 1) / (kSequenceVectorBytes / bytes);
    cycles = fit::EvalVfFormula(*param, halfRepeat, rows, halfRepeat, 0, halfRepeat);
    return true;
}
} // namespace pto::mocker::lightweight::a5
#endif
