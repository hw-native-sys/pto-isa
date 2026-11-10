/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_MOCKER_A5_REDUCTION_COSTMODEL_HPP
#define PTO_MOCKER_A5_REDUCTION_COSTMODEL_HPP

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace pto::mocker::lightweight::a5 {

inline constexpr uint64_t kRowSumElementsPerRepeatFp16 = 64;
inline constexpr double kRowSumSingleRepeatFloorCycles = 63.0;
inline constexpr double kRowSumSingleRepeatBaseCycles = 42.5;
inline constexpr double kRowSumSingleRepeatCyclesPerRow = 6.0;
inline constexpr double kRowSumMultiRepeatBaseCycles = 58.0;
inline constexpr double kRowSumMultiRepeatSetupCycles = 60.0;
inline constexpr double kRowSumMultiRepeatCyclesPerRepeat = 3.0;
inline constexpr double kRowSumMultiRepeatRowBias = 1.0;

struct DualReductionFitParams {
    double flatRepeatSlope;
    double flatBase;
    double rowBaseSlope;
    double rowRepeatSlope;
    double rowOffset;
};

inline constexpr uint64_t kRowReductionElementsPerRepeatFp16 = 128;
inline constexpr DualReductionFitParams kRowProdFit = {9.9344, 124.14, 17.277, 6.5745, 83.493};
inline constexpr double kRowExtremaStartupCycles = 55.0;
inline constexpr double kRowExtremaSingleRepeatCyclesPerRow = 35.0;
inline constexpr double kRowExtremaDualAccumCyclesPerRow = 45.0;
inline constexpr double kRowExtremaOddRepeatCyclesPerRow = 5.0;
inline constexpr uint64_t kRowExtremaPairPipelineDepth = 3;
inline constexpr double kRowExtremaCyclesPerExtraPair = 2.0;
inline constexpr double kRowExtremaTailCyclesPerRow = 1.0;

struct ColReductionFitParams {
    double flatRepeatSlope;
    double flatBase;
    double rowBaseSlope;
    double rowRepeatSlope;
    double rowRepeatOffset;
    double rowOffset;
};

inline constexpr uint64_t kColArithmeticElementsPerRepeatFp16 = 64;
inline constexpr ColReductionFitParams kColSumFit = {23.25, 44.5, 0.0, 2.0, -38.5, 79.5};
inline constexpr ColReductionFitParams kColProdFit = {23.5, 76.0, 0.0, 4.0, -105.0, 146.0};
inline constexpr ColReductionFitParams kColSumHighRepeatSmallRowFit = {4.5919, 43.277, 1.0, 1.2233, 2.2869, 35.158};
inline constexpr ColReductionFitParams kColProdHighRepeatSmallRowFit = {13.18966081, 33.17883043,  1.74494416,
                                                                        2.5504652,   -31.06228086, 65.26532138};
inline constexpr int64_t kColSmallRowLimit = 16;
inline constexpr uint64_t kColHighRepeatStart = 5;
inline constexpr double kColProdSingleRowRepeatSlope = 4.2774;
inline constexpr double kColProdSingleRowBaseCycles = 43.521;
inline constexpr double kColExtremaElementsPerRepeatFp16 = 128.0;
inline constexpr double kColExtremaSingleRowBaseCycles = 33.0;
inline constexpr double kColExtremaMultiRowBaseCycles = 37.0;

inline uint64_t RoundA5SpecialCycles(double cycles)
{
    return cycles <= 0.0 ? 0 : static_cast<uint64_t>(std::llround(cycles));
}

inline bool TryEstimateA5RowSumCycles(const CostModelInput& input, uint64_t& cycles)
{
    if (input.op != PtoOpcode::TROWSUM || input.dtype != DType::Half) {
        return false;
    }
    const int64_t rows_i = input.valid_rows > 0 ? input.valid_rows : input.rows;
    const int64_t cols_i = input.valid_cols > 0 ? input.valid_cols : input.cols;
    if (rows_i <= 0 || cols_i <= 0) {
        return false;
    }

    const double rows = static_cast<double>(rows_i);
    const uint64_t repeats = CeilDivU64(static_cast<uint64_t>(cols_i), kRowSumElementsPerRepeatFp16);
    double value = 0.0;
    if (repeats == 1) {
        value = std::max(
            kRowSumSingleRepeatFloorCycles, kRowSumSingleRepeatBaseCycles + kRowSumSingleRepeatCyclesPerRow * rows);
    } else {
        const double repeat_count = static_cast<double>(repeats);
        const double setup = kRowSumMultiRepeatSetupCycles + kRowSumMultiRepeatCyclesPerRepeat * repeat_count;
        const double row_path = kRowSumMultiRepeatBaseCycles +
                                rows * (kRowSumMultiRepeatCyclesPerRepeat * repeat_count + kRowSumMultiRepeatRowBias);
        value = std::max(setup, row_path);
    }
    cycles = RoundA5SpecialCycles(value);
    return true;
}

inline bool TryResolveA5ReductionShape(const CostModelInput& input, int64_t& rows, int64_t& cols)
{
    rows = input.valid_rows > 0 ? input.valid_rows : input.rows;
    cols = input.valid_cols > 0 ? input.valid_cols : input.cols;
    return input.dtype == DType::Half && rows > 0 && cols > 0;
}

inline double EvalA5DualReduction(const DualReductionFitParams& params, double rows, double repeats)
{
    const double flatPath = params.flatRepeatSlope * rows * repeats + params.flatBase;
    const double rowPath = (params.rowBaseSlope + params.rowRepeatSlope * repeats) * rows + params.rowOffset;
    return std::max(flatPath, rowPath);
}

inline double EvalA5RowExtrema(double rows, uint64_t repeats, bool hasTail)
{
    if (repeats == 1) {
        return kRowExtremaStartupCycles +
               rows * (kRowExtremaSingleRepeatCyclesPerRow + (hasTail ? kRowExtremaTailCyclesPerRow : 0.0));
    }
    const uint64_t repeatPairs = repeats / 2;
    const uint64_t extraPairs =
        repeatPairs > kRowExtremaPairPipelineDepth ? repeatPairs - kRowExtremaPairPipelineDepth : 0;
    double cyclesPerRow = kRowExtremaDualAccumCyclesPerRow;
    if (repeats % 2 != 0) {
        cyclesPerRow += kRowExtremaOddRepeatCyclesPerRow;
    }
    cyclesPerRow += static_cast<double>(extraPairs) * kRowExtremaCyclesPerExtraPair;
    if (hasTail) {
        cyclesPerRow += kRowExtremaTailCyclesPerRow;
    }
    return kRowExtremaStartupCycles + rows * cyclesPerRow;
}

inline bool TryEstimateA5RowReductionCycles(const CostModelInput& input, uint64_t& cycles)
{
    const bool isRowExtrema = input.op == PtoOpcode::TROWMAX || input.op == PtoOpcode::TROWMIN;
    if (!isRowExtrema && input.op != PtoOpcode::TROWPROD) {
        return false;
    }
    int64_t rows_i = 0;
    int64_t cols_i = 0;
    if (!TryResolveA5ReductionShape(input, rows_i, cols_i)) {
        return false;
    }
    const double rows = static_cast<double>(rows_i);
    const uint64_t repeats = CeilDivU64(static_cast<uint64_t>(cols_i), kRowReductionElementsPerRepeatFp16);
    const bool hasTail = static_cast<uint64_t>(cols_i) % kRowReductionElementsPerRepeatFp16 != 0;
    const double value = isRowExtrema ? EvalA5RowExtrema(rows, repeats, hasTail) :
                                        EvalA5DualReduction(kRowProdFit, rows, static_cast<double>(repeats));
    cycles = RoundA5SpecialCycles(value);
    return true;
}

inline double EvalA5ColArithmeticReduction(const ColReductionFitParams& params, double rows, double repeats)
{
    const double flatPath = params.flatRepeatSlope * repeats + params.flatBase;
    const double rowPath = (params.rowBaseSlope + params.rowRepeatSlope * repeats) * rows +
                           params.rowRepeatOffset * repeats + params.rowOffset;
    return std::max(flatPath, rowPath);
}

inline bool TryEstimateA5ColReductionCycles(const CostModelInput& input, uint64_t& cycles)
{
    int64_t rows_i = 0;
    int64_t cols_i = 0;
    if (!TryResolveA5ReductionShape(input, rows_i, cols_i)) {
        return false;
    }
    const double rows = static_cast<double>(rows_i);
    if (input.op == PtoOpcode::TCOLMAX || input.op == PtoOpcode::TCOLMIN) {
        const double repeats = std::max(1.0, static_cast<double>(cols_i) / kColExtremaElementsPerRepeatFp16);
        const double value =
            rows_i == 1 ? kColExtremaSingleRowBaseCycles + repeats : kColExtremaMultiRowBaseCycles + rows * repeats;
        cycles = RoundA5SpecialCycles(value);
        return true;
    }

    const uint64_t repeatCount = CeilDivU64(static_cast<uint64_t>(cols_i), kColArithmeticElementsPerRepeatFp16);
    const bool useHighRepeatSmallRowFit = rows_i < kColSmallRowLimit && repeatCount >= kColHighRepeatStart;
    const ColReductionFitParams* params = nullptr;
    if (input.op == PtoOpcode::TCOLSUM) {
        params = useHighRepeatSmallRowFit ? &kColSumHighRepeatSmallRowFit : &kColSumFit;
    } else if (input.op == PtoOpcode::TCOLPROD) {
        params = useHighRepeatSmallRowFit ? &kColProdHighRepeatSmallRowFit : &kColProdFit;
    } else {
        return false;
    }
    const double repeats = static_cast<double>(repeatCount);
    if (input.op == PtoOpcode::TCOLPROD && rows_i == 1) {
        cycles = RoundA5SpecialCycles(kColProdSingleRowRepeatSlope * repeats + kColProdSingleRowBaseCycles);
        return true;
    }
    cycles = RoundA5SpecialCycles(EvalA5ColArithmeticReduction(*params, rows, repeats));
    return true;
}

inline bool TryEstimateA5ReductionCycles(const CostModelInput& input, uint64_t& cycles)
{
    return TryEstimateA5RowSumCycles(input, cycles) || TryEstimateA5RowReductionCycles(input, cycles) ||
           TryEstimateA5ColReductionCycles(input, cycles);
}

} // namespace pto::mocker::lightweight::a5

#endif // PTO_MOCKER_A5_REDUCTION_COSTMODEL_HPP
