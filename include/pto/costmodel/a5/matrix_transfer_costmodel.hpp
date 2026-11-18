/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_MOCKER_A5_MATRIX_TRANSFER_COSTMODEL_HPP
#define PTO_MOCKER_A5_MATRIX_TRANSFER_COSTMODEL_HPP

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace pto::mocker::lightweight::a5 {

inline bool IsMatrixOpcode(PtoOpcode op)
{
    return op == PtoOpcode::TMATMUL || op == PtoOpcode::TMATMUL_ACC || op == PtoOpcode::TMATMUL_BIAS ||
           op == PtoOpcode::TMATMUL_MX || op == PtoOpcode::TGEMV || op == PtoOpcode::TGEMV_ACC ||
           op == PtoOpcode::TGEMV_BIAS || op == PtoOpcode::TGEMV_MX;
}

// Single MAD completion model; excludes operand transfers and inter-instruction overlap.
// GEMV, accumulation and bias reuse its compute baseline, without an unmeasured surcharge.
inline bool TryEstimateA5MadCompletionCycles(const CostModelInput& input, uint64_t& cycles)
{
    constexpr int64_t kMaxMadDimension = 4095;
    constexpr uint64_t kFractalSide = 16;
    constexpr uint64_t kMinimumLongSideFractals = 3;
    constexpr uint64_t kStartupCycles = 25;
    constexpr uint64_t kMxKAlignment = 64;
    constexpr uint64_t kHalfWorkElements = 16;
    constexpr uint64_t kByteWorkElements = 32;
    constexpr uint64_t kNibbleWorkElements = 64;
    if (!IsMatrixOpcode(input.op) || input.rows <= 0 || input.cols <= 0 || input.k <= 0 ||
        input.rows > kMaxMadDimension || input.cols > kMaxMadDimension || input.k > kMaxMadDimension) {
        return false;
    }
    const bool mx = input.op == PtoOpcode::TMATMUL_MX || input.op == PtoOpcode::TGEMV_MX;
    const bool gemv = input.op == PtoOpcode::TGEMV || input.op == PtoOpcode::TGEMV_ACC ||
                      input.op == PtoOpcode::TGEMV_BIAS || input.op == PtoOpcode::TGEMV_MX;
    const auto fp8 = [](DType t) { return t == DType::Float8E4M3 || t == DType::Float8E5M2; };
    const auto fp4 = [](DType t) { return t == DType::Float4E1M2 || t == DType::Float4E2M1; };
    const DType rhs = input.has_dtype2 ? input.dtype2 : input.dtype;
    uint64_t workElements = 0;
    if (fp8(input.dtype) && fp8(rhs) && input.dst_dtype == DType::Float) {
        workElements = kByteWorkElements;
    } else if (mx && fp4(input.dtype) && fp4(rhs) && input.dst_dtype == DType::Float) {
        workElements = kNibbleWorkElements;
    } else if (!mx && input.dtype == rhs) {
        if (input.dtype == DType::Int8 && input.dst_dtype == DType::Int32) {
            workElements = kByteWorkElements;
        } else if (input.dst_dtype == DType::Float) {
            if (input.dtype == DType::Half || input.dtype == DType::BFloat16)
                workElements = kHalfWorkElements;
            if (input.dtype == DType::HFloat8)
                workElements = kByteWorkElements;
            if (input.dtype == DType::Float)
                workElements = 1; // Full-precision FP32, not HF32.
        }
    }
    if (workElements == 0)
        return false;
    const auto ceilDiv = [](uint64_t value, uint64_t divisor) { return (value - 1) / divisor + 1; };
    const uint64_t m = ceilDiv(gemv ? 1 : input.rows, kFractalSide);
    const uint64_t n = ceilDiv(input.cols, kFractalSide);
    const uint64_t timingK = mx ? ceilDiv(input.k, kMxKAlignment) * kMxKAlignment : input.k;
    cycles =
        kStartupCycles + std::min(m, n) * std::max({m, n, kMinimumLongSideFractals}) * ceilDiv(timingK, workElements);
    return true;
}

inline constexpr double kMatmulMacsPerCycleFp16 = 4096.0;
inline constexpr double kMatmulMinM = 48.0;
inline constexpr double kMatmulMinK = 32.0;
inline constexpr double kMatmulMinCycles = 12.0;
inline constexpr double kMatmulPipelineCycles = 0.4;

inline constexpr uint64_t kLoadLargeTransferMinBytes = 1280;
inline constexpr double kLoadLargeTransferBaseCycles = 0.1092;
inline constexpr double kLoadLargeTransferCyclesPerByte = 0.0099945;
inline constexpr double kStoreBaseCycles = 34.0;
inline constexpr double kStoreBytesPerCycle = 128.0;
inline constexpr uint64_t kReshapeAliasCycles = 1;

inline uint64_t RoundA5MatrixTransferCycles(double cycles)
{
    return cycles <= 0.0 ? 0 : static_cast<uint64_t>(std::llround(cycles));
}

inline bool TryEstimateA5MatmulCycles(const CostModelInput& input, uint64_t& cycles)
{
    if (input.op == PtoOpcode::TMATMUL && input.dtype == input.dtype2 && input.rows > 0 && input.k > 0 &&
        input.cols > 0 &&
        (input.dtype == DType::BFloat16 || input.dtype == DType::Float || input.dtype == DType::Int8)) {
        // The resident-L0 instruction rounds M to one cube block. K and N
        // must already satisfy the calibrated operand blocking contract.
        constexpr uint64_t kCubeBlockM = 16;
        constexpr int64_t kCubeBlockK16 = 16;
        constexpr int64_t kCubeBlockK8 = 32;
        const int64_t blockK = input.dtype == DType::Int8 ? kCubeBlockK8 : kCubeBlockK16;
        if (input.k % blockK != 0 || (input.cols > kCubeBlockK16 && input.cols % kCubeBlockK16 != 0)) {
            return false;
        }
        const fit::VfCurveKey key{
            "TMATMUL",
            DTypeToKey(input.dtype),
            DTypeToKey(input.dst_dtype),
            fit::ShapePath::Path2D,
            "DEFAULT",
            fit::TailKind::Full,
            "resident_l0"};
        const fit::VfFormulaParam* param = nullptr;
        if (!fit::TryLookupVfFormulaParam(key, param)) {
            return false;
        }
        const uint64_t paddedM = ((static_cast<uint64_t>(input.rows) - 1) / kCubeBlockM + 1) * kCubeBlockM;
        cycles = fit::EvalVfFormula(*param, paddedM, input.k, input.cols, 0, 0);
        return cycles > 0;
    }
    if (input.op != PtoOpcode::TMATMUL || input.dtype != DType::Half || input.dtype2 != DType::Half ||
        input.dst_dtype != DType::Float || input.rows <= 0 || input.k <= 0 || input.cols <= 0) {
        return false;
    }

    const double m = static_cast<double>(input.rows);
    const double k = static_cast<double>(input.k);
    const double n = static_cast<double>(input.cols);
    const double ideal = m * k * n / kMatmulMacsPerCycleFp16;
    const double utilization = std::max({1.0, kMatmulMinM / m, kMatmulMinK / k});
    cycles = RoundA5MatrixTransferCycles(std::max(kMatmulMinCycles, ideal * utilization) + kMatmulPipelineCycles);
    return true;
}

inline bool TryEstimateA5LoadCycles(const CostModelInput& input, uint64_t& cycles)
{
    if (input.op != PtoOpcode::TLOAD ||
        (input.dtype != DType::Half && input.dtype != DType::Float && input.dtype != DType::Int32) ||
        input.layout != MemLayout::ND || input.tile_type != TransferTileType::VecTile) {
        return false;
    }
    const int64_t elementCount = input.data_size > 0 ? input.data_size : input.rows * input.cols;
    if (elementCount <= 0) {
        return false;
    }
    const uint64_t elementBytes = input.dtype == DType::Half ? sizeof(uint16_t) : sizeof(uint32_t);
    const uint64_t bytes = static_cast<uint64_t>(elementCount) * elementBytes;
    if (bytes < kLoadLargeTransferMinBytes) {
        return false;
    }
    cycles = RoundA5MatrixTransferCycles(kLoadLargeTransferBaseCycles + kLoadLargeTransferCyclesPerByte * bytes);
    return true;
}

inline bool TryEstimateA5ReshapeCycles(const CostModelInput& input, uint64_t& cycles)
{
    if (input.op != PtoOpcode::TRESHAPE) {
        return false;
    }
    // TRESHAPE aliases the source storage and emits no vector instruction.
    // Charge one scalar bookkeeping cycle, matching perf-sim's scalar-stage convention.
    cycles = kReshapeAliasCycles;
    return true;
}

inline uint64_t GetA5DTypeBytes(DType dtype)
{
    switch (dtype) {
        case DType::Float:
        case DType::Int32:
        case DType::Uint32:
            return sizeof(uint32_t);
        case DType::Half:
        case DType::BFloat16:
        case DType::Int16:
        case DType::Uint16:
            return sizeof(uint16_t);
        case DType::Int8:
        case DType::Uint8:
        case DType::Float8E4M3:
        case DType::Float8E5M2:
        case DType::HFloat8:
            return sizeof(uint8_t);
        default:
            return 0;
    }
}

inline bool TryEstimateA5StoreCycles(const CostModelInput& input, uint64_t& cycles)
{
    if (input.op != PtoOpcode::TSTORE || input.layout != MemLayout::ND ||
        input.tile_type != TransferTileType::VecTile) {
        return false;
    }
    const int64_t elementCount = input.data_size > 0 ? input.data_size : input.rows * input.cols;
    const uint64_t dtypeBytes = GetA5DTypeBytes(input.dtype);
    if (elementCount <= 0 || dtypeBytes == 0) {
        return false;
    }
    const bool isLowPrecisionFloat =
        input.dtype == DType::Float8E4M3 || input.dtype == DType::Float8E5M2 || input.dtype == DType::HFloat8;
    // The byte-sized floating-point profiles validate only the aligned DMA path.
    // Do not extrapolate to odd 32-byte row strides: those profiles diverge.
    constexpr int64_t kAlignedDmaRowBytes = 64;
    if (isLowPrecisionFloat && (input.cols <= 0 || input.cols % kAlignedDmaRowBytes != 0 ||
                                (input.valid_cols > 0 && input.valid_cols != input.cols))) {
        return false;
    }
    const double bytes = static_cast<double>(elementCount) * static_cast<double>(dtypeBytes);
    cycles = RoundA5MatrixTransferCycles(kStoreBaseCycles + bytes / kStoreBytesPerCycle);
    return true;
}

inline bool TryEstimateA5MatrixTransferCycles(const CostModelInput& input, uint64_t& cycles)
{
    // Preserve the existing measured TMATMUL curves; new variants use the completion model.
    if (IsMatrixOpcode(input.op) && input.op != PtoOpcode::TMATMUL) {
        return TryEstimateA5MadCompletionCycles(input, cycles);
    }
    return TryEstimateA5MatmulCycles(input, cycles) || TryEstimateA5LoadCycles(input, cycles) ||
           TryEstimateA5StoreCycles(input, cycles) || TryEstimateA5ReshapeCycles(input, cycles);
}

} // namespace pto::mocker::lightweight::a5

#endif // PTO_MOCKER_A5_MATRIX_TRANSFER_COSTMODEL_HPP
