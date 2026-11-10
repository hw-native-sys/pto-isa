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
    if (input.op != PtoOpcode::TLOAD || input.dtype != DType::Half || input.layout != MemLayout::ND ||
        input.tile_type != TransferTileType::VecTile) {
        return false;
    }
    const int64_t elementCount = input.data_size > 0 ? input.data_size : input.rows * input.cols;
    if (elementCount <= 0) {
        return false;
    }
    const uint64_t bytes = static_cast<uint64_t>(elementCount) * sizeof(uint16_t);
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
    const double bytes = static_cast<double>(elementCount) * static_cast<double>(dtypeBytes);
    cycles = RoundA5MatrixTransferCycles(kStoreBaseCycles + bytes / kStoreBytesPerCycle);
    return true;
}

inline bool TryEstimateA5MatrixTransferCycles(const CostModelInput& input, uint64_t& cycles)
{
    return TryEstimateA5MatmulCycles(input, cycles) || TryEstimateA5LoadCycles(input, cycles) ||
           TryEstimateA5StoreCycles(input, cycles) || TryEstimateA5ReshapeCycles(input, cycles);
}

} // namespace pto::mocker::lightweight::a5

#endif // PTO_MOCKER_A5_MATRIX_TRANSFER_COSTMODEL_HPP
