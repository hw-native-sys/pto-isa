/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/
#ifndef TRANDOM_CPU_HPP
#define TRANDOM_CPU_HPP

#include <array>
#include <cstdint>
#include <type_traits>

namespace pto {
namespace cpu_random {
constexpr uint16_t kTrandomOnceRepeat = 4;
constexpr uint32_t kConst0 = 0xD2511F53u;
constexpr uint32_t kConst1 = 0xCD9E8D57u;
constexpr uint32_t kKeyAdd0 = 0x9E3779B9u;
constexpr uint32_t kKeyAdd1 = 0xBB67AE85u;
constexpr uint32_t kRepeatBytes = 256u;

inline void IncrementCounter(std::array<uint32_t, 4> &counter, uint32_t value)
{
    uint64_t carry = value;
    for (size_t i = 0; i < counter.size(); ++i) {
        carry += static_cast<uint64_t>(counter[i]);
        counter[i] = static_cast<uint32_t>(carry);
        carry >>= 32;
        if (carry == 0) {
            break;
        }
    }
}

template <uint16_t Rounds>
inline std::array<uint32_t, 4> RunRounds(std::array<uint32_t, 4> counter, std::array<uint32_t, 2> key)
{
    for (uint16_t i = 0; i < Rounds; ++i) {
        const uint64_t mul0 = static_cast<uint64_t>(counter[0]) * static_cast<uint64_t>(kConst0);
        const uint64_t mul1 = static_cast<uint64_t>(counter[2]) * static_cast<uint64_t>(kConst1);
        const uint32_t hi0 = static_cast<uint32_t>(mul0 >> 32);
        const uint32_t lo0 = static_cast<uint32_t>(mul0);
        const uint32_t hi1 = static_cast<uint32_t>(mul1 >> 32);
        const uint32_t lo1 = static_cast<uint32_t>(mul1);

        const uint32_t next0 = hi1 ^ counter[1] ^ key[0];
        const uint32_t next1 = lo1;
        const uint32_t next2 = hi0 ^ counter[3] ^ key[1];
        const uint32_t next3 = lo0;

        counter = {next0, next1, next2, next3};
        key[0] += kKeyAdd0;
        key[1] += kKeyAdd1;
    }
    return counter;
}
} // namespace cpu_random

template <uint16_t Rounds = 10, typename DstTile>
PTO_INTERNAL void TRANDOM_IMPL(DstTile &dst, TRandomKey &key, TRandomCounter &counter)
{
    using T = typename DstTile::DType;
    static_assert(std::is_same_v<T, int32_t> || std::is_same_v<T, uint32_t>,
                  "Fix: TRANDOM only support int32_t and uint32_t.");
    static_assert(Rounds == 7 || Rounds == 10, "Fix: TRANDOM Rounds can only  be configured to 7 or 10.");
    static_assert(DstTile::isRowMajor, "Fix: TRANDOM only support row major layout.");

    PTO_CPU_ASSERT(key != nullptr && counter != nullptr, "Fix: TRANDOM key and counter must be provided.");

    constexpr uint32_t elementsPerRepeat = cpu_random::kRepeatBytes / sizeof(T);
    constexpr uint32_t rowStrideChunks = DstTile::RowStride / cpu_random::kTrandomOnceRepeat;
    const uint32_t validRows = static_cast<uint32_t>(dst.GetValidRow());
    const uint32_t validCols = static_cast<uint32_t>(dst.GetValidCol());
    const uint32_t loopCount = (validCols + cpu_random::kTrandomOnceRepeat * elementsPerRepeat - 1) /
                               (cpu_random::kTrandomOnceRepeat * elementsPerRepeat);

    const std::array<uint32_t, 2> baseKey = {key[0], key[1]};
    std::array<uint32_t, 4> rowCounter = {counter[0], counter[1], counter[2], counter[3]};

    for (uint32_t row = 0; row < validRows; ++row) {
        for (uint32_t loop = 0; loop < loopCount; ++loop) {
            for (uint32_t lane = 0; lane < elementsPerRepeat; ++lane) {
                std::array<uint32_t, 4> laneCounter = rowCounter;
                cpu_random::IncrementCounter(laneCounter, lane);
                const auto out = cpu_random::RunRounds<Rounds>(laneCounter, baseKey);
                for (uint32_t repeat = 0; repeat < cpu_random::kTrandomOnceRepeat; ++repeat) {
                    const uint32_t col =
                        loop * cpu_random::kTrandomOnceRepeat * elementsPerRepeat + repeat * elementsPerRepeat + lane;
                    if (col >= validCols) {
                        continue;
                    }
                    dst.data()[GetTileElementOffset<DstTile>(row, col)] = static_cast<T>(out[repeat]);
                }
            }
            cpu_random::IncrementCounter(rowCounter, rowStrideChunks);
        }
    }
}
} // namespace pto

#endif