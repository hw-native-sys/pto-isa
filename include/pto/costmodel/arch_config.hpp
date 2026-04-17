/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_MOCKER_ARCH_CONFIG_HPP
#define PTO_MOCKER_ARCH_CONFIG_HPP

#include <cstdint>
#include <string_view>

namespace pto::mocker::evaluator {

inline constexpr uint64_t kBlockBytes = 32;
inline constexpr long double kBytesPerGb = 1024.0L * 1024.0L * 1024.0L;
inline constexpr long double kMainFrequencyHz = 1.85e9L;

enum class PipeKey
{
    VECTOR,
    CUBE,
    GM_TO_UB,
    GM_TO_L1,
    UB_TO_GM,
    L1_TO_GM,
    UB_TO_UB,
    L0C_TO_GM,
    L0C_TO_L1,
    L1_TO_L0A,
    L1_TO_L0B,
    L1_TO_BT,
    L1_TO_FB,
    L1_FILL,
    COUNT,
};

struct BandwidthTable {
    double GM_TO_UB = 0.0;
    double GM_TO_L1 = 0.0;
    double UB_TO_GM = 0.0;
    double L1_TO_GM = 0.0;
    double UB_TO_UB = 0.0;
    double L0C_TO_GM = 0.0;
    double L0C_TO_L1 = 0.0;
    double L1_TO_L0A = 0.0;
    double L1_TO_L0B = 0.0;
    double L1_TO_BT = 0.0;
    double L1_TO_FB = 0.0;
    double L1_FILL = 0.0;

    constexpr double operator[](PipeKey key) const
    {
        switch (key) {
            case PipeKey::VECTOR:
            case PipeKey::CUBE:
            case PipeKey::COUNT:
                return 0.0;
            case PipeKey::GM_TO_UB:
                return GM_TO_UB;
            case PipeKey::GM_TO_L1:
                return GM_TO_L1;
            case PipeKey::UB_TO_GM:
                return UB_TO_GM;
            case PipeKey::L1_TO_GM:
                return L1_TO_GM;
            case PipeKey::UB_TO_UB:
                return UB_TO_UB;
            case PipeKey::L0C_TO_GM:
                return L0C_TO_GM;
            case PipeKey::L0C_TO_L1:
                return L0C_TO_L1;
            case PipeKey::L1_TO_L0A:
                return L1_TO_L0A;
            case PipeKey::L1_TO_L0B:
                return L1_TO_L0B;
            case PipeKey::L1_TO_BT:
                return L1_TO_BT;
            case PipeKey::L1_TO_FB:
                return L1_TO_FB;
            case PipeKey::L1_FILL:
                return L1_FILL;
        }
        return 0.0;
    }
};

struct ArchConfig {
    std::string_view arch_name;
    long double frequency_hz = kMainFrequencyHz;
    BandwidthTable bandwidth{};
};

inline constexpr ArchConfig kA2A3ArchConfig{
    "a2a3",
    kMainFrequencyHz,
    {
        100.9,
        135.0,
        188.46,
        32.0,
        1024.0,
        70.0,
        128.0,
        441.0,
        220.5,
        32.0,
        32.0,
        32.0,
    },
};

inline const ArchConfig &GetDefaultArchConfig()
{
    return kA2A3ArchConfig;
}

} // namespace pto::mocker::evaluator

#endif
