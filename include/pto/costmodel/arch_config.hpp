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

#include <array>
#include <cstdint>
#include <limits>
#include <string_view>

namespace pto::mocker::evaluator {

inline constexpr uint64_t kBlockBytes = 32;
inline constexpr long double kBytesPerGb = 1024.0L * 1024.0L * 1024.0L;
inline constexpr long double kMainFrequencyHz = 1.85e9L;
inline constexpr long double kMicrosPerSecond = 1.0e6L;

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

inline constexpr bool IsMemoryPipe(PipeKey key)
{
    return key != PipeKey::VECTOR && key != PipeKey::CUBE && key != PipeKey::COUNT;
}

inline constexpr std::array<bool, static_cast<std::size_t>(PipeKey::COUNT)> BuildBandwidthProvided(
    const BandwidthTable &bandwidth)
{
    std::array<bool, static_cast<std::size_t>(PipeKey::COUNT)> provided{};
    provided[static_cast<std::size_t>(PipeKey::GM_TO_UB)] = (bandwidth.GM_TO_UB > 0.0);
    provided[static_cast<std::size_t>(PipeKey::GM_TO_L1)] = (bandwidth.GM_TO_L1 > 0.0);
    provided[static_cast<std::size_t>(PipeKey::UB_TO_GM)] = (bandwidth.UB_TO_GM > 0.0);
    provided[static_cast<std::size_t>(PipeKey::L1_TO_GM)] = (bandwidth.L1_TO_GM > 0.0);
    provided[static_cast<std::size_t>(PipeKey::UB_TO_UB)] = (bandwidth.UB_TO_UB > 0.0);
    provided[static_cast<std::size_t>(PipeKey::L0C_TO_GM)] = (bandwidth.L0C_TO_GM > 0.0);
    provided[static_cast<std::size_t>(PipeKey::L0C_TO_L1)] = (bandwidth.L0C_TO_L1 > 0.0);
    provided[static_cast<std::size_t>(PipeKey::L1_TO_L0A)] = (bandwidth.L1_TO_L0A > 0.0);
    provided[static_cast<std::size_t>(PipeKey::L1_TO_L0B)] = (bandwidth.L1_TO_L0B > 0.0);
    provided[static_cast<std::size_t>(PipeKey::L1_TO_BT)] = (bandwidth.L1_TO_BT > 0.0);
    provided[static_cast<std::size_t>(PipeKey::L1_TO_FB)] = (bandwidth.L1_TO_FB > 0.0);
    provided[static_cast<std::size_t>(PipeKey::L1_FILL)] = (bandwidth.L1_FILL > 0.0);
    return provided;
}

inline thread_local long double gPredictFrequencyMHz = GetDefaultArchConfig().frequency_hz / kMicrosPerSecond;
inline thread_local BandwidthTable gPredictBandwidthBytesPerUs = GetDefaultArchConfig().bandwidth;
inline thread_local std::array<bool, static_cast<std::size_t>(PipeKey::COUNT)> gPredictBandwidthProvided =
    BuildBandwidthProvided(gPredictBandwidthBytesPerUs);

inline void SetPredictFrequencyAndBandwidth(long double frequencyMhz, const BandwidthTable &bandwidth)
{
    gPredictFrequencyMHz = frequencyMhz;
    gPredictBandwidthBytesPerUs = bandwidth;
    gPredictBandwidthProvided = BuildBandwidthProvided(bandwidth);
}

inline void SetPredictArchConfig(const ArchConfig &archConfig)
{
    SetPredictFrequencyAndBandwidth(archConfig.frequency_hz / kMicrosPerSecond, archConfig.bandwidth);
}

inline void ResetPredictArchConfig()
{
    SetPredictArchConfig(GetDefaultArchConfig());
}

inline long double GetPredictFrequencyMHz()
{
    return gPredictFrequencyMHz;
}

inline bool HasPredictBandwidthBytesPerUs(PipeKey key)
{
    if (!IsMemoryPipe(key)) {
        return false;
    }
    return gPredictBandwidthProvided[static_cast<std::size_t>(key)];
}

inline double GetPredictBandwidthBytesPerUs(PipeKey key)
{
    return gPredictBandwidthBytesPerUs[key];
}

inline long double CyclesToUs(uint64_t cycles, long double frequencyMhz)
{
    if (frequencyMhz <= 0.0L) {
        return 0.0L;
    }
    return static_cast<long double>(cycles) / frequencyMhz;
}

inline long double CyclesToUs(uint64_t cycles)
{
    return CyclesToUs(cycles, GetPredictFrequencyMHz());
}

inline long double TransferBytesToUs(uint64_t bytes, double bandwidthBytesPerUs)
{
    if (bytes == 0) {
        return 0.0L;
    }
    if (bandwidthBytesPerUs <= 0.0) {
        return std::numeric_limits<long double>::quiet_NaN();
    }
    return static_cast<long double>(bytes) / static_cast<long double>(bandwidthBytesPerUs);
}

inline long double TransferBytesToUs(uint64_t bytes, PipeKey pipe)
{
    if (!HasPredictBandwidthBytesPerUs(pipe)) {
        return std::numeric_limits<long double>::quiet_NaN();
    }
    return TransferBytesToUs(bytes, GetPredictBandwidthBytesPerUs(pipe));
}

} // namespace pto::mocker::evaluator

#endif
