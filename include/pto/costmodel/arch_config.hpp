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

enum class PipeKey {
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

inline const ArchConfig& GetDefaultArchConfig() { return kA2A3ArchConfig; }

inline constexpr bool IsMemoryPipe(PipeKey key)
{
    return key != PipeKey::VECTOR && key != PipeKey::CUBE && key != PipeKey::COUNT;
}

inline constexpr std::array<bool, static_cast<std::size_t>(PipeKey::COUNT)> BuildBandwidthProvided(
    const BandwidthTable& bandwidth)
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

inline void SetPredictFrequencyAndBandwidth(long double frequencyMhz, const BandwidthTable& bandwidth)
{
    gPredictFrequencyMHz = frequencyMhz;
    gPredictBandwidthBytesPerUs = bandwidth;
    gPredictBandwidthProvided = BuildBandwidthProvided(bandwidth);
}

inline void SetPredictArchConfig(const ArchConfig& archConfig)
{
    SetPredictFrequencyAndBandwidth(archConfig.frequency_hz / kMicrosPerSecond, archConfig.bandwidth);
}

inline void ResetPredictArchConfig() { SetPredictArchConfig(GetDefaultArchConfig()); }

inline long double GetPredictFrequencyMHz() { return gPredictFrequencyMHz; }

inline bool HasPredictBandwidthBytesPerUs(PipeKey key)
{
    if (!IsMemoryPipe(key)) {
        return false;
    }
    return gPredictBandwidthProvided[static_cast<std::size_t>(key)];
}

inline double GetPredictBandwidthBytesPerUs(PipeKey key) { return gPredictBandwidthBytesPerUs[key]; }

// ---------------------------------------------------------------------------
// Hill bandwidth model (single-transfer size saturation + multi-core cap).
//   hill_bw(B) = peak * B / (K + B)                       [GiB/s], saturates at peak
//   bw_eff(B,n) = min(hill_bw(B), total_group / n)        external GM pipes contend
//     read  group {GM_TO_UB, GM_TO_L1}      -> total_read
//     write group {UB_TO_GM, L1_TO_GM, L0C_TO_GM} -> total_write
//   cycle(B,n) = (B / kBytesPerGb) / bw_eff(B,n) * kMainFrequencyHz
// Default model: K=0 (=> hill=peak, constant) + totals<=0 (=> no cap) == the legacy
// flat `bytes / bandwidth[PipeKey]` model. Fitted Hill params are applied via env
// PTO_BW_MODE=fitted (see ApplyHillBandwidthFromEnv). Backward compatible.
// ---------------------------------------------------------------------------
inline constexpr bool IsExternalGmPipe(PipeKey key)
{
    return key == PipeKey::GM_TO_UB || key == PipeKey::GM_TO_L1 || key == PipeKey::UB_TO_GM ||
           key == PipeKey::L1_TO_GM || key == PipeKey::L0C_TO_GM;
}

struct HillBandwidthModel {
    double peak_gibs[static_cast<std::size_t>(PipeKey::COUNT)] = {0.0};
    double k_bytes[static_cast<std::size_t>(PipeKey::COUNT)] = {0.0};
    double total_read_gibs = 0.0;  // <=0 means no read cap
    double total_write_gibs = 0.0; // <=0 means no write cap

    double HillBw(PipeKey key, uint64_t bytes) const
    {
        const auto i = static_cast<std::size_t>(key);
        const double peak = peak_gibs[i];
        const double k = k_bytes[i];
        const double denom = k + static_cast<double>(bytes);
        if (denom <= 0.0) {
            return peak; // bytes==0, K==0 -> any finite bw (cycle computed as 0)
        }
        return peak * static_cast<double>(bytes) / denom;
    }

    double GroupTotal(PipeKey key) const
    {
        if (key == PipeKey::GM_TO_UB || key == PipeKey::GM_TO_L1) {
            return total_read_gibs;
        }
        if (key == PipeKey::UB_TO_GM || key == PipeKey::L1_TO_GM || key == PipeKey::L0C_TO_GM) {
            return total_write_gibs;
        }
        return 0.0; // internal pipe: no GM contention
    }

    double BwEff(PipeKey key, uint64_t bytes, uint32_t ncores) const
    {
        double bw = HillBw(key, bytes);
        if (IsExternalGmPipe(key)) {
            const double total = GroupTotal(key);
            if (total > 0.0 && ncores > 0) {
                const double cap = total / static_cast<double>(ncores);
                if (cap < bw) {
                    bw = cap;
                }
            }
        }
        return bw;
    }
};

// Build the legacy flat model (peak = current table constant, K = 0, no cap).
inline HillBandwidthModel MakeFlatHillModel()
{
    const BandwidthTable& b = GetDefaultArchConfig().bandwidth;
    HillBandwidthModel m;
    m.peak_gibs[static_cast<std::size_t>(PipeKey::GM_TO_UB)] = b.GM_TO_UB;
    m.peak_gibs[static_cast<std::size_t>(PipeKey::GM_TO_L1)] = b.GM_TO_L1;
    m.peak_gibs[static_cast<std::size_t>(PipeKey::UB_TO_GM)] = b.UB_TO_GM;
    m.peak_gibs[static_cast<std::size_t>(PipeKey::L1_TO_GM)] = b.L1_TO_GM;
    m.peak_gibs[static_cast<std::size_t>(PipeKey::UB_TO_UB)] = b.UB_TO_UB;
    m.peak_gibs[static_cast<std::size_t>(PipeKey::L0C_TO_GM)] = b.L0C_TO_GM;
    m.peak_gibs[static_cast<std::size_t>(PipeKey::L0C_TO_L1)] = b.L0C_TO_L1;
    m.peak_gibs[static_cast<std::size_t>(PipeKey::L1_TO_L0A)] = b.L1_TO_L0A;
    m.peak_gibs[static_cast<std::size_t>(PipeKey::L1_TO_L0B)] = b.L1_TO_L0B;
    m.peak_gibs[static_cast<std::size_t>(PipeKey::L1_TO_BT)] = b.L1_TO_BT;
    m.peak_gibs[static_cast<std::size_t>(PipeKey::L1_TO_FB)] = b.L1_TO_FB;
    m.peak_gibs[static_cast<std::size_t>(PipeKey::L1_FILL)] = b.L1_FILL;
    return m; // K=0, totals=0 -> exact legacy flat behaviour
}

// Fitted Hill params — pure on-device B3 fit (see tools/bandwidth_fit/hill_params.json).
// The CAModel (B1) per-event signal was found to disagree with B3 and is NOT used:
// pure-B3 is strictly better in-fit (0.136 vs 0.227) and LOO (0.303 vs 0.364), and fixes
// GM_TO_L1's systematic 0.70x low bias of the mixed fit.
// Caps left at 0 (no cap): FA at n<=4 with small tiles shows no observable GM contention.
inline HillBandwidthModel MakeFittedHillModel()
{
    HillBandwidthModel m = MakeFlatHillModel();
    // Per-pipe fitted params: peak bandwidth (GiB/s) and half-saturation size (k_bytes).
    constexpr double kGmToUbPeakGiBs = 247.16;
    constexpr double kGmToUbKBytes = 30643.0;
    constexpr double kGmToL1PeakGiBs = 28.61;
    constexpr double kGmToL1KBytes = 1107.0;
    constexpr double kUbToGmPeakGiBs = 28.19;
    constexpr double kUbToGmKBytes = 1755.0;
    constexpr double kL0cToGmPeakGiBs = 41.25;
    constexpr double kL0cToGmKBytes = 29104.0;
    auto set = [&](PipeKey k, double peak, double kb) {
        m.peak_gibs[static_cast<std::size_t>(k)] = peak;
        m.k_bytes[static_cast<std::size_t>(k)] = kb;
    };
    set(PipeKey::GM_TO_UB, kGmToUbPeakGiBs, kGmToUbKBytes);
    set(PipeKey::GM_TO_L1, kGmToL1PeakGiBs, kGmToL1KBytes);
    set(PipeKey::UB_TO_GM, kUbToGmPeakGiBs, kUbToGmKBytes);
    set(PipeKey::L0C_TO_GM, kL0cToGmPeakGiBs, kL0cToGmKBytes);
    return m;
}

inline thread_local HillBandwidthModel gHillBandwidth = MakeFlatHillModel();
inline thread_local uint32_t gActiveCoreCount = 1;
inline thread_local bool gHillBandwidthApplied = false;

inline void SetHillBandwidthModel(const HillBandwidthModel& m) { gHillBandwidth = m; }

inline void ResetHillBandwidthModel()
{
    gHillBandwidth = MakeFlatHillModel();
    gActiveCoreCount = 1;
    gHillBandwidthApplied = true; // explicit reset suppresses env auto-apply
}

inline void SetActiveCoreCount(uint32_t n) { gActiveCoreCount = n > 0 ? n : 1; }

// PTO_BW_MODE: "" / "flat" = legacy flat; "fitted" = fitted Hill params. Applied once.
inline void ApplyHillBandwidthFromEnv()
{
    if (gHillBandwidthApplied) {
        return;
    }
    gHillBandwidthApplied = true;
    const char* mode = std::getenv("PTO_BW_MODE");
    if (mode != nullptr && std::string_view(mode) == "fitted") {
        gHillBandwidth = MakeFittedHillModel();
    }
}

inline long double CyclesToUs(uint64_t cycles, long double frequencyMhz)
{
    if (frequencyMhz <= 0.0L) {
        return 0.0L;
    }
    return static_cast<long double>(cycles) / frequencyMhz;
}

inline long double CyclesToUs(uint64_t cycles) { return CyclesToUs(cycles, GetPredictFrequencyMHz()); }

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
