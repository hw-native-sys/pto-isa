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

#include <array>
#include <string>

namespace pto::mocker::lightweight::a5 {

inline constexpr uint64_t kDefaultUbCapacityBytes = 262144;
inline constexpr uint64_t kDefaultBlockBytes = 32;
inline constexpr uint64_t kDefaultVectorBytes = 256;
inline constexpr size_t kDefaultFeatureCount = 16;
inline constexpr size_t kDefaultCoefficientParts = 4;
inline constexpr size_t kDefaultCoefficientsPerPart = 4;

inline bool ValidDefaultStorage(const OperandGeometry& g, uint64_t bytes)
{
    return bytes > 0 && g.rows > 0 && g.cols > 0 && g.valid_rows > 0 && g.valid_cols > 0 && g.valid_rows <= g.rows &&
           g.valid_cols <= g.cols && g.row_stride == g.cols && g.has_address && g.address < kDefaultUbCapacityBytes &&
           g.address % kDefaultBlockBytes == 0 && uint64_t(g.rows) <= kDefaultUbCapacityBytes / bytes &&
           uint64_t(g.cols) <= (kDefaultUbCapacityBytes - g.address) / bytes / uint64_t(g.rows) &&
           uint64_t(g.cols) * bytes % kDefaultBlockBytes == 0;
}

inline bool DefaultStorageDisjoint(const OperandGeometry& a, uint64_t aBytes, const OperandGeometry& b, uint64_t bBytes)
{
    return a.address + uint64_t(a.rows) * uint64_t(a.cols) * aBytes <= b.address ||
           b.address + uint64_t(b.rows) * uint64_t(b.cols) * bBytes <= a.address;
}

inline bool IsDefaultDataType(DType dtype)
{
    return dtype == DType::Float || dtype == DType::Half || dtype == DType::BFloat16 || dtype == DType::Int8 ||
           dtype == DType::Uint8 || dtype == DType::Int16 || dtype == DType::Uint16 || dtype == DType::Int32 ||
           dtype == DType::Uint32;
}

inline bool ValidDefaultLoadGeometry(const CostModelInput& input)
{
    constexpr int64_t kCalibratedTileSide = 16;
    constexpr uint64_t kCalibratedGmAlignmentBytes = 4096;
    const auto& g = input.default_input;
    const auto& d = g.dst;
    const auto& s = g.src;
    // Provisional, explicitly user-approved coverage (2026-11-19), NOT a
    // <=10% accuracy guarantee: rounded training maxima are 11.76055% for
    // half and 10.05648% for pooled float/int32. The measured cycle spread
    // persists at the same shape/cache hint/observable 4KiB GM alignment.
    // Geometry alone does not explain it; physical mapping/cache/memory
    // state are unmeasured possible contributors, not proven root causes.
    // This is a warm same-address MTE2 increment, excluding cold/streaming
    // and end-to-end latency. No card-specific or per-case constants.
    return g.profile == DefaultInputProfile::WarmSameAddressApproximate &&
           (input.dtype == DType::Half || input.dtype == DType::Float || input.dtype == DType::Int32) &&
           d.rows == kCalibratedTileSide && d.cols == kCalibratedTileSide && d.address == 0 &&
           s.rows == kCalibratedTileSide && s.cols == kCalibratedTileSide && s.valid_rows == kCalibratedTileSide &&
           s.valid_cols == kCalibratedTileSide && s.row_stride == kCalibratedTileSide && s.has_address &&
           s.address != 0 && s.address % kCalibratedGmAlignmentBytes == 0;
}

inline bool ValidDefaultIndexedLayout(const DefaultInputGeometry& g, uint64_t bytes, uint64_t& indexBytes)
{
    const auto& d = g.dst;
    const auto& s = g.src;
    const auto& i = g.index;
    const bool knownIndex = g.index_dtype == DType::Int16 || g.index_dtype == DType::Uint16 ||
                            g.index_dtype == DType::Int32 || g.index_dtype == DType::Uint32;
    if (!knownIndex || !TryGetDTypeSizeBytes(g.index_dtype, indexBytes) || !ValidDefaultStorage(s, bytes) ||
        !ValidDefaultStorage(i, indexBytes) || s.valid_rows != s.rows || s.valid_cols != s.cols || i.rows != d.rows ||
        i.valid_rows != d.rows || !DefaultStorageDisjoint(d, bytes, i, indexBytes) ||
        !DefaultStorageDisjoint(s, bytes, i, indexBytes))
        return false;
    // Keep the measured deterministic layout explicit: index at UB zero,
    // followed by source then destination. Do not extrapolate to bank/address
    // schedules that were not part of this default-input experiment.
    if (i.address != 0 || s.address != uint64_t(i.rows) * uint64_t(i.cols) * indexBytes)
        return false;
    return d.address == s.address || d.address == s.address + uint64_t(s.rows) * uint64_t(s.cols) * bytes;
}

inline bool ValidDefaultBlockGatherGeometry(const DefaultInputGeometry& g, uint64_t bytes)
{
    // Bounds and index padding describe the identity-block calibration domain.
    constexpr int64_t kMaxRows = 95;
    constexpr uint64_t kMaxRowBytes = 1568;
    constexpr int64_t kMinIndexCols = 64;
    constexpr int64_t kIndexColAlignment = 8;
    const auto& d = g.dst;
    const auto& s = g.src;
    const auto& i = g.index;
    const uint64_t repeats = CeilDivU64(uint64_t(d.cols) * bytes, kDefaultVectorBytes);
    const uint64_t readEnd = uint64_t(i.rows - 1) * uint64_t(i.cols) * sizeof(uint32_t) +
                             (repeats - 1) * kDefaultBlockBytes + kDefaultVectorBytes;
    return g.profile == DefaultInputProfile::IdentityBlocks && g.index_dtype == DType::Uint32 && s.rows == d.rows &&
           s.cols == d.cols && d.rows <= kMaxRows && uint64_t(d.cols) * bytes <= kMaxRowBytes &&
           i.cols ==
               std::max<int64_t>(
                   kMinIndexCols, (d.cols * int64_t(bytes) / int64_t(kDefaultBlockBytes) + kIndexColAlignment - 1) /
                                      kIndexColAlignment * kIndexColAlignment) &&
           i.valid_cols == i.cols && readEnd <= kDefaultUbCapacityBytes - i.address &&
           DefaultStorageDisjoint(d, bytes, s, bytes);
}

inline bool ValidDefaultZeroGatherGeometry(const DefaultInputGeometry& g, uint64_t bytes, uint64_t indexBytes)
{
    constexpr int64_t kAliasMinRows = 3;
    constexpr int64_t kAliasMaxRows = 256;
    constexpr int64_t kAliasMaxCols = 384;
    const auto& d = g.dst;
    const auto& s = g.src;
    // Only the first source element is read by this assumed profile. The
    // measured same-base alias preserves that value on every output write.
    return (bytes == sizeof(uint16_t) || bytes == sizeof(uint32_t)) &&
           (bytes != sizeof(uint32_t) || indexBytes == sizeof(uint32_t)) && s.rows == 1 &&
           s.cols * bytes == kDefaultBlockBytes &&
           (DefaultStorageDisjoint(d, bytes, s, bytes) ||
            (d.address == s.address && bytes == sizeof(uint16_t) && indexBytes == sizeof(uint16_t) &&
             d.rows >= kAliasMinRows && d.rows <= kAliasMaxRows && d.cols <= kAliasMaxCols));
}

inline bool ValidDefaultZeroIndexGeometry(const CostModelInput& input, uint64_t bytes, uint64_t indexBytes)
{
    const auto& g = input.default_input;
    const auto& d = g.dst;
    const auto& s = g.src;
    const auto& i = g.index;
    if ((input.op != PtoOpcode::TGATHER && input.op != PtoOpcode::TSCATTER) ||
        g.profile != DefaultInputProfile::ZeroIndex || i.valid_cols != d.cols ||
        i.cols !=
            int64_t(CeilDivU64(uint64_t(d.cols) * indexBytes, kDefaultBlockBytes) * kDefaultBlockBytes / indexBytes))
        return false;
    const uint64_t q = CeilDivU64(uint64_t(d.cols) * indexBytes, kDefaultVectorBytes);
    const uint64_t indexReadSpan = uint64_t(i.rows - 1) * uint64_t(i.cols) * indexBytes + q * kDefaultVectorBytes;
    if (indexReadSpan > kDefaultUbCapacityBytes - i.address)
        return false;
    if (input.op == PtoOpcode::TGATHER)
        return ValidDefaultZeroGatherGeometry(g, bytes, indexBytes);
    const uint64_t sourceReadSpan =
        uint64_t(s.rows - 1) * uint64_t(s.cols) * bytes + q * (kDefaultVectorBytes / indexBytes) * bytes;
    return ((bytes <= sizeof(uint16_t) && indexBytes == sizeof(uint16_t)) ||
            (bytes == sizeof(uint32_t) && indexBytes == sizeof(uint32_t))) &&
           s.rows == d.rows && s.cols == d.cols && i.valid_cols == d.cols &&
           sourceReadSpan <= kDefaultUbCapacityBytes - s.address && DefaultStorageDisjoint(d, bytes, s, bytes);
}

inline bool ValidDefaultInputGeometry(const CostModelInput& input, uint64_t& bytes, uint64_t& indexBytes)
{
    const auto& g = input.default_input;
    const auto& d = g.dst;
    if (!g.present || !g.enabled || !g.supported_layout || !g.runtime_valid_shape ||
        input.vf_impl_kind != VFImplKind::VFIMPL_DEFAULT || !TryGetDTypeSizeBytes(input.dtype, bytes) ||
        !ValidDefaultStorage(d, bytes) || d.valid_rows != d.rows || d.valid_cols != d.cols)
        return false;
    if (!IsDefaultDataType(input.dtype))
        return false;
    if (input.op == PtoOpcode::TLOAD)
        return ValidDefaultLoadGeometry(input);
    if (!ValidDefaultIndexedLayout(g, bytes, indexBytes))
        return false;
    if (input.op == PtoOpcode::TGATHERB)
        return ValidDefaultBlockGatherGeometry(g, bytes);
    return ValidDefaultZeroIndexGeometry(input, bytes, indexBytes);
}

inline std::array<double, kDefaultFeatureCount> BuildDefaultInputFeatures(
    const OperandGeometry& d, uint64_t bytes, uint64_t indexBytes)
{
    constexpr double kLargeIndexRepeatKnee = 16.0;
    constexpr double kSmallIndexRepeatKnee = 8.0;
    constexpr uint64_t kScatterPackedElementsPerChunk = 64;
    const double rows = static_cast<double>(d.rows);
    const double q = static_cast<double>(CeilDivU64(uint64_t(d.cols) * indexBytes, kDefaultVectorBytes));
    const double dataQ = static_cast<double>(CeilDivU64(uint64_t(d.cols) * bytes, kDefaultVectorBytes));
    const double init =
        static_cast<double>(CeilDivU64(uint64_t(d.rows) * uint64_t(d.cols) * bytes, kDefaultVectorBytes));
    return {
        1.0,
        q,
        rows,
        rows * q,
        init,
        rows * dataQ,
        dataQ,
        double(d.rows * d.cols) * bytes,
        std::min(rows, dataQ),
        dataQ > rows ? 1.0 : 0.0,
        std::max(q - kLargeIndexRepeatKnee, 0.0),
        std::max(q - kSmallIndexRepeatKnee, 0.0),
        rows * static_cast<double>(CeilDivU64(uint64_t(d.cols), kScatterPackedElementsPerChunk)),
        uint64_t(d.cols) * bytes % kDefaultVectorBytes ? rows : 0.0,
        0.0,
        0.0};
}

inline double EvaluateDefaultInputModel(
    std::string_view model, const std::array<double, kDefaultFeatureCount>& features,
    const std::array<double, kDefaultFeatureCount>& parameters)
{
    const double q = features[1], rows = features[2], init = features[4], dataQ = features[6];
    if (model == "linear") {
        double value = 0;
        for (size_t i = 0; i < features.size(); ++i)
            value += parameters[i] * features[i];
        return value;
    }
    if (model == "max_scatter_packed")
        return std::max(
                   parameters[0] * rows * q + parameters[1] * features[12] + parameters[2],
                   parameters[3] * rows * q + parameters[4] * features[12] + parameters[5]) +
               parameters[6] * init;
    if (model == "max_blocks")
        return std::max(parameters[0] * rows * dataQ + parameters[1], parameters[2] * rows * dataQ + parameters[3]);
    if (model == "max_nested")
        return std::max(
            {parameters[0] * rows * q + parameters[1], parameters[2] * rows * q + parameters[3] * rows + parameters[4],
             parameters[5] * rows * q + parameters[6]});
    // Empirical overlap envelope: startup/repeat work, row work, and
    // sustained repeat work. Not a sum of universal instruction costs.
    double value = std::max(
        {parameters[0] * rows * q + parameters[1], parameters[2] * rows + parameters[3],
         parameters[4] * rows * q + parameters[5]});
    if (model == "max_pipeline_init")
        value += parameters[6] * init;
    return value;
}

inline bool TryEstimateA5DefaultInputCycles(const CostModelInput& input, uint64_t& cycles)
{
    uint64_t bytes = 0, indexBytes = 0;
    if (!ValidDefaultInputGeometry(input, bytes, indexBytes))
        return false;
    const auto features = BuildDefaultInputFeatures(input.default_input.dst, bytes, indexBytes);
    // Signedness and FP representation do not change these bit-movement
    // lowerings. Original dtype combinations are separately hardware checked.
    const std::string_view storage = bytes == sizeof(uint8_t) ? "uint8" : bytes == sizeof(uint16_t) ? "fp16" : "fp32";
    std::string family;
    if (input.op == PtoOpcode::TGATHER || input.op == PtoOpcode::TSCATTER)
        family = std::string("default_zero_i") + (indexBytes == sizeof(uint16_t) ? "16" : "32") +
                 (input.default_input.dst.rows == 1 ? "_row1" : "_grid");
    else if (input.op == PtoOpcode::TLOAD)
        family = "default_warm_load_approximate";
    else
        family = "default_blocks";
    for (const std::string_view model :
         {"linear", "max_blocks", "max_scatter_packed", "max_pipeline", "max_pipeline_init", "max_nested"}) {
        std::array<double, kDefaultFeatureCount> parameters{};
        bool found = true;
        for (size_t part = 0; part < kDefaultCoefficientParts; ++part) {
            const std::string key = family + "_" + std::string(model) + "_" + std::to_string(part);
            const fit::VfFormulaParam* param = nullptr;
            const auto opcode = input.op == PtoOpcode::TLOAD ? std::string_view("TLOAD") : PtoOpcodeToKey(input.op);
            const fit::VfCurveKey curve{
                opcode, storage, "none", fit::ShapePath::Path2D, "POST_UPDATE", fit::TailKind::Full, key};
            if (!fit::TryLookupVfFormulaParam(curve, param)) {
                found = false;
                break;
            }
            parameters[part * kDefaultCoefficientsPerPart] = param->param0;
            parameters[part * kDefaultCoefficientsPerPart + 1] = param->param1;
            parameters[part * kDefaultCoefficientsPerPart + 2] = param->param2;
            parameters[part * kDefaultCoefficientsPerPart + 3] = param->param3;
        }
        if (!found)
            continue;
        cycles = fit::RoundToCycles(EvaluateDefaultInputModel(model, features, parameters));
        return cycles > 0;
    }
    return false;
}

} // namespace pto::mocker::lightweight::a5
