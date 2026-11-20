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

inline bool ValidDefaultStorage(const OperandGeometry& g, uint64_t bytes)
{
    constexpr uint64_t capacity = 262144;
    return bytes > 0 && g.rows > 0 && g.cols > 0 && g.valid_rows > 0 && g.valid_cols > 0 && g.valid_rows <= g.rows &&
           g.valid_cols <= g.cols && g.row_stride == g.cols && g.has_address && g.address < capacity &&
           g.address % 32 == 0 && uint64_t(g.rows) <= capacity / bytes &&
           uint64_t(g.cols) <= (capacity - g.address) / bytes / uint64_t(g.rows) && uint64_t(g.cols) * bytes % 32 == 0;
}

inline bool DefaultStorageDisjoint(const OperandGeometry& a, uint64_t aBytes, const OperandGeometry& b, uint64_t bBytes)
{
    return a.address + uint64_t(a.rows) * uint64_t(a.cols) * aBytes <= b.address ||
           b.address + uint64_t(b.rows) * uint64_t(b.cols) * bBytes <= a.address;
}

inline bool ValidDefaultInputGeometry(const CostModelInput& input, uint64_t& bytes, uint64_t& indexBytes)
{
    const auto& g = input.default_input;
    const auto& d = g.dst;
    const auto& s = g.src;
    const auto& i = g.index;
    if (!g.present || !g.enabled || !g.supported_layout || !g.runtime_valid_shape ||
        input.vf_impl_kind != VFImplKind::VFIMPL_DEFAULT || !TryGetDTypeSizeBytes(input.dtype, bytes) ||
        !ValidDefaultStorage(d, bytes) || d.valid_rows != d.rows || d.valid_cols != d.cols)
        return false;
    const bool knownData = input.dtype == DType::Float || input.dtype == DType::Half ||
                           input.dtype == DType::BFloat16 || input.dtype == DType::Int8 ||
                           input.dtype == DType::Uint8 || input.dtype == DType::Int16 || input.dtype == DType::Uint16 ||
                           input.dtype == DType::Int32 || input.dtype == DType::Uint32;
    if (!knownData)
        return false;
    if (input.op == PtoOpcode::TLOAD) {
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
               d.rows == 16 && d.cols == 16 && d.address == 0 && s.rows == 16 && s.cols == 16 && s.valid_rows == 16 &&
               s.valid_cols == 16 && s.row_stride == 16 && s.has_address && s.address != 0 && s.address % 4096 == 0;
    }
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
    const bool sourceAlias = d.address == s.address;
    if (!sourceAlias && d.address != s.address + uint64_t(s.rows) * uint64_t(s.cols) * bytes)
        return false;
    if (input.op == PtoOpcode::TGATHERB) {
        const uint64_t repeats = CeilDivU64(uint64_t(d.cols) * bytes, 256);
        const uint64_t readEnd = uint64_t(i.rows - 1) * uint64_t(i.cols) * 4 + (repeats - 1) * 32 + 256;
        return g.profile == DefaultInputProfile::IdentityBlocks && g.index_dtype == DType::Uint32 && s.rows == d.rows &&
               s.cols == d.cols && d.rows <= 95 && uint64_t(d.cols) * bytes <= 1568 &&
               i.cols == std::max<int64_t>(64, (d.cols * int64_t(bytes) / 32 + 7) / 8 * 8) && i.valid_cols == i.cols &&
               readEnd <= 262144 - i.address && DefaultStorageDisjoint(d, bytes, s, bytes);
    }
    if ((input.op != PtoOpcode::TGATHER && input.op != PtoOpcode::TSCATTER) ||
        g.profile != DefaultInputProfile::ZeroIndex || i.valid_cols != d.cols ||
        i.cols != int64_t(CeilDivU64(uint64_t(d.cols) * indexBytes, 32) * 32 / indexBytes))
        return false;
    const uint64_t q = CeilDivU64(uint64_t(d.cols) * indexBytes, 256);
    const uint64_t indexReadSpan = uint64_t(i.rows - 1) * uint64_t(i.cols) * indexBytes + q * 256;
    if (indexReadSpan > 262144 - i.address)
        return false;
    if (input.op == PtoOpcode::TGATHER) {
        // Only the first source element is read by this assumed profile. The
        // measured same-base alias preserves that value on every output write.
        return (bytes == 2 || bytes == 4) && (bytes != 4 || indexBytes == 4) && s.rows == 1 && s.cols * bytes == 32 &&
               (DefaultStorageDisjoint(d, bytes, s, bytes) ||
                (sourceAlias && bytes == 2 && indexBytes == 2 && d.rows >= 3 && d.rows <= 256 && d.cols <= 384));
    }
    const uint64_t sourceReadSpan = uint64_t(s.rows - 1) * uint64_t(s.cols) * bytes + q * (256 / indexBytes) * bytes;
    return ((bytes <= 2 && indexBytes == 2) || (bytes == 4 && indexBytes == 4)) && s.rows == d.rows &&
           s.cols == d.cols && i.valid_cols == d.cols && sourceReadSpan <= 262144 - s.address &&
           DefaultStorageDisjoint(d, bytes, s, bytes);
}

inline bool TryEstimateA5DefaultInputCycles(const CostModelInput& input, uint64_t& cycles)
{
    uint64_t bytes = 0, indexBytes = 0;
    if (!ValidDefaultInputGeometry(input, bytes, indexBytes))
        return false;
    const auto& d = input.default_input.dst;
    const double rows = static_cast<double>(d.rows);
    const double q = static_cast<double>(CeilDivU64(uint64_t(d.cols) * indexBytes, 256));
    const double dataQ = static_cast<double>(CeilDivU64(uint64_t(d.cols) * bytes, 256));
    const double init = static_cast<double>(CeilDivU64(uint64_t(d.rows) * uint64_t(d.cols) * bytes, 256));
    const std::array<double, 16> features{
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
        std::max(q - 16.0, 0.0),
        std::max(q - 8.0, 0.0),
        rows * static_cast<double>(CeilDivU64(uint64_t(d.cols), 64)),
        uint64_t(d.cols) * bytes % 256 ? rows : 0.0,
        0.0,
        0.0};
    // Signedness and FP representation do not change these bit-movement
    // lowerings. Original dtype combinations are separately hardware checked.
    const std::string_view storage = bytes == 1 ? "uint8" : bytes == 2 ? "fp16" : "fp32";
    std::string family;
    if (input.op == PtoOpcode::TGATHER || input.op == PtoOpcode::TSCATTER)
        family = std::string("default_zero_i") + (indexBytes == 2 ? "16" : "32") + (d.rows == 1 ? "_row1" : "_grid");
    else if (input.op == PtoOpcode::TLOAD)
        family = "default_warm_load_approximate";
    else
        family = "default_blocks";
    for (const std::string_view model :
         {"linear", "max_blocks", "max_scatter_packed", "max_pipeline", "max_pipeline_init", "max_nested"}) {
        std::array<double, 16> parameters{};
        bool found = true;
        for (size_t part = 0; part < 4; ++part) {
            const std::string key = family + "_" + std::string(model) + "_" + std::to_string(part);
            const fit::VfFormulaParam* param = nullptr;
            const auto opcode = input.op == PtoOpcode::TLOAD ? std::string_view("TLOAD") : PtoOpcodeToKey(input.op);
            const fit::VfCurveKey curve{
                opcode, storage, "none", fit::ShapePath::Path2D, "POST_UPDATE", fit::TailKind::Full, key};
            if (!fit::TryLookupVfFormulaParam(curve, param)) {
                found = false;
                break;
            }
            parameters[part * 4] = param->param0;
            parameters[part * 4 + 1] = param->param1;
            parameters[part * 4 + 2] = param->param2;
            parameters[part * 4 + 3] = param->param3;
        }
        if (!found)
            continue;
        double value = 0;
        if (model == "linear") {
            for (size_t i = 0; i < features.size(); ++i)
                value += parameters[i] * features[i];
        } else if (model == "max_scatter_packed") {
            value = std::max(
                        parameters[0] * rows * q + parameters[1] * features[12] + parameters[2],
                        parameters[3] * rows * q + parameters[4] * features[12] + parameters[5]) +
                    parameters[6] * init;
        } else if (model == "max_blocks") {
            value =
                std::max(parameters[0] * rows * dataQ + parameters[1], parameters[2] * rows * dataQ + parameters[3]);
        } else if (model == "max_nested") {
            value = std::max(
                {parameters[0] * rows * q + parameters[1],
                 parameters[2] * rows * q + parameters[3] * rows + parameters[4],
                 parameters[5] * rows * q + parameters[6]});
        } else {
            // Empirical overlap envelope: startup/repeat work, row work, and
            // sustained repeat work. Not a sum of universal instruction costs.
            value = std::max(
                {parameters[0] * rows * q + parameters[1], parameters[2] * rows + parameters[3],
                 parameters[4] * rows * q + parameters[5]});
            if (model == "max_pipeline_init")
                value += parameters[6] * init;
        }
        cycles = fit::RoundToCycles(value);
        return cycles > 0;
    }
    return false;
}

} // namespace pto::mocker::lightweight::a5
