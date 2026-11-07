/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#include "data_utils.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <stdexcept>

namespace {

std::string GetJsonScalar(const std::string& text, const std::string& key)
{
    const std::string token = "\"" + key + "\":";
    const size_t pos = text.find(token);
    if (pos == std::string::npos) {
        throw std::runtime_error("missing key: " + key);
    }
    size_t begin = pos + token.size();
    while (begin < text.size() && std::isspace(static_cast<unsigned char>(text[begin])) != 0) {
        ++begin;
    }
    if (begin >= text.size()) {
        throw std::runtime_error("invalid value for key: " + key);
    }
    if (text[begin] == '"') {
        const size_t end = text.find('"', begin + 1);
        if (end == std::string::npos) {
            throw std::runtime_error("unterminated string for key: " + key);
        }
        return text.substr(begin + 1, end - begin - 1);
    }
    const size_t end = text.find_first_of(",}\n", begin);
    return text.substr(begin, end - begin);
}

bool TryGetJsonScalar(const std::string& text, const std::string& key, std::string& value)
{
    if (text.find("\"" + key + "\":") == std::string::npos) {
        return false;
    }
    value = GetJsonScalar(text, key);
    return true;
}

uint32_t ParseJsonUInt(const std::string& text, const std::string& key)
{
    std::string value = GetJsonScalar(text, key);
    const size_t end = value.find_last_not_of(" \t\n\r\v\f");
    if (end == std::string::npos) {
        throw std::runtime_error("empty integer for key: " + key);
    }
    value.resize(end + 1U);
    if (value.find_first_not_of("0123456789") != std::string::npos) {
        throw std::runtime_error("invalid unsigned integer for key: " + key);
    }
    const unsigned long long number = std::stoull(value);
    if (number > std::numeric_limits<uint32_t>::max()) {
        throw std::runtime_error("integer exceeds uint32 for key: " + key);
    }
    return static_cast<uint32_t>(number);
}

double ParseJsonDouble(const std::string& text, const std::string& key, double default_value)
{
    std::string value;
    if (!TryGetJsonScalar(text, key, value)) {
        return default_value;
    }
    size_t parsed = 0U;
    const double number = std::stod(value, &parsed);
    if (value.find_first_not_of(" \t\n\r\v\f", parsed) != std::string::npos || !std::isfinite(number) || number < 0.0) {
        throw std::runtime_error("invalid non-negative finite number for key: " + key);
    }
    return number;
}

float Bf16ToFloat(uint16_t value)
{
    const uint32_t bits = static_cast<uint32_t>(value) << 16U;
    float result = 0.0f;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

} // namespace

std::vector<uint8_t> ReadBinaryFile(const std::string& path)
{
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("failed to open: " + path);
    }
    file.seekg(0, std::ios::end);
    const size_t bytes = static_cast<size_t>(file.tellg());
    file.seekg(0, std::ios::beg);
    std::vector<uint8_t> data(bytes);
    file.read(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(bytes));
    return data;
}

void WriteBinaryFile(const std::string& path, const void* data, size_t bytes)
{
    std::ofstream file(path, std::ios::binary | std::ios::trunc);
    if (!file) {
        throw std::runtime_error("failed to open for write: " + path);
    }
    file.write(reinterpret_cast<const char*>(data), static_cast<std::streamsize>(bytes));
}

CaseConfig LoadCaseConfig(const std::string& case_json_path)
{
    const std::vector<uint8_t> raw = ReadBinaryFile(case_json_path);
    const std::string text(raw.begin(), raw.end());
    CaseConfig cfg;
    cfg.data_cache_version = ParseJsonUInt(text, "data_cache_version");
    cfg.m = ParseJsonUInt(text, "m");
    cfg.k = ParseJsonUInt(text, "k");
    cfg.n = ParseJsonUInt(text, "n");
    cfg.topk = ParseJsonUInt(text, "topk");
    cfg.expert_per_rank = ParseJsonUInt(text, "expert_per_rank");
    cfg.world_size = ParseJsonUInt(text, "world_size");
    cfg.max_output_size = ParseJsonUInt(text, "max_output_size");
    cfg.compare_atol = ParseJsonDouble(text, "compare_atol", 1e-3);
    cfg.compare_rtol = ParseJsonDouble(text, "compare_rtol", 1e-3);
    cfg.input_tokens_all_ranks =
        ParseJsonDouble(text, "input_tokens_all_ranks", static_cast<double>(cfg.m) * cfg.world_size);
    cfg.routed_tokens_all_ranks =
        ParseJsonDouble(text, "routed_tokens_all_ranks", static_cast<double>(cfg.m) * cfg.topk * cfg.world_size);
    cfg.remote_routed_tokens_all_ranks = ParseJsonDouble(text, "remote_routed_tokens_all_ranks", 0.0);
    cfg.compute_flops_all_ranks =
        ParseJsonDouble(text, "compute_flops_all_ranks", cfg.routed_tokens_all_ranks * 3.0 * cfg.k * cfg.n);
    cfg.comm_bytes_all_ranks = ParseJsonDouble(text, "comm_bytes_all_ranks", 0.0);
    return cfg;
}

RankFileSet BuildRankFileSet(const std::string& case_dir, int rank)
{
    const std::string prefix = case_dir + "/rank" + std::to_string(rank) + "_";
    return RankFileSet{
        prefix + "x.bin",      prefix + "weight1.bin", prefix + "weight2.bin", prefix + "expert_idx.bin",
        prefix + "scale1.bin", prefix + "scale2.bin",  prefix + "probs.bin",   prefix + "expected_out.bin",
    };
}

AccuracyReport CompareBf16File(
    const std::vector<uint16_t>& expected, const std::vector<uint16_t>& actual, double atol, double rtol)
{
    if (!std::isfinite(atol) || !std::isfinite(rtol) || atol < 0.0 || rtol < 0.0) {
        throw std::invalid_argument("accuracy tolerances must be finite and non-negative");
    }
    AccuracyReport report;
    if (expected.size() != actual.size()) {
        report.pass = false;
        report.mismatch_count = 1;
        return report;
    }
    report.err_threshold =
        rtol >= 1.0 ? expected.size() : static_cast<size_t>(static_cast<double>(expected.size()) * rtol);

    for (size_t i = 0; i < expected.size(); ++i) {
        const float expected_value = Bf16ToFloat(expected[i]);
        const float actual_value = Bf16ToFloat(actual[i]);
        const bool invalid = std::isnan(actual_value) || std::isinf(actual_value) || std::isnan(expected_value) ||
                             std::isinf(expected_value);
        const double abs_err = invalid ? std::numeric_limits<double>::infinity() :
                                         std::fabs(static_cast<double>(actual_value) - expected_value);
        const double tolerance = atol + rtol * std::fabs(static_cast<double>(expected_value));
        const double expectedAbs = std::fabs(static_cast<double>(expected_value));
        const double relDenom = expectedAbs > 1e-7 ? expectedAbs : 1e-7;
        const double rel_err = invalid ? std::numeric_limits<double>::infinity() : abs_err / relDenom;

        if (invalid) {
            report.nan_or_inf_count += 1;
        }
        if (invalid || abs_err > tolerance) {
            report.mismatch_count += 1;
        }

        if (!invalid) {
            report.max_abs_err = std::max(report.max_abs_err, abs_err);
            report.max_rel_err = std::max(report.max_rel_err, rel_err);
        }
    }

    report.pass = (report.mismatch_count <= report.err_threshold) && (report.nan_or_inf_count == 0);
    return report;
}
