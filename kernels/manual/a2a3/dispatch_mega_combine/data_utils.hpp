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

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "tiling_builder.hpp"

struct RankFileSet {
    std::string x;
    std::string weight1;
    std::string weight2;
    std::string expert_idx;
    std::string scale1;
    std::string scale2;
    std::string probs;
    std::string x_active_mask;
    std::string expected_out;
};

struct AccuracyReport {
    bool pass = false;
    size_t mismatch_count = 0;
    size_t err_threshold = 0;
    size_t nan_or_inf_count = 0;
    double max_abs_err = 0.0;
    double max_rel_err = 0.0;
};

CaseConfig LoadCaseConfig(const std::string &case_json_path);
RankFileSet BuildRankFileSet(const std::string &case_dir, int rank);
std::vector<uint8_t> ReadBinaryFile(const std::string &path);
void WriteBinaryFile(const std::string &path, const void *data, size_t bytes);
AccuracyReport CompareFp16File(const std::vector<uint16_t> &expected, const std::vector<uint16_t> &actual, double atol,
                               double rtol);
