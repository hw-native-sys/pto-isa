/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/
#ifndef PTO_MOCKER_A2A3_HYBRID_CONTEXT_H_
#define PTO_MOCKER_A2A3_HYBRID_CONTEXT_H_

#include <cstdint>
#include <string_view>

#include <pto/costmodel/trace.hpp>

namespace pto::mocker::hybrid {

enum class ModelKind {
    NONE,
    CCE_TARGET_PIPE_PROJECTION,
    PTO_COMPOSITE_ANALYTIC,
    PTO_FORMULA,
};

struct PtoCycleContext {
    std::string_view opcode;
    int rows = 0;
    int cols = 0;
    std::string_view dst_dtype;
    std::string_view src_dtype;
    uint64_t stub_cycles = 0;
    const PtoInstrRecord* trace = nullptr;
};

struct CycleEstimate {
    uint64_t cycles = 0;
    ModelKind kind = ModelKind::NONE;

    explicit operator bool() const { return cycles != 0; }
};

inline uint64_t CountCceCalls(const PtoInstrRecord& trace, std::string_view name, uint64_t chargedCycles = 0)
{
    uint64_t count = 0;
    for (const auto& call : trace.cce_calls) {
        if (call.name == name && (chargedCycles == 0 || call.cycles == chargedCycles)) {
            ++count;
        }
    }
    return count;
}

inline uint64_t DTypeBytes(std::string_view dtype)
{
    if (dtype == "int8" || dtype == "uint8") {
        return sizeof(uint8_t);
    }
    if (dtype == "fp16" || dtype == "bf16" || dtype == "int16" || dtype == "uint16") {
        return sizeof(uint16_t);
    }
    if (dtype == "fp32" || dtype == "int32" || dtype == "uint32") {
        return sizeof(uint32_t);
    }
    return 0;
}

} // namespace pto::mocker::hybrid

#endif
