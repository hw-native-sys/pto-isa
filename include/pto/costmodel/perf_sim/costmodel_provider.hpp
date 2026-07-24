/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_PERF_SIM_COSTMODEL_PROVIDER_HPP
#define PTO_PERF_SIM_COSTMODEL_PROVIDER_HPP

#include "config.hpp"
#include "recorder.hpp"
#include "latency.hpp"
#include <pto/costmodel/lightweight_costmodel.hpp>
#include <string>

namespace pto::perf_sim {

// ── Magic number constants ──
static constexpr uint64_t kPercentMultiplier = 100;
static constexpr uint64_t kMatrixLatencyBase = 4;
static constexpr uint64_t kElemsPerMatrixUnit = 16;
static constexpr uint64_t kGMLatencyBase = 3;
static constexpr uint64_t kGMCycleMultiplier = 2;
static constexpr uint64_t kGMElemsPerUnit = 64;
static constexpr uint64_t kMTE1LatencyBase = 1;
static constexpr uint64_t kMTE1ElemsPerUnit = 64;
static constexpr uint64_t kDefaultLatencyBase = 2;
static constexpr uint64_t kDefaultElemsPerUnit = 32;

// ── Runtime context passed to costmodel ──

struct CostModelRuntimeCtx {
    uint32_t block_dim = 1;
    uint32_t npu_arch = 2201;
};

inline CostModelRuntimeCtx& GetCostModelCtx()
{
    static thread_local CostModelRuntimeCtx ctx;
    return ctx;
}

// ── String → lightweight::PtoOpcode mapping (X-macro, single source of truth) ──
// Enum names in lightweight::PtoOpcode match opcode strings exactly.

// clang-format off
#define PTO_PERF_SIM_OPCODE_LIST                                                                                      \
    X(TADD)                                                                                                           \
    X(TSUB)                                                                                                           \
    X(TMUL)                                                                                                           \
    X(TDIV)                                                                                                           \
    X(TADDS)                                                                                                          \
    X(TSUBS)                                                                                                          \
    X(TMULS)                                                                                                          \
    X(TDIVS) X(TMINS) X(TMAXS) X(TABS) X(TNEG) X(TEXP) X(TSQRT) X(TRSQRT) X(TLOG) X(TRELU) X(TLRELU) X(TNOT)          \
        X(TROWSUM) X(TROWMAX) X(TROWMIN) X(TROWPROD) X(TCOLSUM) X(TCOLMAX) X(TCOLMIN) X(TCOLPROD) X(TMATMUL) X(TGEMV) \
            X(TCVT) X(TMOV) X(TLOAD) X(TSTORE) X(TTRANS) X(TPREFETCH) X(TSORT32) X(TMRGSORT) X(TSEL) X(TSCATTER)      \
                X(TEXTRACT) X(TINSERT) X(TROWEXPAND) X(TCOLEXPAND) X(TLOADCONV)
// clang-format on

inline bool TryMapOpcode(const std::string& opcode, ::pto::mocker::lightweight::PtoOpcode& out)
{
#define X(name)                                            \
    if (opcode == #name) {                                 \
        out = ::pto::mocker::lightweight::PtoOpcode::name; \
        return true;                                       \
    }
    PTO_PERF_SIM_OPCODE_LIST
#undef X
    return false;
}

// ── String → lightweight::DType mapping (X-macro, paired str↔enum) ──

#define PTO_PERF_SIM_DTYPE_LIST \
    X("fp32", Float)            \
    X("fp16", Half)             \
    X("int8", Int8)             \
    X("int16", Int16) X("int32", Int32) X("uint8", Uint8) X("uint16", Uint16) X("uint32", Uint32) X("bf16", BFloat16)

inline ::pto::mocker::lightweight::DType MapDType(const std::string& dtype)
{
#define X(str, enum_val) \
    if (dtype == str)    \
        return ::pto::mocker::lightweight::DType::enum_val;
    PTO_PERF_SIM_DTYPE_LIST
#undef X
    return ::pto::mocker::lightweight::DType::Half;
}

// ── LightweightFormula backend ──

inline uint64_t EstimateLightweightCycles(const std::string& opcode, int rows, int cols, const std::string& dtype)
{
    using namespace ::pto::mocker::lightweight;

    PtoOpcode pto_op;
    if (!TryMapOpcode(opcode, pto_op))
        return 0;

    CostModelInput input{};
    input.op = pto_op;
    input.dtype = MapDType(dtype);
    input.rows = rows;
    input.cols = cols;
    input.data_size = static_cast<int64_t>(rows) * cols;

    if (pto_op == PtoOpcode::TMATMUL || pto_op == PtoOpcode::TGEMV) {
        input.k = cols;
    }

    CostModelResult result = EstimateCycles(input);
    if (result.cycles > 0.0)
        return static_cast<uint64_t>(result.cycles);
    return 0;
}

// ── Fallback for unsupported instructions ──

inline uint64_t FallbackCycles(const std::string& opcode, int rows, int cols)
{
    uint64_t elems = static_cast<uint64_t>(rows) * cols;
    PipeStage stage = StaticPipeStageLookup(opcode);

    if (stage == PipeStage::Scalar)
        return 1;
    if (stage == PipeStage::Matrix)
        return kMatrixLatencyBase + elems / kElemsPerMatrixUnit;
    if (IsGMAccessPipe(stage))
        return kGMLatencyBase + elems * kGMCycleMultiplier / kGMElemsPerUnit;
    if (stage == PipeStage::MTE1)
        return kMTE1LatencyBase + elems / kMTE1ElemsPerUnit;
    return kDefaultLatencyBase + elems / kDefaultElemsPerUnit;
}

// ── Unified estimation entry (used when CCE mock trace is unavailable) ──

inline uint64_t EstimateInstrCycles(const std::string& opcode, int rows, int cols, const std::string& dtype)
{
    uint64_t cycles = EstimateLightweightCycles(opcode, rows, cols, dtype);
    return cycles > 0 ? cycles : FallbackCycles(opcode, rows, cols);
}

} // namespace pto::perf_sim

#endif
