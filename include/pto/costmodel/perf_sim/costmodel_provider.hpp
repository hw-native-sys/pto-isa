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
#include <string_view>

namespace pto::perf_sim {

#if defined(__NPU_ARCH__)
inline constexpr uint32_t kDefaultNpuArch = __NPU_ARCH__;
#else
inline constexpr uint32_t kDefaultNpuArch = 2201;
#endif
inline constexpr uint32_t kAscend950Arch = 3101;
inline constexpr uint32_t kAscend950UrmaArch = 3510;

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
    uint32_t npu_arch = kDefaultNpuArch;
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
    X(TRECIP)                                                                                                         \
    X(TMAX)                                                                                                           \
    X(TMIN)                                                                                                           \
    X(TAND)                                                                                                           \
    X(TOR)                                                                                                            \
    X(TANDS) X(TORS) X(TXOR) X(TXORS) X(TFMOD) X(TFMODS) X(TREM) X(TREMS) X(TPOW) X(TPOWS) X(TMADD) X(TMULA)        \
        X(TPRELU) X(TPARTADD) X(TPARTMUL) X(TPARTMAX) X(TPARTMIN)                                                     \
    X(TADDS)                                                                                                          \
    X(TSUBS)                                                                                                          \
    X(TMULS)                                                                                                          \
    X(TDIVS) X(TMINS) X(TMAXS) X(TABS) X(TNEG) X(TEXP) X(TSQRT) X(TRSQRT) X(TLOG) X(TRELU) X(TLRELU) X(TNOT)          \
        X(TROWSUM) X(TROWMAX) X(TROWMIN) X(TROWPROD) X(TCOLSUM) X(TCOLMAX) X(TCOLMIN) X(TCOLPROD) X(TMATMUL) X(TGEMV) \
            X(TCVT) X(TMOV) X(TLOAD) X(TSTORE) X(TTRANS) X(TPREFETCH) X(TSORT32) X(TMRGSORT) X(TSEL) X(TSCATTER)      \
                X(TEXTRACT) X(TINSERT) X(TROWEXPAND) X(TCOLEXPAND) X(TLOADCONV)                                      \
    X(TAXPY) X(TCMP) X(TCMPS) X(TROWARGMAX) X(TROWARGMIN) X(TCOLARGMAX) X(TCOLARGMIN)                                \
        X(TROWEXPANDADD) X(TROWEXPANDSUB) X(TROWEXPANDMUL) X(TROWEXPANDDIV) X(TROWEXPANDMAX)                         \
            X(TROWEXPANDMIN) X(TROWEXPANDEXPDIF) X(TCOLEXPANDADD) X(TCOLEXPANDSUB) X(TCOLEXPANDMUL)                  \
                X(TCOLEXPANDDIV) X(TCOLEXPANDMAX) X(TCOLEXPANDMIN) X(TCOLEXPANDEXPDIF) X(TEXPANDS) X(TSELS)          \
                    X(TCONCAT) X(TGATHER) X(TGATHERB) X(TFILLPAD) X(TFILLPAD_EXPAND) X(TTRI) X(TDEQUANT) X(TRESHAPE) \
                    X(TSHL) X(TSHR) X(TSHLS) X(TSHRS) X(TPARTARGMAX) X(TPARTARGMIN) X(TCI) \
                    X(TINTERLEAVE) X(TDEINTERLEAVE) X(TMATMUL_ACC) X(TMATMUL_BIAS) X(TMATMUL_MX) \
                    X(TGEMV_ACC) X(TGEMV_BIAS) X(TGEMV_MX)
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
    using DType = ::pto::mocker::lightweight::DType;
    if (dtype == "fp8_e4m3")
        return DType::Float8E4M3;
    if (dtype == "fp8_e5m2")
        return DType::Float8E5M2;
    if (dtype == "hif8")
        return DType::HFloat8;
    if (dtype == "fp4_e1m2")
        return DType::Float4E1M2;
    if (dtype == "fp4_e2m1")
        return DType::Float4E2M1;
#define X(str, enum_val)                                    \
    if (dtype == str) {                                     \
        return ::pto::mocker::lightweight::DType::enum_val; \
    }
    PTO_PERF_SIM_DTYPE_LIST
#undef X
    return ::pto::mocker::lightweight::DType::Half;
}

// ── LightweightFormula backend ──

struct MatrixFitGeometry {
    bool present = false;
    int64_t m = 0;
    int64_t k = 0;
    int64_t n = 0;
    ::pto::mocker::lightweight::DType left{};
    ::pto::mocker::lightweight::DType right{};
    ::pto::mocker::lightweight::DType accumulator{};
};

struct LightweightEstimateOptions {
    MatrixFitGeometry matrix{};
    int physical_rows = 0;
    int physical_cols = 0;
    int valid_rows = 0;
    int valid_cols = 0;
    ::pto::mocker::lightweight::A5ShapePathHint a5_shape_path = ::pto::mocker::lightweight::A5ShapePathHint::Auto;
    ::pto::mocker::lightweight::DType dtype2 = ::pto::mocker::lightweight::DType::Float;
    bool has_dtype2 = false;
    ::pto::VFImplKind vf_impl_kind = ::pto::VFImplKind::VFIMPL_DEFAULT;
    ::pto::RoundMode round_mode = ::pto::RoundMode::CAST_NONE;
    ::pto::SaturationMode saturation_mode = ::pto::SaturationMode::ON;
    std::string_view a5_op_params{};
    ::pto::mocker::lightweight::TransposeGeometry transpose{};
    bool has_mask_pattern = false;
    ::pto::MaskPattern mask_pattern = ::pto::MaskPattern::P1111;
    ::pto::GatherAxis gather_axis = ::pto::GatherAxis::GATHER_ROW;
    ::pto::ScatterAxis scatter_axis = ::pto::ScatterAxis::SCATTER_ROW;
    int dst_rows = 0;
    int dst_cols = 0;
    int dst_valid_rows = 0;
    int dst_valid_cols = 0;
    int concat_src0_cols = 0;
    int concat_src1_cols = 0;
    bool concat_dense_sources = false;
};

inline uint64_t EstimateLightweightCycles(
    const std::string& opcode, int rows, int cols, const std::string& src_dtype,
    const std::string& dst_dtype = "unknown", int64_t inner_dimension = 0,
    ::pto::mocker::lightweight::TransferTileType tile_type = ::pto::mocker::lightweight::TransferTileType::Unknown,
    const LightweightEstimateOptions& options = {})
{
    using namespace ::pto::mocker::lightweight;

    PtoOpcode pto_op;
    if (!TryMapOpcode(opcode, pto_op))
        return 0;

    CostModelInput input{};
    input.op = pto_op;
    const uint32_t npu_arch = GetCostModelCtx().npu_arch;
    input.arch =
        (npu_arch == kAscend950Arch || npu_arch == kAscend950UrmaArch) ? CostModelArch::A5 : CostModelArch::A2A3;
    input.dtype = MapDType(src_dtype);
    input.dtype2 = options.has_dtype2 ? options.dtype2 : input.dtype;
    input.has_dtype2 = options.has_dtype2;
    if (dst_dtype != "unknown") {
        input.dst_dtype = MapDType(dst_dtype);
    }
    // A5 vector paths use physical strides to select their implementation.
    // Legacy formulas and matrix M/N dimensions describe valid work instead.
    const bool usePhysicalShape = input.arch == CostModelArch::A5 && !a5::IsMatrixOpcode(pto_op);
    input.rows = usePhysicalShape && options.physical_rows > 0 ? options.physical_rows : rows;
    input.cols = usePhysicalShape && options.physical_cols > 0 ? options.physical_cols : cols;
    input.valid_rows = options.valid_rows > 0 ? options.valid_rows : rows;
    input.valid_cols = options.valid_cols > 0 ? options.valid_cols : cols;
    input.a5_shape_path = options.a5_shape_path;
    input.vf_impl_kind = options.vf_impl_kind;
    input.round_mode = options.round_mode;
    input.saturation_mode = options.saturation_mode;
    input.a5_op_params = options.a5_op_params;
    input.transpose = options.transpose;
    input.has_mask_pattern = options.has_mask_pattern;
    input.mask_pattern = options.mask_pattern;
    input.gather_axis = options.gather_axis;
    input.scatter_axis = options.scatter_axis;
    input.dst_rows = options.dst_rows;
    input.dst_cols = options.dst_cols;
    input.dst_valid_rows = options.dst_valid_rows;
    input.dst_valid_cols = options.dst_valid_cols;
    input.concat_src0_cols = options.concat_src0_cols;
    input.concat_src1_cols = options.concat_src1_cols;
    input.concat_dense_sources = options.concat_dense_sources;
    input.data_size = static_cast<int64_t>(input.valid_rows) * input.valid_cols;
    input.tile_type = tile_type;

    if (a5::IsMatrixOpcode(pto_op)) {
        input.k = inner_dimension > 0 ? inner_dimension : cols;
    }

    if (input.arch == CostModelArch::A5 && a5::IsMatrixOpcode(pto_op) && options.matrix.present) {
        input.rows = options.matrix.m;
        input.k = options.matrix.k;
        input.cols = options.matrix.n;
        input.dtype = options.matrix.left;
        input.dtype2 = options.matrix.right;
        input.has_dtype2 = true;
        input.dst_dtype = options.matrix.accumulator;
    }
    CostModelResult result = EstimateCycles(input);
    if (result.cycles > 0.0)
        return static_cast<uint64_t>(result.cycles);
    return 0;
}

// ── Fallback for unsupported instructions ──

inline uint64_t FallbackCycles(const std::string& opcode, int rows, int cols)
{
    if (opcode == "TROWEXPAND") {
        constexpr uint64_t kIntegerRowCycles = 14;
        constexpr uint64_t kFixedEventCycles = 37;
        // The non-VBRCB A2/A3 implementation serializes every output row through
        // V->S and S->V event pairs before issuing vector_dup.  Existing fitted
        // TROWEXPAND entries reduce to approximately 14 cycles per row plus a
        // 37-cycle fixed cost.  Keep that behavior for integer dtypes and column
        // sizes not present in the formula table instead of falling through to
        // the generic element-throughput estimate, which misses the row events.
        return kIntegerRowCycles * static_cast<uint64_t>(rows) + kFixedEventCycles;
    }
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

inline uint64_t EstimateInstrCycles(
    const std::string& opcode, int rows, int cols, const std::string& src_dtype,
    const std::string& dst_dtype = "unknown", int64_t inner_dimension = 0,
    ::pto::mocker::lightweight::TransferTileType tile_type = ::pto::mocker::lightweight::TransferTileType::Unknown,
    const LightweightEstimateOptions& options = {})
{
    uint64_t cycles =
        EstimateLightweightCycles(opcode, rows, cols, src_dtype, dst_dtype, inner_dimension, tile_type, options);
    const auto arch = GetCostModelCtx().npu_arch;
    if (arch == kAscend950Arch || arch == kAscend950UrmaArch) {
        // An unsupported A5 fit key must not silently become a generic estimate.
        return cycles;
    }
    return cycles > 0 ? cycles : FallbackCycles(opcode, rows, cols);
}

} // namespace pto::perf_sim

#endif
