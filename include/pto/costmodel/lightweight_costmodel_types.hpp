/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/
#ifndef PTO_COSTMODEL_LIGHTWEIGHT_TYPES_H_
#define PTO_COSTMODEL_LIGHTWEIGHT_TYPES_H_

#include <cstdint>
#include <string_view>

#include <pto/common/type.hpp>
#include <pto/costmodel/a2a3/formula_costmodel/formula_backend_transfer.hpp>
#include <pto/costmodel/arch_config.hpp>

namespace pto::mocker::lightweight {

enum class PtoOpcode {
    TADD,
    TSUB,
    TMUL,
    TDIV,
    TRECIP,
    TADDS,
    TSUBS,
    TMULS,
    TDIVS,
    TMINS,
    TMAXS,
    TABS,
    TNEG,
    TEXP,
    TSQRT,
    TRSQRT,
    TLOG,
    TRELU,
    TLRELU,
    TNOT,
    TROWSUM,
    TROWMAX,
    TROWMIN,
    TROWPROD,
    TCOLSUM,
    TCOLMAX,
    TCOLMIN,
    TCOLPROD,
    TMATMUL,
    TGEMV,
    TCVT,
    TMOV,
    TLOAD,
    TSTORE,
    TTRANS,
    TSORT32,
    TMRGSORT,
    TSEL,
    TSCATTER,
    TEXTRACT,
    TINSERT,
    TROWEXPAND,
    TCOLEXPAND,
    TLOADCONV,
    TPREFETCH,
    TMAX,
    TMIN,
    TAND,
    TOR,
    TANDS,
    TORS,
    TXOR,
    TXORS,
    TFMOD,
    TFMODS,
    TREM,
    TREMS,
    TPOW,
    TPOWS,
    TMADD,
    TMULA,
    TPRELU,
    TPARTADD,
    TPARTMUL,
    TPARTMAX,
    TPARTMIN,
    TAXPY,
    TCMP,
    TCMPS,
    TROWARGMAX,
    TROWARGMIN,
    TCOLARGMAX,
    TCOLARGMIN,
    TROWEXPANDADD,
    TROWEXPANDSUB,
    TROWEXPANDMUL,
    TROWEXPANDDIV,
    TROWEXPANDMAX,
    TROWEXPANDMIN,
    TROWEXPANDEXPDIF,
    TCOLEXPANDADD,
    TCOLEXPANDSUB,
    TCOLEXPANDMUL,
    TCOLEXPANDDIV,
    TCOLEXPANDMAX,
    TCOLEXPANDMIN,
    TCOLEXPANDEXPDIF,
    TEXPANDS,
    TSELS,
    TCONCAT,
    TGATHER,
    TGATHERB,
    TFILLPAD,
    TFILLPAD_EXPAND,
    TTRI,
    TDEQUANT,
    TRESHAPE,
    TSHL,
    TSHR,
    TSHLS,
    TSHRS,
    TPARTARGMAX,
    TPARTARGMIN,
    TCI,
    TINTERLEAVE,
    TDEINTERLEAVE,
    TMATMUL_ACC,
    TMATMUL_BIAS,
    TMATMUL_MX,
    TGEMV_ACC,
    TGEMV_BIAS,
    TGEMV_MX,
    TFILLPAD_INPLACE,
    TRANDOM,
    TQUANT,
    THISTOGRAM,
    TIMG2COL,
    SETFMATRIX,
    SET_IMG2COL_RPT,
    SET_IMG2COL_PADDING,
    SET_QUANT_SCALAR,
    SET_QUANT_VECTOR,
    COUNT,
};

enum class DType : uint8_t {
    Float,
    Half,
    Int8,
    Int16,
    Int32,
    Uint8,
    Uint16,
    Uint32,
    BFloat16,
    Float8E4M3,
    Float8E5M2,
    HFloat8,
    Float4E1M2,
    Float4E2M1,
};

using MemLayout = ::pto::Layout;
using RoundMode = ::pto::RoundMode;
using SaturationMode = ::pto::SaturationMode;
using TransferTileType = fit::TransferTileType;
using VFImplKind = ::pto::VFImplKind;

enum class CostModelArch : uint8_t {
    A2A3,
    A5,
};

enum class A5ShapePathHint : uint8_t {
    // Legacy geometry-only inference. Runtime valid_cols == cols does not prove
    // that a multi-row tile has compile-time ValidCol == Cols: dynamic-valid
    // tiles can execute a row loop even when their runtime extent is full.
    // Direct callers must pass the actual kernel path when that distinction
    // matters. Public PTO wrappers derive it from the tile template metadata.
    Auto,
    Path1D,
    Path2D,
};

// Explicit source/destination geometry: transpose dispatch uses physical source
// dimensions, while its loops use valid dimensions and its indices use strides.
struct TransposeGeometry {
    bool ordinary_row_major = false;
    int64_t src_rows = 0;
    int64_t src_cols = 0;
    int64_t src_valid_rows = 0;
    int64_t src_valid_cols = 0;
    int64_t src_row_stride = 0;
    int64_t dst_rows = 0;
    int64_t dst_cols = 0;
    int64_t dst_valid_rows = 0;
    int64_t dst_valid_cols = 0;
    int64_t dst_row_stride = 0;
};

// Observable API metadata, not tensor contents or an inferred cache/queue state.
// Zero valid extents remain distinguishable from missing metadata.
struct OperandGeometry {
    int64_t rows = 0;
    int64_t cols = 0;
    int64_t valid_rows = 0;
    int64_t valid_cols = 0;
    int64_t row_stride = 0;
    bool has_address = false;
    uint64_t address = 0;
};

// A declaration about the NPU binary, not a property of the host argument value.
// Unknown must not select a newly calibrated comparison code-generation path.
enum class ComparisonCodegen { Unknown, ConstantFolded, RuntimeDispatch };

struct MetadataGeometry {
    enum class Kind { Missing, Padding, PaddingExpand, Partial, Select, Triangle, Comparison };
    Kind kind = Kind::Missing;
    bool supported_layout = false;
    OperandGeometry dst{};
    OperandGeometry src0{};
    OperandGeometry src1{};
    OperandGeometry mask{};
    DType data_dtype = DType::Float;
    A5ShapePathHint source_shape_path = A5ShapePathHint::Auto;
    bool runtime_valid_shape = false;
    bool upper = false;
    int64_t diagonal = 0;
    int comparison_mode = -1;
    ComparisonCodegen comparison_codegen = ComparisonCodegen::Unknown;
    bool scalar_rhs = false;
    double scalar_value = 0;
};

// Performance assumptions, not values observed by dereferencing tensor data.
enum class DefaultInputProfile { None, ZeroIndex, IdentityBlocks, WarmSameAddressApproximate };

struct DefaultInputGeometry {
    bool present = false;
    bool enabled = true;
    bool supported_layout = false;
    bool runtime_valid_shape = false;
    DefaultInputProfile profile = DefaultInputProfile::None;
    OperandGeometry dst{}, src{}, index{};
    DType index_dtype = DType::Uint32;
};

struct CostModelInput {
    PtoOpcode op;
    DType dtype;
    int64_t rows;
    int64_t cols;
    CostModelArch arch = CostModelArch::A2A3;

    DType dtype2 = DType::Float;
    bool has_dtype2 = false;
    int64_t k = 0;

    DType dst_dtype = DType::Float;
    RoundMode round_mode = RoundMode::CAST_NONE;

    MemLayout layout = MemLayout::ND;

    int64_t dst_rows = 0;
    int64_t dst_cols = 0;

    int64_t block_len = 0;
    int64_t src_count = 1;

    int64_t channels = 0;
    int64_t height = 0;
    int64_t width = 0;

    TransferTileType tile_type = TransferTileType::Unknown;
    int64_t data_size = 0;

    int64_t valid_rows = 0;
    int64_t valid_cols = 0;
    A5ShapePathHint a5_shape_path = A5ShapePathHint::Auto;
    VFImplKind vf_impl_kind = VFImplKind::VFIMPL_DEFAULT;
    SaturationMode saturation_mode = SaturationMode::ON;
    std::string_view a5_op_params{};
    TransposeGeometry transpose{};
    // Pattern overloads are distinct from indexed gather/scatter, including P1111.
    bool has_mask_pattern = false;
    MaskPattern mask_pattern = MaskPattern::P1111;
    GatherAxis gather_axis = GatherAxis::GATHER_ROW;
    ScatterAxis scatter_axis = ScatterAxis::SCATTER_ROW;
    int64_t dst_valid_rows = 0;
    int64_t dst_valid_cols = 0;
    int64_t concat_src0_cols = 0;
    int64_t concat_src1_cols = 0;
    bool concat_dense_sources = false;
    MetadataGeometry metadata{};
    DefaultInputGeometry default_input{};
};

struct CostModelResult {
    double cycles = 0.0;
    long double latency_us = 0.0L;
    DefaultInputProfile assumed_input_profile = DefaultInputProfile::None;
};

struct PredictRuntimeConfig {
    long double frequency_mhz = 0.0L;
    evaluator::BandwidthTable bandwidth_bytes_per_us{};
};

} // namespace pto::mocker::lightweight

#endif
