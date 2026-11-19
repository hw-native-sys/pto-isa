/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/
#ifndef PTO_INSTR_HPP
#define PTO_INSTR_HPP
#include <bit>
#include <string_view>
#include <tuple>
#include <type_traits>

// Intentionally reuse the common PTO include guard so this header can act as a
// drop-in replacement when <pto/pto-inst.hpp> selects it for __COSTMODEL.

#include "pto/common/debug.h"
#include "pto/common/event.hpp"
#include "pto/common/tassign_check.hpp"
#include "pto/common/utils.hpp"
#include "pto/common/pto_instr_impl.hpp"
#ifdef __COSTMODEL
#include "pto/costmodel/trace.hpp"
#include "pto/costmodel/perf_sim/recorder.hpp"
#include "pto/costmodel/perf_sim/tile_dep_tracker.hpp"
#include "pto/costmodel/perf_sim/latency.hpp"
#include "pto/costmodel/perf_sim/costmodel_provider.hpp"
#include "pto/costmodel/a2a3/hybrid_costmodel.hpp"
namespace perf_sim = ::pto::perf_sim;
#endif
#if !defined(PTO_COMM_NOT_SUPPORTED)
#include "pto/comm/pto_comm_inst.hpp"
#endif

#define TSTORE_FP_IMPL TSTORE_IMPL
#define TEXTRACT_FP_IMPL TEXTRACT_IMPL
#define TINSERT_FP_IMPL TINSERT_IMPL
#define TMOV_FP_IMPL TMOV_IMPL

#ifdef __COSTMODEL
namespace pto::mocker {

inline uint64_t GetCurrentPtoInstrCycles()
{
    auto& trace = GetMutableTrace();
    if (trace.executed_pto.empty()) {
        return 0;
    }

    if (trace.active_pto_stack.size() == 1) {
        FlushAllPendingTailsExceptVector();
    }

    if (trace.active_pto_stack.empty()) {
        return trace.executed_pto.back().total_cycles;
    }
    return trace.executed_pto[trace.active_pto_stack.back()].total_cycles;
}

template <typename T>
inline void InjectTileCycles(T& obj)
{
    if constexpr (requires { obj.SetLastCycle(0.0f); }) {
        obj.SetLastCycle(static_cast<float>(GetCurrentPtoInstrCycles()));
    }
}

// Record one PTO instruction to the pipeline simulator.

template <typename T>
inline constexpr const char* CostModelTileDTypeString()
{
    using TileType = std::remove_cvref_t<T>;
    if constexpr (requires { typename TileType::DType; }) {
        using TileDType = typename TileType::DType;
        // Some host compilers alias bfloat16_t to half. In that configuration
        // the alias cannot identify BF16 and must not relabel every FP16 tile.
        if constexpr (std::is_same_v<TileDType, bfloat16_t> && !std::is_same_v<bfloat16_t, half>) {
            const uint32_t npuArch = perf_sim::GetCostModelCtx().npu_arch;
            if (npuArch == perf_sim::kAscend950Arch || npuArch == perf_sim::kAscend950UrmaArch) {
                return "bf16";
            }
        }
    }
    return perf_sim::TileTraits<TileType>::dtype_str();
}

// Helper: try to extract dimensions + dtype from a single tile; returns true if successful.
template <typename T>
inline bool TryTileInfo(int& rows, int& cols, std::string& dtype, T&& tile)
{
    if constexpr (requires {
                      tile.GetValidRow();
                      tile.GetValidCol();
                  }) {
        rows = static_cast<int>(tile.GetValidRow());
        cols = static_cast<int>(tile.GetValidCol());
        dtype = CostModelTileDTypeString<T>();
        return true;
    } else if constexpr (requires {
                             tile.Rows;
                             tile.Cols;
                         }) {
        rows = tile.Rows;
        cols = tile.Cols;
        dtype = CostModelTileDTypeString<T>();
        return true;
    }
    return false;
}

struct CostModelTileInfo {
    int physical_rows = 0;
    int physical_cols = 0;
    int valid_rows = 0;
    int valid_cols = 0;
    std::string dtype;
    bool static_full_cols = false;
    lightweight::A5ShapePathHint a5_shape_path = lightweight::A5ShapePathHint::Auto;
};

template <typename T>
inline bool TryCostModelTileInfo(CostModelTileInfo& info, T&& tile)
{
    using Tile = std::remove_cvref_t<T>;
    if constexpr (requires {
                      typename Tile::DType;
                      Tile::Rows;
                      Tile::Cols;
                      tile.GetValidRow();
                      tile.GetValidCol();
                  }) {
        info.physical_rows = static_cast<int>(Tile::Rows);
        info.physical_cols = static_cast<int>(Tile::Cols);
        info.valid_rows = static_cast<int>(tile.GetValidRow());
        info.valid_cols = static_cast<int>(tile.GetValidCol());
        info.dtype = CostModelTileDTypeString<T>();
        if constexpr (requires { Tile::ValidCol; }) {
            info.static_full_cols = Tile::ValidCol == Tile::Cols;
            constexpr bool kUsesOneDimensionalVfPath = Tile::Rows == 1 || Tile::ValidCol == Tile::Cols;
            info.a5_shape_path =
                kUsesOneDimensionalVfPath ? lightweight::A5ShapePathHint::Path1D : lightweight::A5ShapePathHint::Path2D;
        }
        return true;
    }
    return false;
}

template <typename T>
inline void ExtractFirstCostModelTileInfo(CostModelTileInfo& info, T&& tile)
{
    TryCostModelTileInfo(info, std::forward<T>(tile));
}

template <typename T, typename T2, typename... Rest>
inline void ExtractFirstCostModelTileInfo(CostModelTileInfo& info, T&& first, T2&& second, Rest&&... rest)
{
    if (!TryCostModelTileInfo(info, std::forward<T>(first))) {
        ExtractFirstCostModelTileInfo(info, std::forward<T2>(second), std::forward<Rest>(rest)...);
    }
}

inline bool UsesSourceWorkShape(std::string_view opcode)
{
    return opcode == "TROWSUM" || opcode == "TROWMAX" || opcode == "TROWMIN" || opcode == "TROWPROD" ||
           opcode == "TROWARGMAX" || opcode == "TROWARGMIN" || opcode == "TCOLSUM" || opcode == "TCOLMAX" ||
           opcode == "TCOLMIN" || opcode == "TCOLPROD" || opcode == "TCOLARGMAX" || opcode == "TCOLARGMIN";
}

inline bool IsIndexReductionOpcode(std::string_view opcode)
{
    return opcode == "TROWARGMAX" || opcode == "TROWARGMIN" || opcode == "TCOLARGMAX" || opcode == "TCOLARGMIN";
}

struct A5SequenceFitTag {
    std::string_view key;
};

template <typename D, typename S>
inline lightweight::TransposeGeometry TransposeFitGeometry(const D& dst, const S& src)
{
    lightweight::TransposeGeometry geometry;
    // Convolution Tiles have a separate implementation and must not be treated
    // as ordinary two-dimensional transposes.
    if constexpr (!is_conv_tile_v<D> && !is_conv_tile_v<S> && requires {
                      D::RowStride;
                      S::RowStride;
                  }) {
        geometry = {
            D::Loc == TileType::Vec && S::Loc == TileType::Vec && D::isRowMajor && S::isRowMajor &&
                std::is_same_v<typename D::DType, typename S::DType>,
            S::Rows,
            S::Cols,
            src.GetValidRow(),
            src.GetValidCol(),
            S::RowStride,
            D::Rows,
            D::Cols,
            dst.GetValidRow(),
            dst.GetValidCol(),
            D::RowStride};
    }
    return geometry;
}

inline bool IsA5SequenceFit()
{
    const auto arch = perf_sim::GetCostModelCtx().npu_arch;
    return arch == perf_sim::kAscend950Arch || arch == perf_sim::kAscend950UrmaArch;
}

template <typename T>
inline lightweight::DType MatrixFitDType()
{
    using D = typename T::DType;
    using lightweight::DType;
    static_assert(
        std::is_same_v<D, half> || std::is_same_v<D, bfloat16_t> || std::is_same_v<D, float> ||
            std::is_same_v<D, int8_t> || std::is_same_v<D, int32_t> || std::is_same_v<D, float8_e4m3_t> ||
            std::is_same_v<D, float8_e5m2_t> || std::is_same_v<D, hifloat8_t> || std::is_same_v<D, float4_e1m2x2_t> ||
            std::is_same_v<D, float4_e2m1x2_t>,
        "Unsupported matrix costmodel operand dtype");
    if constexpr (std::is_same_v<D, float8_e4m3_t>)
        return DType::Float8E4M3;
    if constexpr (std::is_same_v<D, float8_e5m2_t>)
        return DType::Float8E5M2;
    if constexpr (std::is_same_v<D, hifloat8_t>)
        return DType::HFloat8;
    if constexpr (std::is_same_v<D, float4_e1m2x2_t>)
        return DType::Float4E1M2;
    if constexpr (std::is_same_v<D, float4_e2m1x2_t>)
        return DType::Float4E2M1;
    return perf_sim::MapDType(CostModelTileDTypeString<T>());
}

template <typename C, typename A, typename B>
inline perf_sim::MatrixFitGeometry MatrixFitShape(const C&, const A& left, const B& right, bool gemv, bool mx = false)
{
    // Ordinary GEMV reads K from B; MX GEMV reads K from A, as on device.
    return {
        true,
        gemv ? 1 : left.GetValidRow(),
        gemv && !mx ? right.GetValidRow() : left.GetValidCol(),
        right.GetValidCol(),
        MatrixFitDType<A>(),
        MatrixFitDType<B>(),
        MatrixFitDType<C>()};
}

template <typename A, typename B>
inline constexpr bool kLegacyMadTypes =
    std::is_same_v<typename A::DType, typename B::DType> &&
    (std::is_same_v<typename A::DType, half> || std::is_same_v<typename A::DType, float> ||
     std::is_same_v<typename A::DType, bfloat16_t> || std::is_same_v<typename A::DType, int8_t>);

template <typename D, typename S>
inline bool HasSameDenseGeometry(const D& dst, const S& src)
{
    return D::Loc == TileType::Vec && S::Loc == TileType::Vec && D::isRowMajor && S::isRowMajor && D::Rows == S::Rows &&
           D::Cols == S::Cols && dst.GetValidRow() == D::Rows && dst.GetValidCol() == D::Cols &&
           src.GetValidRow() == S::Rows && src.GetValidCol() == S::Cols;
}

template <typename D, typename... S>
inline A5SequenceFitTag DenseSequenceTag(const D& dst, const S&... src)
{
    const bool types = (std::is_same_v<typename D::DType, typename S::DType> && ...);
    return {
        types && HasSameDenseGeometry(dst, dst) && (HasSameDenseGeometry(dst, src) && ...) ? "dense" : "unsupported"};
}

template <typename D, typename S0, typename S1, typename I, typename I0, typename I1>
inline A5SequenceFitTag PartialArgSequenceTag(
    const D& dst, const S0& s0, const S1& s1, const I& idx, const I0& idx0, const I1& idx1)
{
    const bool indices = std::is_same_v<typename I::DType, int32_t> && std::is_same_v<typename I0::DType, int32_t> &&
                         std::is_same_v<typename I1::DType, int32_t>;
    return {
        indices && DenseSequenceTag(dst, s0, s1).key == "dense" && HasSameDenseGeometry(dst, idx) &&
                HasSameDenseGeometry(dst, idx0) && HasSameDenseGeometry(dst, idx1) ?
            "dense_index32" :
            "unsupported"};
}

template <HistByte Byte, typename D, typename S, typename I>
inline A5SequenceFitTag HistogramFitTag(const D& dst, const S& src, const I&)
{
    constexpr int64_t kByteBins = 256;
    constexpr int64_t kSourceElementsPerLoad = 256;
    const auto cols = src.GetValidCol();
    const bool supported =
        Byte == HistByte::BYTE_3 && std::is_same_v<typename S::DType, uint32_t> &&
        std::is_same_v<typename D::DType, uint32_t> && std::is_same_v<typename I::DType, uint8_t> &&
        D::Loc == TileType::Vec && S::Loc == TileType::Vec && I::Loc == TileType::Vec && D::isRowMajor &&
        S::isRowMajor && D::RowStride == D::Cols && S::RowStride == S::Cols && D::Rows == S::Rows &&
        D::Cols == kByteBins && src.GetValidRow() == S::Rows && dst.GetValidRow() == D::Rows &&
        dst.GetValidCol() == kByteBins && cols > 0 &&
        S::Cols == ((cols + kSourceElementsPerLoad - 1) / kSourceElementsPerLoad) * kSourceElementsPerLoad;
    // BYTE_3 does not consume filter indices. Lower-byte passes require
    // separate curves and index geometry, and must not use this tag.
    return {supported ? "u32_byte3_padded256" : "unsupported"};
}

template <auto Quant, typename D, typename S, typename P>
inline A5SequenceFitTag QuantFitTag(const D& dst, const S& src, const P& scale, const P* offset)
{
    constexpr int64_t kScaleBlockElements = 32 / sizeof(float);
    const bool asymmetric = Quant == QuantType::INT8_ASYM;
    const bool types = std::is_same_v<typename S::DType, float> && std::is_same_v<typename P::DType, float> &&
                       ((Quant == QuantType::INT8_SYM && std::is_same_v<typename D::DType, int8_t>) ||
                        (asymmetric && std::is_same_v<typename D::DType, uint8_t>));
    const auto parameterGeometry = [&](const P& value) {
        return P::Loc == TileType::Vec && P::isRowMajor && P::RowStride == P::Cols && P::Rows == 1 &&
               P::Cols == ((S::Rows + kScaleBlockElements - 1) / kScaleBlockElements) * kScaleBlockElements &&
               value.GetValidRow() == P::Rows && value.GetValidCol() == P::Cols;
    };
    const bool supported = types && HasSameDenseGeometry(dst, src) && D::RowStride == D::Cols &&
                           S::RowStride == S::Cols && parameterGeometry(scale) &&
                           (!asymmetric || (offset && parameterGeometry(*offset)));
    return {supported ? (asymmetric ? "dense_fp32_int8_asym" : "dense_fp32_int8_sym") : "unsupported"};
}

// Shadow only the explicit B-bank configuration needed by the calibrated
// manual image path. Configuration execution has a separately calibrated domain.
inline auto& ImageManualFitState() { return perf_sim::GetCostModelCtx().imageManual; }

template <int Kind, auto Mode, typename S>
inline A5SequenceFitTag ImageConfigFitTag(const S& src)
{
    bool supported = (Mode == SetFmatrixMode::FMATRIX_A_MANUAL || Mode == SetFmatrixMode::FMATRIX_B_MANUAL) &&
                     std::is_same_v<typename S::DType, half> && S::Loc == TileType::Mat &&
                     S::layout == Layout::NC1HWC0 && S::totalDimCount == 5 && src.GetShape(0) == 1 &&
                     src.GetShape(1) == 2 && src.GetShape(2) == 1 && src.GetShape(3) == 16 && src.GetShape(4) == 16 &&
                     src.GetFmapH() == 1 && src.GetFilterH() == 1 && src.GetFilterW() == 1 && src.GetStrideH() == 1 &&
                     src.GetStrideW() == 1 && src.GetDilationH() == 1 && src.GetDilationW() == 1 &&
                     src.GetChannelSize() == 32 && !src.GetTranspose() && src.GetRepeatTime() == 1 &&
                     src.GetRepeatMode() == 0 && src.GetRepeatStride() == 0 && src.GetDstMposition() == 0;
    for (int i = 0; i < S::totalDimCount; ++i) {
        supported = supported && src.GetShape(i) == src.GetValidShape(i);
    }
    for (int i = 0; i < 4; ++i) {
        supported = supported && src.GetPadList(i) == 0;
    }
    const float pad = static_cast<float>(src.GetPadValue());
    supported = supported && (Kind == 0 ? src.GetFmapW() >= 8 && src.GetFmapW() <= 65 : src.GetFmapW() == 16) &&
                (Kind == 1 ? src.GetDstStride() >= 1 && src.GetDstStride() <= 10 : src.GetDstStride() == 1) &&
                (Kind == 2 ? pad >= 0 && pad <= 9 && pad == static_cast<int>(pad) : pad == 0);
    return {supported ? "image_config_manual_half" : "unsupported_image_config"};
}

template <auto Mode, typename D, typename S>
inline A5SequenceFitTag ImageFitTag(const D& dst, const S& src, uint16_t posM, uint16_t posK)
{
    const auto& state = ImageManualFitState();
    const auto rows = dst.GetValidRow();
    const auto cols = dst.GetValidCol();
    bool supported =
        Mode == SetFmatrixMode::FMATRIX_B_MANUAL && std::is_same_v<typename D::DType, half> &&
        std::is_same_v<typename S::DType, half> && D::Loc == TileType::Left && !D::isRowMajor &&
        D::SFractal == SLayout::RowMajor && D::SFractalSize == TileConfig::fractalABSize &&
        D::Compact == CompactMode::Null && D::Rows == rows && D::Cols == cols && rows >= 16 && rows <= 96 &&
        rows % 16 == 0 && cols >= 16 && cols <= 128 && cols % 16 == 0 && S::Loc == TileType::Mat &&
        S::layout == Layout::NC1HWC0 && S::totalDimCount == 5 && src.GetShape(0) == 1 && src.GetShape(1) == cols / 16 &&
        src.GetShape(2) == 1 && src.GetShape(3) == rows && src.GetShape(4) == 16 && src.GetFilterH() == 1 &&
        src.GetFilterW() == 1 && src.GetStrideH() == 1 && src.GetStrideW() == 1 && src.GetDilationH() == 1 &&
        src.GetDilationW() == 1 && !src.GetTranspose() && src.GetChannelSize() == cols && posM == 0 && posK == 0 &&
        state.fmapH == 1 && state.fmapW == rows && state.zeroPadding && state.zeroPadValue && state.repeatOne &&
        state.dstStride == rows / 16;
    for (int i = 0; i < S::totalDimCount; ++i) {
        supported = supported && src.GetValidShape(i) == src.GetShape(i);
    }
    return {supported ? "image_left_manual_1x1" : "unsupported_image"};
}

template <bool Insert, auto Relu, typename D, typename S, typename F>
inline A5SequenceFitTag FpSliceFitTag(const D& dst, const S& src, const F& fp, uint16_t row, uint16_t col)
{
    const int64_t rows = src.GetValidRow();
    const int64_t cols = src.GetValidCol() - (Insert ? 0 : 64);
    const bool supported =
        std::is_same_v<typename S::DType, float> && std::is_same_v<typename D::DType, half> &&
        std::is_same_v<typename F::DType, uint64_t> && S::Loc == TileType::Acc && D::Loc == TileType::Vec &&
        F::Loc == TileType::Scaling && !S::isRowMajor && S::SFractal == SLayout::RowMajor &&
        S::SFractalSize == TileConfig::fractalCSize && S::Compact == CompactMode::Null && D::isRowMajor &&
        D::SFractal == SLayout::NoneBox && D::RowStride == D::Cols && F::isRowMajor &&
        F::SFractal == SLayout::NoneBox && F::Rows == 1 && F::Cols == S::Cols && fp.GetValidRow() == 1 &&
        fp.GetValidCol() == src.GetValidCol() && S::Rows == ((rows + 15) / 16) * 16 && S::Cols == src.GetValidCol() &&
        D::Rows == rows + 1 && D::Cols == cols + 96 && dst.GetValidRow() == (Insert ? D::Rows : rows) &&
        dst.GetValidCol() == (Insert ? D::Cols : cols) && rows >= 16 && rows <= 64 && rows % 8 == 0 && cols >= 32 &&
        cols <= 256 && cols % 16 == 0 && row == 0 && col == 64 && Relu == ReluPreMode::NormalRelu;
    // Only the measured Acc-to-Vec vector-quantization path; never a Vec slice or Acc-to-Mat copy.
    return {supported ? "fp_acc_vec_offset64_relu" : "unsupported_fp_slice"};
}

template <typename Small, typename Large>
inline A5SequenceFitTag SliceFitTag(const Small& small, const Large& large, uint16_t row, uint16_t col)
{
    // ConvTile describes multidimensional convolution storage, not a Vec slice.
    if constexpr (is_conv_tile_v<Small> || is_conv_tile_v<Large>) {
        return {"unsupported"};
    } else {
        constexpr int64_t kMeasuredOffsetBytes = 32;
        constexpr int64_t kMeasuredOffsetRows = 1;
        // A static ValidCol selects a DMA or single-chunk specialization instead
        // of the dynamic-column VF used to collect these curves.
        const bool supported =
            Small::ValidCol == DYNAMIC && Large::ValidCol == DYNAMIC && Small::Loc == TileType::Vec &&
            Large::Loc == TileType::Vec && Small::isRowMajor && Large::isRowMajor && Small::RowStride == Small::Cols &&
            Large::RowStride == Large::Cols && std::is_same_v<typename Small::DType, typename Large::DType> &&
            row == kMeasuredOffsetRows && col * sizeof(typename Small::DType) == kMeasuredOffsetBytes &&
            Large::Rows == Small::Rows + row && Large::Cols == Small::Cols + col &&
            small.GetValidRow() == Small::Rows && small.GetValidCol() == Small::Cols &&
            large.GetValidRow() == Large::Rows && large.GetValidCol() == Large::Cols;
        return {supported ? "dynamic_slice_offset1_32B" : "unsupported"};
    }
}

template <uint16_t Rounds, typename D>
inline A5SequenceFitTag RandomFitTag(const D& dst)
{
    if (!HasSameDenseGeometry(dst, dst)) {
        return {"unsupported"};
    }
    return {Rounds == 7 ? "dense_rounds7" : "dense_rounds10"};
}

template <typename D, typename S>
inline A5SequenceFitTag InplacePaddingTag(const D& dst, const S& src)
{
    constexpr int64_t kMeasuredSuffixBytes = 32;
    const bool supported = D::Loc == TileType::Vec && S::Loc == TileType::Vec && D::isRowMajor && S::isRowMajor &&
                           D::RowStride == D::Cols && S::RowStride == S::Cols && D::Rows == S::Rows &&
                           D::Cols == S::Cols && std::is_same_v<typename D::DType, typename S::DType> &&
                           D::PadVal == PadValue::Zero && dst.GetValidRow() == src.GetValidRow() &&
                           dst.GetValidCol() == src.GetValidCol() &&
                           (D::Cols - src.GetValidCol()) * sizeof(typename S::DType) == kMeasuredSuffixBytes;
    return {supported ? "suffix32B_no_bottom" : "unsupported"};
}

template <typename T>
inline constexpr bool IsObservableVecGeometry = requires {
    T::Loc;
    T::Rows;
    T::Cols;
    T::RowStride;
    T::isRowMajor;
    T::SFractal;
};

template <typename T>
inline lightweight::OperandGeometry ObservableOperandGeometry(const T& tile)
{
    if constexpr (IsObservableVecGeometry<T>) {
        lightweight::OperandGeometry geometry{T::Rows, T::Cols, tile.GetValidRow(), tile.GetValidCol(), T::RowStride};
        if constexpr (requires {
                          tile.HasCostModelAssignedAddress();
                          tile.GetCostModelAssignedAddress();
                      }) {
            geometry.has_address = tile.HasCostModelAssignedAddress();
            geometry.address = tile.GetCostModelAssignedAddress();
        }
        return geometry;
    }
    return {};
}

template <typename T>
inline constexpr bool IsMeasuredMetadataLayout()
{
    if constexpr (IsObservableVecGeometry<T>) {
        using D = typename T::DType;
        constexpr bool knownDtype =
            std::is_same_v<D, float> || std::is_same_v<D, half> || std::is_same_v<D, bfloat16_t> ||
            std::is_same_v<D, int8_t> || std::is_same_v<D, uint8_t> || std::is_same_v<D, int16_t> ||
            std::is_same_v<D, uint16_t> || std::is_same_v<D, int32_t> || std::is_same_v<D, uint32_t>;
        return T::Loc == TileType::Vec && T::isRowMajor && T::SFractal == SLayout::NoneBox && T::RowStride == T::Cols &&
               knownDtype;
    }
    return false;
}

template <typename T>
inline constexpr bool HasRuntimeMetadataShape()
{
    if constexpr (requires {
                      T::ValidRow;
                      T::ValidCol;
                  })
        return T::ValidRow == -1 && T::ValidCol == -1;
    return false;
}

template <typename D, typename A, typename B>
inline lightweight::MetadataGeometry PartialGeometryTag(const D& dst, const A& src0, const B& src1)
{
    lightweight::MetadataGeometry g;
    g.kind = lightweight::MetadataGeometry::Kind::Partial;
    g.runtime_valid_shape =
        HasRuntimeMetadataShape<D>() && HasRuntimeMetadataShape<A>() && HasRuntimeMetadataShape<B>();
    g.supported_layout = IsMeasuredMetadataLayout<D>() && IsMeasuredMetadataLayout<A>() &&
                         IsMeasuredMetadataLayout<B>() && std::is_same_v<typename D::DType, typename A::DType> &&
                         std::is_same_v<typename D::DType, typename B::DType>;
    g.dst = ObservableOperandGeometry(dst);
    g.src0 = ObservableOperandGeometry(src0);
    g.src1 = ObservableOperandGeometry(src1);
    return g;
}

template <lightweight::DefaultInputProfile Profile, typename D, typename S, typename I>
inline lightweight::DefaultInputGeometry DefaultIndexedGeometryTag(const D& dst, const S& src, const I& index)
{
    lightweight::DefaultInputGeometry g;
    g.present = true;
    g.profile = Profile;
    g.dst = ObservableOperandGeometry(dst);
    g.src = ObservableOperandGeometry(src);
    g.index = ObservableOperandGeometry(index);
    g.index_dtype = perf_sim::MapDType(CostModelTileDTypeString<I>());
    g.supported_layout = IsMeasuredMetadataLayout<D>() && IsMeasuredMetadataLayout<S>() &&
                         IsMeasuredMetadataLayout<I>() && std::is_same_v<typename D::DType, typename S::DType> &&
                         (std::is_same_v<typename I::DType, int16_t> || std::is_same_v<typename I::DType, uint16_t> ||
                          std::is_same_v<typename I::DType, int32_t> || std::is_same_v<typename I::DType, uint32_t>);
    g.runtime_valid_shape =
        HasRuntimeMetadataShape<D>() && HasRuntimeMetadataShape<S>() && HasRuntimeMetadataShape<I>();
    return g;
}

template <TLoadL2Hint Hint, typename D, typename G>
inline lightweight::DefaultInputGeometry DefaultLoadGeometryTag(const D& dst, G& src)
{
    lightweight::DefaultInputGeometry g;
    g.present = true;
    g.profile = lightweight::DefaultInputProfile::WarmSameAddressApproximate;
    g.dst = ObservableOperandGeometry(dst);
    g.runtime_valid_shape = HasRuntimeMetadataShape<D>();
    if constexpr (requires {
                      G::layout;
                      typename G::RawDType;
                      src.GetShape(0);
                      src.GetStride(0);
                      src.data();
                  }) {
        g.supported_layout = IsMeasuredMetadataLayout<D>() && Hint == TLoadL2Hint::NormalFirstVictim &&
                             G::layout == Layout::ND && std::is_same_v<typename D::DType, typename G::RawDType> &&
                             src.GetShape(0) == 1 && src.GetShape(1) == 1 && src.GetShape(2) == 1 &&
                             src.GetStride(4) == 1;
        g.src.rows = g.src.valid_rows = src.GetShape(3);
        g.src.cols = g.src.valid_cols = src.GetShape(4);
        g.src.row_stride = src.GetStride(3);
        g.src.has_address = src.data() != nullptr;
        g.src.address = reinterpret_cast<uintptr_t>(src.data());
    }
    return g;
}

template <typename D>
inline constexpr bool IsApproximateDefaultLoad()
{
    // Do not instantiate ordinary Tile metadata for convolution tiles.
    if constexpr (is_conv_tile_v<D>) {
        return false;
    } else {
        return D::Loc == TileType::Vec && D::Rows == 16 && D::Cols == 16 &&
               (std::is_same_v<typename D::DType, half> || std::is_same_v<typename D::DType, float> ||
                std::is_same_v<typename D::DType, int32_t>);
    }
}

template <bool Expand, typename D, typename S>
inline lightweight::MetadataGeometry PaddingGeometryTag(const D& dst, const S& src)
{
    lightweight::MetadataGeometry g;
    g.kind = Expand ? lightweight::MetadataGeometry::Kind::PaddingExpand : lightweight::MetadataGeometry::Kind::Padding;
    g.runtime_valid_shape = HasRuntimeMetadataShape<D>() && HasRuntimeMetadataShape<S>();
    g.supported_layout = IsMeasuredMetadataLayout<D>() && IsMeasuredMetadataLayout<S>() &&
                         std::is_same_v<typename D::DType, typename S::DType> && D::PadVal == PadValue::Zero;
    g.dst = ObservableOperandGeometry(dst);
    g.src0 = ObservableOperandGeometry(src);
    return g;
}

template <typename D, typename S, typename R>
inline lightweight::MetadataGeometry ComparisonGeometryTag(const D& dst, const S& src, const R& rhs, CmpMode mode)
{
    lightweight::MetadataGeometry g;
    g.kind = lightweight::MetadataGeometry::Kind::Comparison;
    g.runtime_valid_shape = HasRuntimeMetadataShape<D>() && HasRuntimeMetadataShape<S>();
    g.supported_layout =
        IsMeasuredMetadataLayout<D>() && IsMeasuredMetadataLayout<S>() && std::is_same_v<typename D::DType, uint8_t>;
    g.dst = ObservableOperandGeometry(dst);
    g.src0 = ObservableOperandGeometry(src);
    g.comparison_mode = static_cast<int>(mode);
    g.comparison_codegen = perf_sim::GetCostModelCtx().comparison_codegen;
    g.data_dtype = perf_sim::MapDType(CostModelTileDTypeString<S>());
    CostModelTileInfo sourceInfo;
    TryCostModelTileInfo(sourceInfo, src);
    g.source_shape_path = sourceInfo.a5_shape_path;
    if constexpr (IsObservableVecGeometry<R>) {
        g.runtime_valid_shape = g.runtime_valid_shape && HasRuntimeMetadataShape<R>();
        g.supported_layout =
            g.supported_layout && IsMeasuredMetadataLayout<R>() && std::is_same_v<typename S::DType, typename R::DType>;
        g.src1 = ObservableOperandGeometry(rhs);
    } else {
        g.scalar_rhs = true;
        if constexpr (std::is_same_v<R, bfloat16_t> && !std::is_same_v<bfloat16_t, half>) {
            const uint32_t bits = uint32_t(std::bit_cast<uint16_t>(rhs)) << 16;
            g.scalar_value = double(std::bit_cast<float>(bits));
        } else
            g.scalar_value = double(rhs);
    }
    return g;
}

inline bool HasExplicitComparisonCodegen()
{
    return IsA5SequenceFit() &&
           perf_sim::GetCostModelCtx().comparison_codegen != lightweight::ComparisonCodegen::Unknown;
}

template <bool Binary = false, typename D, typename S>
inline bool UseObservableComparison(const D& dst, const S& src)
{
    // Explicit declarations must not fall back to a legacy, unidentified path,
    // including unsupported dtypes and equal-width destination layouts.
    if (HasExplicitComparisonCodegen())
        return true;
    // Preserve the established legacy equal-width mock route. The new route
    // describes actual packed uint8 output, including unknown-address rejects.
    using T = typename S::DType;
    constexpr bool measured = (std::is_same_v<T, bfloat16_t> && !std::is_same_v<bfloat16_t, half>) ||
                              std::is_same_v<T, float> || std::is_same_v<T, int32_t> ||
                              (!Binary && (std::is_same_v<T, uint32_t> || std::is_same_v<T, uint16_t> ||
                                           std::is_same_v<T, int8_t> || std::is_same_v<T, uint8_t>));
    return measured && IsA5SequenceFit() && (D::Cols != S::Cols || dst.GetValidCol() != src.GetValidCol());
}

template <typename D, typename S>
inline bool PreserveLegacyScalarComparison(const D& dst, const S& src, typename S::DType scalar, CmpMode mode)
{
    if (HasExplicitComparisonCodegen())
        return false;
    using T = typename S::DType;
    if constexpr (std::is_same_v<T, float> || std::is_same_v<T, int32_t> || std::is_same_v<T, uint16_t>) {
        if (mode != CmpMode::EQ || (scalar != T(1) && scalar != T(2)))
            return false;
        const auto d = ObservableOperandGeometry(dst);
        const auto s = ObservableOperandGeometry(src);
        // Existing mock-only EQ profiles also accept host buffers. Preserve
        // that old route when device addresses are not observable; never
        // reinterpret host pointers as UB addresses for the new fitted route.
        return !d.has_address || !s.has_address || d.address >= 262144 || s.address >= 262144;
    }
    return false;
}

template <int Direction, typename D>
inline lightweight::MetadataGeometry TriangleGeometryTag(const D& dst, int diagonal)
{
    lightweight::MetadataGeometry g;
    g.kind = lightweight::MetadataGeometry::Kind::Triangle;
    g.runtime_valid_shape = HasRuntimeMetadataShape<D>();
    g.supported_layout = IsMeasuredMetadataLayout<D>() && (Direction == 0 || Direction == 1);
    g.dst = ObservableOperandGeometry(dst);
    g.upper = Direction == 1;
    g.diagonal = diagonal;
    return g;
}

template <typename D, typename M, typename S>
inline lightweight::MetadataGeometry SelectGeometryTag(const D& dst, const M& mask, const S& src0, const S& src1)
{
    lightweight::MetadataGeometry g;
    g.kind = lightweight::MetadataGeometry::Kind::Select;
    g.runtime_valid_shape =
        HasRuntimeMetadataShape<D>() && HasRuntimeMetadataShape<M>() && HasRuntimeMetadataShape<S>();
    g.supported_layout = IsMeasuredMetadataLayout<D>() && IsMeasuredMetadataLayout<M>() &&
                         IsMeasuredMetadataLayout<S>() && std::is_same_v<typename D::DType, typename S::DType> &&
                         std::is_same_v<typename M::DType, uint8_t>;
    g.dst = ObservableOperandGeometry(dst);
    g.src0 = ObservableOperandGeometry(src0);
    g.src1 = ObservableOperandGeometry(src1);
    g.mask = ObservableOperandGeometry(mask);
    g.data_dtype = perf_sim::MapDType(CostModelTileDTypeString<S>());
    CostModelTileInfo sourceInfo;
    TryCostModelTileInfo(sourceInfo, src0);
    g.source_shape_path = sourceInfo.a5_shape_path;
    return g;
}

template <typename T>
inline void UpdateLightweightEstimateOptions(perf_sim::LightweightEstimateOptions& options, T&& value)
{
    using Value = std::remove_cvref_t<T>;
    if constexpr (std::is_same_v<Value, perf_sim::MatrixFitGeometry>) {
        options.matrix = value;
    } else if constexpr (std::is_same_v<Value, lightweight::TransposeGeometry>) {
        options.transpose = value;
    } else if constexpr (std::is_same_v<Value, lightweight::MetadataGeometry>) {
        options.metadata = value;
    } else if constexpr (std::is_same_v<Value, lightweight::DefaultInputGeometry>) {
        options.default_input = value;
    } else if constexpr (std::is_same_v<Value, A5SequenceFitTag>) {
        options.a5_op_params = value.key;
    } else if constexpr (std::is_same_v<Value, RoundMode>) {
        options.round_mode = value;
    } else if constexpr (std::is_same_v<Value, MaskPattern>) {
        options.has_mask_pattern = true;
        options.mask_pattern = value;
    } else if constexpr (std::is_same_v<Value, GatherAxis>) {
        options.gather_axis = value;
    } else if constexpr (std::is_same_v<Value, ScatterAxis>) {
        options.scatter_axis = value;
    } else if constexpr (std::is_same_v<Value, SaturationMode>) {
        options.saturation_mode = value;
    } else if constexpr (std::is_same_v<Value, VFImplKind>) {
        options.vf_impl_kind = value;
        if (value == VFImplKind::VFIMPL_1D_NO_POST_UPDATE || value == VFImplKind::VFIMPL_1D_POST_UPDATE) {
            options.a5_shape_path = lightweight::A5ShapePathHint::Path1D;
        } else if (value == VFImplKind::VFIMPL_2D_NO_POST_UPDATE || value == VFImplKind::VFIMPL_2D_POST_UPDATE) {
            options.a5_shape_path = lightweight::A5ShapePathHint::Path2D;
        }
    }
}

// Recursive: try tiles in order, extract dims+dtype from first one that has them.
template <typename T>
inline void ExtractFirstTileInfo(int& rows, int& cols, std::string& dtype, T&& tile)
{
    TryTileInfo(rows, cols, dtype, std::forward<T>(tile));
}

template <typename T, typename T2, typename... Rest>
inline void ExtractFirstTileInfo(int& rows, int& cols, std::string& dtype, T&& first, T2&& second, Rest&&... rest)
{
    if (!TryTileInfo(rows, cols, dtype, std::forward<T>(first))) {
        ExtractFirstTileInfo(rows, cols, dtype, std::forward<T2>(second), std::forward<Rest>(rest)...);
    }
}

inline constexpr lightweight::TransferTileType MapTransferTileType(TileType tileType)
{
    switch (tileType) {
        case TileType::Vec:
            return lightweight::TransferTileType::VecTile;
        case TileType::Mat:
            return lightweight::TransferTileType::MatTile;
        case TileType::Acc:
            return lightweight::TransferTileType::AccTile;
        case TileType::Left:
            return lightweight::TransferTileType::LeftTile;
        case TileType::Right:
            return lightweight::TransferTileType::RightTile;
        case TileType::Bias:
            return lightweight::TransferTileType::BiasTile;
        case TileType::Scaling:
            return lightweight::TransferTileType::ScalingTile;
        default:
            return lightweight::TransferTileType::Unknown;
    }
}

template <typename T>
inline constexpr lightweight::TransferTileType CostModelTileType()
{
    using Tile = std::remove_cvref_t<T>;
    return MapTransferTileType(perf_sim::SafeGetTileLoc<Tile>());
}

template <typename First, typename... Rest>
inline lightweight::TransferTileType ResolveCostModelTransferTileType(const char* opcode)
{
    if (std::string_view(opcode) == "TLOAD") {
        return CostModelTileType<First>();
    }
    if constexpr (sizeof...(Rest) > 0) {
        if (std::string_view(opcode) == "TSTORE") {
            using Src = std::tuple_element_t<0, std::tuple<Rest...>>;
            return CostModelTileType<Src>();
        }
    }
    return lightweight::TransferTileType::Unknown;
}

inline void RecordInstr(const char* opcode, auto&& first_tile, auto&&... rest_tiles)
{
    // Look up pipeline stage from opcode name and tile types (TLOAD/TSTORE need tile routing)
    perf_sim::PipeStage stage = perf_sim::ResolvePipeStageArgs(opcode, first_tile, rest_tiles...);

    // 1. Track data dependencies (cross-pipe waits + signal event for this instr)
    auto dep = perf_sim::TileDepTracker::TrackByAddr(opcode, stage, first_tile, rest_tiles...);

    // 2. Build InstrRecord
    perf_sim::InstrRecord r;
    r.opcode = opcode;
    r.stage = stage;
    r.signal_event = dep.signal_event;
    for (int i = 0; i < dep.wait_count && i < perf_sim::InstrRecord::MAX_WAIT_EVENTS; ++i) {
        r.wait_events[i] = dep.wait_events[i];
    }
    r.wait_count = dep.wait_count;

    // Tile dimensions + dtype: try each tile argument in order, use first one with info.
    // For TSTORE(dst=GlobalData, src=TileData), this skips GlobalData and uses TileData.
    // Reduction outputs are 1 x cols. Their work is determined by the source
    // rows x cols, so select the first source tile rather than the destination.
    // This is also the shape contract used by the reduce costmodels.
    if constexpr (sizeof...(rest_tiles) > 0) {
        if (UsesSourceWorkShape(opcode)) {
            if constexpr (sizeof...(rest_tiles) >= 3) {
                if (IsIndexReductionOpcode(opcode)) {
                    auto restTuple = std::forward_as_tuple(rest_tiles...);
                    TryTileInfo(r.rows, r.cols, r.dtype, std::get<1>(restTuple));
                } else {
                    ExtractFirstTileInfo(r.rows, r.cols, r.dtype, rest_tiles...);
                }
            } else {
                ExtractFirstTileInfo(r.rows, r.cols, r.dtype, rest_tiles...);
            }
        } else {
            ExtractFirstTileInfo(r.rows, r.cols, r.dtype, first_tile, rest_tiles...);
        }
    } else {
        ExtractFirstTileInfo(r.rows, r.cols, r.dtype, first_tile);
    }

    int dstRows = 0;
    int dstCols = 0;
    std::string dstDtype;
    TryTileInfo(dstRows, dstCols, dstDtype, first_tile);
    int srcRows = 0;
    int srcCols = 0;
    std::string srcDtype;
    if constexpr (sizeof...(rest_tiles) > 0) {
        if constexpr (sizeof...(rest_tiles) >= 3) {
            if (IsIndexReductionOpcode(opcode)) {
                auto restTuple = std::forward_as_tuple(rest_tiles...);
                TryTileInfo(srcRows, srcCols, srcDtype, std::get<1>(restTuple));
            } else {
                ExtractFirstTileInfo(srcRows, srcCols, srcDtype, rest_tiles...);
            }
        } else {
            ExtractFirstTileInfo(srcRows, srcCols, srcDtype, rest_tiles...);
        }
    }
    const auto transferTileType =
        ResolveCostModelTransferTileType<decltype(first_tile), decltype(rest_tiles)...>(opcode);
    const int64_t innerDimension = std::string_view(opcode) == "TMATMUL" ? srcCols : 0;

    CostModelTileInfo modelTileInfo;
    if constexpr (sizeof...(rest_tiles) > 0) {
        // Comparison destinations pack bits; their width is not the input work size.
        if (UsesSourceWorkShape(opcode) || std::string_view(opcode) == "TCMPS") {
            if constexpr (sizeof...(rest_tiles) >= 3) {
                if (IsIndexReductionOpcode(opcode)) {
                    auto restTuple = std::forward_as_tuple(rest_tiles...);
                    TryCostModelTileInfo(modelTileInfo, std::get<1>(restTuple));
                } else {
                    ExtractFirstCostModelTileInfo(modelTileInfo, rest_tiles...);
                }
            } else {
                ExtractFirstCostModelTileInfo(modelTileInfo, rest_tiles...);
            }
        } else {
            ExtractFirstCostModelTileInfo(modelTileInfo, first_tile, rest_tiles...);
        }
    } else {
        ExtractFirstCostModelTileInfo(modelTileInfo, first_tile);
    }
    perf_sim::LightweightEstimateOptions estimateOptions{
        .physical_rows = modelTileInfo.physical_rows,
        .physical_cols = modelTileInfo.physical_cols,
        .valid_rows = modelTileInfo.valid_rows,
        .valid_cols = modelTileInfo.valid_cols,
        .a5_shape_path = modelTileInfo.a5_shape_path,
    };
    if constexpr (sizeof...(rest_tiles) >= 2) {
        if (std::string_view(opcode) == "TGATHER" || std::string_view(opcode) == "TSCATTER") {
            auto restTuple = std::forward_as_tuple(rest_tiles...);
            CostModelTileInfo indexTileInfo;
            if (TryCostModelTileInfo(indexTileInfo, std::get<1>(restTuple))) {
                estimateOptions.dtype2 = perf_sim::MapDType(indexTileInfo.dtype);
                estimateOptions.has_dtype2 = true;
            }
        }
        if (std::string_view(opcode) == "TSORT32") {
            estimateOptions.a5_op_params = sizeof...(rest_tiles) >= 3 ? "with_tmp" : "no_tmp_subrepeat";
        }
    }
    (UpdateLightweightEstimateOptions(estimateOptions, rest_tiles), ...);
    if (std::string_view(opcode) == "TCVT") {
        CostModelTileInfo sourceInfo;
        CostModelTileInfo destinationInfo;
        if constexpr (sizeof...(rest_tiles) > 0) {
            ExtractFirstCostModelTileInfo(sourceInfo, rest_tiles...);
        }
        TryCostModelTileInfo(destinationInfo, first_tile);
        // Match implTCVT: both operand layouts must permit the flat loop.
        // VFIMPL_1D_POST_UPDATE on padded tiles still executes a 2D loop.
        const bool flatOperands = (sourceInfo.physical_rows == 1 && destinationInfo.physical_rows == 1) ||
                                  (sourceInfo.static_full_cols && destinationInfo.static_full_cols);
        estimateOptions.a5_shape_path =
            flatOperands ? lightweight::A5ShapePathHint::Path1D : lightweight::A5ShapePathHint::Path2D;
        // Opt into dense conversion fits only with both operand geometries known.
        // Do not infer a destination stride from the source work shape.
        if constexpr (sizeof...(rest_tiles) > 0) {
            auto operands = std::forward_as_tuple(rest_tiles...);
            using D = std::remove_cvref_t<decltype(first_tile)>;
            using S = std::remove_cvref_t<decltype(std::get<0>(operands))>;
            if constexpr (requires {
                              D::Loc;
                              S::Loc;
                              D::RowStride;
                              S::RowStride;
                              D::isRowMajor;
                              S::isRowMajor;
                          }) {
                if (HasSameDenseGeometry(first_tile, std::get<0>(operands)) && D::RowStride == D::Cols &&
                    S::RowStride == S::Cols) {
                    estimateOptions.dst_rows = destinationInfo.physical_rows;
                    estimateOptions.dst_cols = destinationInfo.physical_cols;
                    estimateOptions.dst_valid_rows = destinationInfo.valid_rows;
                    estimateOptions.dst_valid_cols = destinationInfo.valid_cols;
                }
            }
        }
    }
    if constexpr (sizeof...(rest_tiles) == 3) {
        if (std::string_view(opcode) == "TCMPS") {
            auto sources = std::forward_as_tuple(rest_tiles...);
            CostModelTileInfo scalarTile;
            if (TryCostModelTileInfo(scalarTile, std::get<1>(sources))) {
                // A scalar tile requires a broadcast load absent from the immediate-scalar fit.
                estimateOptions.a5_op_params = "scalar_tile";
            }
        }
    }
    if constexpr (sizeof...(rest_tiles) == 2) {
        if (std::string_view(opcode) == "TCONCAT") {
            auto sources = std::forward_as_tuple(rest_tiles...);
            CostModelTileInfo source0;
            CostModelTileInfo source1;
            if (TryCostModelTileInfo(source0, std::get<0>(sources)) &&
                TryCostModelTileInfo(source1, std::get<1>(sources))) {
                estimateOptions.concat_src0_cols = source0.valid_cols;
                estimateOptions.concat_src1_cols = source1.valid_cols;
                estimateOptions.concat_dense_sources =
                    source0.physical_cols == source0.valid_cols && source1.physical_cols == source1.valid_cols &&
                    source0.physical_rows == source0.valid_rows && source1.physical_rows == source1.valid_rows &&
                    source0.valid_rows == modelTileInfo.valid_rows && source1.valid_rows == modelTileInfo.valid_rows;
            }
        }
    }
    if (estimateOptions.has_mask_pattern) {
        CostModelTileInfo sourceInfo;
        CostModelTileInfo destinationInfo;
        if constexpr (sizeof...(rest_tiles) > 0) {
            ExtractFirstCostModelTileInfo(sourceInfo, rest_tiles...);
        }
        TryCostModelTileInfo(destinationInfo, first_tile);
        estimateOptions.physical_rows = sourceInfo.physical_rows;
        estimateOptions.physical_cols = sourceInfo.physical_cols;
        estimateOptions.valid_rows = sourceInfo.valid_rows;
        estimateOptions.valid_cols = sourceInfo.valid_cols;
        estimateOptions.a5_shape_path = sourceInfo.a5_shape_path;
        estimateOptions.dst_rows = destinationInfo.physical_rows;
        estimateOptions.dst_cols = destinationInfo.physical_cols;
        estimateOptions.dst_valid_rows = destinationInfo.valid_rows;
        estimateOptions.dst_valid_cols = destinationInfo.valid_cols;
    }

    const uint64_t measured_cycles = GetLastPtoInstrCycles();
    uint64_t cycles = measured_cycles;
    const auto& trace = GetTrace();
    const PtoInstrRecord* currentTrace = trace.executed_pto.empty() ? nullptr : &trace.executed_pto.back();
    const uint32_t npuArch = perf_sim::GetCostModelCtx().npu_arch;
    const bool useA5Fit = npuArch == perf_sim::kAscend950Arch || npuArch == perf_sim::kAscend950UrmaArch;
    const auto hybridEstimate = useA5Fit ? hybrid::CycleEstimate{} :
                                           hybrid::EstimatePtoCycles({
                                               opcode,
                                               r.rows,
                                               r.cols,
                                               dstDtype,
                                               srcDtype.empty() ? r.dtype : srcDtype,
                                               measured_cycles,
                                               currentTrace,
                                           });
    const uint64_t estimated_cycles =
        hybridEstimate ?
            0 :
            perf_sim::EstimateInstrCycles(
                opcode, r.rows, r.cols, srcDtype.empty() ? (r.dtype.empty() ? "unknown" : r.dtype.c_str()) : srcDtype,
                dstDtype.empty() ? "unknown" : dstDtype, innerDimension, transferTileType, estimateOptions);
    // A5 uses calibrated fits only: the mock trace follows the A2/A3 implementation
    // and is not an A5 hardware measurement, even when it reports nonzero cycles.
    const bool useEstimatedCycles = hybridEstimate || useA5Fit || cycles == 0 ||
                                    (std::string_view(opcode) == "TROWEXPAND" && estimated_cycles > cycles) ||
                                    (std::string_view(opcode) == "TDIVS" && (r.dtype == "int16" || r.dtype == "int32"));
    if (useEstimatedCycles) {
        cycles = hybridEstimate ? hybridEstimate.cycles : estimated_cycles;
        auto& trace = ::pto::mocker::GetMutableTrace();
        if (!trace.executed_pto.empty()) {
            trace.executed_pto.back().total_cycles = cycles;
        }
    }
    r.estimated_cycles = cycles;

    perf_sim::CvSyncRecorder::ApplyPending(opcode, r);
    perf_sim::PtoRecorder::Record(std::move(r));
}

// Scalar-stage overload for instructions with no tile arguments
// Scalar-stage ops don't register as data producers (per TileDepTracker design),
// so we skip TrackByAddr and use signal_event=-1.
inline void RecordInstr(const char* opcode)
{
    perf_sim::PipeStage stage = perf_sim::StaticPipeStageLookup(opcode);
    perf_sim::InstrRecord r;
    r.opcode = opcode;
    r.stage = stage;
    r.signal_event = -1;
    r.estimated_cycles = perf_sim::EstimateInstrCycles(opcode, 0, 0, "unknown");
    perf_sim::CvSyncRecorder::ApplyPending(opcode, r);
    perf_sim::PtoRecorder::Record(std::move(r));
}

} // namespace pto::mocker

// RecordInstr variant: first_tile is in the variadic position (MAP_INSTR_IMPL style)
// Accepts any types (tiles or scalars); filters to only tile references before passing to RecordInstr.
template <typename OpcodeStr, typename First, typename... Args>
inline void RecordInstrFromFirst(OpcodeStr&& opcode_str, First&& first, Args&&... args)
{
    ::pto::mocker::RecordInstr(
        std::forward<OpcodeStr>(opcode_str), std::forward<First>(first), std::forward<Args>(args)...);
    // RecordInstr may replace the raw CCE sum with a PTO-level estimate. Store
    // that final value on the output tile so dependency consumers and the
    // exported instruction record observe the same cycle count.
    ::pto::mocker::InjectTileCycles(first);
}

//

// TPUSH: Record push event to CV ring buffer
// pipe: TPipe<FlagID, DirType, ...> or TMPipe<FlagID, ...> instance
// tile: Tile being pushed
// tile_index: pipe.prod.tileIndex (sequential counter)
template <typename Pipe, typename TileProd>
inline void RecordTPushSync(Pipe& pipe, TileProd& tile, int tile_index)
{
    uint64_t key = perf_sim::MakeCvFifoKey<Pipe>();
    perf_sim::CvSyncRecorder::SetPending(perf_sim::CvSyncKind::Push, key);
    (void)pipe;
    (void)tile;
    (void)tile_index;
}

// TPOP: Record pop event to CV ring buffer
// pipe: TPipe<FlagID, DirType, ...> or TMPipe<FlagID, ...> instance
// tile: Tile being popped
// tile_index: pipe.cons.tileIndex (sequential counter)
template <typename Pipe, typename TileCons>
inline void RecordTPopSync(Pipe& pipe, TileCons& tile, int tile_index)
{
    uint64_t key = perf_sim::MakeCvFifoKey<Pipe>();
    perf_sim::CvSyncRecorder::SetPending(perf_sim::CvSyncKind::Pop, key);
    (void)pipe;
    (void)tile;
    (void)tile_index;
}

#define PTO_FIRST_ARG(first, ...) first
#define PTO_SECOND_ARG(_first, second, ...) second
#define PTO_TEMPLATE_ARGS(...) <__VA_ARGS__>

#if defined(PTO_NPU_ARCH_KIRIN9030) || defined(PTO_NPU_ARCH_KIRINX90) || defined(PTO_NPU_ARCH_KIRIN9050)
#define PTO_FORWARD_L2HINT_TO_IMPL 0
#else
#define PTO_FORWARD_L2HINT_TO_IMPL 1
#endif
#define MAP_INSTR_IMPL(API, ...)                   \
    do {                                           \
        ::pto::mocker::PtoInstrScope _scope(#API); \
        API##_IMPL(__VA_ARGS__);                   \
        _scope.Finish();                           \
        ::RecordInstrFromFirst(#API, __VA_ARGS__); \
    } while (0)
// Template calls use a dedicated macro because the preprocessor does not parse
// template commas in a generic `_IMPL(...)` wrapper reliably.
#define MAP_INSTR_IMPL_T(API, TEMPLATE_ARGS, ...)  \
    do {                                           \
        ::pto::mocker::PtoInstrScope _scope(#API); \
        API##_IMPL TEMPLATE_ARGS(__VA_ARGS__);     \
        _scope.Finish();                           \
        ::RecordInstrFromFirst(#API, __VA_ARGS__); \
    } while (0)
#define RECORD_INSTR_ONLY(API, ...)                \
    do {                                           \
        ::pto::mocker::PtoInstrScope _scope(#API); \
        _scope.Finish();                           \
        ::RecordInstrFromFirst(#API, __VA_ARGS__); \
    } while (0)

// TPUSH/TPOP special macro: records FFTS sync events for ring buffer
// First arg is Pipe, second is Tile, third is tile index
#define MAP_INSTR_IMPL_TPUSH_POP(IS_TPUSH, API, TEMPLATE_ARGS, ...)                                                  \
    do {                                                                                                             \
        ::pto::mocker::PtoInstrScope _scope(#API);                                                                   \
        API##_IMPL TEMPLATE_ARGS(__VA_ARGS__);                                                                       \
        _scope.Finish();                                                                                             \
        /* Call RecordTPushSync or RecordTPopSync for FFTS sync */                                                   \
        if constexpr (IS_TPUSH) {                                                                                    \
            ::RecordTPushSync(PTO_FIRST_ARG(__VA_ARGS__), PTO_NTH_ARG(__VA_ARGS__, 2), PTO_NTH_ARG(__VA_ARGS__, 3)); \
        } else {                                                                                                     \
            ::RecordTPopSync(PTO_FIRST_ARG(__VA_ARGS__), PTO_NTH_ARG(__VA_ARGS__, 2), PTO_NTH_ARG(__VA_ARGS__, 3));  \
        }                                                                                                            \
        ::RecordInstrFromFirst(#API, __VA_ARGS__);                                                                   \
    } while (0)

// Special macro for TPUSH: includes CV ring buffer FFTS sync recording
// Args: API, TEMPLATE_ARGS, pipe, tile, [tile_index]
#define MAP_INSTR_IMPL_T_TPUSH(API, TEMPLATE_ARGS, ...)                                                          \
    do {                                                                                                         \
        ::pto::mocker::PtoInstrScope _scope(#API);                                                               \
        API##_IMPL TEMPLATE_ARGS(__VA_ARGS__);                                                                   \
        _scope.Finish();                                                                                         \
        ::RecordTPushSync(                                                                                       \
            PTO_FIRST_ARG(__VA_ARGS__), PTO_SECOND_ARG(__VA_ARGS__), PTO_FIRST_ARG(__VA_ARGS__).prod.tileIndex); \
        ::RecordInstrFromFirst(#API, __VA_ARGS__);                                                               \
    } while (0)

// Special macro for TPOP: includes CV ring buffer FFTS sync recording
#define MAP_INSTR_IMPL_T_TPOP(API, TEMPLATE_ARGS, ...)                                                           \
    do {                                                                                                         \
        ::pto::mocker::PtoInstrScope _scope(#API);                                                               \
        API##_IMPL TEMPLATE_ARGS(__VA_ARGS__);                                                                   \
        _scope.Finish();                                                                                         \
        ::RecordTPopSync(                                                                                        \
            PTO_FIRST_ARG(__VA_ARGS__), PTO_SECOND_ARG(__VA_ARGS__), PTO_FIRST_ARG(__VA_ARGS__).cons.tileIndex); \
        ::RecordInstrFromFirst(#API, __VA_ARGS__);                                                               \
    } while (0)

#define PTO_FIRST_ARG(first, ...) first
#define PTO_SECOND_ARG(_first, second, ...) second
#define PTO_TEMPLATE_ARGS(...) <__VA_ARGS__>
#endif

namespace pto {

namespace detail {
template <Op OpCode>
PTO_INTERNAL void PtoSyncOp()
{
    TSYNC_IMPL<OpCode>();
}

template <typename... WaitEvents>
PTO_INTERNAL void PtoWaitEvents(WaitEvents&... events)
{
    WaitAllEvents(events...);
}

} // namespace detail

template <typename T, typename AddrType>
PTO_INST void TASSIGN(T& obj, AddrType addr)
{
    MAP_INSTR_IMPL(TASSIGN, obj, addr);
}

// Compile-time address overload: TASSIGN<Addr>(tile)
// Performs static bounds and alignment checks when Addr is a compile-time constant.
// Only enabled for Tile / ConvTile types (not GlobalTensor).
template <std::size_t Addr, typename T>
PTO_INST std::enable_if_t<is_tile_data_v<T> || is_conv_tile_v<T>> TASSIGN(T& obj)
{
    // Trigger compile-time checks (static_assert inside tassign_static_check).
    (void)detail::tassign_static_check<std::remove_cv_t<T>, Addr>{};

    // Delegate to the existing runtime TASSIGN path.
    TASSIGN(obj, static_cast<std::size_t>(Addr));
}

template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1, typename... WaitEvents>
PTO_INST RecordEvent TADD(TileDataDst& dst, TileDataSrc0& src0, TileDataSrc1& src1, WaitEvents&... events)
{
    ::pto::detail::PtoWaitEvents(events...);
    MAP_INSTR_IMPL(TADD, dst, src0, src1);
    return RecordEvent{};
}

template <typename TileDataDst, typename TileDataSrc, typename... WaitEvents>
PTO_INST RecordEvent TABS(TileDataDst& dst, TileDataSrc& src, WaitEvents&... events)
{
    ::pto::detail::PtoWaitEvents(events...);
    MAP_INSTR_IMPL(TABS, dst, src);
    return RecordEvent{};
}

template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1, typename... WaitEvents>
PTO_INST RecordEvent TAND(TileDataDst& dst, TileDataSrc0& src0, TileDataSrc1& src1, WaitEvents&... events)
{
    ::pto::detail::PtoWaitEvents(events...);
    MAP_INSTR_IMPL(TAND, dst, src0, src1);
    return RecordEvent{};
}

template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1, typename... WaitEvents>
PTO_INST RecordEvent TOR(TileDataDst& dst, TileDataSrc0& src0, TileDataSrc1& src1, WaitEvents&... events)
{
    ::pto::detail::PtoWaitEvents(events...);
    MAP_INSTR_IMPL(TOR, dst, src0, src1);
    return RecordEvent{};
}

template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1, typename... WaitEvents>
PTO_INST RecordEvent TSUB(TileDataDst& dst, TileDataSrc0& src0, TileDataSrc1& src1, WaitEvents&... events)
{
    ::pto::detail::PtoWaitEvents(events...);
    MAP_INSTR_IMPL(TSUB, dst, src0, src1);
    return RecordEvent{};
}

template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1, typename... WaitEvents>
PTO_INST RecordEvent TMUL(TileDataDst& dst, TileDataSrc0& src0, TileDataSrc1& src1, WaitEvents&... events)
{
    ::pto::detail::PtoWaitEvents(events...);
    MAP_INSTR_IMPL(TMUL, dst, src0, src1);
    return RecordEvent{};
}

template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1, typename... WaitEvents>
PTO_INST RecordEvent TMADD(TileDataDst& dst, TileDataSrc0& src0, TileDataSrc1& src1, WaitEvents&... events)
{
    ::pto::detail::PtoWaitEvents(events...);
    MAP_INSTR_IMPL(TMADD, dst, src0, src1);
    return RecordEvent{};
}

#if defined(PTO_NPU_ARCH_A6)
// A6 sign-variant overload retaining the TMADD name: TMADD<NegProduct, NegAddend>(dst, src0, src1).
template <
    bool NegProduct, bool NegAddend, typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1,
    typename... WaitEvents>
PTO_INST RecordEvent TMADD(TileDataDst& dst, TileDataSrc0& src0, TileDataSrc1& src1, WaitEvents&... events)
{
    ::pto::detail::PtoWaitEvents(events...);
    MAP_INSTR_IMPL_T(TMADD, PTO_TEMPLATE_ARGS(NegProduct, NegAddend), dst, src0, src1);
    return RecordEvent{};
}
#endif

template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1, typename... WaitEvents>
PTO_INST RecordEvent TMULA(TileDataDst& dst, TileDataSrc0& src0, TileDataSrc1& src1, WaitEvents&... events)
{
    ::pto::detail::PtoWaitEvents(events...);
    MAP_INSTR_IMPL(TMULA, dst, src0, src1);
    return RecordEvent{};
}

#if defined(PTO_NPU_ARCH_A6)
// A6 sign-variant overload retaining the TMULA name: TMULA<NegProduct, NegAcc>(dst, src0, src1).
template <
    bool NegProduct, bool NegAcc, typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1,
    typename... WaitEvents>
PTO_INST RecordEvent TMULA(TileDataDst& dst, TileDataSrc0& src0, TileDataSrc1& src1, WaitEvents&... events)
{
    ::pto::detail::PtoWaitEvents(events...);
    MAP_INSTR_IMPL_T(TMULA, PTO_TEMPLATE_ARGS(NegProduct, NegAcc), dst, src0, src1);
    return RecordEvent{};
}
#endif

template <
    auto PrecisionType = PowAlgorithm::DEFAULT, typename DstTile, typename BaseTile, typename ExpTile, typename TmpTile,
    typename... WaitEvents>
PTO_INST RecordEvent TPOW(DstTile& dst, BaseTile& base, ExpTile& exp, TmpTile& tmp, WaitEvents&... events)
{
    ::pto::detail::PtoWaitEvents(events...);
    MAP_INSTR_IMPL_T(TPOW, PTO_TEMPLATE_ARGS(PrecisionType), dst, base, exp, tmp);
    return RecordEvent{};
}

template <
    auto PrecisionType = PowAlgorithm::DEFAULT, typename DstTile, typename BaseTile, typename TmpTile,
    typename... WaitEvents>
PTO_INST RecordEvent
TPOWS(DstTile& dst, BaseTile& base, typename DstTile::DType exp, TmpTile& tmp, WaitEvents&... events)
{
    ::pto::detail::PtoWaitEvents(events...);
    MAP_INSTR_IMPL_T(TPOWS, PTO_TEMPLATE_ARGS(PrecisionType), dst, base, exp, tmp);
    return RecordEvent{};
}

template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1, typename... WaitEvents>
PTO_INST RecordEvent TMIN(TileDataDst& dst, TileDataSrc0& src0, TileDataSrc1& src1, WaitEvents&... events)
{
    ::pto::detail::PtoWaitEvents(events...);
    MAP_INSTR_IMPL(TMIN, dst, src0, src1);
    return RecordEvent{};
}

template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1, typename... WaitEvents>
PTO_INST RecordEvent TMAX(TileDataDst& dst, TileDataSrc0& src0, TileDataSrc1& src1, WaitEvents&... events)
{
    ::pto::detail::PtoWaitEvents(events...);
    MAP_INSTR_IMPL(TMAX, dst, src0, src1);
    return RecordEvent{};
}

template <typename TileData, typename... WaitEvents>
PTO_INST RecordEvent TEXPANDS(TileData& dst, typename TileData::DType scalar, WaitEvents&... events)
{
    ::pto::detail::PtoWaitEvents(events...);
    MAP_INSTR_IMPL(TEXPANDS, dst, scalar);
    return RecordEvent{};
}

template <typename TileData, typename GlobalData, typename... WaitEvents>
PTO_INST RecordEvent TLOAD(TileData& dst, GlobalData& src, WaitEvents&... events)
{
    ::pto::detail::PtoWaitEvents(events...);
    if constexpr (::pto::mocker::IsApproximateDefaultLoad<TileData>()) {
        if (::pto::mocker::IsA5SequenceFit()) {
            const auto tag = ::pto::mocker::DefaultLoadGeometryTag<TLoadL2Hint::NormalFirstVictim>(dst, src);
            ::RecordInstrFromFirst("TLOAD", dst, tag);
            return RecordEvent{};
        }
    }
#if PTO_FORWARD_L2HINT_TO_IMPL
    MAP_INSTR_IMPL_T(TLOAD, PTO_TEMPLATE_ARGS(TLoadL2Hint::NormalFirstVictim), dst, src);
#else
    // Kirin IMPLs keep pre-L2-hint signatures.
    MAP_INSTR_IMPL(TLOAD, dst, src);
#endif
    return RecordEvent{};
}

template <
    TLoadL2Hint l2Control, typename TileData, typename GlobalData, typename... WaitEvents,
    std::enable_if_t<all_events_v<WaitEvents...>, int> = 0>
PTO_INST RecordEvent TLOAD(TileData& dst, GlobalData& src, WaitEvents&... events)
{
    ::pto::detail::PtoWaitEvents(events...);
    if constexpr (::pto::mocker::IsApproximateDefaultLoad<TileData>()) {
        if (::pto::mocker::IsA5SequenceFit()) {
            const auto tag = ::pto::mocker::DefaultLoadGeometryTag<l2Control>(dst, src);
            ::RecordInstrFromFirst("TLOAD", dst, tag);
            return RecordEvent{};
        }
    }
#if PTO_FORWARD_L2HINT_TO_IMPL
    MAP_INSTR_IMPL_T(TLOAD, PTO_TEMPLATE_ARGS(l2Control), dst, src);
#else
    (void)static_cast<uint8_t>(l2Control);
    MAP_INSTR_IMPL(TLOAD, dst, src);
#endif
    return RecordEvent{};
}

template <typename TileData, typename GlobalData>
PTO_INST RecordEvent TPREFETCH(TileData& dst, GlobalData& src)
{
    MAP_INSTR_IMPL(TPREFETCH, dst, src);
    return RecordEvent{};
}

template <typename TileDataOut, typename TileDataIn, typename TileDataTmp, typename... WaitEvents>
PTO_INST RecordEvent TROWSUM(TileDataOut& dst, TileDataIn& src, TileDataTmp& tmp, WaitEvents&... events)
{
    ::pto::detail::PtoWaitEvents(events...);
    MAP_INSTR_IMPL(TROWSUM, dst, src, tmp);
    return RecordEvent{};
}

template <typename TileDataOut, typename TileDataIn, typename TileDataTmp, typename... WaitEvents>
PTO_INST RecordEvent TROWPROD(TileDataOut& dst, TileDataIn& src, TileDataTmp& tmp, WaitEvents&... events)
{
    ::pto::detail::PtoWaitEvents(events...);
    MAP_INSTR_IMPL(TROWPROD, dst, src, tmp);
    return RecordEvent{};
}

template <typename TileDataOut, typename TileDataIn, typename... WaitEvents>
PTO_INST RecordEvent TCOLSUM(TileDataOut& dst, TileDataIn& src, WaitEvents&... events)
{
    ::pto::detail::PtoWaitEvents(events...);
    MAP_INSTR_IMPL(TCOLSUM, dst, src);
    return RecordEvent{};
}

template <typename TileDataOut, typename TileDataIn, typename TileDataTmp, typename... WaitEvents>
PTO_INST RecordEvent TCOLSUM(TileDataOut& dst, TileDataIn& src, TileDataTmp& tmp, bool isBinary, WaitEvents&... events)
{
    ::pto::detail::PtoWaitEvents(events...);
    MAP_INSTR_IMPL(TCOLSUM, dst, src, tmp, isBinary);
    return RecordEvent{};
}

template <typename TileDataOut, typename TileDataIn, typename... WaitEvents>
PTO_INST RecordEvent TCOLPROD(TileDataOut& dst, TileDataIn& src, WaitEvents&... events)
{
    ::pto::detail::PtoWaitEvents(events...);
    MAP_INSTR_IMPL(TCOLPROD, dst, src);
    return RecordEvent{};
}

template <typename TileDataOut, typename TileDataIn, typename... WaitEvents>
PTO_INST RecordEvent TCOLMAX(TileDataOut& dst, TileDataIn& src, WaitEvents&... events)
{
    ::pto::detail::PtoWaitEvents(events...);
    MAP_INSTR_IMPL(TCOLMAX, dst, src);
    return RecordEvent{};
}

template <typename TileDataOut, typename TileDataIn, typename TileDataTmp, typename... WaitEvents>
PTO_INST RecordEvent TROWMAX(TileDataOut& dst, TileDataIn& src, TileDataTmp& tmp, WaitEvents&... events)
{
    ::pto::detail::PtoWaitEvents(events...);
    MAP_INSTR_IMPL(TROWMAX, dst, src, tmp);
    return RecordEvent{};
}

template <typename TileDataOut, typename TileDataIn, typename TileDataTmp, typename... WaitEvents>
PTO_INST RecordEvent TROWARGMAX(TileDataOut& dst, TileDataIn& src, TileDataTmp& tmp, WaitEvents&... events)
{
    ::pto::detail::PtoWaitEvents(events...);
    MAP_INSTR_IMPL(TROWARGMAX, dst, src, tmp);
    return RecordEvent{};
}

template <
    typename TileDataOutVal, typename TileDataOutIdx, typename TileDataIn, typename TileDataTmp, typename... WaitEvents>
PTO_INST RecordEvent
TROWARGMAX(TileDataOutVal& dstVal, TileDataOutIdx& dstIdx, TileDataIn& src, TileDataTmp& tmp, WaitEvents&... events)
{
    ::pto::detail::PtoWaitEvents(events...);
    MAP_INSTR_IMPL(TROWARGMAX, dstVal, dstIdx, src, tmp);
    return RecordEvent{};
}

template <typename TileDataOut, typename TileDataIn, typename... WaitEvents>
PTO_INST RecordEvent TRESHAPE(TileDataOut& dst, TileDataIn& src, WaitEvents&... events)
{
    ::pto::detail::PtoWaitEvents(events...);
    MAP_INSTR_IMPL(TRESHAPE, dst, src);
    return RecordEvent{};
}

template <typename TileDataOut, typename TileDataIn, typename TileDataTmp, typename... WaitEvents>
PTO_INST RecordEvent TROWMIN(TileDataOut& dst, TileDataIn& src, TileDataTmp& tmp, WaitEvents&... events)
{
    ::pto::detail::PtoWaitEvents(events...);
    MAP_INSTR_IMPL(TROWMIN, dst, src, tmp);
    return RecordEvent{};
}

template <typename TileDataOut, typename TileDataIn, typename TileDataTmp, typename... WaitEvents>
PTO_INST RecordEvent TROWARGMIN(TileDataOut& dst, TileDataIn& src, TileDataTmp& tmp, WaitEvents&... events)
{
    ::pto::detail::PtoWaitEvents(events...);
    MAP_INSTR_IMPL(TROWARGMIN, dst, src, tmp);
    return RecordEvent{};
}

template <
    typename TileDataOutVal, typename TileDataOutIdx, typename TileDataIn, typename TileDataTmp, typename... WaitEvents>
PTO_INST RecordEvent
TROWARGMIN(TileDataOutVal& dstVal, TileDataOutIdx& dstIdx, TileDataIn& src, TileDataTmp& tmp, WaitEvents&... events)
{
    ::pto::detail::PtoWaitEvents(events...);
    MAP_INSTR_IMPL(TROWARGMIN, dstVal, dstIdx, src, tmp);
    return RecordEvent{};
}

template <
    typename TileDataDst, typename TileDataMask, typename TileDataSrc, typename TileDataTmp, typename... WaitEvents>
PTO_INST RecordEvent TSELS(
    TileDataDst& dst, TileDataMask& mask, TileDataSrc& src, TileDataTmp& tmp, typename TileDataSrc::DType scalar,
    WaitEvents&... events)
{
    ::pto::detail::PtoWaitEvents(events...);
    MAP_INSTR_IMPL(TSELS, dst, mask, src, tmp, scalar);
    return RecordEvent{};
}

template <typename TileData, typename MaskTile, typename TmpTile, typename... WaitEvents>
PTO_INST RecordEvent
TSEL(TileData& dst, MaskTile& selMask, TileData& src0, TileData& src1, TmpTile& tmp, WaitEvents&... events)
{
    ::pto::detail::PtoWaitEvents(events...);
    if constexpr (
        std::is_same_v<typename TileData::DType, uint16_t> || std::is_same_v<typename TileData::DType, uint32_t>) {
        if (::pto::perf_sim::GetCostModelCtx().npu_arch == ::pto::perf_sim::kAscend950Arch) {
            // A5 does not use tmp for these dtypes. Keep packed-mask and data
            // metadata separate instead of selecting the mask dtype.
            RECORD_INSTR_ONLY(
                TSEL, dst, selMask, src0, src1, tmp, ::pto::mocker::SelectGeometryTag(dst, selMask, src0, src1));
            return RecordEvent{};
        }
    }
    MAP_INSTR_IMPL(TSEL, dst, selMask, src0, src1, tmp);
    return RecordEvent{};
}

template <
    typename TileDataDst, typename MaskTile, typename TileDataSrc0, typename TileDataSrc1, typename TmpTile,
    typename... WaitEvents>
PTO_INST RecordEvent TPARTSEL(
    TileDataDst& dst, MaskTile& selMask, TileDataSrc0& src0, TileDataSrc1& src1, TmpTile& tmp, WaitEvents&... events)
{
    ::pto::detail::PtoWaitEvents(events...);
    MAP_INSTR_IMPL(TSEL, dst, selMask, src0, src1, tmp);
    return RecordEvent{};
}

template <typename TileDataDst, typename TileDataSrc, typename TileDataTmp, typename... WaitEvents>
PTO_INST RecordEvent TTRANS(TileDataDst& dst, TileDataSrc& src, TileDataTmp& tmp, WaitEvents&... events)
{
    ::pto::detail::PtoWaitEvents(events...);
    if (::pto::mocker::IsA5SequenceFit()) {
        RECORD_INSTR_ONLY(TTRANS, dst, src, tmp, ::pto::mocker::TransposeFitGeometry(dst, src));
    } else {
        MAP_INSTR_IMPL(TTRANS, dst, src, tmp);
    }
    return RecordEvent{};
}

template <typename TileDataDst, typename TileDataSrc, typename... WaitEvents>
PTO_INST RecordEvent
TMINS(TileDataDst& dst, TileDataSrc& src, typename TileDataSrc::DType scalar, WaitEvents&... events)
{
    ::pto::detail::PtoWaitEvents(events...);
    MAP_INSTR_IMPL(TMINS, dst, src, scalar);
    return RecordEvent{};
}

template <typename TileDataDst, typename TileDataSrc, typename... WaitEvents>
PTO_INST RecordEvent TROWEXPAND(TileDataDst& dst, TileDataSrc& src, WaitEvents&... events)
{
    ::pto::detail::PtoWaitEvents(events...);
    MAP_INSTR_IMPL(TROWEXPAND, dst, src);
    return RecordEvent{};
}

template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1, typename... WaitEvents>
PTO_INST RecordEvent TROWEXPANDDIV(TileDataDst& dst, TileDataSrc0& src0, TileDataSrc1& src1, WaitEvents&... events)
{
    ::pto::detail::PtoWaitEvents(events...);
    MAP_INSTR_IMPL(TROWEXPANDDIV, dst, src0, src1);
    return RecordEvent{};
}

template <
    typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1, typename TileDataTmp, typename... WaitEvents>
PTO_INST RecordEvent
TROWEXPANDDIV(TileDataDst& dst, TileDataSrc0& src0, TileDataSrc1& src1, TileDataTmp& tmp, WaitEvents&... events)
{
    ::pto::detail::PtoWaitEvents(events...);
    MAP_INSTR_IMPL(TROWEXPANDDIV, dst, src0, src1, tmp);
    return RecordEvent{};
}

template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1, typename... WaitEvents>
PTO_INST RecordEvent TROWEXPANDMUL(TileDataDst& dst, TileDataSrc0& src0, TileDataSrc1& src1, WaitEvents&... events)
{
    ::pto::detail::PtoWaitEvents(events...);
    MAP_INSTR_IMPL(TROWEXPANDMUL, dst, src0, src1);
    return RecordEvent{};
}

template <
    typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1, typename TileDataTmp, typename... WaitEvents>
PTO_INST RecordEvent
TROWEXPANDMUL(TileDataDst& dst, TileDataSrc0& src0, TileDataSrc1& src1, TileDataTmp& tmp, WaitEvents&... events)
{
    ::pto::detail::PtoWaitEvents(events...);
    MAP_INSTR_IMPL(TROWEXPANDMUL, dst, src0, src1, tmp);
    return RecordEvent{};
}

template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1, typename... WaitEvents>
PTO_INST RecordEvent TROWEXPANDSUB(TileDataDst& dst, TileDataSrc0& src0, TileDataSrc1& src1, WaitEvents&... events)
{
    ::pto::detail::PtoWaitEvents(events...);
    MAP_INSTR_IMPL(TROWEXPANDSUB, dst, src0, src1);
    return RecordEvent{};
}

template <
    typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1, typename TileDataTmp, typename... WaitEvents>
PTO_INST RecordEvent
TROWEXPANDSUB(TileDataDst& dst, TileDataSrc0& src0, TileDataSrc1& src1, TileDataTmp& tmp, WaitEvents&... events)
{
    ::pto::detail::PtoWaitEvents(events...);
    MAP_INSTR_IMPL(TROWEXPANDSUB, dst, src0, src1, tmp);
    return RecordEvent{};
}

template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1, typename... WaitEvents>
PTO_INST RecordEvent TROWEXPANDADD(TileDataDst& dst, TileDataSrc0& src0, TileDataSrc1& src1, WaitEvents&... events)
{
    ::pto::detail::PtoWaitEvents(events...);
    MAP_INSTR_IMPL(TROWEXPANDADD, dst, src0, src1);
    return RecordEvent{};
}

template <
    typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1, typename TileDataTmp, typename... WaitEvents>
PTO_INST RecordEvent
TROWEXPANDADD(TileDataDst& dst, TileDataSrc0& src0, TileDataSrc1& src1, TileDataTmp& tmp, WaitEvents&... events)
{
    ::pto::detail::PtoWaitEvents(events...);
    MAP_INSTR_IMPL(TROWEXPANDADD, dst, src0, src1, tmp);
    return RecordEvent{};
}

template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1, typename... WaitEvents>
PTO_INST RecordEvent TROWEXPANDMAX(TileDataDst& dst, TileDataSrc0& src0, TileDataSrc1& src1, WaitEvents&... events)
{
    ::pto::detail::PtoWaitEvents(events...);
    MAP_INSTR_IMPL(TROWEXPANDMAX, dst, src0, src1);
    return RecordEvent{};
}

template <
    typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1, typename TileDataTmp, typename... WaitEvents>
PTO_INST RecordEvent
TROWEXPANDMAX(TileDataDst& dst, TileDataSrc0& src0, TileDataSrc1& src1, TileDataTmp& tmp, WaitEvents&... events)
{
    ::pto::detail::PtoWaitEvents(events...);
    MAP_INSTR_IMPL(TROWEXPANDMAX, dst, src0, src1, tmp);
    return RecordEvent{};
}

template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1, typename... WaitEvents>
PTO_INST RecordEvent TROWEXPANDMIN(TileDataDst& dst, TileDataSrc0& src0, TileDataSrc1& src1, WaitEvents&... events)
{
    ::pto::detail::PtoWaitEvents(events...);
    MAP_INSTR_IMPL(TROWEXPANDMIN, dst, src0, src1);
    return RecordEvent{};
}

template <
    typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1, typename TileDataTmp, typename... WaitEvents>
PTO_INST RecordEvent
TROWEXPANDMIN(TileDataDst& dst, TileDataSrc0& src0, TileDataSrc1& src1, TileDataTmp& tmp, WaitEvents&... events)
{
    ::pto::detail::PtoWaitEvents(events...);
    MAP_INSTR_IMPL(TROWEXPANDMIN, dst, src0, src1, tmp);
    return RecordEvent{};
}

template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1, typename... WaitEvents>
PTO_INST RecordEvent TROWEXPANDEXPDIF(TileDataDst& dst, TileDataSrc0& src0, TileDataSrc1& src1, WaitEvents&... events)
{
    ::pto::detail::PtoWaitEvents(events...);
    MAP_INSTR_IMPL(TROWEXPANDEXPDIF, dst, src0, src1);
    return RecordEvent{};
}

template <
    typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1, typename TileDataTmp, typename... WaitEvents>
PTO_INST RecordEvent
TROWEXPANDEXPDIF(TileDataDst& dst, TileDataSrc0& src0, TileDataSrc1& src1, TileDataTmp& tmp, WaitEvents&... events)
{
    ::pto::detail::PtoWaitEvents(events...);
    MAP_INSTR_IMPL(TROWEXPANDEXPDIF, dst, src0, src1, tmp);
    return RecordEvent{};
}

template <typename TileDataDst, typename TileDataSrc, typename... WaitEvents>
PTO_INST RecordEvent TRSQRT(TileDataDst& dst, TileDataSrc& src, WaitEvents&... events)
{
    ::pto::detail::PtoWaitEvents(events...);
    MAP_INSTR_IMPL(TRSQRT, dst, src);
    return RecordEvent{};
}

template <
    typename TileDataDst, typename TileDataSrc, typename TileDataTmp, typename... WaitEvents,
    typename = std::void_t<typename TileDataTmp::DType>>
PTO_INST RecordEvent TRSQRT(TileDataDst& dst, TileDataSrc& src, TileDataTmp& tmp, WaitEvents&... events)
{
    ::pto::detail::PtoWaitEvents(events...);
    MAP_INSTR_IMPL(TRSQRT, dst, src, tmp);
    return RecordEvent{};
}

template <typename TileDataDst, typename TileDataSrc, typename... WaitEvents>
PTO_INST RecordEvent TSQRT(TileDataDst& dst, TileDataSrc& src, WaitEvents&... events)
{
    ::pto::detail::PtoWaitEvents(events...);
    MAP_INSTR_IMPL(TSQRT, dst, src);
    return RecordEvent{};
}

template <typename TileDataDst, typename TileDataSrc, typename... WaitEvents>
PTO_INST RecordEvent TEXP(TileDataDst& dst, TileDataSrc& src, WaitEvents&... events)
{
    ::pto::detail::PtoWaitEvents(events...);
    MAP_INSTR_IMPL(TEXP, dst, src);
    return RecordEvent{};
}

template <typename TileDataDst, typename TileDataSrc, typename... WaitEvents>
PTO_INST RecordEvent TNOT(TileDataDst& dst, TileDataSrc& src, WaitEvents&... events)
{
    ::pto::detail::PtoWaitEvents(events...);
    MAP_INSTR_IMPL(TNOT, dst, src);
    return RecordEvent{};
}

template <typename TileDataDst, typename TileDataSrc, typename... WaitEvents>
PTO_INST RecordEvent TRELU(TileDataDst& dst, TileDataSrc& src, WaitEvents&... events)
{
    ::pto::detail::PtoWaitEvents(events...);
    MAP_INSTR_IMPL(TRELU, dst, src);
    return RecordEvent{};
}

template <typename TileDataDst, typename TileDataSrc, typename TileDataOffset, typename... WaitEvents>
PTO_INST RecordEvent TGATHERB(TileDataDst& dst, TileDataSrc& src, TileDataOffset& offset, WaitEvents&... events)
{
    ::pto::detail::PtoWaitEvents(events...);
    if (::pto::mocker::IsA5SequenceFit()) {
        const auto tag =
            ::pto::mocker::DefaultIndexedGeometryTag<::pto::mocker::lightweight::DefaultInputProfile::IdentityBlocks>(
                dst, src, offset);
        ::RecordInstrFromFirst("TGATHERB", dst, tag);
        return RecordEvent{};
    }
    MAP_INSTR_IMPL(TGATHERB, dst, src, offset);
    return RecordEvent{};
}

template <typename TileDataDst, typename TileDataSrc, typename... WaitEvents>
PTO_INST RecordEvent
TADDS(TileDataDst& dst, TileDataSrc& src0, typename TileDataSrc::DType scalar, WaitEvents&... events)
{
    ::pto::detail::PtoWaitEvents(events...);
    MAP_INSTR_IMPL(TADDS, dst, src0, scalar);
    return RecordEvent{};
}

template <typename TileDataDst, typename TileDataSrc, typename... WaitEvents>
PTO_INST RecordEvent
TAXPY(TileDataDst& dst, TileDataSrc& src0, typename TileDataSrc::DType scalar, WaitEvents&... events)
{
    ::pto::detail::PtoWaitEvents(events...);
    MAP_INSTR_IMPL(TAXPY, dst, src0, scalar);
    return RecordEvent{};
}

template <typename TileDataDst, typename TileDataSrc, typename... WaitEvents>
PTO_INST RecordEvent
TSUBS(TileDataDst& dst, TileDataSrc& src0, typename TileDataSrc::DType scalar, WaitEvents&... events)
{
    ::pto::detail::PtoWaitEvents(events...);
    MAP_INSTR_IMPL(TSUBS, dst, src0, scalar);
    return RecordEvent{};
}

template <typename TileDataDst, typename TileDataSrc, typename... WaitEvents>
PTO_INST RecordEvent
TDIVS(TileDataDst& dst, TileDataSrc& src0, typename TileDataSrc::DType scalar, WaitEvents&... events)
{
    ::pto::detail::PtoWaitEvents(events...);
    MAP_INSTR_IMPL(TDIVS, dst, src0, scalar);
    return RecordEvent{};
}

template <typename TileDataDst, typename TileDataSrc, typename... WaitEvents>
PTO_INST RecordEvent
TMULS(TileDataDst& dst, TileDataSrc& src0, typename TileDataSrc::DType scalar, WaitEvents&... events)
{
    ::pto::detail::PtoWaitEvents(events...);
    MAP_INSTR_IMPL(TMULS, dst, src0, scalar);
    return RecordEvent{};
}

template <typename TileDataDst, typename TileDataSrc, typename... WaitEvents>
PTO_INST RecordEvent
TDIVS(TileDataDst& dst, typename TileDataDst::DType scalar, TileDataSrc& src0, WaitEvents&... events)
{
    ::pto::detail::PtoWaitEvents(events...);
    MAP_INSTR_IMPL(TDIVS, dst, scalar, src0);
    return RecordEvent{};
}

template <typename TileDataDst, typename TileDataSrc, typename... WaitEvents>
PTO_INST RecordEvent
TFMODS(TileDataDst& dst, TileDataSrc& src, typename TileDataSrc::DType scalar, WaitEvents&... events)
{
    ::pto::detail::PtoWaitEvents(events...);
    MAP_INSTR_IMPL(TFMODS, dst, src, scalar);
    return RecordEvent{};
}

template <
    auto PrecisionType = RemSAlgorithm::DEFAULT, typename TileDataDst, typename TileDataSrc, typename TileDataTmp,
    typename... WaitEvents>
PTO_INST RecordEvent
TREMS(TileDataDst& dst, TileDataSrc& src, typename TileDataSrc::DType scalar, TileDataTmp& tmp, WaitEvents&... events)
{
    ::pto::detail::PtoWaitEvents(events...);
    MAP_INSTR_IMPL_T(TREMS, PTO_TEMPLATE_ARGS(PrecisionType), dst, src, scalar, tmp);
    return RecordEvent{};
}

template <typename TileDataDst, typename TileDataSrc, typename... WaitEvents>
PTO_INST RecordEvent
TMAXS(TileDataDst& dst, TileDataSrc& src, typename TileDataSrc::DType scalar, WaitEvents&... events)
{
    ::pto::detail::PtoWaitEvents(events...);
    MAP_INSTR_IMPL(TMAXS, dst, src, scalar);
    return RecordEvent{};
}

template <typename TileDataDst, typename TileDataSrc, typename... WaitEvents>
PTO_INST RecordEvent
TANDS(TileDataDst& dst, TileDataSrc& src, typename TileDataDst::DType scalar, WaitEvents&... events)
{
    ::pto::detail::PtoWaitEvents(events...);
    MAP_INSTR_IMPL(TANDS, dst, src, scalar);
    return RecordEvent{};
}

template <typename TileDataDst, typename TileDataSrc, typename... WaitEvents>
PTO_INST RecordEvent TORS(TileDataDst& dst, TileDataSrc& src, typename TileDataDst::DType scalar, WaitEvents&... events)
{
    ::pto::detail::PtoWaitEvents(events...);
    MAP_INSTR_IMPL(TORS, dst, src, scalar);
    return RecordEvent{};
}

template <typename TileDataDst, typename TileDataSrc, typename... WaitEvents>
PTO_INST RecordEvent
TSHLS(TileDataDst& dst, TileDataSrc& src, typename TileDataDst::DType scalar, WaitEvents&... events)
{
    ::pto::detail::PtoWaitEvents(events...);
    if (::pto::mocker::IsA5SequenceFit()) {
        RECORD_INSTR_ONLY(TSHLS, dst, src, scalar, ::pto::mocker::DenseSequenceTag(dst, src));
    } else {
        MAP_INSTR_IMPL(TSHLS, dst, src, scalar);
    }
    return RecordEvent{};
}

template <typename TileDataDst, typename TileDataSrc, typename... WaitEvents>
PTO_INST RecordEvent
TSHRS(TileDataDst& dst, TileDataSrc& src, typename TileDataDst::DType scalar, WaitEvents&... events)
{
    ::pto::detail::PtoWaitEvents(events...);
    if (::pto::mocker::IsA5SequenceFit()) {
        RECORD_INSTR_ONLY(TSHRS, dst, src, scalar, ::pto::mocker::DenseSequenceTag(dst, src));
    } else {
        MAP_INSTR_IMPL(TSHRS, dst, src, scalar);
    }
    return RecordEvent{};
}

template <typename TileDataDst, typename TileDataSrc, typename TileDataTmp, typename... WaitEvents>
PTO_INST RecordEvent
TXORS(TileDataDst& dst, TileDataSrc& src0, typename TileDataSrc::DType scalar, TileDataTmp& tmp, WaitEvents&... events)
{
    ::pto::detail::PtoWaitEvents(events...);
    MAP_INSTR_IMPL(TXORS, dst, src0, scalar, tmp);
    return RecordEvent{};
}

template <typename TileDataDst, typename TileDataSrc, typename... WaitEvents>
PTO_INST RecordEvent
TLRELU(TileDataDst& dst, TileDataSrc& src, typename TileDataSrc::DType scalar, WaitEvents&... events)
{
    ::pto::detail::PtoWaitEvents(events...);
    MAP_INSTR_IMPL(TLRELU, dst, src, scalar);
    return RecordEvent{};
}

template <typename TileDataOut, typename TileDataIn, typename... WaitEvents>
PTO_INST RecordEvent TCOLMIN(TileDataOut& dst, TileDataIn& src, WaitEvents&... events)
{
    ::pto::detail::PtoWaitEvents(events...);
    MAP_INSTR_IMPL(TCOLMIN, dst, src);
    return RecordEvent{};
}

template <typename TileDataD, typename TileDataS, typename TileDataI, typename... WaitEvents>
PTO_INST RecordEvent TSCATTER(TileDataD& dst, TileDataS& src, TileDataI& indexes, WaitEvents&... events)
{
    ::pto::detail::PtoWaitEvents(events...);
    if (::pto::mocker::IsA5SequenceFit()) {
        const auto tag =
            ::pto::mocker::DefaultIndexedGeometryTag<::pto::mocker::lightweight::DefaultInputProfile::ZeroIndex>(
                dst, src, indexes);
        ::RecordInstrFromFirst("TSCATTER", dst, tag);
        return RecordEvent{};
    }
    MAP_INSTR_IMPL(TSCATTER, dst, src, indexes);
    return RecordEvent{};
}

template <
    MaskPattern maskPattern = MaskPattern::P1111, auto ScatterType = ScatterAxis::SCATTER_ROW, typename DstTileData,
    typename SrcTileData, typename... WaitEvents>
PTO_INST RecordEvent TSCATTER(DstTileData& dst, SrcTileData& src, WaitEvents&... events)
{
    ::pto::detail::PtoWaitEvents(events...);
    ::pto::mocker::PtoInstrScope scope("TSCATTER");
    TSCATTER_IMPL<maskPattern, ScatterType>(dst, src);
    scope.Finish();
    ::RecordInstrFromFirst("TSCATTER", dst, src, maskPattern, ScatterType);
    return RecordEvent{};
}

template <typename TileDataDst, typename TileDataSrc, typename... WaitEvents>
PTO_INST RecordEvent TCOLEXPAND(TileDataDst& dst, TileDataSrc& src, WaitEvents&... events)
{
    ::pto::detail::PtoWaitEvents(events...);
    MAP_INSTR_IMPL(TCOLEXPAND, dst, src);
    return RecordEvent{};
}

template <typename TileDst, typename GlobalData, typename TileInd, typename... WaitEvents>
PTO_INST RecordEvent MGATHER(TileDst& dst, GlobalData& src, TileInd& indexes, WaitEvents&... events)
{
    ::pto::detail::PtoWaitEvents(events...);
    MAP_INSTR_IMPL(MGATHER, dst, src, indexes);
    return RecordEvent{};
}

template <typename GlobalData, typename TileSrc, typename TileInd, typename... WaitEvents>
PTO_INST RecordEvent MSCATTER(GlobalData& dst, TileSrc& src, TileInd& indexes, WaitEvents&... events)
{
    ::pto::detail::PtoWaitEvents(events...);
    MAP_INSTR_IMPL(MSCATTER, dst, src, indexes);
    return RecordEvent{};
}

template <typename TileDataDst, typename TileDataSrc, typename... WaitEvents>
PTO_INST RecordEvent TNEG(TileDataDst& dst, TileDataSrc& src, WaitEvents&... events)
{
    ::pto::detail::PtoWaitEvents(events...);
    MAP_INSTR_IMPL(TNEG, dst, src);
    return RecordEvent{};
}

template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1, typename... WaitEvents>
PTO_INST RecordEvent TCOLEXPANDDIV(TileDataDst& dst, TileDataSrc0& src0, TileDataSrc1& src1, WaitEvents&... events)
{
    ::pto::detail::PtoWaitEvents(events...);
    MAP_INSTR_IMPL(TCOLEXPANDDIV, dst, src0, src1);
    return RecordEvent{};
}

template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1, typename... WaitEvents>
PTO_INST RecordEvent TCOLEXPANDMUL(TileDataDst& dst, TileDataSrc0& src0, TileDataSrc1& src1, WaitEvents&... events)
{
    ::pto::detail::PtoWaitEvents(events...);
    MAP_INSTR_IMPL(TCOLEXPANDMUL, dst, src0, src1);
    return RecordEvent{};
}

template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1, typename... WaitEvents>
PTO_INST RecordEvent TCOLEXPANDADD(TileDataDst& dst, TileDataSrc0& src0, TileDataSrc1& src1, WaitEvents&... events)
{
    ::pto::detail::PtoWaitEvents(events...);
    MAP_INSTR_IMPL(TCOLEXPANDADD, dst, src0, src1);
    return RecordEvent{};
}

template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1, typename... WaitEvents>
PTO_INST RecordEvent TCOLEXPANDMAX(TileDataDst& dst, TileDataSrc0& src0, TileDataSrc1& src1, WaitEvents&... events)
{
    ::pto::detail::PtoWaitEvents(events...);
    MAP_INSTR_IMPL(TCOLEXPANDMAX, dst, src0, src1);
    return RecordEvent{};
}

template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1, typename... WaitEvents>
PTO_INST RecordEvent TCOLEXPANDMIN(TileDataDst& dst, TileDataSrc0& src0, TileDataSrc1& src1, WaitEvents&... events)
{
    ::pto::detail::PtoWaitEvents(events...);
    MAP_INSTR_IMPL(TCOLEXPANDMIN, dst, src0, src1);
    return RecordEvent{};
}

template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1, typename... WaitEvents>
PTO_INST RecordEvent TCOLEXPANDSUB(TileDataDst& dst, TileDataSrc0& src0, TileDataSrc1& src1, WaitEvents&... events)
{
    ::pto::detail::PtoWaitEvents(events...);
    MAP_INSTR_IMPL(TCOLEXPANDSUB, dst, src0, src1);
    return RecordEvent{};
}

template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1, typename... WaitEvents>
PTO_INST RecordEvent TCOLEXPANDEXPDIF(TileDataDst& dst, TileDataSrc0& src0, TileDataSrc1& src1, WaitEvents&... events)
{
    ::pto::detail::PtoWaitEvents(events...);
    MAP_INSTR_IMPL(TCOLEXPANDEXPDIF, dst, src0, src1);
    return RecordEvent{};
}

template <typename TileDataDst, typename TileDataSrc, typename TileDataPara, typename... WaitEvents>
PTO_INST RecordEvent
TDEQUANT(TileDataDst& dst, TileDataSrc& src, TileDataPara& scale, TileDataPara& offset, WaitEvents&... events)
{
    ::pto::detail::PtoWaitEvents(events...);
    MAP_INSTR_IMPL(TDEQUANT, dst, src, scale, offset);
    return RecordEvent{};
}

template <
    auto PrecisionType = RemAlgorithm::DEFAULT, typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1,
    typename TileDataTmp, typename... WaitEvents>
PTO_INST RecordEvent
TREM(TileDataDst& dst, TileDataSrc0& src0, TileDataSrc1& src1, TileDataTmp& tmp, WaitEvents&... events)
{
    ::pto::detail::PtoWaitEvents(events...);
    MAP_INSTR_IMPL_T(TREM, PTO_TEMPLATE_ARGS(PrecisionType), dst, src0, src1, tmp);
    return RecordEvent{};
}

template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1, typename... WaitEvents>
PTO_INST RecordEvent TFMOD(TileDataDst& dst, TileDataSrc0& src0, TileDataSrc1& src1, WaitEvents&... events)
{
    ::pto::detail::PtoWaitEvents(events...);
    MAP_INSTR_IMPL(TFMOD, dst, src0, src1);
    return RecordEvent{};
}

template <typename TileDataDst, typename TileDataSrc, typename... WaitEvents>
PTO_INST RecordEvent
TCMPS(TileDataDst& dst, TileDataSrc& src0, typename TileDataSrc::DType src1, CmpMode mode, WaitEvents&... events)
{
    ::pto::detail::PtoWaitEvents(events...);
    if (::pto::mocker::HasExplicitComparisonCodegen()) {
        RECORD_INSTR_ONLY(TCMPS, dst, src0, src1, mode, ::pto::mocker::ComparisonGeometryTag(dst, src0, src1, mode));
        return RecordEvent{};
    }
    using SrcType = typename TileDataSrc::DType;
    if constexpr (
        std::is_same_v<SrcType, bfloat16_t> || std::is_same_v<SrcType, float> || std::is_same_v<SrcType, int32_t> ||
        std::is_same_v<SrcType, uint32_t> || std::is_same_v<SrcType, int16_t> || std::is_same_v<SrcType, uint16_t> ||
        std::is_same_v<SrcType, int8_t> || std::is_same_v<SrcType, uint8_t>) {
        if (::pto::mocker::UseObservableComparison(dst, src0) &&
            !::pto::mocker::PreserveLegacyScalarComparison(dst, src0, src1, mode)) {
            RECORD_INSTR_ONLY(
                TCMPS, dst, src0, src1, mode, ::pto::mocker::ComparisonGeometryTag(dst, src0, src1, mode));
            return RecordEvent{};
        }
    }
    if constexpr (std::is_same_v<SrcType, bfloat16_t> && !std::is_same_v<bfloat16_t, half>) {
        // The A2/A3 mock cannot instantiate native BF16. Only the guarded A5
        // metadata route above can describe this host type; unsupported native
        // compilation paths deliberately return no prediction.
        ::pto::mocker::PtoInstrScope scope("TCMPS");
        scope.Finish();
    } else if constexpr (
        std::is_same_v<SrcType, int8_t> || std::is_same_v<SrcType, uint8_t> || std::is_same_v<SrcType, uint32_t>) {
        // These A5 source types cannot instantiate the A2/A3 mock lowering.
        // Keep the existing calibrated backend, without inventing a mock cost.
        // This bridge covers the measured immediate-EQ, full row-major fixture.
        ::pto::mocker::PtoInstrScope scope("TCMPS");
        scope.Finish();
        if constexpr (
            TileDataDst::Loc == TileType::Vec && TileDataSrc::Loc == TileType::Vec && TileDataDst::isRowMajor &&
            TileDataSrc::isRowMajor && std::is_same_v<typename TileDataDst::DType, uint8_t>) {
            if (::pto::mocker::IsA5SequenceFit() && mode == CmpMode::EQ && src1 == static_cast<SrcType>(2) &&
                src0.GetValidRow() == TileDataSrc::Rows && src0.GetValidCol() == TileDataSrc::Cols &&
                dst.GetValidRow() == src0.GetValidRow() && dst.GetValidCol() == src0.GetValidCol() &&
                TileDataDst::Rows == TileDataSrc::Rows && TileDataDst::Cols == TileDataSrc::Cols &&
                TileDataDst::RowStride == TileDataDst::Cols && TileDataSrc::RowStride == TileDataSrc::Cols) {
                ::RecordInstrFromFirst("TCMPS", dst, src0, src1, mode);
            }
        }
    } else {
        MAP_INSTR_IMPL(TCMPS, dst, src0, src1, mode);
    }
    return RecordEvent{};
}

template <
    typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1,
    typename = std::void_t<typename TileDataSrc1::DType>, typename... WaitEvents>
PTO_INST RecordEvent
TCMPS(TileDataDst& dst, TileDataSrc0& src0, TileDataSrc1& src1, CmpMode mode, WaitEvents&... events)
{
    ::pto::detail::PtoWaitEvents(events...);
    if (::pto::mocker::UseObservableComparison(dst, src0)) {
        RECORD_INSTR_ONLY(TCMPS, dst, src0, src1, mode, ::pto::mocker::ComparisonGeometryTag(dst, src0, src1, mode));
        return RecordEvent{};
    }
    using T = typename TileDataSrc0::DType;
    if constexpr (
        std::is_same_v<T, int8_t> || std::is_same_v<T, uint8_t> || std::is_same_v<T, uint32_t> ||
        (std::is_same_v<T, bfloat16_t> && !std::is_same_v<bfloat16_t, half>)) {
        // These A5-only types cannot instantiate the A2/A3 mock implementation.
        ::pto::mocker::PtoInstrScope scope("TCMPS");
        scope.Finish();
    } else {
        MAP_INSTR_IMPL(TCMPS, dst, src0, src1, mode);
    }
    return RecordEvent{};
}

template <typename TileDataDst, typename TileDataSrc, typename... WaitEvents>
PTO_INST RecordEvent
TCMP(TileDataDst& dst, TileDataSrc& src0, TileDataSrc& src1, CmpMode cmpMode, WaitEvents&... events)
{
    ::pto::detail::PtoWaitEvents(events...);
    if (::pto::mocker::UseObservableComparison<true>(dst, src0)) {
        RECORD_INSTR_ONLY(
            TCMP, dst, src0, src1, cmpMode, ::pto::mocker::ComparisonGeometryTag(dst, src0, src1, cmpMode));
        return RecordEvent{};
    }
    if constexpr (std::is_same_v<typename TileDataSrc::DType, bfloat16_t> && !std::is_same_v<bfloat16_t, half>) {
        ::pto::mocker::PtoInstrScope scope("TCMP");
        scope.Finish();
    } else {
        MAP_INSTR_IMPL(TCMP, dst, src0, src1, cmpMode);
    }
    return RecordEvent{};
}

template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1, typename... WaitEvents>
PTO_INST RecordEvent TCONCAT(TileDataDst& dst, TileDataSrc0& src0, TileDataSrc1& src1, WaitEvents&... events)
{
    ::pto::detail::PtoWaitEvents(events...);
    MAP_INSTR_IMPL(TCONCAT, dst, src0, src1);
    return RecordEvent{};
}

template <typename TileData, typename GlobalData, typename... WaitEvents>
PTO_INST RecordEvent TSTORE(GlobalData& dst, TileData& src, WaitEvents&... events)
{
    ::pto::detail::PtoWaitEvents(events...);
    MAP_INSTR_IMPL_T(TSTORE, PTO_TEMPLATE_ARGS(TileData, GlobalData, AtomicType::AtomicNone), dst, src);
    return RecordEvent{};
}

// UF-aware overload: allow selecting unit-flag phase while keeping the TSTORE name.
template <STPhase Phase, typename TileData, typename GlobalData, typename... WaitEvents>
PTO_INST RecordEvent TSTORE(GlobalData& dst, TileData& src, WaitEvents&... events)
{
    ::pto::detail::PtoWaitEvents(events...);
    MAP_INSTR_IMPL_T(TSTORE, PTO_TEMPLATE_ARGS(TileData, GlobalData, AtomicType::AtomicNone, Phase), dst, src);
    return RecordEvent{};
}

template <typename TileData, typename GlobalData, AtomicType atomicType, typename... WaitEvents>
PTO_INST RecordEvent TSTORE(GlobalData& dst, TileData& src, WaitEvents&... events)
{
    ::pto::detail::PtoWaitEvents(events...);
    MAP_INSTR_IMPL_T(TSTORE, PTO_TEMPLATE_ARGS(TileData, GlobalData, atomicType), dst, src);
    return RecordEvent{};
}

template <STPhase Phase, typename TileData, typename GlobalData, AtomicType atomicType, typename... WaitEvents>
PTO_INST RecordEvent TSTORE(GlobalData& dst, TileData& src, WaitEvents&... events)
{
    ::pto::detail::PtoWaitEvents(events...);
    MAP_INSTR_IMPL_T(TSTORE, PTO_TEMPLATE_ARGS(TileData, GlobalData, atomicType, Phase), dst, src);
    return RecordEvent{};
}

template <
    typename TileData, typename GlobalData, AtomicType atomicType = AtomicType::AtomicNone, ReluPreMode reluPreMode,
    typename... WaitEvents>
PTO_INST RecordEvent TSTORE(GlobalData& dst, TileData& src, WaitEvents&... events)
{
    ::pto::detail::PtoWaitEvents(events...);
    MAP_INSTR_IMPL_T(TSTORE, PTO_TEMPLATE_ARGS(TileData, GlobalData, atomicType, reluPreMode), dst, src);
    return RecordEvent{};
}

template <
    STPhase Phase, typename TileData, typename GlobalData, AtomicType atomicType = AtomicType::AtomicNone,
    ReluPreMode reluPreMode, typename... WaitEvents>
PTO_INST RecordEvent TSTORE(GlobalData& dst, TileData& src, WaitEvents&... events)
{
    ::pto::detail::PtoWaitEvents(events...);
    MAP_INSTR_IMPL_T(TSTORE, PTO_TEMPLATE_ARGS(TileData, GlobalData, atomicType, reluPreMode, Phase), dst, src);
    return RecordEvent{};
}

template <
    typename TileData, typename GlobalData, AtomicType atomicType = AtomicType::AtomicNone,
    ReluPreMode reluPreMode = ReluPreMode::NoRelu, typename... WaitEvents>
PTO_INST RecordEvent TSTORE(GlobalData& dst, TileData& src, uint64_t preQuantScalar, WaitEvents&... events)
{
    ::pto::detail::PtoWaitEvents(events...);
    MAP_INSTR_IMPL_T(
        TSTORE, PTO_TEMPLATE_ARGS(TileData, GlobalData, atomicType, reluPreMode), dst, src, preQuantScalar);
    return RecordEvent{};
}

template <
    STPhase Phase, typename TileData, typename GlobalData, AtomicType atomicType = AtomicType::AtomicNone,
    ReluPreMode reluPreMode = ReluPreMode::NoRelu, typename... WaitEvents>
PTO_INST RecordEvent TSTORE(GlobalData& dst, TileData& src, uint64_t preQuantScalar, WaitEvents&... events)
{
    ::pto::detail::PtoWaitEvents(events...);
    MAP_INSTR_IMPL_T(
        TSTORE, PTO_TEMPLATE_ARGS(TileData, GlobalData, atomicType, reluPreMode, Phase), dst, src, preQuantScalar);
    return RecordEvent{};
}

template <
    typename TileData, typename GlobalData, typename FpTileData, AtomicType atomicType = AtomicType::AtomicNone,
    ReluPreMode reluPreMode = ReluPreMode::NoRelu, typename... WaitEvents,
    std::enable_if_t<
        is_tile_data_v<FpTileData> && (FpTileData::Loc == TileType::Scaling) && all_events_v<WaitEvents...>, int> = 0>
PTO_INST RecordEvent TSTORE(GlobalData& dst, TileData& src, FpTileData& fp, WaitEvents&... events)
{
    ::pto::detail::PtoWaitEvents(events...);
    MAP_INSTR_IMPL_T(
        TSTORE, PTO_TEMPLATE_ARGS(TileData, GlobalData, FpTileData, atomicType, reluPreMode), dst, src, fp);
    return RecordEvent{};
}

template <
    STPhase Phase, typename TileData, typename GlobalData, typename FpTileData,
    AtomicType atomicType = AtomicType::AtomicNone, ReluPreMode reluPreMode = ReluPreMode::NoRelu,
    typename... WaitEvents,
    std::enable_if_t<
        is_tile_data_v<FpTileData> && (FpTileData::Loc == TileType::Scaling) && all_events_v<WaitEvents...>, int> = 0>
PTO_INST RecordEvent TSTORE(GlobalData& dst, TileData& src, FpTileData& fp, WaitEvents&... events)
{
    ::pto::detail::PtoWaitEvents(events...);
#if defined(PTO_NPU_ARCH_A2A3)
    MAP_INSTR_IMPL_T(
        TSTORE, PTO_TEMPLATE_ARGS(TileData, GlobalData, FpTileData, atomicType, reluPreMode, Phase), dst, src, fp);
#else
    (void)Phase;
    RECORD_INSTR_ONLY(TSTORE, dst, src, fp);
#endif
    return RecordEvent{};
}

template <
    typename TileData, typename GlobalData, typename FpTileData, AtomicType atomicType = AtomicType::AtomicNone,
    ReluPreMode reluPreMode = ReluPreMode::NoRelu, typename... WaitEvents,
    std::enable_if_t<is_tile_data_v<FpTileData> && all_events_v<WaitEvents...>, int> = 0>
PTO_INST RecordEvent TSTORE_FP(GlobalData& dst, TileData& src, FpTileData& fp, WaitEvents&... events)
{
    ::pto::detail::PtoWaitEvents(events...);
    MAP_INSTR_IMPL_T(
        TSTORE, PTO_TEMPLATE_ARGS(TileData, GlobalData, FpTileData, atomicType, reluPreMode), dst, src, fp);
    return RecordEvent{};
}

template <
    STPhase Phase, typename TileData, typename GlobalData, typename FpTileData,
    AtomicType atomicType = AtomicType::AtomicNone, ReluPreMode reluPreMode = ReluPreMode::NoRelu,
    typename... WaitEvents, std::enable_if_t<is_tile_data_v<FpTileData> && all_events_v<WaitEvents...>, int> = 0>
PTO_INST RecordEvent TSTORE_FP(GlobalData& dst, TileData& src, FpTileData& fp, WaitEvents&... events)
{
    ::pto::detail::PtoWaitEvents(events...);
#if defined(PTO_NPU_ARCH_A2A3)
    MAP_INSTR_IMPL_T(
        TSTORE, PTO_TEMPLATE_ARGS(TileData, GlobalData, FpTileData, atomicType, reluPreMode, Phase), dst, src, fp);
#else
    (void)Phase;
    RECORD_INSTR_ONLY(TSTORE, dst, src, fp);
#endif
    return RecordEvent{};
}

template <TStoreL2Hint l2Control, typename TileData, typename GlobalData, typename... WaitEvents>
PTO_INST RecordEvent TSTORE(GlobalData& dst, TileData& src, WaitEvents&... events)
{
    detail::PtoWaitEvents(events...);
#if PTO_FORWARD_L2HINT_TO_IMPL
    MAP_INSTR_IMPL_T(
        TSTORE, PTO_TEMPLATE_ARGS(TileData, GlobalData, AtomicType::AtomicNone, STPhase::Unspecified, l2Control), dst,
        src);
#else
    (void)static_cast<uint8_t>(l2Control);
    MAP_INSTR_IMPL_T(TSTORE, PTO_TEMPLATE_ARGS(TileData, GlobalData), dst, src);
#endif
    return {};
}

template <TStoreL2Hint l2Control, STPhase Phase, typename TileData, typename GlobalData, typename... WaitEvents>
PTO_INST RecordEvent TSTORE(GlobalData& dst, TileData& src, WaitEvents&... events)
{
    detail::PtoWaitEvents(events...);
#if PTO_FORWARD_L2HINT_TO_IMPL
    MAP_INSTR_IMPL_T(
        TSTORE, PTO_TEMPLATE_ARGS(TileData, GlobalData, AtomicType::AtomicNone, Phase, l2Control), dst, src);
#else
    (void)static_cast<uint8_t>(l2Control);
    MAP_INSTR_IMPL_T(TSTORE, PTO_TEMPLATE_ARGS(TileData, GlobalData, AtomicType::AtomicNone, Phase), dst, src);
#endif
    return {};
}

template <TStoreL2Hint l2Control, typename TileData, typename GlobalData, AtomicType atomicType, typename... WaitEvents>
PTO_INST RecordEvent TSTORE(GlobalData& dst, TileData& src, WaitEvents&... events)
{
    detail::PtoWaitEvents(events...);
#if PTO_FORWARD_L2HINT_TO_IMPL
    MAP_INSTR_IMPL_T(
        TSTORE, PTO_TEMPLATE_ARGS(TileData, GlobalData, atomicType, STPhase::Unspecified, l2Control), dst, src);
#else
    (void)static_cast<uint8_t>(l2Control);
    MAP_INSTR_IMPL_T(TSTORE, PTO_TEMPLATE_ARGS(TileData, GlobalData, atomicType), dst, src);
#endif
    return {};
}

template <
    TStoreL2Hint l2Control, STPhase Phase, typename TileData, typename GlobalData, AtomicType atomicType,
    typename... WaitEvents>
PTO_INST RecordEvent TSTORE(GlobalData& dst, TileData& src, WaitEvents&... events)
{
    detail::PtoWaitEvents(events...);
#if PTO_FORWARD_L2HINT_TO_IMPL
    MAP_INSTR_IMPL_T(TSTORE, PTO_TEMPLATE_ARGS(TileData, GlobalData, atomicType, Phase, l2Control), dst, src);
#else
    (void)static_cast<uint8_t>(l2Control);
    MAP_INSTR_IMPL_T(TSTORE, PTO_TEMPLATE_ARGS(TileData, GlobalData, atomicType, Phase), dst, src);
#endif
    return {};
}

template <
    TStoreL2Hint l2Control, typename TileData, typename GlobalData, typename FpTileData,
    AtomicType atomicType = AtomicType::AtomicNone, ReluPreMode reluPreMode = ReluPreMode::NoRelu,
    typename... WaitEvents,
    std::enable_if_t<
        is_tile_data_v<FpTileData> && (FpTileData::Loc == TileType::Scaling) && all_events_v<WaitEvents...>, int> = 0>
PTO_INST RecordEvent TSTORE(GlobalData& dst, TileData& src, FpTileData& fp, WaitEvents&... events)
{
    detail::PtoWaitEvents(events...);
#if PTO_FORWARD_L2HINT_TO_IMPL
#if defined(PTO_NPU_ARCH_A5) || defined(PTO_NPU_ARCH_A2A3) || defined(__CPU_SIM)
    MAP_INSTR_IMPL_T(
        TSTORE,
        PTO_TEMPLATE_ARGS(TileData, GlobalData, FpTileData, atomicType, reluPreMode, STPhase::Unspecified, l2Control),
        dst, src, fp);
#else
    MAP_INSTR_IMPL_T(
        TSTORE, PTO_TEMPLATE_ARGS(TileData, GlobalData, FpTileData, atomicType, reluPreMode, l2Control), dst, src, fp);
#endif
#else
    (void)static_cast<uint8_t>(l2Control);
    MAP_INSTR_IMPL_T(
        TSTORE, PTO_TEMPLATE_ARGS(TileData, GlobalData, FpTileData, atomicType, reluPreMode), dst, src, fp);
#endif
    return {};
}

#if defined(PTO_NPU_ARCH_A5) || defined(PTO_NPU_ARCH_A2A3) || defined(PTO_NPU_ARCH_KIRIN9030) || \
    defined(PTO_NPU_ARCH_KIRIN9050) || defined(__CPU_SIM)
template <
    TStoreL2Hint l2Control, STPhase Phase, typename TileData, typename GlobalData, typename FpTileData,
    AtomicType atomicType = AtomicType::AtomicNone, ReluPreMode reluPreMode = ReluPreMode::NoRelu,
    typename... WaitEvents,
    std::enable_if_t<
        is_tile_data_v<FpTileData> && (FpTileData::Loc == TileType::Scaling) && all_events_v<WaitEvents...>, int> = 0>
PTO_INST RecordEvent TSTORE(GlobalData& dst, TileData& src, FpTileData& fp, WaitEvents&... events)
{
    detail::PtoWaitEvents(events...);
#if PTO_FORWARD_L2HINT_TO_IMPL
    MAP_INSTR_IMPL_T(
        TSTORE, PTO_TEMPLATE_ARGS(TileData, GlobalData, FpTileData, atomicType, reluPreMode, Phase, l2Control), dst,
        src, fp);
#else
    (void)static_cast<uint8_t>(l2Control);
    MAP_INSTR_IMPL_T(
        TSTORE, PTO_TEMPLATE_ARGS(TileData, GlobalData, FpTileData, atomicType, reluPreMode, Phase), dst, src, fp);
#endif
    return {};
}
#endif

template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1, typename... WaitEvents>
PTO_INST RecordEvent TDIV(TileDataDst& dst, TileDataSrc0& src0, TileDataSrc1& src1, WaitEvents&... events)
{
    ::pto::detail::PtoWaitEvents(events...);
    MAP_INSTR_IMPL(TDIV, dst, src0, src1);
    return RecordEvent{};
}

template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1, typename... WaitEvents>
PTO_INST RecordEvent TSHL(TileDataDst& dst, TileDataSrc0& src0, TileDataSrc1& src1, WaitEvents&... events)
{
    ::pto::detail::PtoWaitEvents(events...);
    if (::pto::mocker::IsA5SequenceFit()) {
        RECORD_INSTR_ONLY(TSHL, dst, src0, src1, ::pto::mocker::DenseSequenceTag(dst, src0, src1));
    } else {
        MAP_INSTR_IMPL(TSHL, dst, src0, src1);
    }
    return RecordEvent{};
}

template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1, typename... WaitEvents>
PTO_INST RecordEvent TSHR(TileDataDst& dst, TileDataSrc0& src0, TileDataSrc1& src1, WaitEvents&... events)
{
    ::pto::detail::PtoWaitEvents(events...);
    if (::pto::mocker::IsA5SequenceFit()) {
        RECORD_INSTR_ONLY(TSHR, dst, src0, src1, ::pto::mocker::DenseSequenceTag(dst, src0, src1));
    } else {
        MAP_INSTR_IMPL(TSHR, dst, src0, src1);
    }
    return RecordEvent{};
}

template <typename TileData, typename... WaitEvents>
PTO_INST RecordEvent TAND(TileData& dst, TileData& src0, TileData& src1, WaitEvents&... events)
{
    ::pto::detail::PtoWaitEvents(events...);
    MAP_INSTR_IMPL(TAND, dst, src0, src1);
    return RecordEvent{};
}

template <typename TileData, typename... WaitEvents>
PTO_INST RecordEvent TOR(TileData& dst, TileData& src0, TileData& src1, WaitEvents&... events)
{
    ::pto::detail::PtoWaitEvents(events...);
    MAP_INSTR_IMPL(TOR, dst, src0, src1);
    return RecordEvent{};
}

template <
    typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1, typename TileDataTmp, typename... WaitEvents>
PTO_INST RecordEvent
TXOR(TileDataDst& dst, TileDataSrc0& src0, TileDataSrc1& src1, TileDataTmp& tmp, WaitEvents&... events)
{
    ::pto::detail::PtoWaitEvents(events...);
    MAP_INSTR_IMPL(TXOR, dst, src0, src1, tmp);
    return RecordEvent{};
}

template <typename TileDataDst, typename TileDataSrc, typename... WaitEvents>
PTO_INST RecordEvent TLOG(TileDataDst& dst, TileDataSrc& src, WaitEvents&... events)
{
    ::pto::detail::PtoWaitEvents(events...);
    MAP_INSTR_IMPL(TLOG, dst, src);
    return RecordEvent{};
}

template <
    auto PrecisionType = RecipAlgorithm::DEFAULT, typename TileDataDst, typename TileDataSrc, typename... WaitEvents>
PTO_INST RecordEvent TRECIP(TileDataDst& dst, TileDataSrc& src, WaitEvents&... events)
{
    constexpr int kReciprocalNumerator = 1;
    ::pto::detail::PtoWaitEvents(events...);
    // TRECIP lowers to TDIVS, but it is still one public TRECIP instruction.
    // Preserve the public opcode in both traces so consumers can retrieve its
    // cycle instead of silently observing zero records named TRECIP.
    ::pto::mocker::PtoInstrScope scope("TRECIP");
    TDIVS_IMPL<static_cast<DivAlgorithm>(PrecisionType)>(dst, kReciprocalNumerator, src);
    scope.Finish();
    ::RecordInstrFromFirst("TRECIP", dst, src);
    return RecordEvent{};
}

template <
    typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1, typename TileDataTmp, typename... WaitEvents>
PTO_INST RecordEvent
TPRELU(TileDataDst& dst, TileDataSrc0& src0, TileDataSrc1& src1, TileDataTmp& tmp, WaitEvents&... events)
{
    ::pto::detail::PtoWaitEvents(events...);
    MAP_INSTR_IMPL(TPRELU, dst, src0, src1, tmp);
    return RecordEvent{};
}

template <typename TileData, typename... WaitEvents>
PTO_INST RecordEvent TPRINT(TileData& src, WaitEvents&... events)
{
    ::pto::detail::PtoWaitEvents(events...);
    MAP_INSTR_IMPL(TPRINT, src);
    return RecordEvent{};
}

template <typename TileRes, typename TileLeft, typename TileRight, typename... WaitEvents>
PTO_INST RecordEvent TMATMUL(TileRes& cMatrix, TileLeft& aMatrix, TileRight& bMatrix, WaitEvents&... events)
{
    ::pto::detail::PtoWaitEvents(events...);
    MAP_INSTR_IMPL(TMATMUL, cMatrix, aMatrix, bMatrix);
    return RecordEvent{};
}

template <
    typename TileRes, typename TileLeft, typename TileLeftScale, typename TileRight, typename TileRightScale,
    typename... WaitEvents>
PTO_INST RecordEvent TGEMV_MX(
    TileRes& cMatrix, TileLeft& aMatrix, TileLeftScale& aScaleMatrix, TileRight& bMatrix, TileRightScale& bScaleMatrix,
    WaitEvents&... events)
{
    detail::PtoWaitEvents(events...);
    RECORD_INSTR_ONLY(
        TGEMV_MX, cMatrix, aMatrix, aScaleMatrix, bMatrix, bScaleMatrix,
        ::pto::mocker::MatrixFitShape(cMatrix, aMatrix, bMatrix, true, true));
    return {};
}

template <
    AccPhase Phase, typename TileRes, typename TileLeft, typename TileLeftScale, typename TileRight,
    typename TileRightScale, typename... WaitEvents>
PTO_INST RecordEvent TGEMV_MX(
    TileRes& cMatrix, TileLeft& aMatrix, TileLeftScale& aScaleMatrix, TileRight& bMatrix, TileRightScale& bScaleMatrix,
    WaitEvents&... events)
{
    detail::PtoWaitEvents(events...);
    RECORD_INSTR_ONLY(
        TGEMV_MX, cMatrix, aMatrix, aScaleMatrix, bMatrix, bScaleMatrix,
        ::pto::mocker::MatrixFitShape(cMatrix, aMatrix, bMatrix, true, true));
    return {};
}

template <
    typename TileRes, typename TileLeft, typename TileLeftScale, typename TileRight, typename TileRightScale,
    typename... WaitEvents>
PTO_INST RecordEvent TGEMV_MX(
    TileRes& cOutMatrix, TileRes& cInMatrix, TileLeft& aMatrix, TileLeftScale& aScaleMatrix, TileRight& bMatrix,
    TileRightScale& bScaleMatrix, WaitEvents&... events)
{
    detail::PtoWaitEvents(events...);
    RECORD_INSTR_ONLY(
        TGEMV_MX, cOutMatrix, cInMatrix, aMatrix, aScaleMatrix, bMatrix, bScaleMatrix,
        ::pto::mocker::MatrixFitShape(cOutMatrix, aMatrix, bMatrix, true, true));
    return {};
}

template <
    AccPhase Phase, typename TileRes, typename TileLeft, typename TileLeftScale, typename TileRight,
    typename TileRightScale, typename... WaitEvents>
PTO_INST RecordEvent TGEMV_MX(
    TileRes& cOutMatrix, TileRes& cInMatrix, TileLeft& aMatrix, TileLeftScale& aScaleMatrix, TileRight& bMatrix,
    TileRightScale& bScaleMatrix, WaitEvents&... events)
{
    detail::PtoWaitEvents(events...);
    RECORD_INSTR_ONLY(
        TGEMV_MX, cOutMatrix, cInMatrix, aMatrix, aScaleMatrix, bMatrix, bScaleMatrix,
        ::pto::mocker::MatrixFitShape(cOutMatrix, aMatrix, bMatrix, true, true));
    return {};
}

template <
    typename TileRes, typename TileLeft, typename TileLeftScale, typename TileRight, typename TileRightScale,
    typename TileBias, typename... WaitEvents>
PTO_INST RecordEvent TGEMV_MX(
    TileRes& cMatrix, TileLeft& aMatrix, TileLeftScale& aScaleMatrix, TileRight& bMatrix, TileRightScale& bScaleMatrix,
    TileBias& biasData, WaitEvents&... events)
{
    detail::PtoWaitEvents(events...);
    RECORD_INSTR_ONLY(
        TGEMV_MX, cMatrix, aMatrix, aScaleMatrix, bMatrix, bScaleMatrix, biasData,
        ::pto::mocker::MatrixFitShape(cMatrix, aMatrix, bMatrix, true, true));
    return {};
}

template <
    AccPhase Phase, typename TileRes, typename TileLeft, typename TileLeftScale, typename TileRight,
    typename TileRightScale, typename TileBias, typename... WaitEvents>
PTO_INST RecordEvent TGEMV_MX(
    TileRes& cMatrix, TileLeft& aMatrix, TileLeftScale& aScaleMatrix, TileRight& bMatrix, TileRightScale& bScaleMatrix,
    TileBias& biasData, WaitEvents&... events)
{
    detail::PtoWaitEvents(events...);
    RECORD_INSTR_ONLY(
        TGEMV_MX, cMatrix, aMatrix, aScaleMatrix, bMatrix, bScaleMatrix, biasData,
        ::pto::mocker::MatrixFitShape(cMatrix, aMatrix, bMatrix, true, true));
    return {};
}

template <
    typename TileRes, typename TileLeft, typename TileLeftScale, typename TileRight, typename TileRightScale,
    typename... WaitEvents>
PTO_INST RecordEvent TMATMUL_MX(
    TileRes& cMatrix, TileLeft& aMatrix, TileLeftScale& aScaleMatrix, TileRight& bMatrix, TileRightScale& bScaleMatrix,
    WaitEvents&... events)
{
    detail::PtoWaitEvents(events...);
    RECORD_INSTR_ONLY(
        TMATMUL_MX, cMatrix, aMatrix, aScaleMatrix, bMatrix, bScaleMatrix,
        ::pto::mocker::MatrixFitShape(cMatrix, aMatrix, bMatrix, false, true));
    return {};
}

// UF-aware overload enabling unit-flag selection via AccPhase while retaining the TMATMUL name.
template <
    AccPhase Phase, typename TileRes, typename TileLeft, typename TileLeftScale, typename TileRight,
    typename TileRightScale, typename... WaitEvents>
PTO_INST RecordEvent TMATMUL_MX(
    TileRes& cMatrix, TileLeft& aMatrix, TileLeftScale& aScaleMatrix, TileRight& bMatrix, TileRightScale& bScaleMatrix,
    WaitEvents&... events)
{
    detail::PtoWaitEvents(events...);
    RECORD_INSTR_ONLY(
        TMATMUL_MX, cMatrix, aMatrix, aScaleMatrix, bMatrix, bScaleMatrix,
        ::pto::mocker::MatrixFitShape(cMatrix, aMatrix, bMatrix, false));
    return {};
}

template <
    typename TileRes, typename TileLeft, typename TileLeftScale, typename TileRight, typename TileRightScale,
    typename... WaitEvents>
PTO_INST RecordEvent TMATMUL_MX(
    TileRes& cOutMatrix, TileRes& cInMatrix, TileLeft& aMatrix, TileLeftScale& aScaleMatrix, TileRight& bMatrix,
    TileRightScale& bScaleMatrix, WaitEvents&... events)
{
    detail::PtoWaitEvents(events...);
    RECORD_INSTR_ONLY(
        TMATMUL_MX, cOutMatrix, cInMatrix, aMatrix, aScaleMatrix, bMatrix, bScaleMatrix,
        ::pto::mocker::MatrixFitShape(cOutMatrix, aMatrix, bMatrix, false));
    return {};
}

template <
    AccPhase Phase, typename TileRes, typename TileLeft, typename TileLeftScale, typename TileRight,
    typename TileRightScale, typename... WaitEvents>
PTO_INST RecordEvent TMATMUL_MX(
    TileRes& cOutMatrix, TileRes& cInMatrix, TileLeft& aMatrix, TileLeftScale& aScaleMatrix, TileRight& bMatrix,
    TileRightScale& bScaleMatrix, WaitEvents&... events)
{
    detail::PtoWaitEvents(events...);
    RECORD_INSTR_ONLY(
        TMATMUL_MX, cOutMatrix, cInMatrix, aMatrix, aScaleMatrix, bMatrix, bScaleMatrix,
        ::pto::mocker::MatrixFitShape(cOutMatrix, aMatrix, bMatrix, false));
    return {};
}

template <
    typename TileRes, typename TileLeft, typename TileLeftScale, typename TileRight, typename TileRightScale,
    typename TileBias, typename... WaitEvents>
PTO_INST RecordEvent TMATMUL_MX(
    TileRes& cMatrix, TileLeft& aMatrix, TileLeftScale& aScaleMatrix, TileRight& bMatrix, TileRightScale& bScaleMatrix,
    TileBias& biasData, WaitEvents&... events)
{
    detail::PtoWaitEvents(events...);
    RECORD_INSTR_ONLY(
        TMATMUL_MX, cMatrix, aMatrix, aScaleMatrix, bMatrix, bScaleMatrix, biasData,
        ::pto::mocker::MatrixFitShape(cMatrix, aMatrix, bMatrix, false));
    return {};
}

template <
    AccPhase Phase, typename TileRes, typename TileLeft, typename TileLeftScale, typename TileRight,
    typename TileRightScale, typename TileBias, typename... WaitEvents>
PTO_INST RecordEvent TMATMUL_MX(
    TileRes& cMatrix, TileLeft& aMatrix, TileLeftScale& aScaleMatrix, TileRight& bMatrix, TileRightScale& bScaleMatrix,
    TileBias& biasData, WaitEvents&... events)
{
    detail::PtoWaitEvents(events...);
    RECORD_INSTR_ONLY(
        TMATMUL_MX, cMatrix, aMatrix, aScaleMatrix, bMatrix, bScaleMatrix, biasData,
        ::pto::mocker::MatrixFitShape(cMatrix, aMatrix, bMatrix, false));
    return {};
}

// UF-aware overload enabling unit-flag selection via AccPhase while retaining the TMATMUL name.
template <AccPhase Phase, typename TileRes, typename TileLeft, typename TileRight, typename... WaitEvents>
PTO_INST RecordEvent TMATMUL(TileRes& cMatrix, TileLeft& aMatrix, TileRight& bMatrix, WaitEvents&... events)
{
    ::pto::detail::PtoWaitEvents(events...);
    MAP_INSTR_IMPL_T(TMATMUL, PTO_TEMPLATE_ARGS(Phase), cMatrix, aMatrix, bMatrix);
    return RecordEvent{};
}

template <typename TileRes, typename TileLeft, typename TileRight, typename... WaitEvents>
PTO_INST RecordEvent
TMATMUL_ACC(TileRes& cOutMatrix, TileRes& cInMatrix, TileLeft& aMatrix, TileRight& bMatrix, WaitEvents&... events)
{
    ::pto::detail::PtoWaitEvents(events...);
    if (::pto::mocker::IsA5SequenceFit()) {
        RECORD_INSTR_ONLY(
            TMATMUL_ACC, cOutMatrix, cInMatrix, aMatrix, bMatrix,
            ::pto::mocker::MatrixFitShape(cOutMatrix, aMatrix, bMatrix, false));
    } else if constexpr (::pto::mocker::kLegacyMadTypes<TileLeft, TileRight>) {
        MAP_INSTR_IMPL(TMATMUL_ACC, cOutMatrix, cInMatrix, aMatrix, bMatrix);
    } else {
        RECORD_INSTR_ONLY(TMATMUL_ACC, cOutMatrix, cInMatrix, aMatrix, bMatrix);
    }
    return RecordEvent{};
}

// UF-aware overloads for TMATMUL_ACC: explicit input/output or shared accumulator tile.
template <AccPhase Phase, typename TileRes, typename TileLeft, typename TileRight, typename... WaitEvents>
PTO_INST RecordEvent
TMATMUL_ACC(TileRes& cOutMatrix, TileRes& cInMatrix, TileLeft& aMatrix, TileRight& bMatrix, WaitEvents&... events)
{
    ::pto::detail::PtoWaitEvents(events...);
    if (::pto::mocker::IsA5SequenceFit()) {
        RECORD_INSTR_ONLY(
            TMATMUL_ACC, cOutMatrix, cInMatrix, aMatrix, bMatrix,
            ::pto::mocker::MatrixFitShape(cOutMatrix, aMatrix, bMatrix, false));
    } else if constexpr (::pto::mocker::kLegacyMadTypes<TileLeft, TileRight>) {
        MAP_INSTR_IMPL_T(TMATMUL_ACC, PTO_TEMPLATE_ARGS(Phase), cOutMatrix, cInMatrix, aMatrix, bMatrix);
    } else {
        RECORD_INSTR_ONLY(TMATMUL_ACC, cOutMatrix, cInMatrix, aMatrix, bMatrix);
    }
    return RecordEvent{};
}

template <
    AccPhase Phase = AccPhase::Unspecified, typename TileRes, typename TileLeft, typename TileRight,
    typename... WaitEvents>
PTO_INST RecordEvent TMATMUL_ACC(TileRes& cMatrix, TileLeft& aMatrix, TileRight& bMatrix, WaitEvents&... events)
{
    ::pto::detail::PtoWaitEvents(events...);
    if (::pto::mocker::IsA5SequenceFit()) {
        RECORD_INSTR_ONLY(
            TMATMUL_ACC, cMatrix, aMatrix, bMatrix, ::pto::mocker::MatrixFitShape(cMatrix, aMatrix, bMatrix, false));
    } else if constexpr (::pto::mocker::kLegacyMadTypes<TileLeft, TileRight>) {
        MAP_INSTR_IMPL_T(TMATMUL_ACC, PTO_TEMPLATE_ARGS(Phase), cMatrix, aMatrix, bMatrix);
    } else {
        RECORD_INSTR_ONLY(TMATMUL_ACC, cMatrix, aMatrix, bMatrix);
    }
    return RecordEvent{};
}

template <typename TileRes, typename TileLeft, typename TileRight, typename TileBias, typename... WaitEvents>
PTO_INST RecordEvent
TMATMUL_BIAS(TileRes& cMatrix, TileLeft& aMatrix, TileRight& bMatrix, TileBias& biasData, WaitEvents&... events)
{
    ::pto::detail::PtoWaitEvents(events...);
    if (::pto::mocker::IsA5SequenceFit()) {
        RECORD_INSTR_ONLY(
            TMATMUL_BIAS, cMatrix, aMatrix, bMatrix, biasData,
            ::pto::mocker::MatrixFitShape(cMatrix, aMatrix, bMatrix, false));
    } else if constexpr (::pto::mocker::kLegacyMadTypes<TileLeft, TileRight>) {
        MAP_INSTR_IMPL(TMATMUL_BIAS, cMatrix, aMatrix, bMatrix, biasData);
    } else {
        RECORD_INSTR_ONLY(TMATMUL_BIAS, cMatrix, aMatrix, bMatrix, biasData);
    }
    return RecordEvent{};
}

// UF-aware overload enabling unit-flag selection for bias matmul while keeping the TMATMUL_BIAS name.
template <
    AccPhase Phase, typename TileRes, typename TileLeft, typename TileRight, typename TileBias, typename... WaitEvents>
PTO_INST RecordEvent
TMATMUL_BIAS(TileRes& cMatrix, TileLeft& aMatrix, TileRight& bMatrix, TileBias& biasData, WaitEvents&... events)
{
    ::pto::detail::PtoWaitEvents(events...);
    if (::pto::mocker::IsA5SequenceFit()) {
        RECORD_INSTR_ONLY(
            TMATMUL_BIAS, cMatrix, aMatrix, bMatrix, biasData,
            ::pto::mocker::MatrixFitShape(cMatrix, aMatrix, bMatrix, false));
    } else if constexpr (::pto::mocker::kLegacyMadTypes<TileLeft, TileRight>) {
        MAP_INSTR_IMPL_T(TMATMUL_BIAS, PTO_TEMPLATE_ARGS(Phase), cMatrix, aMatrix, bMatrix, biasData);
    } else {
        RECORD_INSTR_ONLY(TMATMUL_BIAS, cMatrix, aMatrix, bMatrix, biasData);
    }
    return RecordEvent{};
}

template <typename TileRes, typename TileLeft, typename TileRight, typename... WaitEvents>
PTO_INST RecordEvent TGEMV(TileRes& cMatrix, TileLeft& aMatrix, TileRight& bMatrix, WaitEvents&... events)
{
    ::pto::detail::PtoWaitEvents(events...);
    if (::pto::mocker::IsA5SequenceFit()) {
        RECORD_INSTR_ONLY(
            TGEMV, cMatrix, aMatrix, bMatrix, ::pto::mocker::MatrixFitShape(cMatrix, aMatrix, bMatrix, true));
    } else if constexpr (::pto::mocker::kLegacyMadTypes<TileLeft, TileRight>) {
        MAP_INSTR_IMPL(TGEMV, cMatrix, aMatrix, bMatrix);
    } else {
        RECORD_INSTR_ONLY(TGEMV, cMatrix, aMatrix, bMatrix);
    }
    return RecordEvent{};
}

template <AccPhase Phase, typename TileRes, typename TileLeft, typename TileRight, typename... WaitEvents>
PTO_INST RecordEvent TGEMV(TileRes& cMatrix, TileLeft& aMatrix, TileRight& bMatrix, WaitEvents&... events)
{
    ::pto::detail::PtoWaitEvents(events...);
    if (::pto::mocker::IsA5SequenceFit()) {
        RECORD_INSTR_ONLY(
            TGEMV, cMatrix, aMatrix, bMatrix, ::pto::mocker::MatrixFitShape(cMatrix, aMatrix, bMatrix, true));
    } else if constexpr (::pto::mocker::kLegacyMadTypes<TileLeft, TileRight>) {
        MAP_INSTR_IMPL_T(TGEMV, PTO_TEMPLATE_ARGS(Phase), cMatrix, aMatrix, bMatrix);
    } else {
        RECORD_INSTR_ONLY(TGEMV, cMatrix, aMatrix, bMatrix);
    }
    return RecordEvent{};
}

template <typename TileRes, typename TileLeft, typename TileRight, typename... WaitEvents>
PTO_INST RecordEvent
TGEMV_ACC(TileRes& cOutMatrix, TileRes& cInMatrix, TileLeft& aMatrix, TileRight& bMatrix, WaitEvents&... events)
{
    ::pto::detail::PtoWaitEvents(events...);
    if (::pto::mocker::IsA5SequenceFit()) {
        RECORD_INSTR_ONLY(
            TGEMV_ACC, cOutMatrix, cInMatrix, aMatrix, bMatrix,
            ::pto::mocker::MatrixFitShape(cOutMatrix, aMatrix, bMatrix, true));
    } else if constexpr (::pto::mocker::kLegacyMadTypes<TileLeft, TileRight>) {
        MAP_INSTR_IMPL(TGEMV_ACC, cOutMatrix, cInMatrix, aMatrix, bMatrix);
    } else {
        RECORD_INSTR_ONLY(TGEMV_ACC, cOutMatrix, cInMatrix, aMatrix, bMatrix);
    }
    return RecordEvent{};
}

template <AccPhase Phase, typename TileRes, typename TileLeft, typename TileRight, typename... WaitEvents>
PTO_INST RecordEvent
TGEMV_ACC(TileRes& cOutMatrix, TileRes& cInMatrix, TileLeft& aMatrix, TileRight& bMatrix, WaitEvents&... events)
{
    ::pto::detail::PtoWaitEvents(events...);
    if (::pto::mocker::IsA5SequenceFit()) {
        RECORD_INSTR_ONLY(
            TGEMV_ACC, cOutMatrix, cInMatrix, aMatrix, bMatrix,
            ::pto::mocker::MatrixFitShape(cOutMatrix, aMatrix, bMatrix, true));
    } else if constexpr (::pto::mocker::kLegacyMadTypes<TileLeft, TileRight>) {
        MAP_INSTR_IMPL_T(TGEMV_ACC, PTO_TEMPLATE_ARGS(Phase), cOutMatrix, cInMatrix, aMatrix, bMatrix);
    } else {
        RECORD_INSTR_ONLY(TGEMV_ACC, cOutMatrix, cInMatrix, aMatrix, bMatrix);
    }
    return RecordEvent{};
}

template <typename TileRes, typename TileLeft, typename TileRight, typename TileBias, typename... WaitEvents>
PTO_INST RecordEvent
TGEMV_BIAS(TileRes& cMatrix, TileLeft& aMatrix, TileRight& bMatrix, TileBias& biasData, WaitEvents&... events)
{
    ::pto::detail::PtoWaitEvents(events...);
    if (::pto::mocker::IsA5SequenceFit()) {
        RECORD_INSTR_ONLY(
            TGEMV_BIAS, cMatrix, aMatrix, bMatrix, biasData,
            ::pto::mocker::MatrixFitShape(cMatrix, aMatrix, bMatrix, true));
    } else if constexpr (::pto::mocker::kLegacyMadTypes<TileLeft, TileRight>) {
        MAP_INSTR_IMPL(TGEMV_BIAS, cMatrix, aMatrix, bMatrix, biasData);
    } else {
        RECORD_INSTR_ONLY(TGEMV_BIAS, cMatrix, aMatrix, bMatrix, biasData);
    }
    return RecordEvent{};
}

template <
    AccPhase Phase, typename TileRes, typename TileLeft, typename TileRight, typename TileBias, typename... WaitEvents>
PTO_INST RecordEvent
TGEMV_BIAS(TileRes& cMatrix, TileLeft& aMatrix, TileRight& bMatrix, TileBias& biasData, WaitEvents&... events)
{
    ::pto::detail::PtoWaitEvents(events...);
    if (::pto::mocker::IsA5SequenceFit()) {
        RECORD_INSTR_ONLY(
            TGEMV_BIAS, cMatrix, aMatrix, bMatrix, biasData,
            ::pto::mocker::MatrixFitShape(cMatrix, aMatrix, bMatrix, true));
    } else if constexpr (::pto::mocker::kLegacyMadTypes<TileLeft, TileRight>) {
        MAP_INSTR_IMPL_T(TGEMV_BIAS, PTO_TEMPLATE_ARGS(Phase), cMatrix, aMatrix, bMatrix, biasData);
    } else {
        RECORD_INSTR_ONLY(TGEMV_BIAS, cMatrix, aMatrix, bMatrix, biasData);
    }
    return RecordEvent{};
}

template <
    typename DstTileData, typename TmpTileData, typename Src0TileData, typename Src1TileData, typename Src2TileData,
    typename Src3TileData, bool exhausted, typename... WaitEvents>
PTO_INST RecordEvent TMRGSORT(
    DstTileData& dst, MrgSortExecutedNumList& executedNumList, TmpTileData& tmp, Src0TileData& src0, Src1TileData& src1,
    Src2TileData& src2, Src3TileData& src3, WaitEvents&... events)
{
    ::pto::detail::PtoWaitEvents(events...);
    MAP_INSTR_IMPL_T(
        TMRGSORT,
        PTO_TEMPLATE_ARGS(DstTileData, TmpTileData, Src0TileData, Src1TileData, Src2TileData, Src3TileData, exhausted),
        dst, executedNumList, tmp, src0, src1, src2, src3);
    return RecordEvent{};
}

template <
    typename DstTileData, typename TmpTileData, typename Src0TileData, typename Src1TileData, typename Src2TileData,
    bool exhausted, typename... WaitEvents>
PTO_INST RecordEvent TMRGSORT(
    DstTileData& dst, MrgSortExecutedNumList& executedNumList, TmpTileData& tmp, Src0TileData& src0, Src1TileData& src1,
    Src2TileData& src2, WaitEvents&... events)
{
    ::pto::detail::PtoWaitEvents(events...);
    MAP_INSTR_IMPL_T(
        TMRGSORT, PTO_TEMPLATE_ARGS(DstTileData, TmpTileData, Src0TileData, Src1TileData, Src2TileData, exhausted), dst,
        executedNumList, tmp, src0, src1, src2);
    return RecordEvent{};
}

template <
    typename DstTileData, typename TmpTileData, typename Src0TileData, typename Src1TileData, bool exhausted,
    typename... WaitEvents>
PTO_INST RecordEvent TMRGSORT(
    DstTileData& dst, MrgSortExecutedNumList& executedNumList, TmpTileData& tmp, Src0TileData& src0, Src1TileData& src1,
    WaitEvents&... events)
{
    ::pto::detail::PtoWaitEvents(events...);
    MAP_INSTR_IMPL_T(
        TMRGSORT, PTO_TEMPLATE_ARGS(DstTileData, TmpTileData, Src0TileData, Src1TileData, exhausted), dst,
        executedNumList, tmp, src0, src1);
    return RecordEvent{};
}

template <typename DstTileData, typename SrcTileData, typename... WaitEvents>
PTO_INST RecordEvent TMRGSORT(DstTileData& dst, SrcTileData& src, uint32_t blockLen, WaitEvents&... events)
{
    ::pto::detail::PtoWaitEvents(events...);
    MAP_INSTR_IMPL(TMRGSORT, dst, src, blockLen);
    return RecordEvent{};
}

template <typename DstTileData, typename SrcTileData, typename... WaitEvents>
PTO_INST RecordEvent
TEXTRACT(DstTileData& dst, SrcTileData& src, uint16_t indexRow = 0, uint16_t indexCol = 0, WaitEvents&... events)
{
    ::pto::detail::PtoWaitEvents(events...);
    if constexpr (is_conv_tile_v<DstTileData> || is_conv_tile_v<SrcTileData>) {
        MAP_INSTR_IMPL(TEXTRACT, dst, src, indexRow, indexCol);
    } else if (::pto::mocker::IsA5SequenceFit()) {
        RECORD_INSTR_ONLY(
            TEXTRACT, dst, src, indexRow, indexCol, ::pto::mocker::SliceFitTag(dst, src, indexRow, indexCol));
    } else {
        MAP_INSTR_IMPL(TEXTRACT, dst, src, indexRow, indexCol);
    }
    return RecordEvent{};
}

template <typename DstTileData, typename SrcTileData, ReluPreMode reluMode, typename... WaitEvents>
PTO_INST RecordEvent
TEXTRACT(DstTileData& dst, SrcTileData& src, uint16_t indexRow, uint16_t indexCol, WaitEvents&... events)
{
    ::pto::detail::PtoWaitEvents(events...);
    MAP_INSTR_IMPL_T(TEXTRACT, PTO_TEMPLATE_ARGS(DstTileData, SrcTileData, reluMode), dst, src, indexRow, indexCol);
    return RecordEvent{};
}

#if defined(PTO_NPU_ARCH_A5) || defined(PTO_NPU_ARCH_A2A3)
template <
    STPhase Phase, typename DstTileData, typename SrcTileData, ReluPreMode reluMode = ReluPreMode::NoRelu,
    typename... WaitEvents>
PTO_INST RecordEvent
TEXTRACT(DstTileData& dst, SrcTileData& src, uint16_t indexRow, uint16_t indexCol, WaitEvents&... events)
{
    ::pto::detail::PtoWaitEvents(events...);
    MAP_INSTR_IMPL_T(
        TEXTRACT, PTO_TEMPLATE_ARGS(DstTileData, SrcTileData, reluMode, Phase), dst, src, indexRow, indexCol);
    return RecordEvent{};
}
#endif

template <
    typename DstTileData, typename SrcTileData, ReluPreMode reluMode = ReluPreMode::NoRelu, typename... WaitEvents>
PTO_INST RecordEvent TEXTRACT(
    DstTileData& dst, SrcTileData& src, uint64_t preQuantScalar, uint16_t indexRow, uint16_t indexCol,
    WaitEvents&... events)
{
    ::pto::detail::PtoWaitEvents(events...);
    MAP_INSTR_IMPL_T(
        TEXTRACT, PTO_TEMPLATE_ARGS(DstTileData, SrcTileData, reluMode), dst, src, preQuantScalar, indexRow, indexCol);
    return RecordEvent{};
}

#if defined(PTO_NPU_ARCH_A5) || defined(PTO_NPU_ARCH_A2A3)
template <
    STPhase Phase, typename DstTileData, typename SrcTileData, ReluPreMode reluMode = ReluPreMode::NoRelu,
    typename... WaitEvents>
PTO_INST RecordEvent TEXTRACT(
    DstTileData& dst, SrcTileData& src, uint64_t preQuantScalar, uint16_t indexRow, uint16_t indexCol,
    WaitEvents&... events)
{
    ::pto::detail::PtoWaitEvents(events...);
    MAP_INSTR_IMPL_T(
        TEXTRACT, PTO_TEMPLATE_ARGS(DstTileData, SrcTileData, reluMode, Phase), dst, src, preQuantScalar, indexRow,
        indexCol);
    return RecordEvent{};
}
#endif

template <
    typename DstTileData, typename SrcTileData, typename FpTileData, ReluPreMode reluMode = ReluPreMode::NoRelu,
    typename... WaitEvents,
    std::enable_if_t<
        is_tile_data_v<FpTileData> && (FpTileData::Loc == TileType::Scaling) && all_events_v<WaitEvents...>, int> = 0>
PTO_INST RecordEvent TEXTRACT(
    DstTileData& dst, SrcTileData& src, FpTileData& fp, uint16_t indexRow, uint16_t indexCol, WaitEvents&... events)
{
    ::pto::detail::PtoWaitEvents(events...);
    MAP_INSTR_IMPL_T(
        TEXTRACT, PTO_TEMPLATE_ARGS(DstTileData, SrcTileData, FpTileData, reluMode), dst, src, fp, indexRow, indexCol);
    return RecordEvent{};
}

#if defined(PTO_NPU_ARCH_A5) || defined(PTO_NPU_ARCH_A2A3)
template <
    STPhase Phase, typename DstTileData, typename SrcTileData, typename FpTileData,
    ReluPreMode reluMode = ReluPreMode::NoRelu, typename... WaitEvents,
    std::enable_if_t<
        is_tile_data_v<FpTileData> && (FpTileData::Loc == TileType::Scaling) && all_events_v<WaitEvents...>, int> = 0>
PTO_INST RecordEvent TEXTRACT(
    DstTileData& dst, SrcTileData& src, FpTileData& fp, uint16_t indexRow, uint16_t indexCol, WaitEvents&... events)
{
    ::pto::detail::PtoWaitEvents(events...);
    MAP_INSTR_IMPL_T(
        TEXTRACT, PTO_TEMPLATE_ARGS(DstTileData, SrcTileData, FpTileData, reluMode, Phase), dst, src, fp, indexRow,
        indexCol);
    return RecordEvent{};
}
#endif

template <
    typename DstTileData, typename SrcTileData, typename FpTileData, ReluPreMode reluMode = ReluPreMode::NoRelu,
    typename... WaitEvents, std::enable_if_t<is_tile_data_v<FpTileData> && all_events_v<WaitEvents...>, int> = 0>
PTO_INST RecordEvent TEXTRACT_FP(
    DstTileData& dst, SrcTileData& src, FpTileData& fp, uint16_t indexRow, uint16_t indexCol, WaitEvents&... events)
{
    ::pto::detail::PtoWaitEvents(events...);
    if (::pto::mocker::IsA5SequenceFit()) {
        const auto tag = ::pto::mocker::FpSliceFitTag<false, reluMode>(dst, src, fp, indexRow, indexCol);
        RECORD_INSTR_ONLY(TEXTRACT, dst, tag);
    } else {
#if defined(PTO_NPU_ARCH_A2A3)
        // The A5 formula harness also compiles against A2/A3 stubs. Its Vec
        // destination must not instantiate the A2/A3 Acc-to-Mat implementation.
        if constexpr (DstTileData::Loc == TileType::Mat) {
#endif
            MAP_INSTR_IMPL_T(
                TEXTRACT, PTO_TEMPLATE_ARGS(DstTileData, SrcTileData, FpTileData, reluMode), dst, src, fp, indexRow,
                indexCol);
#if defined(PTO_NPU_ARCH_A2A3)
        } else {
            RECORD_INSTR_ONLY(TEXTRACT, dst, ::pto::mocker::A5SequenceFitTag{"unsupported_fp_slice"});
        }
#endif
    }
    return RecordEvent{};
}

#if defined(PTO_NPU_ARCH_A5) || defined(PTO_NPU_ARCH_A2A3)
template <
    STPhase Phase, typename DstTileData, typename SrcTileData, typename FpTileData,
    ReluPreMode reluMode = ReluPreMode::NoRelu, typename... WaitEvents,
    std::enable_if_t<is_tile_data_v<FpTileData> && all_events_v<WaitEvents...>, int> = 0>
PTO_INST RecordEvent TEXTRACT_FP(
    DstTileData& dst, SrcTileData& src, FpTileData& fp, uint16_t indexRow, uint16_t indexCol, WaitEvents&... events)
{
    ::pto::detail::PtoWaitEvents(events...);
    MAP_INSTR_IMPL_T(
        TEXTRACT, PTO_TEMPLATE_ARGS(DstTileData, SrcTileData, FpTileData, reluMode, Phase), dst, src, fp, indexRow,
        indexCol);
    return RecordEvent{};
}
#endif

template <
    typename DstTileData, typename SrcTileData, typename FpTileData, AccToVecMode mode,
    ReluPreMode reluMode = ReluPreMode::NoRelu, typename... WaitEvents,
    std::enable_if_t<
        is_tile_data_v<FpTileData> && (FpTileData::Loc == TileType::Scaling) && all_events_v<WaitEvents...>, int> = 0>
PTO_INST RecordEvent TEXTRACT(
    DstTileData& dst, SrcTileData& src, FpTileData& fp, uint16_t indexRow, uint16_t indexCol, WaitEvents&... events)
{
    ::pto::detail::PtoWaitEvents(events...);
    (void)mode;
    RECORD_INSTR_ONLY(TEXTRACT, dst, src, fp, indexRow, indexCol);
    return RecordEvent{};
}

template <
    typename TileData, typename ConvTileData, SetFmatrixMode FmatrixMode = SetFmatrixMode::FMATRIX_A_MANUAL,
    typename... WaitEvents>
PTO_INST RecordEvent
TIMG2COL(TileData& dst, ConvTileData& src, uint16_t posM = 0, uint16_t posK = 0, WaitEvents&... events)
{
    ::pto::detail::PtoWaitEvents(events...);
    if (::pto::mocker::IsA5SequenceFit()) {
        const auto tag = ::pto::mocker::ImageFitTag<FmatrixMode>(dst, src, posM, posK);
        RECORD_INSTR_ONLY(TIMG2COL, dst, tag);
        if constexpr (FmatrixMode == SetFmatrixMode::FMATRIX_B_AUTO) {
            ::pto::mocker::ImageManualFitState() = {};
        }
    } else {
        if constexpr (TileData::isRowMajor) {
            MAP_INSTR_IMPL_T(TIMG2COL, PTO_TEMPLATE_ARGS(TileData, ConvTileData, FmatrixMode), dst, src, posM, posK);
        } else {
            const ::pto::mocker::A5SequenceFitTag tag{"unsupported_image"};
            RECORD_INSTR_ONLY(TIMG2COL, dst, tag);
        }
    }
    return RecordEvent{};
}

template <typename FpTileData, typename... WaitEvents>
PTO_INST RecordEvent SET_QUANT_VECTOR(FpTileData& fpTile, WaitEvents&... events)
{
    detail::PtoWaitEvents(events...);
    using F = std::remove_cvref_t<FpTileData>;
    bool supported = false;
#ifndef __PTO_AUTO__
    if constexpr (
        std::is_same_v<typename F::DType, uint64_t> && F::Loc == TileType::Scaling && F::isRowMajor &&
        F::SFractal == SLayout::NoneBox && F::Rows == 1 && F::Cols == 128 && F::RowStride == 128) {
        // In the costmodel's NPU TASSIGN implementation, a nonzero explicit
        // address is preserved as a pointer value. Unassigned tiles instead
        // allocate host storage in data(); these are outside this measured domain.
        const auto address = reinterpret_cast<std::uintptr_t>(fpTile.data());
        supported = fpTile.GetValidRow() == 1 && fpTile.GetValidCol() == 128 && address >= 1024 && address <= 3072 &&
                    address % 128 == 0;
    }
#endif
    const pto::mocker::A5SequenceFitTag tag{supported ? "quant_vector_scaling_u64_128" : "unsupported_quant_vector"};
    RECORD_INSTR_ONLY(SET_QUANT_VECTOR, fpTile, tag);
    return RecordEvent{};
}

template <typename OutType, typename... WaitEvents>
PTO_INST RecordEvent SET_QUANT_SCALAR(float preQuantScalar, WaitEvents&... events)
{
    ::pto::detail::PtoWaitEvents(events...);
    // Positive finite scales in the measured half-output domain. Comparisons also reject NaN/Inf.
    const bool supported = std::is_same_v<OutType, half> && preQuantScalar >= 0.5f && preQuantScalar <= 7.0f;
    const ::pto::mocker::A5SequenceFitTag tag{supported ? "quant_scalar_half_positive" : "unsupported_quant_scalar"};
    RECORD_INSTR_ONLY(SET_QUANT_SCALAR, preQuantScalar, tag);
    return RecordEvent{};
}

template <typename ConvTileData, SetFmatrixMode FmatrixMode = SetFmatrixMode::FMATRIX_A_MANUAL, typename... WaitEvents>
PTO_INST RecordEvent SETFMATRIX(ConvTileData& src, WaitEvents&... events)
{
    ::pto::detail::PtoWaitEvents(events...);
    if constexpr (FmatrixMode == SetFmatrixMode::FMATRIX_B_MANUAL) {
        auto& state = ::pto::mocker::ImageManualFitState();
        state.fmapH = src.GetFmapH();
        state.fmapW = src.GetFmapW();
        state.zeroPadding = true;
        for (int i = 0; i < 4; ++i) {
            state.zeroPadding = state.zeroPadding && src.GetPadList(i) == 0;
        }
    }
    if (::pto::perf_sim::GetCostModelCtx().npu_arch == ::pto::perf_sim::kAscend950Arch) {
        const auto tag = ::pto::mocker::ImageConfigFitTag<0, FmatrixMode>(src);
        RECORD_INSTR_ONLY(SETFMATRIX, src, tag);
    } else {
        MAP_INSTR_IMPL_T(SETFMATRIX, PTO_TEMPLATE_ARGS(ConvTileData, FmatrixMode), src);
    }
    return RecordEvent{};
}

#if defined(PTO_NPU_ARCH_A2A3) || defined(PTO_NPU_ARCH_A5)
template <typename ConvTileData, SetFmatrixMode FmatrixMode = SetFmatrixMode::FMATRIX_A_MANUAL, typename... WaitEvents>
PTO_INST RecordEvent SET_IMG2COL_RPT(ConvTileData& src, WaitEvents&... events)
{
    ::pto::detail::PtoWaitEvents(events...);
    if constexpr (FmatrixMode == SetFmatrixMode::FMATRIX_B_MANUAL) {
        auto& state = ::pto::mocker::ImageManualFitState();
        state.repeatOne = src.GetRepeatTime() == 1 && src.GetRepeatMode() == 0 && src.GetRepeatStride() == 0 &&
                          src.GetDstMposition() == 0;
        state.dstStride = src.GetDstStride();
    }
    if (::pto::mocker::IsA5SequenceFit()) {
        const auto tag = ::pto::mocker::ImageConfigFitTag<1, FmatrixMode>(src);
        RECORD_INSTR_ONLY(SET_IMG2COL_RPT, src, tag);
    } else {
        MAP_INSTR_IMPL_T(SET_IMG2COL_RPT, PTO_TEMPLATE_ARGS(ConvTileData, FmatrixMode), src);
    }
    return RecordEvent{};
}

template <typename ConvTileData, SetFmatrixMode FmatrixMode = SetFmatrixMode::FMATRIX_A_MANUAL, typename... WaitEvents>
PTO_INST RecordEvent SET_IMG2COL_PADDING(ConvTileData& src, WaitEvents&... events)
{
    ::pto::detail::PtoWaitEvents(events...);
    if constexpr (FmatrixMode == SetFmatrixMode::FMATRIX_B_MANUAL) {
        ::pto::mocker::ImageManualFitState().zeroPadValue = src.GetPadValue() == 0;
    }
    if (::pto::mocker::IsA5SequenceFit()) {
        const auto tag = ::pto::mocker::ImageConfigFitTag<2, FmatrixMode>(src);
        RECORD_INSTR_ONLY(SET_IMG2COL_PADDING, src, tag);
    } else {
        MAP_INSTR_IMPL_T(SET_IMG2COL_PADDING, PTO_TEMPLATE_ARGS(ConvTileData, FmatrixMode), src);
    }
    return RecordEvent{};
}
#endif
template <typename DstTileData, typename SrcTileData, typename... WaitEvents>
PTO_INST RecordEvent
TINSERT(DstTileData& dst, SrcTileData& src, uint16_t indexRow, uint16_t indexCol, WaitEvents&... events)
{
    ::pto::detail::PtoWaitEvents(events...);
    if constexpr (is_conv_tile_v<DstTileData> || is_conv_tile_v<SrcTileData>) {
        MAP_INSTR_IMPL(TINSERT, dst, src, indexRow, indexCol);
    } else if (::pto::mocker::IsA5SequenceFit()) {
        RECORD_INSTR_ONLY(
            TINSERT, dst, src, indexRow, indexCol, ::pto::mocker::SliceFitTag(src, dst, indexRow, indexCol));
    } else {
        MAP_INSTR_IMPL(TINSERT, dst, src, indexRow, indexCol);
    }
    return RecordEvent{};
}

template <typename DstTileData, typename SrcTileData, ReluPreMode reluMode, typename... WaitEvents>
PTO_INST RecordEvent
TINSERT(DstTileData& dst, SrcTileData& src, uint16_t indexRow, uint16_t indexCol, WaitEvents&... events)
{
    ::pto::detail::PtoWaitEvents(events...);
    MAP_INSTR_IMPL_T(TINSERT, PTO_TEMPLATE_ARGS(DstTileData, SrcTileData, reluMode), dst, src, indexRow, indexCol);
    return RecordEvent{};
}

#if defined(PTO_NPU_ARCH_A5) || defined(PTO_NPU_ARCH_A2A3)
template <
    STPhase Phase, typename DstTileData, typename SrcTileData, ReluPreMode reluMode = ReluPreMode::NoRelu,
    typename... WaitEvents>
PTO_INST RecordEvent
TINSERT(DstTileData& dst, SrcTileData& src, uint16_t indexRow, uint16_t indexCol, WaitEvents&... events)
{
    ::pto::detail::PtoWaitEvents(events...);
    MAP_INSTR_IMPL_T(
        TINSERT, PTO_TEMPLATE_ARGS(DstTileData, SrcTileData, reluMode, Phase), dst, src, indexRow, indexCol);
    return RecordEvent{};
}
#endif

template <
    typename DstTileData, typename SrcTileData, ReluPreMode reluMode = ReluPreMode::NoRelu, typename... WaitEvents>
PTO_INST RecordEvent TINSERT(
    DstTileData& dst, SrcTileData& src, uint64_t preQuantScalar, uint16_t indexRow, uint16_t indexCol,
    WaitEvents&... events)
{
    ::pto::detail::PtoWaitEvents(events...);
    MAP_INSTR_IMPL_T(
        TINSERT, PTO_TEMPLATE_ARGS(DstTileData, SrcTileData, reluMode), dst, src, preQuantScalar, indexRow, indexCol);
    return RecordEvent{};
}

#if defined(PTO_NPU_ARCH_A5) || defined(PTO_NPU_ARCH_A2A3)
template <
    STPhase Phase, typename DstTileData, typename SrcTileData, ReluPreMode reluMode = ReluPreMode::NoRelu,
    typename... WaitEvents>
PTO_INST RecordEvent TINSERT(
    DstTileData& dst, SrcTileData& src, uint64_t preQuantScalar, uint16_t indexRow, uint16_t indexCol,
    WaitEvents&... events)
{
    ::pto::detail::PtoWaitEvents(events...);
    MAP_INSTR_IMPL_T(
        TINSERT, PTO_TEMPLATE_ARGS(DstTileData, SrcTileData, reluMode, Phase), dst, src, preQuantScalar, indexRow,
        indexCol);
    return RecordEvent{};
}
#endif

template <
    typename DstTileData, typename SrcTileData, typename FpTileData, ReluPreMode reluMode = ReluPreMode::NoRelu,
    typename... WaitEvents,
    std::enable_if_t<
        is_tile_data_v<FpTileData> && (FpTileData::Loc == TileType::Scaling) && all_events_v<WaitEvents...>, int> = 0>
PTO_INST RecordEvent
TINSERT(DstTileData& dst, SrcTileData& src, FpTileData& fp, uint16_t indexRow, uint16_t indexCol, WaitEvents&... events)
{
    ::pto::detail::PtoWaitEvents(events...);
    MAP_INSTR_IMPL_T(
        TINSERT, PTO_TEMPLATE_ARGS(DstTileData, SrcTileData, FpTileData, reluMode), dst, src, fp, indexRow, indexCol);
    return RecordEvent{};
}

#if defined(PTO_NPU_ARCH_A5) || defined(PTO_NPU_ARCH_A2A3)
template <
    STPhase Phase, typename DstTileData, typename SrcTileData, typename FpTileData,
    ReluPreMode reluMode = ReluPreMode::NoRelu, typename... WaitEvents,
    std::enable_if_t<
        is_tile_data_v<FpTileData> && (FpTileData::Loc == TileType::Scaling) && all_events_v<WaitEvents...>, int> = 0>
PTO_INST RecordEvent
TINSERT(DstTileData& dst, SrcTileData& src, FpTileData& fp, uint16_t indexRow, uint16_t indexCol, WaitEvents&... events)
{
    ::pto::detail::PtoWaitEvents(events...);
    MAP_INSTR_IMPL_T(
        TINSERT, PTO_TEMPLATE_ARGS(DstTileData, SrcTileData, FpTileData, reluMode, Phase), dst, src, fp, indexRow,
        indexCol);
    return RecordEvent{};
}
#endif

template <
    typename DstTileData, typename SrcTileData, typename FpTileData, ReluPreMode reluMode = ReluPreMode::NoRelu,
    typename... WaitEvents, std::enable_if_t<is_tile_data_v<FpTileData> && all_events_v<WaitEvents...>, int> = 0>
PTO_INST RecordEvent TINSERT_FP(
    DstTileData& dst, SrcTileData& src, FpTileData& fp, uint16_t indexRow, uint16_t indexCol, WaitEvents&... events)
{
    ::pto::detail::PtoWaitEvents(events...);
    if (::pto::mocker::IsA5SequenceFit()) {
        const auto tag = ::pto::mocker::FpSliceFitTag<true, reluMode>(dst, src, fp, indexRow, indexCol);
        RECORD_INSTR_ONLY(TINSERT, dst, tag);
    } else {
#if defined(PTO_NPU_ARCH_A2A3)
        // Preserve native Acc-to-Mat execution without instantiating it for
        // the A5-only Acc-to-Vec formula inputs in the shared test harness.
        if constexpr (DstTileData::Loc == TileType::Mat) {
#endif
            MAP_INSTR_IMPL_T(
                TINSERT, PTO_TEMPLATE_ARGS(DstTileData, SrcTileData, FpTileData, reluMode), dst, src, fp, indexRow,
                indexCol);
#if defined(PTO_NPU_ARCH_A2A3)
        } else {
            RECORD_INSTR_ONLY(TINSERT, dst, ::pto::mocker::A5SequenceFitTag{"unsupported_fp_slice"});
        }
#endif
    }
    return RecordEvent{};
}

#if defined(PTO_NPU_ARCH_A5) || defined(PTO_NPU_ARCH_A2A3)
template <
    STPhase Phase, typename DstTileData, typename SrcTileData, typename FpTileData,
    ReluPreMode reluMode = ReluPreMode::NoRelu, typename... WaitEvents,
    std::enable_if_t<is_tile_data_v<FpTileData> && all_events_v<WaitEvents...>, int> = 0>
PTO_INST RecordEvent TINSERT_FP(
    DstTileData& dst, SrcTileData& src, FpTileData& fp, uint16_t indexRow, uint16_t indexCol, WaitEvents&... events)
{
    ::pto::detail::PtoWaitEvents(events...);
    MAP_INSTR_IMPL_T(
        TINSERT, PTO_TEMPLATE_ARGS(DstTileData, SrcTileData, FpTileData, reluMode, Phase), dst, src, fp, indexRow,
        indexCol);
    return RecordEvent{};
}
#endif

template <
    typename DstTileData, typename SrcTileData, typename FpTileData, AccToVecMode mode,
    ReluPreMode reluMode = ReluPreMode::NoRelu, typename... WaitEvents,
    std::enable_if_t<
        is_tile_data_v<FpTileData> && (FpTileData::Loc == TileType::Scaling) && all_events_v<WaitEvents...>, int> = 0>
PTO_INST RecordEvent
TINSERT(DstTileData& dst, SrcTileData& src, FpTileData& fp, uint16_t indexRow, uint16_t indexCol, WaitEvents&... events)
{
    ::pto::detail::PtoWaitEvents(events...);
    (void)mode;
    RECORD_INSTR_ONLY(TINSERT, dst, src, fp, indexRow, indexCol);
    return RecordEvent{};
}

template <
    typename TileData, PadValue PadVal = PadValue::Zero, std::enable_if_t<(TileData::Loc == TileType::Mat), int> = 0,
    typename... WaitEvents>
PTO_INST RecordEvent TFILLPAD(TileData& dst, TileData& src, WaitEvents&... events)
{
    ::pto::detail::PtoWaitEvents(events...);
    MAP_INSTR_IMPL_T(TFILLPAD, PTO_TEMPLATE_ARGS(TileData, PadVal), dst, src);
    return RecordEvent{};
}

template <
    TFillPadMode mode = TFillPadMode::Normal, typename DstTileData, typename SrcTileData,
    std::enable_if_t<(DstTileData::Loc == TileType::Vec) && (SrcTileData::Loc == TileType::Vec), int> = 0,
    typename... WaitEvents>
PTO_INST RecordEvent TFILLPAD(DstTileData& dst, SrcTileData& src, WaitEvents&... events)
{
    static_assert(
        mode == TFillPadMode::Normal || mode == TFillPadMode::InPlace || mode == TFillPadMode::Expand,
        "TFILLPAD: invalid mode.");
    ::pto::detail::PtoWaitEvents(events...);
    if constexpr (mode == TFillPadMode::Normal) {
        // The new metadata calibration does not replace the pre-existing
        // fp16 padding contract. Its original lowering/routing stays intact.
        if (::pto::mocker::IsA5SequenceFit() && !std::is_same_v<typename DstTileData::DType, half>) {
            RECORD_INSTR_ONLY(TFILLPAD, dst, src, ::pto::mocker::PaddingGeometryTag<false>(dst, src));
        } else {
            MAP_INSTR_IMPL_T(TFILLPAD, PTO_TEMPLATE_ARGS(DstTileData, SrcTileData), dst, src);
        }
    } else if constexpr (mode == TFillPadMode::InPlace) {
        if (::pto::mocker::IsA5SequenceFit()) {
            RECORD_INSTR_ONLY(TFILLPAD_INPLACE, dst, src, ::pto::mocker::InplacePaddingTag(dst, src));
        } else {
            MAP_INSTR_IMPL(TFILLPAD_INPLACE, dst, src);
        }
    } else if constexpr (mode == TFillPadMode::Expand) {
        if (::pto::mocker::IsA5SequenceFit() && !std::is_same_v<typename DstTileData::DType, half>) {
            RECORD_INSTR_ONLY(TFILLPAD_EXPAND, dst, src, ::pto::mocker::PaddingGeometryTag<true>(dst, src));
        } else {
            MAP_INSTR_IMPL(TFILLPAD_EXPAND, dst, src);
        }
    }
    return RecordEvent{};
}

template <uint16_t Rounds = 10, typename DstTile, typename... WaitEvents>
PTO_INST RecordEvent TRANDOM(DstTile& dst, TRandomKey& key, TRandomCounter& counter, WaitEvents&... events)
{
    static_assert(Rounds == 7 || Rounds == 10, "TRANDOM supports 7 or 10 rounds.");
    static_assert(
        std::is_same_v<typename DstTile::DType, int32_t> || std::is_same_v<typename DstTile::DType, uint32_t>,
        "TRANDOM requires int32 or uint32.");
    static_assert(DstTile::isRowMajor, "TRANDOM requires row major layout.");
    ::pto::detail::PtoWaitEvents(events...);
    // Philox key/counter values do not change its fixed arithmetic instruction path.
    (void)key;
    (void)counter;
    RECORD_INSTR_ONLY(TRANDOM, dst, ::pto::mocker::RandomFitTag<Rounds>(dst));
    return RecordEvent{};
}

template <typename DstTileData, typename SrcTileData, typename... WaitEvents>
PTO_INST RecordEvent TFILLPAD_INPLACE(DstTileData& dst, SrcTileData& src, WaitEvents&... events)
{
    return TFILLPAD<TFillPadMode::InPlace>(dst, src, events...);
}

template <typename DstTileData, typename SrcTileData, typename... WaitEvents>
PTO_INST RecordEvent TFILLPAD_EXPAND(DstTileData& dst, SrcTileData& src, WaitEvents&... events)
{
    return TFILLPAD<TFillPadMode::Expand>(dst, src, events...);
}

template <typename DstTileData, typename SrcTileData, typename IdxTileData, typename TmpTileData>
PTO_INST RecordEvent TSORT32(DstTileData& dst, SrcTileData& src, IdxTileData& idx, TmpTileData& tmp)
{
    MAP_INSTR_IMPL(TSORT32, dst, src, idx, tmp);
    return RecordEvent{};
}

template <typename DstTileData, typename SrcTileData, typename IdxTileData>
PTO_INST RecordEvent TSORT32(DstTileData& dst, SrcTileData& src, IdxTileData& idx)
{
    MAP_INSTR_IMPL(TSORT32, dst, src, idx);
    return RecordEvent{};
}

template <typename TileDataD, typename TileDataS0, typename TileDataS1, typename TileDataTmp, typename... WaitEvents>
PTO_INST RecordEvent
TGATHER(TileDataD& dst, TileDataS0& src0, TileDataS1& src1, TileDataTmp& tmp, WaitEvents&... events)
{
    ::pto::detail::PtoWaitEvents(events...);
    if (::pto::mocker::IsA5SequenceFit()) {
        const auto tag =
            ::pto::mocker::DefaultIndexedGeometryTag<::pto::mocker::lightweight::DefaultInputProfile::ZeroIndex>(
                dst, src0, src1);
        ::RecordInstrFromFirst("TGATHER", dst, tag);
        return RecordEvent{};
    }
    if constexpr (
        (sizeof(typename TileDataD::DType) == 2 || sizeof(typename TileDataD::DType) == 4) &&
        sizeof(typename TileDataS1::DType) == 4 &&
        std::is_same_v<typename TileDataTmp::DType, typename TileDataS1::DType> &&
        std::is_same_v<typename TileDataD::DType, typename TileDataS0::DType>) {
        MAP_INSTR_IMPL(TGATHER, dst, src0, src1, tmp);
    } else {
        // A5 does not use tmp and accepts b16 indices. Those calls must not
        // instantiate the incompatible A2/A3 mock in the formula test harness.
        ::pto::mocker::PtoInstrScope scope("TGATHER");
        scope.Finish();
    }
    return RecordEvent{};
}

template <
    typename TileDataD, typename TileDataS, typename TileDataC, typename TileDataTmp, CmpMode cmpMode, int offset,
    typename... WaitEvents>
PTO_INST RecordEvent TGATHER(
    TileDataD& dst, TileDataS& src0, typename TileDataS::DType k_value, TileDataC& cdst, TileDataTmp& tmp,
    WaitEvents&... events)
{
    ::pto::detail::PtoWaitEvents(events...);
    MAP_INSTR_IMPL_T(
        TGATHER, PTO_TEMPLATE_ARGS(TileDataD, TileDataS, TileDataC, TileDataTmp, cmpMode, offset), dst, src0, k_value,
        cdst, tmp);
    return RecordEvent{};
}

template <typename TileData, typename T, int descending, typename... WaitEvents>
PTO_INST RecordEvent TCI(TileData& dst, T start, WaitEvents&... events)
{
    ::pto::detail::PtoWaitEvents(events...);
    if (::pto::mocker::IsA5SequenceFit()) {
        const bool supported = (descending == 0 || descending == 1) && std::is_same_v<typename TileData::DType, T> &&
                               ::pto::mocker::DenseSequenceTag(dst).key == "dense";
        RECORD_INSTR_ONLY(TCI, dst, start, (::pto::mocker::A5SequenceFitTag{supported ? "ci_scalar" : "unsupported"}));
    } else {
        RECORD_INSTR_ONLY(TCI, dst, start, (::pto::mocker::A5SequenceFitTag{"unsupported"}));
    }
    return RecordEvent{};
}

template <typename TileData, typename TileDataTmp, typename T, int descending, typename... WaitEvents>
PTO_INST RecordEvent TCI(TileData& dst, T start, TileDataTmp& tmp, WaitEvents&... events)
{
    ::pto::detail::PtoWaitEvents(events...);
    const bool supported = ::pto::mocker::IsA5SequenceFit() && (descending == 0 || descending == 1) &&
                           std::is_same_v<typename TileData::DType, T> &&
                           ::pto::mocker::DenseSequenceTag(dst, tmp).key == "dense";
    RECORD_INSTR_ONLY(TCI, dst, start, tmp, (::pto::mocker::A5SequenceFitTag{supported ? "ci_vector" : "unsupported"}));
    return RecordEvent{};
}

template <typename TileDataDst, typename TileDataSrc, typename... WaitEvents>
PTO_INST RecordEvent
TINTERLEAVE(TileDataDst& dst1, TileDataDst& dst0, TileDataSrc& src1, TileDataSrc& src0, WaitEvents&... events)
{
    ::pto::detail::PtoWaitEvents(events...);
    RECORD_INSTR_ONLY(TINTERLEAVE, dst1, dst0, src1, src0, ::pto::mocker::DenseSequenceTag(dst1, dst0, src1, src0));
    return RecordEvent{};
}

template <typename TileDataDst, typename TileDataSrc, typename... WaitEvents>
PTO_INST RecordEvent
TDEINTERLEAVE(TileDataDst& dst1, TileDataDst& dst0, TileDataSrc& src1, TileDataSrc& src0, WaitEvents&... events)
{
    ::pto::detail::PtoWaitEvents(events...);
    RECORD_INSTR_ONLY(TDEINTERLEAVE, dst1, dst0, src1, src0, ::pto::mocker::DenseSequenceTag(dst1, dst0, src1, src0));
    return RecordEvent{};
}

template <typename TileData, int isUpperOrLower, typename... WaitEvents>
PTO_INST RecordEvent TTRI(TileData& dst, int diagonal, WaitEvents&... events)
{
    ::pto::detail::PtoWaitEvents(events...);
    if constexpr (
        std::is_same_v<typename TileData::DType, float> || std::is_same_v<typename TileData::DType, half> ||
        std::is_same_v<typename TileData::DType, int16_t> || std::is_same_v<typename TileData::DType, int32_t>) {
        if (::pto::mocker::IsA5SequenceFit()) {
            RECORD_INSTR_ONLY(TTRI, dst, diagonal, ::pto::mocker::TriangleGeometryTag<isUpperOrLower>(dst, diagonal));
            return RecordEvent{};
        }
    }
    MAP_INSTR_IMPL_T(TTRI, PTO_TEMPLATE_ARGS(TileData, isUpperOrLower), dst, diagonal);
    return RecordEvent{};
}

template <
    typename DstTileData, typename SrcTileData, MaskPattern maskPattern = MaskPattern::P1111,
    auto gatherType = GatherAxis::GATHER_ROW, typename... WaitEvents>
PTO_INST RecordEvent TGATHER(DstTileData& dst, SrcTileData& src, WaitEvents&... events)
{
    ::pto::detail::PtoWaitEvents(events...);
    ::pto::mocker::PtoInstrScope scope("TGATHER");
    TGATHER_IMPL<DstTileData, SrcTileData, maskPattern, gatherType>(dst, src);
    scope.Finish();
    ::RecordInstrFromFirst("TGATHER", dst, src, maskPattern, gatherType);
    return RecordEvent{};
}

template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1, typename... WaitEvents>
PTO_INST RecordEvent TPARTADD(TileDataDst& dst, TileDataSrc0& src0, TileDataSrc1& src1, WaitEvents&... events)
{
    ::pto::detail::PtoWaitEvents(events...);
    if (::pto::mocker::IsA5SequenceFit() && !std::is_same_v<typename TileDataDst::DType, half>) {
        RECORD_INSTR_ONLY(TPARTADD, dst, src0, src1, ::pto::mocker::PartialGeometryTag(dst, src0, src1));
    } else {
        MAP_INSTR_IMPL(TPARTADD, dst, src0, src1);
    }
    return RecordEvent{};
}

template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1, typename... WaitEvents>
PTO_INST RecordEvent TPARTMUL(TileDataDst& dst, TileDataSrc0& src0, TileDataSrc1& src1, WaitEvents&... events)
{
    ::pto::detail::PtoWaitEvents(events...);
    if (::pto::mocker::IsA5SequenceFit() && !std::is_same_v<typename TileDataDst::DType, half>) {
        RECORD_INSTR_ONLY(TPARTMUL, dst, src0, src1, ::pto::mocker::PartialGeometryTag(dst, src0, src1));
    } else {
        MAP_INSTR_IMPL(TPARTMUL, dst, src0, src1);
    }
    return RecordEvent{};
}

template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1, typename... WaitEvents>
PTO_INST RecordEvent TPARTMAX(TileDataDst& dst, TileDataSrc0& src0, TileDataSrc1& src1, WaitEvents&... events)
{
    ::pto::detail::PtoWaitEvents(events...);
    if (::pto::mocker::IsA5SequenceFit() && !std::is_same_v<typename TileDataDst::DType, half>) {
        RECORD_INSTR_ONLY(TPARTMAX, dst, src0, src1, ::pto::mocker::PartialGeometryTag(dst, src0, src1));
    } else {
        if constexpr (
            std::is_same_v<typename TileDataDst::DType, int8_t> ||
            std::is_same_v<typename TileDataDst::DType, uint8_t>) {
            ::pto::mocker::PtoInstrScope scope("TPARTMAX");
            scope.Finish();
        } else {
            MAP_INSTR_IMPL(TPARTMAX, dst, src0, src1);
        }
    }
    return RecordEvent{};
}

template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1, typename... WaitEvents>
PTO_INST RecordEvent TPARTMIN(TileDataDst& dst, TileDataSrc0& src0, TileDataSrc1& src1, WaitEvents&... events)
{
    ::pto::detail::PtoWaitEvents(events...);
    if (::pto::mocker::IsA5SequenceFit() && !std::is_same_v<typename TileDataDst::DType, half>) {
        RECORD_INSTR_ONLY(TPARTMIN, dst, src0, src1, ::pto::mocker::PartialGeometryTag(dst, src0, src1));
    } else {
        if constexpr (
            std::is_same_v<typename TileDataDst::DType, int8_t> ||
            std::is_same_v<typename TileDataDst::DType, uint8_t>) {
            ::pto::mocker::PtoInstrScope scope("TPARTMIN");
            scope.Finish();
        } else {
            MAP_INSTR_IMPL(TPARTMIN, dst, src0, src1);
        }
    }
    return RecordEvent{};
}

template <typename TileDataD, typename TileDataS, typename TmpTileData, typename... WaitEvents>
PTO_INST RecordEvent
TCVT(TileDataD& dst, TileDataS& src, TmpTileData& tmp, RoundMode mode, SaturationMode satMode, WaitEvents&... events)
{
    ::pto::detail::PtoWaitEvents(events...);
    MAP_INSTR_IMPL(TCVT, dst, src, tmp, mode, satMode);
    return RecordEvent{};
}

template <typename TileDataD, typename TileDataS, typename TmpTileData, typename... WaitEvents>
PTO_INST RecordEvent TCVT(TileDataD& dst, TileDataS& src, TmpTileData& tmp, RoundMode mode, WaitEvents&... events)
{
    ::pto::detail::PtoWaitEvents(events...);
    MAP_INSTR_IMPL(TCVT, dst, src, tmp, mode);
    return RecordEvent{};
}

template <typename TileDataD, typename TileDataS, typename... WaitEvents>
PTO_INST RecordEvent TCVT(TileDataD& dst, TileDataS& src, RoundMode mode, SaturationMode satMode, WaitEvents&... events)
{
    ::pto::detail::PtoWaitEvents(events...);
    MAP_INSTR_IMPL(TCVT, dst, src, mode, satMode);
    return RecordEvent{};
}

template <typename TileDataD, typename TileDataS, typename... WaitEvents>
PTO_INST RecordEvent TCVT(TileDataD& dst, TileDataS& src, RoundMode mode, WaitEvents&... events)
{
    ::pto::detail::PtoWaitEvents(events...);
    MAP_INSTR_IMPL(TCVT, dst, src, mode);
    return RecordEvent{};
}

template <typename DstTileData, typename SrcTileData, typename... WaitEvents>
PTO_INST RecordEvent TMOV(DstTileData& dst, SrcTileData& src, WaitEvents&... events)
{
    ::pto::detail::PtoWaitEvents(events...);
    MAP_INSTR_IMPL(TMOV, dst, src);
    return RecordEvent{};
}

template <STPhase Phase, typename DstTileData, typename SrcTileData, typename... WaitEvents>
PTO_INST RecordEvent TMOV(DstTileData& dst, SrcTileData& src, WaitEvents&... events)
{
    ::pto::detail::PtoWaitEvents(events...);
    MAP_INSTR_IMPL_T(TMOV, PTO_TEMPLATE_ARGS(DstTileData, SrcTileData, ReluPreMode::NoRelu, Phase), dst, src);
    return RecordEvent{};
}

template <typename DstTileData, typename SrcTileData, ReluPreMode reluMode, typename... WaitEvents>
PTO_INST RecordEvent TMOV(DstTileData& dst, SrcTileData& src, WaitEvents&... events)
{
    ::pto::detail::PtoWaitEvents(events...);
    MAP_INSTR_IMPL_T(TMOV, PTO_TEMPLATE_ARGS(DstTileData, SrcTileData, reluMode), dst, src);
    return RecordEvent{};
}

template <STPhase Phase, typename DstTileData, typename SrcTileData, ReluPreMode reluMode, typename... WaitEvents>
PTO_INST RecordEvent TMOV(DstTileData& dst, SrcTileData& src, WaitEvents&... events)
{
    ::pto::detail::PtoWaitEvents(events...);
    MAP_INSTR_IMPL_T(TMOV, PTO_TEMPLATE_ARGS(DstTileData, SrcTileData, reluMode, Phase), dst, src);
    return RecordEvent{};
}

template <
    typename DstTileData, typename SrcTileData, AccToVecMode mode, ReluPreMode reluMode = ReluPreMode::NoRelu,
    typename... WaitEvents>
PTO_INST RecordEvent TMOV(DstTileData& dst, SrcTileData& src, WaitEvents&... events)
{
    ::pto::detail::PtoWaitEvents(events...);
    MAP_INSTR_IMPL_T(TMOV, PTO_TEMPLATE_ARGS(DstTileData, SrcTileData, mode, reluMode), dst, src);
    return RecordEvent{};
}

template <
    STPhase Phase, typename DstTileData, typename SrcTileData, AccToVecMode mode,
    ReluPreMode reluMode = ReluPreMode::NoRelu, typename... WaitEvents>
PTO_INST RecordEvent TMOV(DstTileData& dst, SrcTileData& src, WaitEvents&... events)
{
    ::pto::detail::PtoWaitEvents(events...);
    MAP_INSTR_IMPL_T(TMOV, PTO_TEMPLATE_ARGS(DstTileData, SrcTileData, mode, reluMode, Phase), dst, src);
    return RecordEvent{};
}

template <
    typename DstTileData, typename SrcTileData, typename FpTileData, ReluPreMode reluMode = ReluPreMode::NoRelu,
    typename... WaitEvents,
    std::enable_if_t<
        is_tile_data_v<FpTileData> && (FpTileData::Loc == TileType::Scaling) && all_events_v<WaitEvents...>, int> = 0>
PTO_INST RecordEvent TMOV(DstTileData& dst, SrcTileData& src, FpTileData& fp, WaitEvents&... events)
{
    ::pto::detail::PtoWaitEvents(events...);
    MAP_INSTR_IMPL_T(TMOV, PTO_TEMPLATE_ARGS(DstTileData, SrcTileData, FpTileData, reluMode), dst, src, fp);
    return RecordEvent{};
}

template <
    STPhase Phase, typename DstTileData, typename SrcTileData, typename FpTileData,
    ReluPreMode reluMode = ReluPreMode::NoRelu, typename... WaitEvents,
    std::enable_if_t<
        is_tile_data_v<FpTileData> && (FpTileData::Loc == TileType::Scaling) && all_events_v<WaitEvents...>, int> = 0>
PTO_INST RecordEvent TMOV(DstTileData& dst, SrcTileData& src, FpTileData& fp, WaitEvents&... events)
{
    ::pto::detail::PtoWaitEvents(events...);
#if defined(PTO_NPU_ARCH_A2A3)
    MAP_INSTR_IMPL_T(TMOV, PTO_TEMPLATE_ARGS(DstTileData, SrcTileData, FpTileData, reluMode, Phase), dst, src, fp);
#else
    (void)Phase;
    RECORD_INSTR_ONLY(TMOV, dst, src, fp);
#endif
    return RecordEvent{};
}

template <
    typename DstTileData, typename SrcTileData, typename FpTileData, ReluPreMode reluMode = ReluPreMode::NoRelu,
    typename... WaitEvents, std::enable_if_t<is_tile_data_v<FpTileData> && all_events_v<WaitEvents...>, int> = 0>
PTO_INST RecordEvent TMOV_FP(DstTileData& dst, SrcTileData& src, FpTileData& fp, WaitEvents&... events)
{
    ::pto::detail::PtoWaitEvents(events...);
    MAP_INSTR_IMPL_T(TMOV, PTO_TEMPLATE_ARGS(DstTileData, SrcTileData, FpTileData, reluMode), dst, src, fp);
    return RecordEvent{};
}

template <
    STPhase Phase, typename DstTileData, typename SrcTileData, typename FpTileData,
    ReluPreMode reluMode = ReluPreMode::NoRelu, typename... WaitEvents,
    std::enable_if_t<is_tile_data_v<FpTileData> && all_events_v<WaitEvents...>, int> = 0>
PTO_INST RecordEvent TMOV_FP(DstTileData& dst, SrcTileData& src, FpTileData& fp, WaitEvents&... events)
{
    ::pto::detail::PtoWaitEvents(events...);
#if defined(PTO_NPU_ARCH_A2A3)
    MAP_INSTR_IMPL_T(TMOV, PTO_TEMPLATE_ARGS(DstTileData, SrcTileData, FpTileData, reluMode, Phase), dst, src, fp);
#else
    (void)Phase;
    RECORD_INSTR_ONLY(TMOV, dst, src, fp);
#endif
    return RecordEvent{};
}

template <
    typename DstTileData, typename SrcTileData, typename FpTileData, AccToVecMode mode,
    ReluPreMode reluMode = ReluPreMode::NoRelu, typename... WaitEvents,
    std::enable_if_t<
        is_tile_data_v<FpTileData> && (FpTileData::Loc == TileType::Scaling) && all_events_v<WaitEvents...>, int> = 0>
PTO_INST RecordEvent TMOV(DstTileData& dst, SrcTileData& src, FpTileData& fp, WaitEvents&... events)
{
    ::pto::detail::PtoWaitEvents(events...);
    (void)mode;
    RECORD_INSTR_ONLY(TMOV, dst, src, fp);
    return RecordEvent{};
}

template <
    STPhase Phase, typename DstTileData, typename SrcTileData, typename FpTileData, AccToVecMode mode,
    ReluPreMode reluMode = ReluPreMode::NoRelu, typename... WaitEvents,
    std::enable_if_t<
        is_tile_data_v<FpTileData> && (FpTileData::Loc == TileType::Scaling) && all_events_v<WaitEvents...>, int> = 0>
PTO_INST RecordEvent TMOV(DstTileData& dst, SrcTileData& src, FpTileData& fp, WaitEvents&... events)
{
    ::pto::detail::PtoWaitEvents(events...);
    (void)Phase;
    (void)mode;
    RECORD_INSTR_ONLY(TMOV, dst, src, fp);
    return RecordEvent{};
}

template <
    typename DstTileData, typename SrcTileData, ReluPreMode reluMode = ReluPreMode::NoRelu, typename... WaitEvents>
PTO_INST RecordEvent TMOV(DstTileData& dst, SrcTileData& src, uint64_t preQuantScalar, WaitEvents&... events)
{
    ::pto::detail::PtoWaitEvents(events...);
    MAP_INSTR_IMPL_T(TMOV, PTO_TEMPLATE_ARGS(DstTileData, SrcTileData, reluMode), dst, src, preQuantScalar);
    return RecordEvent{};
}

template <
    STPhase Phase, typename DstTileData, typename SrcTileData, ReluPreMode reluMode = ReluPreMode::NoRelu,
    typename... WaitEvents>
PTO_INST RecordEvent TMOV(DstTileData& dst, SrcTileData& src, uint64_t preQuantScalar, WaitEvents&... events)
{
    ::pto::detail::PtoWaitEvents(events...);
    MAP_INSTR_IMPL_T(TMOV, PTO_TEMPLATE_ARGS(DstTileData, SrcTileData, reluMode, Phase), dst, src, preQuantScalar);
    return RecordEvent{};
}

template <
    typename DstTileData, typename SrcTileData, AccToVecMode mode, ReluPreMode reluMode = ReluPreMode::NoRelu,
    typename... WaitEvents>
PTO_INST RecordEvent TMOV(DstTileData& dst, SrcTileData& src, uint64_t preQuantScalar, WaitEvents&... events)
{
    ::pto::detail::PtoWaitEvents(events...);
    MAP_INSTR_IMPL_T(TMOV, PTO_TEMPLATE_ARGS(DstTileData, SrcTileData, mode, reluMode), dst, src, preQuantScalar);
    return RecordEvent{};
}

template <
    STPhase Phase, typename DstTileData, typename SrcTileData, AccToVecMode mode,
    ReluPreMode reluMode = ReluPreMode::NoRelu, typename... WaitEvents>
PTO_INST RecordEvent TMOV(DstTileData& dst, SrcTileData& src, uint64_t preQuantScalar, WaitEvents&... events)
{
    ::pto::detail::PtoWaitEvents(events...);
    MAP_INSTR_IMPL_T(
        TMOV, PTO_TEMPLATE_ARGS(DstTileData, SrcTileData, mode, reluMode, Phase), dst, src, preQuantScalar);
    return RecordEvent{};
}

template <
    typename Pipe, typename TileProd, TileSplitAxis Split, std::enable_if_t<!is_global_data_v<TileProd>, int> = 0,
    typename... WaitEvents>
PTO_INST RecordEvent TPUSH(Pipe& pipe, TileProd& tile, WaitEvents&... events)
{
    ::pto::detail::PtoWaitEvents(events...);
    MAP_INSTR_IMPL_T_TPUSH(TPUSH, PTO_TEMPLATE_ARGS(Pipe, TileProd, Split), pipe, tile);
    return RecordEvent{};
}

template <typename TileData, typename Pipe, typename... WaitEvents>
PTO_INST RecordEvent TPUSH(TileData& tile, Pipe& pipe, WaitEvents&... events)
{
    ::pto::detail::PtoWaitEvents(events...);
    MAP_INSTR_IMPL_T(TPUSH, PTO_TEMPLATE_ARGS(TileData, Pipe), tile, pipe);
    return RecordEvent{};
}

template <
    typename Pipe, typename TileCons, TileSplitAxis Split, std::enable_if_t<!is_global_data_v<TileCons>, int> = 0,
    typename... WaitEvents>
PTO_INST RecordEvent TPOP(Pipe& pipe, TileCons& tile, WaitEvents&... events)
{
    ::pto::detail::PtoWaitEvents(events...);
    MAP_INSTR_IMPL_T_TPOP(TPOP, PTO_TEMPLATE_ARGS(Pipe, TileCons, Split), pipe, tile);
    return RecordEvent{};
}

template <typename TileData, typename Pipe, typename... WaitEvents>
PTO_INST RecordEvent TPOP(TileData& tile, Pipe& pipe, WaitEvents&... events)
{
    ::pto::detail::PtoWaitEvents(events...);
    MAP_INSTR_IMPL_T(TPOP, PTO_TEMPLATE_ARGS(TileData, Pipe), tile, pipe);
    return RecordEvent{};
}

template <typename Pipe, TileSplitAxis Split, typename... WaitEvents>
PTO_INST RecordEvent TFREE(Pipe& pipe, WaitEvents&... events)
{
    ::pto::detail::PtoWaitEvents(events...);
    MAP_INSTR_IMPL_T(TFREE, PTO_TEMPLATE_ARGS(Pipe, Split), pipe);
    return RecordEvent{};
}

template <typename Pipe, typename... WaitEvents>
PTO_INST RecordEvent TFREE(Pipe& pipe, WaitEvents&... events)
{
    ::pto::detail::PtoWaitEvents(events...);
    MAP_INSTR_IMPL_T(TFREE, PTO_TEMPLATE_ARGS(Pipe), pipe);
    return RecordEvent{};
}

template <
    typename Pipe, typename GlobalData, TileSplitAxis Split, std::enable_if_t<is_global_data_v<GlobalData>, int> = 0,
    typename... WaitEvents>
PTO_INST RecordEvent TALLOC(Pipe& pipe, GlobalData& gmTensor, WaitEvents&... events)
{
    ::pto::detail::PtoWaitEvents(events...);
    MAP_INSTR_IMPL_T(TALLOC, PTO_TEMPLATE_ARGS(Pipe, GlobalData, Split), pipe, gmTensor);
    return RecordEvent{};
}

template <
    typename Pipe, typename GlobalData, TileSplitAxis Split, std::enable_if_t<is_global_data_v<GlobalData>, int> = 0,
    typename... WaitEvents>
PTO_INST RecordEvent TPUSH(Pipe& pipe, GlobalData& gmTensor, WaitEvents&... events)
{
    ::pto::detail::PtoWaitEvents(events...);
    MAP_INSTR_IMPL_T_TPUSH(TPUSH, PTO_TEMPLATE_ARGS(Pipe, GlobalData, Split), pipe, gmTensor);
    return RecordEvent{};
}

template <
    typename Pipe, typename GlobalData, TileSplitAxis Split, std::enable_if_t<is_global_data_v<GlobalData>, int> = 0,
    typename... WaitEvents>
PTO_INST RecordEvent TPOP(Pipe& pipe, GlobalData& gmTensor, WaitEvents&... events)
{
    ::pto::detail::PtoWaitEvents(events...);
    MAP_INSTR_IMPL_T_TPOP(TPOP, PTO_TEMPLATE_ARGS(Pipe, GlobalData, Split), pipe, gmTensor);
    return RecordEvent{};
}

template <
    typename Pipe, typename GlobalData, TileSplitAxis Split, std::enable_if_t<is_global_data_v<GlobalData>, int> = 0,
    typename... WaitEvents>
PTO_INST RecordEvent TFREE(Pipe& pipe, GlobalData& gmTensor, WaitEvents&... events)
{
    ::pto::detail::PtoWaitEvents(events...);
    MAP_INSTR_IMPL_T(TFREE, PTO_TEMPLATE_ARGS(Pipe, GlobalData, Split), pipe, gmTensor);
    return RecordEvent{};
}

template <HistByte byte, typename TileDataDst, typename TileDataSrc, typename TileDataIdx, typename... WaitEvents>
PTO_INST RecordEvent THISTOGRAM(TileDataDst& dst, TileDataSrc& src, TileDataIdx& idx, WaitEvents&... events)
{
    ::pto::detail::PtoWaitEvents(events...);
    const auto tag = ::pto::mocker::IsA5SequenceFit() ? ::pto::mocker::HistogramFitTag<byte>(dst, src, idx) :
                                                        ::pto::mocker::A5SequenceFitTag{"unsupported"};
    // The workload is the source matrix; the destination always has 256 bins.
    RECORD_INSTR_ONLY(THISTOGRAM, src, tag);
    return RecordEvent{};
}

template <auto quant_type, typename TileDataOut, typename TileDataSrc, typename TileDataPara, typename... WaitEvents>
PTO_INST RecordEvent
TQUANT(TileDataOut& dst, TileDataSrc& src, TileDataPara& scale, TileDataPara* offset = nullptr, WaitEvents&... events)
{
    ::pto::detail::PtoWaitEvents(events...);
    if (::pto::mocker::IsA5SequenceFit()) {
        RECORD_INSTR_ONLY(TQUANT, dst, ::pto::mocker::QuantFitTag<quant_type>(dst, src, scale, offset));
    } else {
        // No stub implementation exists for this interface; an unmeasured
        // backend must remain unsupported rather than instantiate device code.
        RECORD_INSTR_ONLY(TQUANT, dst, (::pto::mocker::A5SequenceFitTag{"unsupported"}));
    }
    return RecordEvent{};
}

template <
    typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1, typename TileDataDstIdx,
    typename TileDataSrc0Idx, typename TileDataSrc1Idx, typename... WaitEvents>
PTO_INST RecordEvent TPARTARGMAX(
    TileDataDst& dst, TileDataSrc0& src0, TileDataSrc1& src1, TileDataDstIdx& dstIdx, TileDataSrc0Idx& src0Idx,
    TileDataSrc1Idx& src1Idx, WaitEvents&... events)
{
    ::pto::detail::PtoWaitEvents(events...);
    if (::pto::mocker::IsA5SequenceFit()) {
        RECORD_INSTR_ONLY(
            TPARTARGMAX, dst, src0, src1, dstIdx, src0Idx, src1Idx,
            ::pto::mocker::PartialArgSequenceTag(dst, src0, src1, dstIdx, src0Idx, src1Idx));
    } else {
        MAP_INSTR_IMPL(TPARTARGMAX, dst, src0, src1, dstIdx, src0Idx, src1Idx);
    }
    return RecordEvent{};
}

template <
    typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1, typename TileDataDstIdx,
    typename TileDataSrc0Idx, typename TileDataSrc1Idx, typename... WaitEvents>
PTO_INST RecordEvent TPARTARGMIN(
    TileDataDst& dst, TileDataSrc0& src0, TileDataSrc1& src1, TileDataDstIdx& dstIdx, TileDataSrc0Idx& src0Idx,
    TileDataSrc1Idx& src1Idx, WaitEvents&... events)
{
    ::pto::detail::PtoWaitEvents(events...);
    if (::pto::mocker::IsA5SequenceFit()) {
        RECORD_INSTR_ONLY(
            TPARTARGMIN, dst, src0, src1, dstIdx, src0Idx, src1Idx,
            ::pto::mocker::PartialArgSequenceTag(dst, src0, src1, dstIdx, src0Idx, src1Idx));
    } else {
        MAP_INSTR_IMPL(TPARTARGMIN, dst, src0, src1, dstIdx, src0Idx, src1Idx);
    }
    return RecordEvent{};
}

template <typename TileDataOut, typename TileDataIn, typename TileDataTmp, typename... WaitEvents>
PTO_INST RecordEvent TCOLARGMAX(TileDataOut& dst, TileDataIn& src, TileDataTmp& tmp, WaitEvents&... events)
{
    ::pto::detail::PtoWaitEvents(events...);
    MAP_INSTR_IMPL(TCOLARGMAX, dst, src, tmp);
    return RecordEvent{};
}

template <typename TileDataOut, typename TileDataIn, typename TileDataTmp, typename... WaitEvents>
PTO_INST RecordEvent TCOLARGMIN(TileDataOut& dst, TileDataIn& src, TileDataTmp& tmp, WaitEvents&... events)
{
    ::pto::detail::PtoWaitEvents(events...);
    MAP_INSTR_IMPL(TCOLARGMIN, dst, src, tmp);
    return RecordEvent{};
}

} // namespace pto
#endif
