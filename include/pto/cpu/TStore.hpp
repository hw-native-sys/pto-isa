/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef TSTORE_HPP
#define TSTORE_HPP

#include <pto/common/constants.hpp>
#include <cassert>
#include "pto/cpu/parallel.hpp"
#include "common.hpp"
#include "nz_utils.hpp"

namespace pto {

template <typename GlobalData, typename TileData, QuantModeCPU_t quantMode, bool applyRelu,
          std::enable_if_t<TileData::isRowMajor, int> = 0>
__tf__ PTO_INLINE void StorePlainMatrix(typename GlobalData::DType __out__ *dst,
                                        typename TileData::TileDType __in__ src, const std::vector<uint64_t> &scalars,
                                        int gShape3, int gShape4, int gStride3, int gStride4, int validRow,
                                        int validCol, size_t idx3)
{
    size_t offsetSrcBase = idx3 * gShape3 * TileData::Cols;
    using D = typename GlobalData::DType;
    using S = typename TileData::DType;
    cpu::parallel_for_1d(
        0, static_cast<std::size_t>(gShape3), static_cast<std::size_t>(gShape3) * gShape4, [&](std::size_t r) {
            const std::size_t srcBase = offsetSrcBase + r * TileData::Cols;
            const std::size_t dstBase = r * static_cast<std::size_t>(gStride3);
            PTO_CPU_VECTORIZE_LOOP
            for (std::size_t c = 0; c < static_cast<std::size_t>(gShape4); c++) {
                int dstIdx = dstBase + c * static_cast<std::size_t>(gStride4);
                if constexpr (quantMode != QuantModeCPU_t::NoQuant) {
                    uint64_t scalar = scalars[c];
                    dst[dstIdx] = quantize_element<D, S, quantMode, applyRelu>(src[srcBase + c], scalar);
                } else {
                    S val = src[srcBase + c];
                    if constexpr (applyRelu) {
                        val = ReLU(val);
                    }
                    dst[dstIdx] = static_cast<D>(val);
                }
            }
        });
}

template <typename GlobalData, typename TileData, QuantModeCPU_t quantMode, bool applyRelu,
          std::enable_if_t<!TileData::isRowMajor, int> = 0>
__tf__ PTO_INLINE void StorePlainMatrix(typename GlobalData::DType __out__ *dst,
                                        typename TileData::TileDType __in__ src, const std::vector<uint64_t> &scalars,
                                        int gShape3, int gShape4, int gStride3, int gStride4, int validRow,
                                        int validCol, size_t idx3)
{
    size_t offsetSrcBase = idx3 * gShape4 * TileData::Rows;
    using D = typename GlobalData::DType;
    using S = typename TileData::DType;
    cpu::parallel_for_1d(
        0, static_cast<std::size_t>(gShape4), static_cast<std::size_t>(gShape3) * gShape4, [&](std::size_t c) {
            const std::size_t srcBase = offsetSrcBase + c * TileData::Rows;
            const std::size_t dstStride4 = static_cast<std::size_t>(gStride4);
            PTO_CPU_VECTORIZE_LOOP
            for (std::size_t r = 0; r < static_cast<std::size_t>(gShape3); r++) {
                int dstIdx = r * static_cast<std::size_t>(gStride3) + c * dstStride4;
                if constexpr (quantMode != QuantModeCPU_t::NoQuant) {
                    uint64_t scalar = scalars[r];
                    dst[dstIdx] = quantize_element<D, S, quantMode, applyRelu>(src[srcBase + r], scalar);
                } else {
                    S val = src[srcBase + r];
                    if constexpr (applyRelu) {
                        val = ReLU(val);
                    }
                    dst[dstIdx] = static_cast<D>(val);
                }
            }
        });
}

template <typename GlobalData, typename TileData, QuantModeCPU_t quantMode, bool applyRelu>
__tf__ PTO_INLINE void StorePlain(typename GlobalData::DType __out__ *dst, typename TileData::TileDType __in__ src,
                                  const std::vector<uint64_t> &scalars, int gShape0, int gShape1, int gShape2,
                                  int gShape3, int gShape4, int gStride0, int gStride1, int gStride2, int gStride3,
                                  int gStride4, int validRow, int validCol)
{
    int64_t srcStride1 = gShape2;
    int64_t srcStride0 = gShape1 * srcStride1;
    for (uint32_t i = 0; i < gShape0; i++) {
        int64_t srcAddr0 = i * srcStride0;
        int64_t dstAddr0 = i * gStride0;
        for (uint32_t j = 0; j < gShape1; j++) {
            int64_t srcAddr1 = j * srcStride1;
            int64_t dstAddr1 = j * gStride1;
            for (uint32_t k = 0; k < gShape2; k++) {
                size_t offsetDstBase = dstAddr0 + dstAddr1 + k * gStride2;
                StorePlainMatrix<GlobalData, TileData, quantMode, applyRelu>(dst + offsetDstBase, src, scalars, gShape3,
                                                                             gShape4, gStride3, gStride4, validRow,
                                                                             validCol, srcAddr0 + srcAddr1 + k);
            }
        }
    }
}

template <typename GlobalData, typename TileData, QuantModeCPU_t quantMode, bool applyRelu>
__tf__ PTO_INLINE void StoreSubfractalMatrix(typename GlobalData::DType __out__ *dst,
                                             typename TileData::TileDType __in__ src,
                                             const std::vector<uint64_t> &scalars, int gShape3, int gShape4,
                                             int gStride3, int gStride4, int validRow, int validCol)
{
    using D = typename GlobalData::DType;
    using S = typename TileData::DType;
    cpu::parallel_for_1d(
        0, static_cast<std::size_t>(gShape4), static_cast<std::size_t>(gShape3) * gShape4, [&](std::size_t c) {
            size_t subTileC = c / TileData::InnerCols;
            size_t innerC = c % TileData::InnerCols;
            for (size_t r = 0; r < static_cast<std::size_t>(gShape3); r++) {
                size_t subTileR = r / TileData::InnerRows;
                size_t innerR = r % TileData::InnerRows;

                size_t tile_idx = GetTileElementOffsetSubfractals<TileData>(subTileR, innerR, subTileC, innerC);

                size_t gd_idx = r * static_cast<std::size_t>(gStride3) + c * static_cast<std::size_t>(gStride4);
                StoreElement<D, S, TileData, quantMode, applyRelu>(dst, gd_idx, src[tile_idx], r, c, scalars);
            }
        });
}

template <typename GlobalData, typename TileData, QuantModeCPU_t quantMode, bool applyRelu>
__tf__ PTO_INLINE void TStore(typename GlobalData::DType __out__ *dst, typename TileData::TileDType __in__ src,
                              const std::vector<uint64_t> &scalars, int gShape0, int gShape1, int gShape2, int gShape3,
                              int gShape4, int gStride0, int gStride1, int gStride2, int gStride3, int gStride4,
                              int validRow, int validCol)
{
    if constexpr (GlobalData::layout == pto::Layout::NZ) {
        assert(validRow == gShape2 * gShape3 && validCol == gShape0 * gShape1 * gShape4);
    } else {
        assert(gShape0 * gShape1 * gShape2 * gShape3 * gShape4 >= validRow * validCol);
    }
    if constexpr (GlobalData::layout == pto::Layout::NZ) {
        using D = typename GlobalData::DType;
        using S = typename TileData::DType;
        ForEachNZElement<TileData>(validRow, validCol, gShape1, gShape3, gShape4, gStride0, gStride1, gStride2,
                                   gStride3, gStride4, [&](size_t r, size_t c, size_t tile_idx, size_t gd_idx) {
                                       StoreElement<D, S, TileData, quantMode, applyRelu>(dst, gd_idx, src[tile_idx], r,
                                                                                          c, scalars);
                                   });
    } else if (TileData::SFractal == SLayout::NoneBox) {
        StorePlain<GlobalData, TileData, quantMode, applyRelu>(dst, src, scalars, gShape0, gShape1, gShape2, gShape3,
                                                               gShape4, gStride0, gStride1, gStride2, gStride3,
                                                               gStride4, validRow, validCol);
    } else {
        assert(gShape0 == 1 && gShape1 == 1 && gShape2 == 1 && "Nz,Zn -> ND,DN convertion does support only 2D GMs");
        StoreSubfractalMatrix<GlobalData, TileData, quantMode, applyRelu>(dst, src, scalars, gShape3, gShape4, gStride3,
                                                                          gStride4, validRow, validCol);
    }
}

template <typename TileData, typename GlobalData, QuantModeCPU_t quantMode, bool applyRelu>
PTO_INTERNAL void TSTORE_IMPL(GlobalData &dst, TileData &src, const std::vector<uint64_t> &scalars = {})
{
    static_assert(GlobalData::layout == pto::Layout::ND || GlobalData::layout == pto::Layout::DN ||
                      GlobalData::layout == pto::Layout::NZ,
                  "Only ND, DN and NZ GLobal Tensors are currently supported");
    TStore<GlobalData, TileData, quantMode, applyRelu>(
        dst.data(), src.data(), scalars, dst.GetShape(pto::GlobalTensorDim::DIM_0),
        dst.GetShape(pto::GlobalTensorDim::DIM_1), dst.GetShape(pto::GlobalTensorDim::DIM_2),
        dst.GetShape(pto::GlobalTensorDim::DIM_3), dst.GetShape(pto::GlobalTensorDim::DIM_4),
        dst.GetStride(pto::GlobalTensorDim::DIM_0), dst.GetStride(pto::GlobalTensorDim::DIM_1),
        dst.GetStride(pto::GlobalTensorDim::DIM_2), dst.GetStride(pto::GlobalTensorDim::DIM_3),
        dst.GetStride(pto::GlobalTensorDim::DIM_4), src.GetValidRow(), src.GetValidCol());
}

template <typename TileData, typename GlobalData, AtomicType atomicType>
PTO_INTERNAL void TSTORE_IMPL(GlobalData &dst, TileData &src)
{
    TSTORE_IMPL<TileData, GlobalData, QuantModeCPU_t::NoQuant, false>(dst, src);
}

template <typename TileData, typename GlobalData, AtomicType atomicType, STPhase Phase>
__aicore__ void TSTORE_IMPL(GlobalData &dst, TileData &src)
{
    (void)Phase;
    TSTORE_IMPL<TileData, GlobalData, QuantModeCPU_t::NoQuant, false>(dst, src);
}

template <typename TileData, typename GlobalData, AtomicType atomicType, ReluPreMode reluPreMode>
__aicore__ void TSTORE_IMPL(GlobalData &dst, TileData &src)
{
    constexpr bool useRelu = reluPreMode == ReluPreMode::NormalRelu;
    TSTORE_IMPL<TileData, GlobalData, QuantModeCPU_t::NoQuant, useRelu>(dst, src);
}

template <typename TileData, typename GlobalData, AtomicType atomicType, ReluPreMode reluPreMode, STPhase Phase>
__aicore__ void TSTORE_IMPL(GlobalData &dst, TileData &src)
{
    (void)Phase;
    constexpr bool useRelu = reluPreMode == ReluPreMode::NormalRelu;
    TSTORE_IMPL<TileData, GlobalData, QuantModeCPU_t::NoQuant, useRelu>(dst, src);
}

template <typename TileData, typename GlobalData, AtomicType atomicType, ReluPreMode reluPreMode>
__aicore__ void TSTORE_IMPL(GlobalData &dst, TileData &src, uint64_t preQuantScalar)
{
    constexpr QuantModeCPU_t quantPre = GetScalarPreQuantMode<typename TileData::DType, typename GlobalData::DType>();
    constexpr bool useRelu = reluPreMode == ReluPreMode::NormalRelu;
    size_t vector_size = 0;
    if constexpr (TileData::isRowMajor) {
        vector_size = src.GetValidCol();
    } else {
        vector_size = src.GetValidRow();
    }
    std::vector<uint64_t> scalars(vector_size, preQuantScalar);
    TSTORE_IMPL<TileData, GlobalData, quantPre, useRelu>(dst, src, scalars);
}

template <typename TileData, typename GlobalData, AtomicType atomicType, ReluPreMode reluPreMode, STPhase Phase>
__aicore__ void TSTORE_IMPL(GlobalData &dst, TileData &src, uint64_t preQuantScalar)
{
    (void)Phase;
    TSTORE_IMPL<TileData, GlobalData, atomicType, reluPreMode>(dst, src, preQuantScalar);
}

template <typename TileData, typename GlobalData, typename FpTileData, AtomicType atomicType, ReluPreMode reluPreMode>
__aicore__ void TSTORE_IMPL(GlobalData &dst, TileData &src, FpTileData &fp)
{
    constexpr QuantModeCPU_t quantPre = GetScalarPreQuantMode<typename TileData::DType, typename GlobalData::DType>();
    constexpr bool useRelu = reluPreMode == ReluPreMode::NormalRelu;

    std::vector<uint64_t> scalars(fp.GetValidCol(), 0);
    for (size_t i = 0; i < fp.GetValidCol(); i++) {
        const size_t quantTileIdx = GetTileElementOffset<FpTileData>(0, i);
        scalars[i] = fp.data()[quantTileIdx];
    }
    TSTORE_IMPL<TileData, GlobalData, quantPre, useRelu>(dst, src, scalars);
}
} // namespace pto
#endif
