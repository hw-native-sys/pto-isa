/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_NPU_A6R2_TMOV_HPP
#define PTO_NPU_A6R2_TMOV_HPP

#include <pto/common/constants.hpp>
#include <pto/common/fixpipe.hpp>

namespace pto {

// Pack A6R2 FIXP MaxTrack sideband for set_fixp_max_cfg:
//   rowmax_compare_en [0]
//   rowmax_output     [1]
//   group_num         [5:2]
//   abs_en            [8]
//   groupmax_addr     [31:16]  — UB offset in C0 (32B) units
//   rowmax_addr       [47:32]  — UB offset in C0 (32B) units
// Byte addresses must be converted: field = byteAddr / C0_SIZE_BYTE.
// Passing raw bytes (e.g. 0x28000) truncates via &0xFFFF to 0x8000; if HW
// treats that as 32B units it becomes 0x100000 and trips "fixp write ub overflow".
PTO_INTERNAL uint64_t MakeFixpMaxConfig(
    bool rowmaxCompareEn, bool rowmaxOutput, uint8_t groupNum, bool absEn, uint32_t groupmaxAddr, uint32_t rowmaxAddr)
{
    PTO_CPU_ASSERT((groupmaxAddr % C0_SIZE_BYTE) == 0, "Fix: groupmax UB addr must be 32B-aligned.");
    PTO_CPU_ASSERT((rowmaxAddr % C0_SIZE_BYTE) == 0, "Fix: rowmax UB addr must be 32B-aligned.");
    const uint32_t groupmaxC0 = groupmaxAddr / static_cast<uint32_t>(C0_SIZE_BYTE);
    const uint32_t rowmaxC0 = rowmaxAddr / static_cast<uint32_t>(C0_SIZE_BYTE);
    uint64_t cfg = 0;
    cfg |= (rowmaxCompareEn ? 1ULL : 0ULL) << 0;
    cfg |= (rowmaxOutput ? 1ULL : 0ULL) << 1;
    cfg |= (static_cast<uint64_t>(groupNum) & 0xFULL) << 2;
    cfg |= (absEn ? 1ULL : 0ULL) << 8;
    cfg |= (static_cast<uint64_t>(groupmaxC0) & 0xFFFFULL) << 16;
    cfg |= (static_cast<uint64_t>(rowmaxC0) & 0xFFFFULL) << 32;
    return cfg;
}

// A6R2 Acc->UB MaxTrack: dst/src plus groupmax / rowmax Vec tiles for set_fixp_max_cfg.
template <
    typename DstTileData, typename SrcTileData, typename GMaxTileData, typename RMaxTileData, AccToVecMode mode,
    QuantMode_t quantPre, ReluPreMode reluMode, STPhase Phase, bool absEn, bool rowMaxEn, bool groupMaxEn,
    bool rowmaxCompareEn, bool rowmaxOutput, uint8_t groupNum>
__tf__ AICORE void TMovCcToUb(
    typename DstTileData::TileDType __out__ dst, typename GMaxTileData::TileDType __out__ gMax,
    typename RMaxTileData::TileDType __out__ rMax, typename SrcTileData::TileDType __in__ src, uint16_t validRow,
    uint16_t validCol)
{
    using dstType = typename DstTileData::DType;
    using srcType = typename SrcTileData::DType;
    static_assert(mode == AccToVecMode::SingleModeVec0, "A6R2 L0C2UB TMOV only supports AccToVecMode::SingleModeVec0.");
    static_assert(
        IsMaxTrackDType<dstType>, "Fix: MaxTrack Acc→UB only supports float / half / bfloat16_t destination dtype.");
    static_assert(GMaxTileData::Loc == TileType::Vec, "Fix: MaxTrack groupmax tile must be Vec.");
    static_assert(RMaxTileData::Loc == TileType::Vec, "Fix: MaxTrack rowmax tile must be Vec.");
    constexpr int32_t c0Size = BLOCK_BYTE_SIZE / sizeof(dstType);
    constexpr bool subBlockId = (mode == AccToVecMode::SingleModeVec1);
    constexpr uint8_t dualDstCtl = GetDualDstCtl<DstTileData, SrcTileData, mode, quantPre>();
    constexpr uint8_t unitFlagCtrl = static_cast<uint8_t>(Phase);
    constexpr bool enableNz2Nd = (DstTileData::isRowMajor && DstTileData::SFractal == SLayout::NoneBox);
    constexpr bool enableNz2Dn = (!DstTileData::isRowMajor && DstTileData::SFractal == SLayout::NoneBox);
    constexpr bool enableNz2Nz = (!DstTileData::isRowMajor && DstTileData::SFractal == SLayout::RowMajor);
    static_assert(
        !groupMaxEn || enableNz2Nd || enableNz2Dn,
        "Fix: MaxTrack groupmax_en / rowmax_en are only effective when NZ2ND or NZ2DN is enabled.");
    // HW: N/group_num <= 16 when groupmax_en && NZ2ND.
    static_assert(
        !(groupMaxEn && enableNz2Nd) || ((SrcTileData::Cols / groupNum) <= 16),
        "Fix: MaxTrack (N/group_num) must be <= 16 when groupmax_en=1 && NZ2ND.");
    constexpr bool channelSplitEnable = (!DstTileData::isRowMajor && (DstTileData::SFractal == SLayout::RowMajor)) &&
                                        (std::is_same_v<typename DstTileData::DType, float>) &&
                                        (DstTileData::SFractalSize == 512);
    constexpr uint32_t dstStride = GetTmovAccDstStride<DstTileData, SrcTileData>();
    static_assert(
        ((dstStride * sizeof(dstType) % C0_SIZE_BYTE == 0) && ((dstStride) > 0)),
        "Dst Tile Cols * sizeof(dstT) must be multiples of 32 and not 0 when nz2nd. \
            Dst Tile Rows * sizeof(dstT) must be multiples of 32 and not 0 when nz2dn. \
            Dst Tile Cols * sizeof(dstType) must be multiples of 32 and not 0 when nz2nz.");

    // HW: validRow <= 128 when rowmax_en, or when groupmax_en && NZ2DN.
    if constexpr (rowMaxEn || (groupMaxEn && enableNz2Dn)) {
        PTO_ASSERT(validRow <= 128, "Fix: MaxTrack srcTile validRow must be <= 128 when rowmax_en=1 or NZ2DN.");
    }

    if constexpr (enableNz2Nz) {
        validRow = SrcTileData::Rows;
        if constexpr ((mode == AccToVecMode::SingleModeVec0 || mode == AccToVecMode::SingleModeVec1)) {
            if constexpr (std::is_same_v<typename DstTileData::DType, float>) {
                constexpr int32_t align = channelSplitEnable ? c0Size : FRACTAL_NZ_ROW;
                validCol = CeilDivision(validCol, align) * align;
            } else {
                validCol = CeilDivision(validCol, c0Size) * c0Size;
            }
        } else if constexpr (mode == AccToVecMode::DualModeSplitM) {
            validCol = CeilDivision(validCol, FRACTAL_NZ_ROW) * FRACTAL_NZ_ROW;
        } else {
            validCol = CeilDivision(validCol, BLOCK_BYTE_SIZE) * BLOCK_BYTE_SIZE;
        }
    }
    if constexpr (enableNz2Nd) {
        SetLoop3Para();
    } else if constexpr (enableNz2Dn) {
        SetLoop3Para();
        constexpr uint64_t channelPara = static_cast<uint64_t>(1) << 48;
        set_channel_para(channelPara);
    }

    const uint32_t groupmaxAddr = static_cast<uint32_t>(reinterpret_cast<uint64_t>(__cce_get_tile_ptr(gMax)));
    uint32_t rowmaxAddr = 0;
    if constexpr (rowMaxEn) {
        rowmaxAddr = static_cast<uint32_t>(reinterpret_cast<uint64_t>(__cce_get_tile_ptr(rMax)));
    }
    const uint64_t fixpMaxConfig =
        MakeFixpMaxConfig(rowmaxCompareEn, rowmaxOutput, groupNum, absEn, groupmaxAddr, rowmaxAddr);
    set_fixp_max_cfg(fixpMaxConfig);

    uint32_t srcStride = GetTmovAccSrcStride<SrcTileData>(validRow);
    __ubuf__ dstType* dstAddr = (__ubuf__ dstType*)__cce_get_tile_ptr(dst);
    __cc__ srcType* srcData = (__cc__ srcType*)__cce_get_tile_ptr(src);
    // A6R2 order: dual, sub, row_max_en, group_max_en, quant_pre_rnd, clip_relu_pre, ...
    pto_copy_matrix_cc_to_ub(
        dstAddr, srcData, validCol, validRow, dstStride, srcStride, dualDstCtl, subBlockId, rowMaxEn, groupMaxEn,
        false /* quant_pre_rnd */, 0 /* clip_relu_pre */, unitFlagCtrl, quantPre, static_cast<uint8_t>(reluMode),
        channelSplitEnable, enableNz2Nd, 0, 0, false, false, 0, false, false, false, false, false, enableNz2Dn);
}

template <
    typename DstTileData, typename SrcTileData, typename GMaxTileData, typename RMaxTileData, AccToVecMode mode,
    QuantMode_t quantPre, ReluPreMode reluMode, STPhase Phase, bool absEn, bool rowMaxEn, bool groupMaxEn,
    bool rowmaxCompareEn, bool rowmaxOutput, uint8_t groupNum>
PTO_INTERNAL void TMovCcToUb(
    DstTileData& dst, GMaxTileData& gMax, RMaxTileData& rMax, SrcTileData& src, uint16_t validRow, uint16_t validCol)
{
    TMovCcToUb<
        DstTileData, SrcTileData, GMaxTileData, RMaxTileData, mode, quantPre, reluMode, Phase, absEn, rowMaxEn,
        groupMaxEn, rowmaxCompareEn, rowmaxOutput, groupNum>(
        dst.data(), gMax.data(), rMax.data(), src.data(), validRow, validCol);
}

} // namespace pto

#endif // PTO_NPU_A6R2_TMOV_HPP
