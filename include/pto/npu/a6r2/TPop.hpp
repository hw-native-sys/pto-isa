/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_NPU_A6R2_TPOP_HPP
#define PTO_NPU_A6R2_TPOP_HPP

#include <type_traits>
#include <pto/common/fifo.hpp>
#include <pto/common/fixpipe.hpp>

namespace pto {

// A6R2 MaxTrack C2V TPOP:
//   groupmax_en=true,  rowmax_en=false → TPOP(pipe, popTile, gMaxTile)
//   groupmax_en=true,  rowmax_en=true  → TPOP(pipe, popTile, gMaxTile, rMaxTile)

template <typename GMaxTile, typename RingFiFo>
PTO_INTERNAL void PopGMaxTileFromFiFo(RingFiFo& fifo, uint32_t tileIndex, int entryOffset, GMaxTile& gMaxTile)
{
    using T = typename GMaxTile::DType;
    constexpr uint32_t kGMaxSlotSize = static_cast<uint32_t>(GMaxTile::Rows * GMaxTile::Cols * sizeof(T));
    TASSIGN_IMPL(
        gMaxTile,
        fifo.C2V_GROUPMAX_BUF + (tileIndex % RingFiFo::SLOT_NUM) * kGMaxSlotSize + static_cast<uint64_t>(entryOffset));
}

// RowMax is m-strip slotted: address follows fifo.rowmaxCons (not tileIndex).
template <typename RMaxTile, typename RingFiFo>
PTO_INTERNAL void PopRMaxTileFromFiFo(RingFiFo& fifo, int entryOffset, RMaxTile& rMaxTile)
{
    using T = typename RMaxTile::DType;
    constexpr uint32_t kRMaxSlotSize = static_cast<uint32_t>(RMaxTile::Rows * RMaxTile::Cols * sizeof(T));
    TASSIGN_IMPL(
        rMaxTile, fifo.C2V_ROWMAX_BUF + (fifo.rowmaxCons % RingFiFo::SLOT_NUM) * kRMaxSlotSize +
                      static_cast<uint64_t>(entryOffset));
}

// Infer GroupMax valid shape from popTile (fixpipe src) valid shape + TConfig::group_num:
//   NZ2ND (RowMajor): [srcValidRow, srcValidCol/group_num]
//   NZ2DN (ColMajor): [srcValidCol/group_num, srcValidRow]
template <typename PopTile, typename GMaxTile, typename TConfig>
PTO_INTERNAL void SetMaxTrackGMaxValidShape(PopTile& popTile, GMaxTile& gMaxTile)
{
    static_assert(
        GMaxTile::ValidRow == DYNAMIC && GMaxTile::ValidCol == DYNAMIC,
        "Fix: MaxTrack gMaxTile ValidRow/ValidCol must be DYNAMIC.");
    constexpr unsigned groupNum = static_cast<unsigned>(TConfig::group_num);
    static_assert(groupNum > 0, "Fix: MaxTrack TConfig::group_num must be > 0.");
    static_assert(
        (PopTile::Cols % groupNum) == 0, "Fix: MaxTrack popTile Cols must be a multiple of TConfig::group_num.");
    const unsigned srcValidRow = popTile.GetValidRow();
    const unsigned srcValidCol = popTile.GetValidCol();
    if constexpr (PopTile::isRowMajor) {
        static_assert(
            GMaxTile::Cols == PopTile::Cols / groupNum,
            "Fix: MaxTrack NZ2ND gMaxTile Cols must equal popTile Cols / TConfig::group_num.");
        gMaxTile.SetValidShape(srcValidRow, srcValidCol / groupNum);
    } else {
        static_assert(
            GMaxTile::Rows == PopTile::Cols / groupNum,
            "Fix: MaxTrack NZ2DN gMaxTile Rows must equal popTile Cols / TConfig::group_num.");
        gMaxTile.SetValidShape(srcValidCol / groupNum, srcValidRow);
    }
}

// RowMax is 1D: ValidRow=1, ValidCol=srcValidRow.
template <typename PopTile, typename RMaxTile>
PTO_INTERNAL void SetMaxTrackRMaxValidShape(PopTile& popTile, RMaxTile& rMaxTile)
{
    static_assert(RMaxTile::Rows == 1 && RMaxTile::ValidRow == 1, "Fix: MaxTrack rMaxTile must be 1-row.");
    static_assert(RMaxTile::ValidCol == DYNAMIC, "Fix: MaxTrack rMaxTile ValidCol must be DYNAMIC.");
    rMaxTile.SetValidCol(popTile.GetValidRow());
}

template <
    typename Pipe, typename PopTile, typename GMaxTile, typename TConfig, TileSplitAxis Split,
    std::enable_if_t<is_tile_data_v<PopTile> && is_tile_data_v<GMaxTile>, int> = 0>
PTO_INTERNAL void TPOP_IMPL(Pipe& pipe, PopTile& popTile, GMaxTile& gMaxTile)
{
    static_assert(Pipe::is_c2v_ub || Pipe::is_both, "Fix: MaxTrack TPOP requires C2V UB FIFO.");
    static_assert(
        IsMaxTrackDType<typename PopTile::DType>, "Fix: MaxTrack TPOP only supports float / half / bfloat16_t.");
    static_assert(
        PopTile::SFractal == SLayout::NoneBox, "Fix: MaxTrack TPOP requires NZ2ND/NZ2DN popTile (no fractal).");
    static_assert(
        std::is_same_v<typename PopTile::DType, typename GMaxTile::DType>,
        "Fix: MaxTrack TPOP popTile and gMaxTile DType must match.");
    bool isWait = pipe.cons.getWaitStatus();
    if (isWait) {
        pipe.cons.template wait<Split>();
    }

    bool reqFree;
    if constexpr (Split == TileSplitAxis::TILE_NO_SPLIT) {
        reqFree = pipe.cons.template pop<PopTile, Split>(pipe.fifo, popTile, 0);
    } else {
        reqFree = pipe.cons.template pop<PopTile, Split>(pipe.fifo, popTile, get_subblockid());
    }
    PopGMaxTileFromFiFo(pipe.fifo, pipe.cons.tileIndex, pipe.cons.entryOffset, gMaxTile);
    SetMaxTrackGMaxValidShape<PopTile, GMaxTile, TConfig>(popTile, gMaxTile);

    pipe.cons.tileIndex++;
    if (reqFree) {
        bool isFree = pipe.cons.getFreeStatus() && Pipe::shouldNotifyFree(pipe.cons.freeIndex);
        pipe.cons.freeIndex++;
        if (isFree) {
            pipe.cons.template free<Split>();
        }
    }
}

// EndRowMaxCmp=true: this pop finishes the current m-strip RowMax slot (set on last nTile).
// rMaxTile keeps the bound UB address for a subsequent TADD; only fifo.rowmaxCons advances for later pops.
template <
    typename Pipe, typename PopTile, typename GMaxTile, typename RMaxTile, typename TConfig, TileSplitAxis Split,
    bool EndRowMaxCmp = false,
    std::enable_if_t<is_tile_data_v<PopTile> && is_tile_data_v<GMaxTile> && is_tile_data_v<RMaxTile>, int> = 0>
PTO_INTERNAL void TPOP_IMPL(Pipe& pipe, PopTile& popTile, GMaxTile& gMaxTile, RMaxTile& rMaxTile)
{
    static_assert(Pipe::is_c2v_ub || Pipe::is_both, "Fix: MaxTrack TPOP requires C2V UB FIFO.");
    static_assert(
        IsMaxTrackDType<typename PopTile::DType>, "Fix: MaxTrack TPOP only supports float / half / bfloat16_t.");
    static_assert(
        PopTile::SFractal == SLayout::NoneBox, "Fix: MaxTrack TPOP requires NZ2ND/NZ2DN popTile (no fractal).");
    static_assert(
        std::is_same_v<typename PopTile::DType, typename GMaxTile::DType> &&
            std::is_same_v<typename PopTile::DType, typename RMaxTile::DType>,
        "Fix: MaxTrack TPOP popTile / gMaxTile / rMaxTile DType must match.");
    using RingFiFo = typename Pipe::RingFiFo;
    bool isWait = pipe.cons.getWaitStatus();
    if (isWait) {
        pipe.cons.template wait<Split>();
    }

    bool reqFree;
    if constexpr (Split == TileSplitAxis::TILE_NO_SPLIT) {
        reqFree = pipe.cons.template pop<PopTile, Split>(pipe.fifo, popTile, 0);
    } else {
        reqFree = pipe.cons.template pop<PopTile, Split>(pipe.fifo, popTile, get_subblockid());
    }
    PopGMaxTileFromFiFo(pipe.fifo, pipe.cons.tileIndex, pipe.cons.entryOffset, gMaxTile);
    PopRMaxTileFromFiFo(pipe.fifo, pipe.cons.entryOffset, rMaxTile);
    SetMaxTrackGMaxValidShape<PopTile, GMaxTile, TConfig>(popTile, gMaxTile);
    SetMaxTrackRMaxValidShape(popTile, rMaxTile);

    pipe.cons.tileIndex++;
    if (reqFree) {
        bool isFree = pipe.cons.getFreeStatus() && Pipe::shouldNotifyFree(pipe.cons.freeIndex);
        pipe.cons.freeIndex++;
        if (isFree) {
            pipe.cons.template free<Split>();
        }
    }

    if constexpr (EndRowMaxCmp) {
        pipe.fifo.rowmaxCons++;
        // Must be >= (not >): seed waits when prod >= SLOT_NUM need the credit from the
        // previous strip's EndRowMaxCmp. Using ">" delays notify until the current strip
        // ends and deadlocks (Push waits on Pop of the same strip).
        if (pipe.fifo.rowmaxCons >= RingFiFo::SLOT_NUM) {
            NotifyRowMaxSlot(pipe);
        }
    }
}

} // namespace pto

#endif // PTO_NPU_A6R2_TPOP_HPP
