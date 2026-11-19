/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

// When PTO_A6R2_TPUSH_CTOR_ONLY is set, this file is injected into pto::TPipe via
// PTO_TPIPE_EXTRA_CTORS_HEADER (ctor only). Otherwise it provides A6R2 MaxTrack
// TPUSH after a5/TPush.hpp has been included. MaxTrack TPOP lives in a6r2/TPop.hpp.
#ifdef PTO_A6R2_TPUSH_CTOR_ONLY

// Injected into pto::TPipe: bind C2V group/row-max workspace bases into the ring FIFO.
PTO_INTERNAL explicit TPipe(
    __gm__ void* GM_SLOT_BUFFER, uint32_t C2V_CONSUMER_BUF, uint32_t V2C_CONSUMER_BUF, uint32_t C2V_GROUPMAX_BUF,
    uint32_t C2V_ROWMAX_BUF)
    : fifo(GM_SLOT_BUFFER, C2V_CONSUMER_BUF, V2C_CONSUMER_BUF, C2V_GROUPMAX_BUF, C2V_ROWMAX_BUF), prod(), cons()
{
    InitPipeAfterConstruct();
}

#else

#ifndef PTO_NPU_A6R2_TPUSH_HPP
#define PTO_NPU_A6R2_TPUSH_HPP

#include <pto/common/constants.hpp>
#include <pto/common/fixpipe.hpp>

namespace pto {

// MaxTrack RowMax m-strip credit uses FlagID+2 (tile free uses FlagID+1).
// Wait on seed when prod >= SLOT_NUM; Notify on EndRowMaxCmp when cons >= SLOT_NUM.
// That notify is one generation earlier than "cons > SLOT_NUM" (avoids Push_k waiting on Pop_k).
// Side effect: one leftover notify when seedCount >= SLOT_NUM — Cube drains it once after the loop.
template <typename Pipe>
PTO_INTERNAL void WaitRowMaxSlot(Pipe& pipe)
{
    static_assert(Pipe::FlagIDPlusTwo <= MAX_SYC_ID, "Fix: MaxTrack RowMax credit needs FlagID+2 <= MAX_SYC_ID.");
    (void)pipe;
#ifdef PTO_COMPILE_CUBE
    wait_intra_block(PIPE_FIX, Pipe::FlagIDPlusTwo);
#endif
}

template <typename Pipe>
PTO_INTERNAL void NotifyRowMaxSlot(Pipe& pipe)
{
    static_assert(Pipe::FlagIDPlusTwo <= MAX_SYC_ID, "Fix: MaxTrack RowMax credit needs FlagID+2 <= MAX_SYC_ID.");
    (void)pipe;
#ifdef PTO_COMPILE_VEC
    set_intra_block(PIPE_V, Pipe::FlagIDPlusTwo);
#endif
}

// MaxTrack C2V TPUSH outputs (FixpipeParams<..., MaxTrack<...>>):
//   groupmax_en=true,  rowmax_en=false → vecTile + gMaxTile
//   groupmax_en=true,  rowmax_en=true  → vecTile + gMaxTile + rMaxTile
//   groupmax_en=false                  → a5 TPUSH (vecTile only)
// RowMax UB is indexed by m-strip (rowmaxProd), not tileIndex:
//   rowmax_compare_en=false → seed new slot (wait if needed); true → reuse previous slot.
template <typename Pipe, typename TileProd, typename TConfig>
PTO_INTERNAL void PushAcc2VecFiFoMaxTrack(Pipe& pipe, TileProd& tileProd)
{
    using T = FixpipeConsDType_t<TConfig::QuantPre, typename TileProd::DType>;
    using RingFiFo = typename Pipe::RingFiFo;
    using TileCons = std::conditional_t<
        TConfig::LayoutMode == LayoutMode_t::NZ2ND,
        Tile<TileType::Vec, T, TileProd::Rows, TileProd::Cols, BLayout::RowMajor, TileProd::Rows, TileProd::Cols>,
        std::conditional_t<
            TConfig::LayoutMode == LayoutMode_t::NZ2DN,
            Tile<TileType::Vec, T, TileProd::Rows, TileProd::Cols, BLayout::ColMajor, TileProd::Rows, TileProd::Cols>,
            Tile<
                TileType::Vec, T, TileProd::Rows, TileProd::Cols, BLayout::ColMajor, TileProd::Rows, TileProd::Cols,
                SLayout::RowMajor> > >;
    // GroupMax (ValidRow/ValidCol DYNAMIC; set from src valid shape):
    //   NZ2ND: [srcValidRow, srcValidCol/group_num]
    //   NZ2DN: transposed [srcValidCol/group_num, srcValidRow]
    using GroupMaxTile = std::conditional_t<
        TConfig::LayoutMode == LayoutMode_t::NZ2DN,
        Tile<
            TileType::Vec, T, TileProd::Cols / TConfig::group_num, TileProd::Rows, BLayout::RowMajor, DYNAMIC, DYNAMIC>,
        Tile<
            TileType::Vec, T, TileProd::Rows, TileProd::Cols / TConfig::group_num, BLayout::RowMajor, DYNAMIC,
            DYNAMIC> >;
    // RowMax (1D): ValidRow=1, ValidCol=DYNAMIC; construct as RMaxTile(srcValidRow).
    // Storage Cols is 32B-ceil of TileProd::Rows.
    constexpr int rowMaxStorageCols = ((TileProd::Rows + static_cast<int>(C0_SIZE_BYTE / sizeof(T)) - 1) /
                                       static_cast<int>(C0_SIZE_BYTE / sizeof(T))) *
                                      static_cast<int>(C0_SIZE_BYTE / sizeof(T));
    using RMaxTile = Tile<TileType::Vec, T, 1, rowMaxStorageCols, BLayout::RowMajor, 1, DYNAMIC>;

    static_assert(
        IsMaxTrackDType<T>, "Fix: MaxTrack TPUSH only supports float / half / bfloat16_t (after FixpipeConsDType).");
    static_assert(
        TConfig::LayoutMode == LayoutMode_t::NZ2ND || TConfig::LayoutMode == LayoutMode_t::NZ2DN,
        "Fix: MaxTrack groupmax_en / rowmax_en are only effective with LayoutMode NZ2ND or NZ2DN.");
    static_assert(
        TConfig::group_num > 0 && (TileProd::Cols % TConfig::group_num) == 0,
        "Fix: MaxTrack groupmax requires ProdN multiple of group_num.");
    static_assert(
        (GroupMaxTile::Cols * sizeof(T)) % C0_SIZE_BYTE == 0, "Fix: MaxTrack gMaxTile row bytes must be 32B-aligned.");
    // HW: N/group_num <= 16 when groupmax_en && NZ2ND.
    static_assert(
        !(TConfig::LayoutMode == LayoutMode_t::NZ2ND) || ((TileProd::Cols / TConfig::group_num) <= 16),
        "Fix: MaxTrack (N/group_num) must be <= 16 when groupmax_en=1 && NZ2ND.");

    const uint16_t validRow = tileProd.GetValidRow();
    const uint16_t validCol = tileProd.GetValidCol();
    // HW: validRow <= 128 when rowmax_en, or when groupmax_en && NZ2DN.
    if constexpr (TConfig::rowmax_en || TConfig::LayoutMode == LayoutMode_t::NZ2DN) {
        PTO_ASSERT(validRow <= 128, "Fix: MaxTrack srcTile validRow must be <= 128 when rowmax_en=1 or NZ2DN.");
    }

    TileCons consTile;
    const uint64_t entryBase = (pipe.prod.tileIndex % RingFiFo::SLOT_NUM) * RingFiFo::SLOT_SIZE;
    TASSIGN(consTile, (uint64_t)(pipe.fifo.C2V_CONSUMER_BUF + entryBase + pipe.prod.entryOffset));

    GroupMaxTile gMaxTile;
    // 1D RowMax: ValidRow=1 (static), ValidCol=srcValidRow (ctor).
    RMaxTile rMaxTile(validRow);
    constexpr uint32_t kGMaxSlotSize = static_cast<uint32_t>(GroupMaxTile::Rows * GroupMaxTile::Cols * sizeof(T));
    TASSIGN(
        gMaxTile, pipe.fifo.C2V_GROUPMAX_BUF + (pipe.prod.tileIndex % RingFiFo::SLOT_NUM) * kGMaxSlotSize +
                      static_cast<uint64_t>(pipe.prod.entryOffset));
    if constexpr (TConfig::LayoutMode == LayoutMode_t::NZ2DN) {
        gMaxTile.SetValidShape(validCol / TConfig::group_num, validRow);
    } else {
        gMaxTile.SetValidShape(validRow, validCol / TConfig::group_num);
    }
    if constexpr (TConfig::rowmax_en) {
        constexpr uint32_t kRMaxSlotSize = static_cast<uint32_t>(RMaxTile::Rows * RMaxTile::Cols * sizeof(T));
        uint32_t rmaxSlot = 0;
        if constexpr (!TConfig::rowmax_compare_en) {
            // Seed a new m-strip RowMax slot; block if that slot may still be in use.
            if (pipe.fifo.rowmaxProd >= RingFiFo::SLOT_NUM) {
                WaitRowMaxSlot(pipe);
            }
            rmaxSlot = pipe.fifo.rowmaxProd % RingFiFo::SLOT_NUM;
            pipe.fifo.rowmaxProd++;
        } else {
            // Compare path: keep writing the current m-strip slot.
            PTO_ASSERT(
                pipe.fifo.rowmaxProd > 0, "Fix: rowmax_compare_en=true requires a prior seed (compare_en=false).");
            rmaxSlot = (pipe.fifo.rowmaxProd - 1) % RingFiFo::SLOT_NUM;
        }
        TASSIGN(
            rMaxTile,
            pipe.fifo.C2V_ROWMAX_BUF + rmaxSlot * kRMaxSlotSize + static_cast<uint64_t>(pipe.prod.entryOffset));
    } else {
        TASSIGN(rMaxTile, 0);
    }

    // A6R2: set_fixp_max_cfg from gMax/rMax Vec tiles + copy_matrix row/group max enables.
    TMovCcToUb<
        TileCons, TileProd, GroupMaxTile, RMaxTile, (AccToVecMode) static_cast<uint8_t>(TConfig::SubBlockId),
        TConfig::QuantPre, TConfig::ReluMode, TConfig::Phase, TConfig::abs_en, TConfig::rowmax_en, TConfig::groupmax_en,
        TConfig::rowmax_compare_en, TConfig::rowmax_output, TConfig::group_num>(
        consTile, gMaxTile, rMaxTile, tileProd, validRow, validCol);
}

template <typename Pipe, typename TileProd, typename TConfig, std::enable_if_t<TConfig::groupmax_en, int> = 0>
PTO_INTERNAL void TPUSH_IMPL(Pipe& pipe, TileProd& tileProd)
{
    using ConsT = FixpipeConsDType_t<TConfig::QuantPre, typename TileProd::DType>;
    static_assert(
        Pipe::is_c2v_ub || Pipe::is_both, "Fix: MaxTrack TPUSH (groupmax_en) is only supported on C2V UB FIFO.");
    static_assert(
        IsMaxTrackDType<ConsT>,
        "Fix: MaxTrack TPUSH only supports float / half / bfloat16_t (after FixpipeConsDType).");
    static_assert(
        TConfig::LayoutMode == LayoutMode_t::NZ2ND || TConfig::LayoutMode == LayoutMode_t::NZ2DN,
        "Fix: MaxTrack groupmax_en / rowmax_en are only effective with LayoutMode NZ2ND or NZ2DN.");

    bool isAllocate = pipe.prod.getAllocateStatus() && Pipe::shouldWaitFree(pipe.prod.tileIndex);
    if (isAllocate) {
        pipe.prod.template allocate<TileSplitAxis::TILE_NO_SPLIT>();
    }

    PushAcc2VecFiFoMaxTrack<Pipe, TileProd, TConfig>(pipe, tileProd);
    pipe.prod.tileIndex++;

    bool isRecord = pipe.prod.getRecordStatus();
    if (isRecord) {
        pipe.prod.template record<TileSplitAxis::TILE_NO_SPLIT>();
    }
}

} // namespace pto

#endif // PTO_NPU_A6R2_TPUSH_HPP
#endif // PTO_A6R2_TPUSH_CTOR_ONLY
