/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef SPARSE_FLASH_MLA_CUSTOM_TPIPE_HPP_
#define SPARSE_FLASH_MLA_CUSTOM_TPIPE_HPP_

#include "../flash_mla_common.h"

#if defined(__CCE_AICORE__)
namespace flash_mla_bf16::custom {

// A TPipe policy with caller-owned slots, credits, and acquire/ready/release
// handoffs. PTO's TPUSH_IMPL/TPOP_IMPL dispatch and TPipe binding machinery
// operate on explicitly addressed endpoints and insertion offsets.
// DestinationTile retains physical pitches (padded NZ and GU256 subviews).
// Consumers pass subBlockId=0 to TPOP for their bound local view.
// The ready wait covers both AIV endpoints for a split V2C transfer.
template <Direction Dir, typename DestinationTile, STPhase Phase = STPhase::Unspecified>
struct ExplicitTilePipe {
    static_assert(Dir == DIR_C2V || Dir == DIR_V2C);
    static_assert(Dir != DIR_C2V || DestinationTile::Loc == TileType::Vec);
    static_assert(Dir != DIR_V2C || DestinationTile::Loc == TileType::Mat);
    using Base = TPipe<0, Dir, 1, 1>;
    using RingFiFo = typename Base::RingFiFo;
    static constexpr bool is_c2v_gm = false;
    static constexpr bool is_v2c_gm = false;
    static constexpr bool is_both_gm = false;
    static constexpr bool shouldWaitFree(uint32_t) { return false; }
    static constexpr bool shouldNotifyFree(uint32_t) { return false; }

    // Final unit-flag copies use two single-destination commands. Each copy
    // clears only its complete 512-byte L0C units, retaining the source pitch.
    // Each command uses STPhase::Final.
    template <typename SrcTile, uint16_t RowOffset, bool Vec1>
    __tf__ AICORE static inline void copyFinal(
        typename DestinationTile::TileDType __out__ dst, typename SrcTile::TileDType __in__ src)
    {
        static_assert(std::is_same_v<typename DestinationTile::DType, float>);
        static_assert(std::is_same_v<typename SrcTile::DType, float>);
        static_assert(SrcTile::Rows == 2 * DestinationTile::Rows);
        static_assert(DestinationTile::Cols == SrcTile::Cols);
        static_assert(RowOffset % 32 == 0 && DestinationTile::Rows % 32 == 0);
        set_loop3_para(1);
        auto* dstAddr = reinterpret_cast<__ubuf__ float*>(__cce_get_tile_ptr(dst));
        auto* srcAddr = reinterpret_cast<__cc__ float*>(__cce_get_tile_ptr(src)) + RowOffset * ACC_C0_SIZE;
        pto_copy_matrix_cc_to_ub(
            dstAddr, srcAddr, 0, DestinationTile::Cols, DestinationTile::Rows, DestinationTile::Cols, SrcTile::Rows, 0,
            Vec1, 0, static_cast<uint8_t>(STPhase::Final), QuantMode_t::NoQuant, 0, false, true, 0, 0, false, false, 0,
            false, false, false, false, false, false);
    }

    struct Producer : Base::Producer {
        uint16_t row = 0;
        uint16_t col = 0;

        AICORE Producer()
        {
            this->setAllocateStatus(false);
            this->setRecordStatus(false);
        }

        template <typename SrcTile, TileSplitAxis Split>
        AICORE void push(RingFiFo& fifo, SrcTile& src, int32_t)
        {
            DestinationTile dst;
            if constexpr (Dir == DIR_C2V) {
                static_assert(SrcTile::Loc == TileType::Acc);
                static_assert(Split == TILE_UP_DOWN);
                TASSIGN(dst, static_cast<uint64_t>(fifo.C2V_CONSUMER_BUF));
                if constexpr (Phase == STPhase::Final) {
                    copyFinal<SrcTile, 0, false>(dst.data(), src.data());
                    copyFinal<SrcTile, DestinationTile::Rows, true>(dst.data(), src.data());
                } else {
                    static_assert(Phase == STPhase::Unspecified);
                    TEXTRACT<DestinationTile, SrcTile, AccToVecMode::DualModeSplitM>(dst, src, 0, 0);
                }
            } else {
                static_assert(SrcTile::Loc == TileType::Vec);
                static_assert(Phase == STPhase::Unspecified);
                TASSIGN(dst, static_cast<uint64_t>(fifo.V2C_CONSUMER_BUF));
                TINSERT(dst, src, row, col);
            }
        }
    };

    struct Consumer : Base::Consumer {
        uint32_t readyEvent = 0;
        AICORE Consumer()
        {
            this->setWaitStatus(false);
            this->setFreeStatus(false);
        }

        template <TileSplitAxis Split>
        AICORE void wait() const
        {
            if constexpr (Dir == DIR_C2V) {
                wait_intra_block(PIPE_V, readyEvent);
            } else if constexpr (Dir == DIR_V2C) {
                Base::Consumer::template waitIntraBlockBySplit<PIPE_MTE1, Split>(readyEvent);
            }
        }

        template <typename DstTile, TileSplitAxis Split>
        AICORE bool pop(RingFiFo& fifo, DstTile& dst, int32_t)
        {
            if constexpr (Dir == DIR_C2V) {
                static_assert(DstTile::Loc == TileType::Vec);
                static_assert(std::is_same_v<typename DstTile::DType, typename DestinationTile::DType>);
                static_assert(DstTile::Rows == DestinationTile::Rows && DstTile::Cols == DestinationTile::Cols);
                this->template popTileFromVecFiFo<DstTile, Split>(fifo, dst);
            } else {
                static_assert(DstTile::Loc == TileType::Mat);
                this->template popTileFromMatFiFo<DstTile, Split>(fifo, dst);
            }
            return false;
        }
    };

    RingFiFo fifo;
    Producer prod;
    Consumer cons;

    AICORE explicit ExplicitTilePipe(uint32_t address, uint16_t row = 0, uint16_t col = 0)
        : fifo(nullptr, Dir == DIR_C2V ? address : 0, Dir == DIR_V2C ? address : 0)
    {
        prod.row = row;
        prod.col = col;
    }

    AICORE void waitReady(uint32_t event)
    {
        cons.readyEvent = event;
        cons.setWaitStatus(true);
    }
};

template <typename DstTile, STPhase Phase = STPhase::Unspecified>
using C2V = ExplicitTilePipe<DIR_C2V, DstTile, Phase>;
template <typename DstTile>
using V2C = ExplicitTilePipe<DIR_V2C, DstTile>;

} // namespace flash_mla_bf16::custom
#endif // __CCE_AICORE__
#endif // SPARSE_FLASH_MLA_CUSTOM_TPIPE_HPP_
