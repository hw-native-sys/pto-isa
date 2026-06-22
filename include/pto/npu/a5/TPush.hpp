/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef TPUSH_HPP
#define TPUSH_HPP

#include <type_traits>
#include <pto/common/fifo.hpp>
#include <pto/common/fixpipe.hpp>
#include <pto/npu/a5/TStore.hpp>
#include <pto/npu/a5/TLoad.hpp>
#include <pto/npu/a5/TMov.hpp>
#include <pto/common/debug.h>

namespace pto {

template <uint8_t FlagID, uint8_t DirType, uint32_t SlotSize, uint32_t SlotNum, uint32_t LocalSlotNum = 2,
          bool IsNoSplit = false, bool EN_UNIT_FLAG = false>
struct TPipe {
    static constexpr uint8_t DIR_MASK = 0x7;
    static constexpr uint8_t DIR_TYPE = DIR_MASK & DirType;
    static constexpr bool is_c2v_ub = (DIR_TYPE == Direction::DIR_C2V);        // 1
    static constexpr bool is_v2c_mat = (DIR_TYPE == Direction::DIR_V2C);       // 2
    static constexpr bool is_both = (DIR_TYPE == Direction::DIR_BOTH);         // 3
    static constexpr bool is_v2c_ctrl = (DIR_TYPE == Direction::DIR_V2C_CTRL); // 4
    static constexpr bool is_c2v_gm = (DIR_TYPE == Direction::DIR_C2V_GM);     // 5
    static constexpr bool is_v2c_gm = (DIR_TYPE == Direction::DIR_V2C_GM);     // 6
    static constexpr bool is_both_gm = (DIR_TYPE == Direction::DIR_BOTH_GM);   // 7
    static constexpr uint32_t SyncPeriod = (SlotNum <= 2) ? SlotNum : SlotNum / 2;
    static constexpr bool is_c2v = is_c2v_gm || is_c2v_ub;
    static constexpr bool is_v2c = is_v2c_gm || is_v2c_mat || is_v2c_ctrl;
    static_assert(is_c2v || is_v2c || is_both || is_both_gm,
                  "Fix: TPipe only supports C2V or V2C or Both communication on A5.");
    static constexpr uint8_t VEC_CORE_ID_OFFSET = 16;
    static_assert(FlagIDPlusOne < 16,
                  "Fix: With Both direction, FlagID + 1 must be less than 16 due to hardware limit.");

    // -------------------------------------------------------------------------
    // RingFiFo
    // -------------------------------------------------------------------------
    using RingFiFo = RingFIFO<SlotSize, SlotNum, LocalSlotNum>;

    PTO_INTERNAL static bool shouldWaitFree(uint32_t tileIndex)
    {
        // Startup protection: Don't check flags when buffer is empty
        if (tileIndex < SlotNum) {
            return false;
        }
        // Sparse sync: Only check flags periodically to reduce overhead
        return (tileIndex % SyncPeriod) == 0;
    }

    PTO_INTERNAL static bool shouldNotifyFree(uint32_t tileIndex)
    {
        // Notify consumer to free buffer when producer is one slot ahead of consumer
        return ((tileIndex + 1) % SyncPeriod) == 0;
    }

    // -------------------------------------------------------------------------
    // Producer Interface
    // -------------------------------------------------------------------------
    struct Producer {
        uint32_t subTileIndex = 0;
        uint32_t tileIndex = 0;
        bool isRecord = true;
        bool isAllocate = true;
        int entryOffset = 0;

        PTO_INTERNAL Producer() = default;

        PTO_INTERNAL void setTileId(int tIndex, int subIndex)
        {
            subTileIndex = subIndex;
            tileIndex = tIndex;
        }

        PTO_INTERNAL int getTileId() const
        {
            return tileIndex;
        }

        PTO_INTERNAL int getSubTileId() const
        {
            return subTileIndex;
        }

        PTO_INTERNAL void setAllocateStatus(bool allocate)
        {
            isAllocate = allocate;
        }

        PTO_INTERNAL bool getAllocateStatus() const
        {
            return isAllocate;
        }

        PTO_INTERNAL void setRecordStatus(bool record)
        {
            isRecord = record;
        }

        PTO_INTERNAL bool getRecordStatus() const
        {
            return isRecord;
        }

        PTO_INTERNAL void setEntryOffset(int offset)
        {
            entryOffset = offset;
        }

        template <pipe_t Pipe, TileSplitAxis Split>
        PTO_INTERNAL static void setIntraBlockBySplit(uint8_t flagId)
        {
            set_intra_block(Pipe, flagId);
            if constexpr (Split != TileSplitAxis::TILE_NO_SPLIT) {
                set_intra_block(Pipe, flagId + VEC_CORE_ID_OFFSET);
            }
        }

        template <pipe_t Pipe, TileSplitAxis Split>
        PTO_INTERNAL static void waitIntraBlockBySplit(uint8_t flagId)
        {
            wait_intra_block(Pipe, flagId);
            if constexpr (Split != TileSplitAxis::TILE_NO_SPLIT) {
                wait_intra_block(Pipe, flagId + VEC_CORE_ID_OFFSET);
            }
        }

        /**
         * alloc: Request space in FIFO
         * 1. (iter >= Depth): Startup protection. Don't check flags when buffer is empty.
         * 2. (iter % Period == 0): Sparse sync. Only check flag periodically.
         */
        template <TileSplitAxis Split = TileSplitAxis::TILE_UP_DOWN>
        PTO_INTERNAL void allocate() const
        {
            if constexpr (is_c2v) {
#ifdef __DAV_CUBE__
                waitIntraBlockBySplit<PIPE_FIX, Split>(FlagIDPlusOne);
#endif
            } else if constexpr (is_v2c_gm || is_v2c_mat) { // is_v2c (both gm and mat)
#ifdef __DAV_VEC__
                wait_intra_block(PIPE_MTE3, FlagIDPlusOne);
#endif
            } else if constexpr (is_both) {
#ifdef __DAV_CUBE__
                waitIntraBlockBySplit<PIPE_FIX, Split>(FlagIDPlusOne);
#endif
#ifdef __DAV_VEC__
                static_assert(FlagIDPlusThree < 16,
                              "Fix: With Both direction, FlagID + 3 must be less than 16 due to hardware limit.");
                wait_intra_block(PIPE_MTE3, FlagIDPlusThree);
#endif
            } else if constexpr (is_v2c_ctrl) {
#ifdef __DAV_VEC__
                wait_intra_block(PIPE_S, FlagIDPlusOne);
#endif
            } else if constexpr (is_both_gm) {
#ifdef __DAV_CUBE__
                waitIntraBlockBySplit<PIPE_FIX, Split>(FlagIDPlusOne);
#endif
#ifdef __DAV_VEC__
                static_assert(FlagIDPlusThree < 16,
                              "Fix: With Both direction, FlagID + 3 must be less than 16 due to hardware limit.");
                wait_intra_block(PIPE_MTE3, FlagIDPlusThree);
#endif
            }
        }

        // Forward dependency: record (producer) and wait (consumer)
        /**
         * record - Producer signals that data is ready
         * Called by the producer after completing the operation (TSTORE_C2GM or TSTORE_V2GM)
         */
        template <TileSplitAxis Split = TileSplitAxis::TILE_UP_DOWN>
        PTO_INTERNAL void record() const
        {
            if constexpr (is_c2v) {
#ifdef __DAV_CUBE__
                setIntraBlockBySplit<PIPE_FIX, Split>(FlagID);
#endif
            } else if constexpr (is_v2c_gm || is_v2c_mat) {
                set_intra_block(PIPE_MTE3, FlagID);
            } else if constexpr (is_both) {
#ifdef __DAV_CUBE__
                setIntraBlockBySplit<PIPE_FIX, Split>(FlagID);
#endif
#ifdef __DAV_VEC__
                set_intra_block(PIPE_MTE3, FlagIDPlusTwo);
#endif
            } else if constexpr (is_v2c_ctrl) {
                set_intra_block(PIPE_S, FlagID);
            } else if constexpr (is_both_gm) {
#ifdef __DAV_VEC__
                set_intra_block(PIPE_MTE3, FlagIDPlusTwo);
#endif
#ifdef __DAV_CUBE__
                setIntraBlockBySplit<PIPE_FIX, Split>(FlagID);
#endif
            }
        }

        template <typename TileProd, TileSplitAxis Split>
        PTO_INTERNAL void pushAcc2VecFiFo(RingFiFo &fifo, TileProd &tile)
        {
            using T = typename TileProd::DType;
            constexpr int ProdM = TileProd::Rows;
            constexpr int ProdN = TileProd::Cols;
            constexpr uint32_t splitNum = 2;
            constexpr int ConsM = (Split == TileSplitAxis::TILE_UP_DOWN) ? (ProdM / splitNum) : ProdM;
            constexpr int ConsN = (Split == TileSplitAxis::TILE_LEFT_RIGHT) ? (ProdN / splitNum) : ProdN;
            using TileCons = Tile<TileType::Vec, T, ConsM, ConsN, BLayout::RowMajor, ConsM, ConsN>;
            TileCons vecTile;
            uint64_t entryBase = (tileIndex % RingFiFo::SLOT_NUM) * RingFiFo::SLOT_SIZE; // ConsM * ConsN * sizeof(T);
            TASSIGN(vecTile, (uint64_t)(fifo.C2V_CONSUMER_BUF + entryBase + entryOffset));

            if constexpr (Split == TileSplitAxis::TILE_NO_SPLIT) {
                TMOV_IMPL<TileCons, TileProd, AccToVecMode::SingleModeVec0>(vecTile, tile);
            } else if constexpr (Split == TileSplitAxis::TILE_UP_DOWN) {
                static_assert((ProdM % 2 == 0) && (sizeof(T) == 4),
                              "Fix: For C2V(L0C-> UB), only support up-down split with ProdM being multiple of 2 due "
                              "to hardware requirement.");
                PTO_ASSERT((tile.GetValidRow() % 2 == 0) && (sizeof(T) == 4),
                           "Fix: For C2V(L0C-> UB), only support up-down split with ProdM being multiple of 2 due to "
                           "hardware requirement.");
                TMOV_IMPL<TileCons, TileProd, AccToVecMode::DualModeSplitM>(vecTile, tile);
            } else if constexpr (Split == TileSplitAxis::TILE_LEFT_RIGHT) {
                static_assert((ProdN % 32 == 0) && (sizeof(T) == 4),
                              "Fix: For C2V(L0C-> UB), only support left-right split with ProdN being multiple of 32 "
                              "due to hardware requirement.");
                PTO_ASSERT((tile.GetValidCol() % 32 == 0) && (sizeof(T) == 4),
                           "Fix: For C2V(L0C-> UB), only support left-right split with ProdN being multiple of 32 due "
                           "to hardware requirement.");
                TMOV_IMPL<TileCons, TileProd, AccToVecMode::DualModeSplitN>(vecTile, tile);
            }
        }

        template <typename TileProd, TileSplitAxis Split>
        PTO_INTERNAL void pushVec2MatFiFo(RingFiFo &fifo, TileProd &tile, int32_t subBlockId)
        {
            using T = typename TileProd::DType;
            constexpr int ProdM = TileProd::Rows;
            constexpr int ProdN = TileProd::Cols;
            constexpr uint32_t splitNum = 2;
            constexpr int ConsM = (Split == TileSplitAxis::TILE_UP_DOWN) ? ProdM * splitNum : ProdM;
            constexpr int ConsN = (Split == TileSplitAxis::TILE_LEFT_RIGHT) ? ProdN * splitNum : ProdN;
            Tile<TileType::Mat, T, ConsM, ConsN, BLayout::ColMajor, ConsM, ConsN> matTile;
            uint64_t entryBase = (tileIndex % RingFiFo::SLOT_NUM) * RingFiFo::SLOT_SIZE; // ConsM * ConsN * sizeof(T);
            TASSIGN_IMPL(matTile, (uint64_t)(fifo.V2C_CONSUMER_BUF + entryBase + entryOffset));
            if constexpr (Split == TileSplitAxis::TILE_NO_SPLIT) {
                // single vector core
                TINSERT_IMPL(matTile, tile, static_cast<uint16_t>(0), static_cast<uint16_t>(0));
            } else if constexpr (Split == TileSplitAxis::TILE_UP_DOWN) {
                int rowIndex = ProdM * static_cast<size_t>(get_subblockid());
                TINSERT_IMPL(matTile, tile, static_cast<uint16_t>(rowIndex), static_cast<uint16_t>(0));
            } else if constexpr (Split == TileSplitAxis::TILE_LEFT_RIGHT) {
                PTO_ASSERT(tile.GetValidCol() * sizeof(T) % 32 == 0,
                           "Fix: For V2C(UB->L1), tile's valid column must be multiple of 32 bytes due to hardware "
                           "requirement.");
                uint32_t colIndex = ProdN * static_cast<size_t>(get_subblockid());
                TINSERT_IMPL(matTile, tile, static_cast<uint16_t>(0), static_cast<uint16_t>(colIndex));
            }
        }

        template <typename TileProd>
        PTO_INTERNAL void pushAcc2GMFiFo(RingFiFo &fifo, TileProd &tile)
        {
            using T = typename TileProd::DType;
            int ValidC = tile.GetValidCol();
            int ValidR = tile.GetValidRow();
            using GlobalStride = pto::Stride<1, 1, 1, -1, 1>;
            using GlobalShape = pto::Shape<1, 1, 1, -1, -1>;
            using GlobalData = GlobalTensor<T, GlobalShape, GlobalStride>;
            size_t entryBase = (tileIndex % RingFiFo::SLOT_NUM) * RingFiFo::SLOT_SIZE;
            __gm__ T *addr = (__gm__ T *)((uint64_t)fifo.GM_SLOT_BUFFER + entryBase + entryOffset);
            GlobalData gmData(addr, GlobalShape(ValidR, ValidC), GlobalStride(ValidC));
            if constexpr (EN_UNIT_FLAG) {
                TSTORE_IMPL<TileProd, GlobalData, AtomicType::AtomicNone, STPhase::Final>(gmData, tile);
            } else { // disable unit flag
                TSTORE_IMPL(gmData, tile);
            }
        }

        template <typename TileProd, TileSplitAxis Split>
        PTO_INTERNAL void pushVec2GMFiFo(RingFiFo &fifo, TileProd &tile, int32_t subBlockId)
        {
            using T = typename TileProd::DType;
            constexpr int splitNum = 2;
            const int gmValidR = tile.GetValidRow();
            const int gmValidC = tile.GetValidCol();
            const int gmStrideR = (Split == TileSplitAxis::TILE_LEFT_RIGHT) ? gmValidC * splitNum : gmValidC;
            size_t entryBase = (tileIndex % RingFiFo::SLOT_NUM) * RingFiFo::SLOT_SIZE;
            size_t subAIVOffset = 0;
            if constexpr (Split == TileSplitAxis::TILE_NO_SPLIT) {
                subAIVOffset = 0;
            } else if constexpr (Split == TileSplitAxis::TILE_UP_DOWN) {
                subAIVOffset = subBlockId * gmValidR * gmValidC * sizeof(T);
            } else { // TILE_LEFT_RIGHT
                subAIVOffset = subBlockId * gmValidC * sizeof(T);
            }
            using GlobalShape = pto::Shape<1, 1, 1, -1, -1>;
            using GlobalStride = pto::Stride<1, 1, 1, -1, 1>;
            using GlobalData = GlobalTensor<T, GlobalShape, GlobalStride>;
            __gm__ T *addr = (__gm__ T *)((uint64_t)fifo.GM_SLOT_BUFFER + entryBase + subAIVOffset + entryOffset);
            GlobalData globalData(addr, GlobalShape(gmValidR, gmValidC), GlobalStride(gmStrideR));
            TSTORE_IMPL(globalData, tile);
        }

        template <typename TileProd>
        PTO_INTERNAL void pushVec2CtrlFiFo(RingFiFo &fifo, TileProd &tile)
        {
            uint32_t slotIndex = (tileIndex % RingFiFo::SLOT_NUM);
            uint64_t entryBase = slotIndex * sizeof(uint32_t);
            __ssbuf__ uint32_t *ctrlBuf = (__ssbuf__ uint32_t *)(fifo.V2C_CONSUMER_BUF + entryBase + entryOffset);
            uint32_t ctrlSignal = *(tile.data());
            *(ctrlBuf) = ctrlSignal;
        }

        template <typename TileProd, TileSplitAxis Split>
        PTO_INTERNAL void pushVec2FiFoByDir(RingFiFo &fifo, TileProd &tile)
        {
            if constexpr (is_v2c_mat || is_both) {
                pushVec2MatFiFo<TileProd, Split>(fifo, tile);
            } else if constexpr (is_v2c_gm) {
                pushVec2GMFiFo<TileProd, Split>(fifo, tile);
            } else if constexpr (is_v2c_ctrl) {
                pushVec2CtrlFiFo<TileProd>(fifo, tile);
            }
        }

        // NoQuant with Split and NZ2ND, need to support NZ2NZ
        template <typename TileProd, TileSplitAxis Split>
        PTO_INTERNAL void push(RingFiFo &fifo, TileProd &tile)
        {
            if constexpr (TileProd::Loc == TileType::Acc) {
                if constexpr (is_c2v_ub || is_both) {
                    pushAcc2VecFiFo<TileProd, Split>(fifo, tile);
                } else if constexpr (is_c2v_gm) {
                    pushAcc2GMFiFo<TileProd>(fifo, tile);
                }
            } else if constexpr (TileProd::Loc == TileType::Vec) {
                pushVec2FiFoByDir<TileProd, Split>(fifo, tile);
            }
        }

        //---------------------------fixpipe features support---------------------------
        template <typename TileProd, typename TConfig>
        using FixpipeConsType = FixpipeConsDType_t<TConfig::QuantPre, typename TileProd::DType>;

        template <typename TileProd, typename T, LayoutMode_t LayoutMode>
        using FixpipeVecTile = std::conditional_t<
            LayoutMode == LayoutMode_t::NZ2ND,
            Tile<TileType::Vec, T, TileProd::Rows, TileProd::Cols, BLayout::RowMajor, TileProd::Rows, TileProd::Cols>,
            std::conditional_t<LayoutMode == LayoutMode_t::NZ2DN,
                               Tile<TileType::Vec, T, TileProd::Rows, TileProd::Cols, BLayout::ColMajor, TileProd::Rows,
                                    TileProd::Cols>,
                               Tile<TileType::Vec, T, TileProd::Rows, TileProd::Cols, BLayout::ColMajor, TileProd::Rows,
                                    TileProd::Cols, SLayout::RowMajor>>>;

        template <typename TileProd, typename TConfig>
        using FixpipeGlobalData =
            GlobalTensor<FixpipeConsType<TileProd, TConfig>, pto::Shape<1, 1, 1, TileProd::Rows, TileProd::Cols>,
                         pto::Stride<1, 1, 1, TileProd::Cols, 1>, TConfig::LayoutMode>;

        template <typename TileProd, typename TConfig>
        PTO_INTERNAL void pushAcc2VecFiFo(RingFiFo &fifo, TileProd &tile)
        {
            using T = FixpipeConsType<TileProd, TConfig>;
            using TileCons = FixpipeVecTile<TileProd, T, TConfig::LayoutMode>;
            TileCons consTile;
            uint64_t entryBase = (tileIndex % RingFiFo::SLOT_NUM) * RingFiFo::SLOT_SIZE;
            TASSIGN(consTile, (uint64_t)(fifo.C2V_CONSUMER_BUF + entryBase + entryOffset));
            TMovCcToUb<TileCons, TileProd, (AccToVecMode) static_cast<uint8_t>(TConfig::SubBlockId), TConfig::QuantPre,
                       TConfig::ReluMode, TConfig::Phase>(consTile.data(), tile.data(), tile.GetValidRow(),
                                                          tile.GetValidCol());
        }

        template <typename TileProd, typename TConfig>
        PTO_INTERNAL void pushAcc2GMFiFo(RingFiFo &fifo, TileProd &tile)
        {
            using T = FixpipeConsType<TileProd, TConfig>;
            using GlobalData = FixpipeGlobalData<TileProd, TConfig>;
            size_t entryBase = (tileIndex % RingFiFo::SLOT_NUM) * RingFiFo::SLOT_SIZE;
            GlobalData globalTensor((__gm__ T *)((uint64_t)fifo.GM_SLOT_BUFFER + entryBase + entryOffset));

            if constexpr (TConfig::AtomicT == AtomicType::AtomicAdd) {
                SetAtomicAdd<typename GlobalData::DType>();
            }
            TStoreAcc<GlobalData, TileProd, TConfig::QuantPre, TConfig::ReluMode, TConfig::Phase>(
                globalTensor.data(), tile.data(), globalTensor.GetShape(GlobalTensorDim::DIM_0),
                globalTensor.GetShape(GlobalTensorDim::DIM_1), globalTensor.GetShape(GlobalTensorDim::DIM_2),
                globalTensor.GetShape(GlobalTensorDim::DIM_3), globalTensor.GetShape(GlobalTensorDim::DIM_4),
                globalTensor.GetStride(GlobalTensorDim::DIM_0), globalTensor.GetStride(GlobalTensorDim::DIM_1),
                globalTensor.GetStride(GlobalTensorDim::DIM_2), globalTensor.GetStride(GlobalTensorDim::DIM_3),
                globalTensor.GetStride(GlobalTensorDim::DIM_4), tile.GetValidRow(), tile.GetValidCol());
            if constexpr (TConfig::AtomicT == AtomicType::AtomicAdd) {
                set_atomic_none();
            }
        }

        template <typename TileProd, typename TConfig>
        PTO_INTERNAL void pushAcc2FiFoByDir(RingFiFo &fifo, TileProd &tile)
        {
            if constexpr (is_c2v_ub || is_both) {
                pushAcc2VecFiFo<TileProd, TConfig>(fifo, tile);
            } else if constexpr (is_c2v_gm) {
                pushAcc2GMFiFo<TileProd, TConfig>(fifo, tile);
            }
        }

        // cast quant, scalar quant, vector quant, NZ2ND, NZ2DN, NZ2NZ
        template <typename TileProd, typename TConfig>
        PTO_INTERNAL void push(RingFiFo &fifo, TileProd &tile)
        {
            static_assert(TileProd::Loc == TileType::Acc,
                          "Fix: the push interface with cast quant mode only suppport Acc tile type!");
            pushAcc2FiFoByDir<TileProd, TConfig>(fifo, tile);
        } // end of push
    };

    // -------------------------------------------------------------------------
    // Consumer Interface
    // -------------------------------------------------------------------------
    struct Consumer {
        uint32_t tileIndex = 0;
        uint32_t subTileIndex = 0;
        bool isWait = true;
        bool isFree = true;
        int entryOffset = 0;

        PTO_INTERNAL Consumer() = default;

        PTO_INTERNAL void setTileId(int tid, int sub_tid)
        {
            tileIndex = tid;
            subTileIndex = sub_tid;
        }

        PTO_INTERNAL int getTileId() const
        {
            return tileIndex;
        }

        PTO_INTERNAL int getSubTileId() const
        {
            return subTileIndex;
        }

        PTO_INTERNAL void setWaitStatus(bool wait)
        {
            isWait = wait;
        }

        PTO_INTERNAL bool getWaitStatus() const
        {
            return isWait;
        }

        PTO_INTERNAL void setFreeStatus(bool free)
        {
            isFree = free;
        }

        PTO_INTERNAL bool getFreeStatus() const
        {
            return isFree;
        }

        PTO_INTERNAL void setEntryOffset(int offset)
        {
            entryOffset = offset;
        }

        template <pipe_t Pipe, TileSplitAxis Split>
        PTO_INTERNAL static void waitIntraBlockBySplit(uint8_t flagId)
        {
            wait_intra_block(Pipe, flagId);
            if constexpr (Split != TileSplitAxis::TILE_NO_SPLIT) {
                wait_intra_block(Pipe, flagId + VEC_CORE_ID_OFFSET);
            }
        }

        template <pipe_t Pipe, TileSplitAxis Split>
        PTO_INTERNAL static void setIntraBlockBySplit(uint8_t flagId)
        {
            set_intra_block(Pipe, flagId);
            if constexpr (Split != TileSplitAxis::TILE_NO_SPLIT) {
                set_intra_block(Pipe, flagId + VEC_CORE_ID_OFFSET);
            }
        }

        /**
         * wait: Block until data is ready
         */
        template <TileSplitAxis Split = TileSplitAxis::TILE_UP_DOWN>
        PTO_INTERNAL void wait() const
        {
            if constexpr (is_c2v_gm) {
                wait_intra_block(PIPE_MTE2, FlagID);
            } else if constexpr (is_c2v_ub) {
#ifdef __DAV_VEC__
                wait_intra_block(PIPE_V, FlagID);
#endif
            } else if constexpr (is_v2c_gm) {
                waitIntraBlockBySplit<PIPE_MTE2, Split>(FlagID);
            } else if constexpr (is_v2c_mat) {
#ifdef __DAV_CUBE__
                waitIntraBlockBySplit<PIPE_MTE1, Split>(FlagID);
#endif
            } else if constexpr (is_both) {
#ifdef __DAV_VEC__ // c2v_ub
                wait_intra_block(PIPE_V, FlagID);
#endif
#ifdef __DAV_CUBE__ // v2c_mat
                waitIntraBlockBySplit<PIPE_MTE1, Split>(FlagIDPlusTwo);
#endif
            } else if constexpr (is_v2c_ctrl) {
                waitIntraBlockBySplit<PIPE_S, Split>(FlagID);
            } else if constexpr (is_both_gm) {
#ifdef __DAV_VEC__
                wait_intra_block(PIPE_MTE2, FlagIDPlusOne);
#endif
#ifdef __DAV_CUBE__
                waitIntraBlockBySplit<PIPE_MTE2, Split>(FlagIDPlusTwo);
#endif
            }
        }

        /**
         * free: Release space in FIFO
         */
        template <TileSplitAxis Split = TileSplitAxis::TILE_UP_DOWN>
        PTO_INTERNAL void free() const
        {
            if constexpr (is_c2v_gm) {
#ifdef __DAV_VEC__
                set_intra_block(PIPE_MTE2, FlagIDPlusOne);
#endif
            } else if constexpr (is_c2v_ub) {
#ifdef __DAV_VEC__
                set_intra_block(PIPE_V, FlagIDPlusOne);
#endif
            } else if constexpr (is_both) {
#ifdef __DAV_VEC__
                set_intra_block(PIPE_V, FlagIDPlusOne);
#endif
#ifdef __DAV_CUBE__
                static_assert(FlagIDPlusThree < 16,
                              "Fix: With Both direction, FlagID + 3 must be less than 16 due to hardware limit.");
                setIntraBlockBySplit<PIPE_MTE1, Split>(FlagIDPlusThree);
#endif
            } else if constexpr (is_v2c_gm || is_v2c_mat) {
#ifdef __DAV_CUBE__
                setIntraBlockBySplit<PIPE_MTE1, Split>(FlagIDPlusOne);
#endif
            } else if constexpr (is_v2c_ctrl) {
#ifdef __DAV_CUBE__
                setIntraBlockBySplit<PIPE_S, Split>(FlagIDPlusOne);
#endif
            } else if constexpr (is_both_gm) {
#ifdef __DAV_VEC__
                set_intra_block(PIPE_MTE2, FlagIDPlusOne);
#endif
#ifdef __DAV_CUBE__
                static_assert(FlagIDPlusThree < 16,
                              "Fix: With Both direction, FlagID + 3 must be less than 16 due to hardware limit.");
                setIntraBlockBySplit<PIPE_MTE1, Split>(FlagIDPlusThree);
#endif
            }
        }

        template <typename TileCons, TileSplitAxis Split>
        PTO_INTERNAL void popTileFromVecFiFo(RingFiFo &fifo, TileCons &tile)
        {
            using T = typename TileCons::DType;
            constexpr int ConsM = TileCons::Rows;
            constexpr int ConsN = TileCons::Cols;
            uint32_t entryBase = (tileIndex % RingFiFo::SLOT_NUM) * RingFiFo::SLOT_SIZE;
            uint64_t localTileBase = fifo.C2V_CONSUMER_BUF + entryBase + entryOffset;
            TASSIGN_IMPL(tile, localTileBase);
        }

        template <typename TileCons, TileSplitAxis Split>
        PTO_INTERNAL void popTileFromMatFiFo(RingFiFo &fifo, TileCons &tile)
        {
            using T = typename TileCons::DType;
            constexpr int ConsM = TileCons::Rows;
            constexpr int ConsN = TileCons::Cols;
            uint32_t entryBase = (tileIndex % RingFiFo::SLOT_NUM) * RingFiFo::SLOT_SIZE;
            uint64_t localTileBase = fifo.V2C_CONSUMER_BUF + entryBase + entryOffset;
            TASSIGN_IMPL(tile, localTileBase);
        }

        template <typename TileCons, TileSplitAxis Split>
        PTO_INTERNAL void popVecTileFromGMFiFo(RingFiFo &fifo, TileCons &tile, int32_t subBlockId)
        {
            constexpr int splitNum = 2;
            using T = typename TileCons::DType;
            constexpr int ConsN = TileCons::Cols;
            constexpr int ConsM = TileCons::Rows;
            constexpr int ProdN = (Split == TileSplitAxis::TILE_LEFT_RIGHT) ? ConsN * splitNum : ConsN;
            constexpr int ProdM = (Split == TileSplitAxis::TILE_UP_DOWN) ? ConsM * splitNum : ConsM;

            // global tensor
            size_t entryBase = (static_cast<size_t>(tileIndex) % RingFiFo::SLOT_NUM) *
                               RingFiFo::SLOT_SIZE; // ProdM * ProdN * sizeof(T);
            constexpr int gmValidC = ConsN;
            constexpr int gmValidR = ConsM;
            constexpr int gmStrideR = ProdN;
            size_t subAIVOffset = 0;
            if constexpr (Split == TileSplitAxis::TILE_UP_DOWN) {
                // TILE_UP_DOWN  : Vec1 starts at the second row-block → offset = VEC_M * ProdN * sizeof(T)
                subAIVOffset = subBlockId * ConsM * ConsN * sizeof(T);
            } else if constexpr (Split == TileSplitAxis::TILE_LEFT_RIGHT) { // TILE_LEFT_RIGHT
                // TILE_LEFT_RIGHT: Vec1 starts at column ConsN within row 0 → offset = ConsN * sizeof(T)
                subAIVOffset = subBlockId * ConsN * sizeof(T);
            } else if constexpr (Split == TileSplitAxis::TILE_NO_SPLIT) {
                subAIVOffset = 0; // TILE_NO_SPLIT : single reader, no offset needed
            }
            using GlobalData =
                GlobalTensor<T, pto::Shape<1, 1, 1, gmValidR, gmValidC>, pto::Stride<1, 1, 1, gmStrideR, 1>>;
            __gm__ T *addr = (__gm__ T *)((uint64_t)fifo.GM_SLOT_BUFFER + entryBase + subAIVOffset + entryOffset);
            GlobalData globalTensor(addr);

            uint64_t localTileBase =
                fifo.C2V_CONSUMER_BUF +
                (static_cast<size_t>(tileIndex) % RingFiFo::LOCAL_SLOT_NUM) * ConsM * ConsN * sizeof(T);
            TASSIGN_IMPL(tile, localTileBase);
            TLOAD_IMPL(tile, globalTensor);
        }

        template <typename TileCons, TileSplitAxis Split>
        PTO_INTERNAL void popMatTileFromGMFiFo(RingFiFo &fifo, TileCons &tile)
        {
            using T = typename TileCons::DType;
            constexpr int ConsN = TileCons::Cols;
            constexpr int ConsM = TileCons::Rows;
            using GlobalData = GlobalTensor<T, pto::Shape<1, 1, 1, ConsM, ConsN>, pto::Stride<1, 1, 1, ConsN, 1>>;
            uint32_t entryBase = (tileIndex % RingFiFo::SLOT_NUM) * RingFiFo::SLOT_SIZE;
            GlobalData globalTensor((__gm__ T *)((uint64_t)fifo.GM_SLOT_BUFFER + entryBase + entryOffset));

            uint64_t localTileBase =
                fifo.V2C_CONSUMER_BUF + (tileIndex % RingFiFo::LOCAL_SLOT_NUM) * ConsM * ConsN * sizeof(T);
            TASSIGN_IMPL(tile, localTileBase);
            TLOAD_IMPL(tile, globalTensor);
        }

        PTO_INTERNAL void popCtrlFromCtrlFiFo(RingFiFo &fifo)
        {
            uint32_t slotIndex = (tileIndex % fifo.SLOT_NUM);
            size_t entryBase = slotIndex * sizeof(uint32_t);
            uint64_t ctrlTileBase = fifo.fifoBase + entryBase + entryOffset;
            fifo.ctrlSignal = (*(__ssbuf__ uint32_t *)(ctrlTileBase) == 1) ? true : false;
        }

        template <typename TileCons, TileSplitAxis Split>
        PTO_INTERNAL bool pop(RingFiFo &fifo, TileCons &tile, int32_t subBlockId)
        {
            static_assert(
                TileCons::Loc == TileType::Vec || TileCons::Loc == TileType::Mat || TileCons::Loc == TileType::Ctrl,
                "Fix: TPOP has unsupported tile type!");
            if constexpr (TileCons::Loc == TileType::Vec) {
                if constexpr (is_c2v_ub || is_both) {
                    popTileFromVecFiFo<TileCons, Split>(fifo, tile);
                    return false;
                } else if constexpr (is_c2v_gm) {
                    popVecTileFromGMFiFo<TileCons, Split>(fifo, tile, subBlockId);
                    return true;
                }
            } else if constexpr (TileCons::Loc == TileType::Mat) {
                if constexpr (is_v2c_mat || is_both) {
                    popTileFromMatFiFo<TileCons, Split>(fifo, tile);
                    return false;
                } else if constexpr (is_v2c_gm) {
                    popMatTileFromGMFiFo<TileCons, Split>(fifo, tile);
                    return true;
                }
            } else { // pop ctrl tile
                popCtrlFromCtrlFiFo(fifo, tile);
                return false;
            }
        }
    };

    RingFiFo fifo;
    Producer prod;
    Consumer cons;

    PTO_INTERNAL explicit TPipe(__gm__ void *GM_SLOT_BUFFER, uint32_t C2V_CONSUMER_BUF, uint32_t V2C_CONSUMER_BUF)
        : fifo(GM_SLOT_BUFFER, C2V_CONSUMER_BUF, V2C_CONSUMER_BUF), prod(), cons()
    {
        for (uint32_t i = 0; i < SyncPeriod; ++i) {
            if constexpr (IsNoSplit) {
                cons.template free<TileSplitAxis::TILE_NO_SPLIT>();
            } else {
                cons.template free<TileSplitAxis::TILE_UP_DOWN>();
            }
        }
    }

    // Destructor for TPipe
    PTO_INTERNAL ~TPipe()
    {
        for (uint32_t i = 0; i < SyncPeriod; ++i) {
            if constexpr (IsNoSplit) {
                prod.template allocate<TileSplitAxis::TILE_NO_SPLIT>();
            } else {
                prod.template allocate<TileSplitAxis::TILE_UP_DOWN>();
            }
        }
    }
};

/**
 * TPUSH: Push Tile to FIFO
 * * Flow:
 * 1. [Alloc]   Check GM space (Cross-Core)
 * 2. [Store]   Write data to GM
 * 3. [Commit]  Signal Consumer (Cross-Core)
 */
template <typename Pipe, typename TileProd, TileSplitAxis Split, std::enable_if_t<is_tile_data_v<TileProd>, int> = 0>
PTO_INTERNAL void TPUSH_IMPL(Pipe &pipe, TileProd &tile)
{
    // 1. Cross-Core: Wait for space
    bool isAllocate = pipe.prod.getAllocateStatus() && Pipe::shouldWaitFree(pipe.prod.tileIndex);
    if (isAllocate) {
        pipe.prod.template allocate<Split>();
    }

    // 2. Address Calculation
    pipe.prod.template push<TileProd, Split>(pipe.fifo, tile, get_subblockid());
    pipe.prod.tileIndex++;

    // 3. Cross-Core: Commit & Signal
    bool isRecord = pipe.prod.getRecordStatus();
    if (isRecord) {
        pipe.prod.template record<Split>();
    }
}

// TPUSH interface when push data from GM FIFO
template <typename Pipe, typename GlobalData, TileSplitAxis Split,
          std::enable_if_t<is_global_data_v<GlobalData>, int> = 0>
PTO_INTERNAL void TPUSH_IMPL(Pipe &pipe, GlobalData &gmTensor)
{
    (void)gmTensor;
    static_assert(Pipe::is_c2v_gm || Pipe::is_v2c_gm || Pipe::is_both_gm,
                  "Fix: TPUSH with GlobalTensor is only supported by GM FIFO directions on A5.");
    pipe.prod.template record<Split>();
}

// TPUSH interface with quant(Noquant, cast, scalar, vector, scalar)
template <typename Pipe, typename TileProd, typename TConfig>
PTO_INTERNAL void TPUSH_IMPL(Pipe &pipe, TileProd &tile)
{
    bool isAllocate = pipe.prod.getAllocateStatus() && Pipe::shouldWaitFree(pipe.prod.tileIndex);
    if (isAllocate) {
        pipe.prod.template allocate<TileSplitAxis::TILE_NO_SPLIT>();
    }

    pipe.prod.template push<TileProd, TConfig>(pipe.fifo, tile);
    pipe.prod.tileIndex++;

    bool isRecord = pipe.prod.getRecordStatus();
    if (isRecord) {
        pipe.prod.template record<TileSplitAxis::TILE_NO_SPLIT>();
    }
}

//------------------------multiple pipe------------------------
template <uint8_t FlagID, FIFOType FiFoType, uint8_t FiFoDepth, uint8_t FiFoSyncT, typename TileDataProd,
          typename TileDataCons, bool EN_UNIT_FLAG = false, uint8_t LocalFiFoDepth = 2,
          VecCubeRatio VCRatio = VecCubeRatio::V2C1_VECS>
struct TMPipe {
    static constexpr uint8_t FlagIDPlusOne = FlagID + 1;
    static constexpr bool is_c2v_gm =
        (FiFoType == FIFOType::GM_FIFO) && (TileDataProd::Loc == TileType::Acc) && (TileDataCons::Loc == TileType::Vec);
    static constexpr bool is_c2v_ub = (FiFoType == FIFOType::VEC_FIFO) && (TileDataProd::Loc == TileType::Acc) &&
                                      (TileDataCons::Loc == TileType::Vec);
    static constexpr bool is_c2v = is_c2v_gm || is_c2v_ub;
    static constexpr bool is_v2c_gm =
        (FiFoType == FIFOType::GM_FIFO) && (TileDataProd::Loc == TileType::Vec) && (TileDataCons::Loc == TileType::Mat);
    static constexpr bool is_v2c_mat = (FiFoType == FIFOType::MAT_FIFO) && (TileDataProd::Loc == TileType::Vec) &&
                                       (TileDataCons::Loc == TileType::Mat);
    static constexpr bool is_v2c_ctrl = (FiFoType == FIFOType::CTRL_FIFO) && (TileDataProd::Loc == TileType::Vec) &&
                                        (TileDataCons::Loc == TileType::Ctrl);
    static constexpr bool is_v2c = is_v2c_gm || is_v2c_mat || is_v2c_ctrl;
    static_assert(
        is_c2v || is_v2c,
        "TPipe currently only supports Cube-to-Vec or Vec-to-Cube communication with specified tile and FIFO types.");
    static constexpr int VEC_CORE_ID_OFFSET = 16;
    static_assert(FlagIDPlusOne < 16,
                  "Fix: With single direction, FlagID + 1 must be less than 16 due to hardware limit.");

    using DataFiFo =
        std::conditional_t<(FiFoType == FIFOType::GM_FIFO),
                           DataFIFO<typename TileDataCons::DType, FiFoType, FiFoDepth, FiFoSyncT, LocalFiFoDepth>,
                           DataFIFO<TileDataCons, FiFoType, FiFoDepth, FiFoSyncT>>;

    // -------------------------------------------------------------------------
    // Producer Interface
    // -------------------------------------------------------------------------
    struct Producer {
        int tile_id = 0;
        int sub_tile_id = 0;
        bool isAllocate = true;
        bool isRecord = true;
        int entryOffset = 0;

        PTO_INTERNAL Producer() = default;

        PTO_INTERNAL void setTileId(int t_id, int sub_t_id)
        {
            tile_id = t_id;
            sub_tile_id = sub_t_id;
        }

        PTO_INTERNAL int getTileId() const
        {
            return tile_id;
        }

        PTO_INTERNAL int getSubTileId() const
        {
            return sub_tile_id;
        }

        PTO_INTERNAL void setAllocateStatus(bool allocate)
        {
            isAllocate = allocate;
        }

        PTO_INTERNAL bool getAllocateStatus() const
        {
            return isAllocate;
        }

        PTO_INTERNAL void setRecordStatus(bool record)
        {
            isRecord = record;
        }

        PTO_INTERNAL bool getRecordStatus() const
        {
            return isRecord;
        }

        PTO_INTERNAL void setEntryOffset(int offset)
        {
            entryOffset = offset;
        }

        /**
         * alloc: Request space in FIFO
         * 1. (iter >= Depth): Startup protection. Don't check flags when buffer is empty.
         * 2. (iter % Period == 0): Sparse sync. Only check flag periodically.
         */
        PTO_INTERNAL void allocate() const
        {
            if constexpr (is_c2v) {
                // Cube producer waits for Vec consumer to free buffer
                // Vec signals on flag_id+1 only, but Cube must wait on BOTH
                // (because Vec0 signals flag_id+1, Vec1 signals flag_id+1+16 from Cube's view)
#ifdef __DAV_CUBE__
                wait_intra_block(PIPE_FIX, FlagIDPlusOne);
                wait_intra_block(PIPE_FIX, FlagIDPlusOne + VEC_CORE_ID_OFFSET);
#endif
            } else if constexpr (is_v2c_gm || is_v2c_mat) {
                // is_v2c (both gm and mat)
                // Vec producer waits for Cube consumer to free buffer
                // Cube signals on BOTH, Vec waits on flag_id+1 only
#ifdef __DAV_VEC__
                wait_intra_block(PIPE_MTE3, FlagIDPlusOne);
#endif
            } else if constexpr (is_v2c_ctrl) {
                // is_v2c_ctrl
                // Control signals from Vec to Cube: Vec signals on flag_id, Cube waits on flag_id only
#ifdef __DAV_VEC__
                wait_intra_block(PIPE_S, FlagIDPlusOne);
#endif
            }
        }

        // Forward dependency: record (producer) and wait (consumer)
        /**
         * record - Producer signals that data is ready
         * Called by the producer after completing the operation (TSTORE_C2GM or TSTORE_V2GM)
         */
        PTO_INTERNAL void record() const
        {
            if constexpr (is_c2v) {
#ifdef __DAV_CUBE__
                // Cube -> Vec: Cube sets BOTH flags on PIPE_FIX
                set_intra_block(PIPE_FIX, FlagID);
                set_intra_block(PIPE_FIX, FlagID + VEC_CORE_ID_OFFSET);
#endif
            } else if constexpr (is_v2c_gm || is_v2c_mat) { // is_v2c (both gm and mat)
                // Vec -> Cube: Vec sets flag_id only on PIPE_MTE3
                // Each Vec subblock executes this; hardware maps subblock 1's flag to flag_id+16
                set_intra_block(PIPE_MTE3, FlagID);
            } else { // is_v2c_ctrl
                // Control signals from Vec to Cube: Vec signals on flag_id, Cube waits on flag_id only
                set_intra_block(PIPE_S, FlagID);
            }
        }

        template <typename T, int ProdM, int ProdN, int ConsM, int ConsN>
        PTO_INTERNAL void pushAcc2GMFiFo(DataFiFo &fifo, TileDataProd &tile)
        {
            // calculate base address in GM FIFO for this tile
            constexpr int kTileFactor = ConsN / ProdN;
            uint32_t bufIndex = static_cast<uint32_t>(tile_id % DataFiFo::fifoDepth);
            size_t entryBase = bufIndex * kTileFactor * ProdM * ProdN * sizeof(T);
            using GlobalData = GlobalTensor<T, pto::Shape<1, 1, 1, ProdM, ProdN>, pto::Stride<1, 1, 1, ProdN, 1>>;
            GlobalData globalTensor((__gm__ T *)((uint64_t)fifo.fifoBase + entryBase + entryOffset));
            // store tile to GM FIFO, enable unit-flag or disable unit-flag
            if constexpr (EN_UNIT_FLAG) {
                TSTORE_IMPL<TileDataProd, GlobalData, AtomicType::AtomicNone, STPhase::Final>(globalTensor, tile);
            } else { // disable unit flag
                TSTORE_IMPL(globalTensor, tile);
            }
        }

        template <typename T, int ProdM, int ProdN, int ConsM, int ConsN, int VEC_CORES>
        PTO_INTERNAL void pushAcc2VecFiFo(DataFiFo &fifo, TileDataProd &tile)
        {
            static_assert((TileDataProd::Loc == TileType::Acc) && (DataFiFo::fifoType == FIFOType::VEC_FIFO),
                          "Fix: TPUSH has unsupported fifo type!");
            constexpr bool isSplitM = (ProdM != ConsM && ProdN == ConsN && VEC_CORES == 2);
            constexpr bool isSplitN = (ProdM == ConsM && ProdN != ConsN && VEC_CORES == 2);
            constexpr bool nonSplit = (ProdM == ConsM && ProdN == ConsN && VEC_CORES == 1);
            // Note: make sure the vecTile is stored in VEC_FIFO continuously.
            // dual vector cores(1c:2v)
            if constexpr (isSplitM) {
                // split M between two vectors
                constexpr int kTileFactor = ConsN / ProdN;
                constexpr uint32_t VecM = ProdM / VEC_CORES / kTileFactor;
                using TileDataVec = Tile<TileType::Vec, T, VecM, ProdN, BLayout::RowMajor, VecM, ProdN>;
                TileDataVec vecTile;
                uint64_t entryBase = (tile_id % DataFiFo::fifoDepth) * VecM * ProdN * sizeof(T);
                TASSIGN(vecTile, fifo.fifoBase + entryBase + entryOffset);
                TMOV_IMPL<TileDataVec, TileDataProd, AccToVecMode::DualModeSplitM>(vecTile, tile);
            } else if constexpr (isSplitN) {
                // split N between two vectors
                constexpr int kTileFactor = ConsN / ProdN;
                constexpr uint32_t VecN = ProdN / VEC_CORES / kTileFactor;
                using TileDataVec = Tile<TileType::Vec, T, ProdM, VecN, BLayout::RowMajor, ProdM, VecN>;
                TileDataVec vecTile;
                uint64_t entryBase = (tile_id % DataFiFo::fifoDepth) * ProdM * VecN * sizeof(T);
                TASSIGN(vecTile, fifo.fifoBase + entryBase + entryOffset);
                TMOV_IMPL<TileDataVec, TileDataProd, AccToVecMode::DualModeSplitN>(vecTile, tile);
            } else if constexpr (nonSplit) {
                // single vector core (1v:1v)
                TileDataCons vecTile;
                uint64_t entryBase = (tile_id % DataFiFo::fifoDepth) * ProdM * ProdN * sizeof(T);
                TASSIGN(vecTile, fifo.fifoBase + entryBase + entryOffset);
                TMOV_IMPL<TileDataCons, TileDataProd, AccToVecMode::SingleModeVec0>(vecTile, tile);
            } else {
                static_assert(isSplitM || isSplitN || nonSplit,
                              "Fix: TPUSH(pushAcc2VecFiFo) has unsupported split mode!");
            }
        }

        template <typename T, int ProdM, int ProdN, int ConsM, int ConsN>
        PTO_INTERNAL void pushVec2GMFiFo(DataFiFo &fifo, TileDataProd &tile)
        {
            static_assert(DataFiFo::fifoType == FIFOType::GM_FIFO, "Fix: TPUSH: unsupported fifoType!");
            // calculate base address in GM FIFO for this tile
            constexpr int kTileFactor = ProdN / ConsN;
            uint32_t bufIndex = static_cast<uint32_t>(tile_id % DataFiFo::fifoDepth);
            size_t entryBase = bufIndex * kTileFactor * ConsM * ConsN * sizeof(T);
            using GlobalDataSub = GlobalTensor<T, pto::Shape<1, 1, 1, ProdM, ConsN>, pto::Stride<1, 1, 1, ConsN, 1>>;
            using TileDataSub = Tile<TileType::Vec, T, ProdM, ProdN, BLayout::RowMajor, ProdM, ConsN>;
            TileDataSub subTile;
            __gm__ T *addr = (__gm__ T *)((uint64_t)fifo.fifoBase + entryBase + entryOffset);
            // store tile to GM FIFO in sub-tiles if needed (when Tile_S1 > Cube_S1)
            for (int sub_col = 0; sub_col < kTileFactor; ++sub_col) {
                __gm__ T *addrSub = addr + sub_col * ConsM * ConsN;
                GlobalDataSub globalDataSub((__gm__ T *)(addrSub));
                uint64_t col_byte_offset = static_cast<uint64_t>(sub_col * ConsN * sizeof(T));
                TASSIGN_IMPL(subTile, (uint64_t)tile.data() + col_byte_offset);
                TSTORE_IMPL(globalDataSub, subTile);
            }
        }

        template <typename T, int ProdM, int ProdN, int ConsM, int ConsN, int VEC_CORES>
        PTO_INTERNAL void pushVec2MatFiFo(DataFiFo &fifo, TileDataProd &tile)
        {
            static_assert((TileDataProd::Loc == TileType::Vec) && (DataFiFo::fifoType == FIFOType::MAT_FIFO),
                          "Fix: TPUSH has unsupported fifo type!");
            constexpr bool isSplitM = (ProdM != ConsM && ProdN == ConsN && VEC_CORES == 2);
            constexpr bool isSplitN = (ProdM == ConsM && ProdN != ConsN && VEC_CORES == 2);
            constexpr bool nonSplit = (ProdM == ConsM && ProdN == ConsN && VEC_CORES == 1);
            // dual vector cores
            if constexpr (isSplitM) {
                // split M between vectors
                constexpr uint32_t VecM = ConsM / VEC_CORES;
                int row_offset = VecM * static_cast<size_t>(get_subblockid());
                uint64_t entryBase = (tile_id % DataFiFo::fifoDepth) * ConsM * ConsN * sizeof(T);
                TileDataCons matTile;
                TASSIGN_IMPL(matTile, fifo.fifoBase + entryBase);
                TINSERT_IMPL(matTile, tile, static_cast<uint16_t>(row_offset), static_cast<uint16_t>(0));
            } else if constexpr (isSplitN) {
                // split N between vectors
                int col_index = ProdN * static_cast<size_t>(get_subblockid());
                uint64_t entryBase = (tile_id % DataFiFo::fifoDepth) * ConsM * ConsN * sizeof(T);
                TileDataCons matTile;
                TASSIGN_IMPL(matTile, fifo.fifoBase + entryBase);
                TINSERT_IMPL(matTile, tile, static_cast<uint16_t>(0), static_cast<uint16_t>(col_index));
            } else if constexpr (nonSplit) {
                // single vector core
                TileDataCons matTile;
                uint64_t entryBase = (tile_id % DataFiFo::fifoDepth) * ConsM * ConsN * sizeof(T);
                TASSIGN_IMPL(matTile, fifo.fifoBase + entryBase);
                TINSERT_IMPL(matTile, tile, static_cast<uint16_t>(0), static_cast<uint16_t>(0));
            } else {
                static_assert(isSplitM || isSplitN || nonSplit,
                              "Fix: TPUSH(pushVec2MatFiFo) has unsupported split mode!");
            }
        }

        template <typename T, int ProdM, int ProdN, int ConsM, int ConsN>
        PTO_INTERNAL void pushVec2CtrlFiFo(DataFiFo &fifo, TileDataProd &tile)
        {
            static_assert(DataFiFo::fifoType == FIFOType::CTRL_FIFO,
                          "Fix: TPUSH(pushVec2CtrlFiFo) has unsupported fifoType!");
            uint32_t bufIndex = static_cast<uint32_t>(tile_id % DataFiFo::fifoDepth);
            uint64_t entryBase = bufIndex * sizeof(uint32_t);
            __ssbuf__ uint32_t *ctrlBuf = (__ssbuf__ uint32_t *)(fifo.fifoBase + entryBase + entryOffset);
            uint32_t ctrlSignal = *(tile.data());
            *(ctrlBuf) = ctrlSignal;
        }

        PTO_INTERNAL void push(DataFiFo &fifo, TileDataProd &tile)
        {
            // get tile shape and valid shape
            using T = typename TileDataProd::DType;
            constexpr int ProdM = TileDataProd::Rows;
            constexpr int ProdN = TileDataProd::Cols;
            constexpr int ConsM = TileDataCons::Rows;
            constexpr int ConsN = TileDataCons::Cols;
            constexpr int VEC_CORES = (VCRatio == VecCubeRatio::V2C1_VECS) ? 2 : 1;

            static_assert(TileDataProd::Loc == TileType::Acc || TileDataProd::Loc == TileType::Vec,
                          "Fix: TPUSH has unsupported tile type!");
            if constexpr (TileDataProd::Loc == TileType::Acc) {
                static_assert((DataFiFo::fifoType == FIFOType::GM_FIFO) || (DataFiFo::fifoType == FIFOType::VEC_FIFO),
                              "Fix: TPUSH has unsupported fifo type!");
                if constexpr (DataFiFo::fifoType == FIFOType::GM_FIFO) {
                    pushAcc2GMFiFo<T, ProdM, ProdN, ConsM, ConsN>(fifo, tile);
                } else if constexpr (DataFiFo::fifoType == FIFOType::VEC_FIFO) {
                    pushAcc2VecFiFo<T, ProdM, ProdN, ConsM, ConsN, VEC_CORES>(fifo, tile);
                }
            } else if constexpr (TileDataProd::Loc == TileType::Vec) {
                static_assert(DataFiFo::fifoType == FIFOType::GM_FIFO || DataFiFo::fifoType == FIFOType::MAT_FIFO ||
                                  DataFiFo::fifoType == FIFOType::CTRL_FIFO,
                              "Fix: TPUSH has unsupported fifo type!");
                if constexpr (DataFiFo::fifoType == FIFOType::GM_FIFO) {
                    pushVec2GMFiFo<T, ProdM, ProdN, ConsM, ConsN>(fifo, tile);
                } else if constexpr (DataFiFo::fifoType == FIFOType::MAT_FIFO) {
                    pushVec2MatFiFo<T, ProdM, ProdN, ConsM, ConsN, VEC_CORES>(fifo, tile);
                } else if constexpr (DataFiFo::fifoType == FIFOType::CTRL_FIFO) {
                    pushVec2CtrlFiFo<T, ProdM, ProdN, ConsM, ConsN>(fifo, tile);
                }
            } // end of store
        }
    };

    // -------------------------------------------------------------------------
    // Consumer Interface
    // -------------------------------------------------------------------------
    struct Consumer {
        int tile_id = 0;
        int sub_tile_id = 0;
        bool isWait = true;
        bool isFree = true;
        int entryOffset = 0;

        PTO_INTERNAL Consumer() = default;

        PTO_INTERNAL void setTileId(int tid, int sub_tid)
        {
            tile_id = tid;
            sub_tile_id = sub_tid;
        }

        PTO_INTERNAL int getTileId() const
        {
            return tile_id;
        }

        PTO_INTERNAL int getSubTileId() const
        {
            return sub_tile_id;
        }

        PTO_INTERNAL void setWaitStatus(bool wait)
        {
            isWait = wait;
        }

        PTO_INTERNAL bool getWaitStatus() const
        {
            return isWait;
        }

        PTO_INTERNAL void setFreeStatus(bool free)
        {
            isFree = free;
        }

        PTO_INTERNAL bool getFreeStatus() const
        {
            return isFree;
        }

        PTO_INTERNAL void setEntryOffset(int offset)
        {
            entryOffset = offset;
        }

        /**
         * wait: Block until data is ready
         * Consumers strictly wait for data (no sparse optimization for safety).
         */
        PTO_INTERNAL void wait() const
        {
            if constexpr (is_c2v_gm) {
                // Cube -> Vec (GM path): Vec waits on PIPE_MTE2 (data loaded from GM)
                wait_intra_block(PIPE_MTE2, FlagID);
            } else if constexpr (is_c2v_ub) {
                // Cube -> Vec (UB path): Vec waits on PIPE_V before vector ops on UB data
                // Cube sets PIPE_FIX, Vec waits PIPE_V (Vec does vector ops, not TLOAD)
#ifdef __DAV_VEC__
                wait_intra_block(PIPE_V, FlagID);
#endif
            } else if constexpr (is_v2c_gm) {
                // Vec -> Cube (GM path): Cube waits on PIPE_MTE2, BOTH flags
                wait_intra_block(PIPE_MTE2, FlagID);
                wait_intra_block(PIPE_MTE2, FlagID + VEC_CORE_ID_OFFSET);
            } else if constexpr (is_v2c_mat) { // is_v2c_mat
                                               // Vec -> Cube (UB path - TINSERT): Cube waits on PIPE_MTE1, BOTH flags
#ifdef __DAV_CUBE__
                wait_intra_block(PIPE_MTE1, FlagID);
                wait_intra_block(PIPE_MTE1, FlagID + VEC_CORE_ID_OFFSET);
#endif
            } else { // is_v2c_ctrl
                wait_intra_block(PIPE_S, FlagID);
                wait_intra_block(PIPE_S, FlagID + VEC_CORE_ID_OFFSET);
            }
        }

        /**
         * free: Release space in FIFO
         * 1. (iter >= Depth - Period): Silence at start. Don't signal if Producer
         * is still enjoying the initial free buffer space.
         * 2. (is_sync_step): Accumulate free slots and signal in batches.
         */
        PTO_INTERNAL void free() const
        {
            if constexpr (is_c2v_gm) {
                // Vec consumer frees buffer for Cube - signals on PIPE_MTE2, flag_id+1 only
#ifdef __DAV_VEC__
                uint8_t freeCubeID = FlagIDPlusOne;
                set_intra_block(PIPE_MTE2, freeCubeID);
#endif
            } else if constexpr (is_c2v_ub) {
                // Vec consumer frees buffer for Cube - signals on PIPE_V, flag_id+1 only
                // Vec signals after vector ops complete (PIPE_V)
#ifdef __DAV_VEC__
                uint8_t freeCubeID = FlagIDPlusOne;
                set_intra_block(PIPE_V, freeCubeID);
#endif
            } else if constexpr (is_v2c_gm || is_v2c_mat) { // is_v2c (both gm and mat)
                // Cube consumer frees buffer for Vec - signals BOTH flags on PIPE_MTE1
#ifdef __DAV_CUBE__
                uint8_t freeVec0ID = FlagIDPlusOne;
                uint8_t freeVec1ID = FlagIDPlusOne + VEC_CORE_ID_OFFSET;
                set_intra_block(PIPE_MTE1, freeVec0ID);
                set_intra_block(PIPE_MTE1, freeVec1ID);
#endif
            } else { // is_v2c_ctrl
                     // Control signals from Vec to Cube: Vec signals on flag_id, Cube waits on flag_id only
#ifdef __DAV_CUBE__
                uint8_t freeVec0ID = FlagIDPlusOne;
                uint8_t freeVec1ID = FlagIDPlusOne + VEC_CORE_ID_OFFSET;
                set_intra_block(PIPE_S, freeVec0ID);
                set_intra_block(PIPE_S, freeVec1ID);
#endif
            }
        }

        template <typename T, int ProdM, int ProdN, int ConsM, int ConsN>
        PTO_INTERNAL void popVecTileFromGMFiFo(DataFiFo &fifo, TileDataCons &tile)
        {
            size_t bufIndex = static_cast<size_t>(tile_id) % fifo.fifoDepth;
            constexpr int kTileFactor = ConsN / ProdN;
            size_t entryBase = static_cast<size_t>(bufIndex) * kTileFactor * ProdM * ProdN * sizeof(T);
            __gm__ T *addr = (__gm__ T *)((uint64_t)fifo.fifoBase + entryBase + entryOffset);

            if constexpr (DataFiFo::useLocalFiFo) {
                uint64_t localTileBase =
                    (uint64_t)fifo.localFiFoBase +
                    (static_cast<size_t>(tile_id) % fifo.localFiFoDepth) * ConsM * ConsN * sizeof(T);
                TASSIGN_IMPL(tile, localTileBase);
            }

            using GlobalDataSub = GlobalTensor<T, pto::Shape<1, 1, 1, ConsM, ProdN>, pto::Stride<1, 1, 1, ProdN, 1>>;
            using TileDataSub = Tile<TileType::Vec, T, ConsM, ConsN, BLayout::RowMajor, ConsM, ProdN>;
            TileDataSub tileSub;
            for (int sub_col = 0; sub_col < kTileFactor; ++sub_col) {
                __gm__ T *addrSub = addr + sub_col * ProdM * ProdN;
                uint64_t col_byte_offset = sub_col * ProdN * sizeof(T);
                GlobalDataSub globalTensorSub(addrSub);
                TASSIGN_IMPL(tileSub, (uint64_t)tile.data() + col_byte_offset);
                TLOAD_IMPL(tileSub, globalTensorSub);
            }
        }

        template <typename T, int ProdM, int ProdN, int ConsM, int ConsN>
        PTO_INTERNAL void popTileFromLocalFiFo(DataFiFo &fifo, TileDataCons &tile)
        {
            uint32_t bufIndex = static_cast<uint32_t>(tile_id % DataFiFo::fifoDepth);
            size_t entryBase = bufIndex * ConsM * ConsN * sizeof(T);
            uint64_t localTileBase = fifo.fifoBase + entryBase + entryOffset;
            TASSIGN_IMPL(tile, localTileBase);
        }

        template <typename T, int ConsM, int ConsN, int ProdN>
        PTO_INTERNAL void popMatTileFromGMFiFo(DataFiFo &fifo, TileDataCons &tile)
        {
            uint32_t bufIndex = static_cast<uint32_t>(tile_id % fifo.fifoDepth);
            size_t entryBase = bufIndex * ConsM * ProdN * sizeof(T);
            using GlobalData = GlobalTensor<T, pto::Shape<1, 1, 1, ConsM, ConsN>, pto::Stride<1, 1, 1, ConsN, 1>>;
            GlobalData globalTensor((__gm__ T *)((uint64_t)fifo.fifoBase + entryBase + entryOffset));

            if constexpr (DataFiFo::useLocalFiFo) {
                uint64_t localTileBase =
                    (uint64_t)fifo.localFiFoBase +
                    (static_cast<size_t>(tile_id) % fifo.localFiFoDepth) * ConsM * ConsN * sizeof(T);
                TASSIGN_IMPL(tile, localTileBase);
            }
            TLOAD_IMPL(tile, globalTensor);
        }

        PTO_INTERNAL void popCtrlFromCtrlFiFo(DataFiFo &fifo)
        {
            uint32_t bufIndex = static_cast<uint32_t>(tile_id % fifo.fifoDepth);
            size_t entryBase = bufIndex * sizeof(uint32_t);
            uint64_t ctrlTileBase = fifo.fifoBase + entryBase + entryOffset;
            fifo.ctrlSignal = (*(__ssbuf__ uint32_t *)(ctrlTileBase) == 1) ? true : false;
        }

        PTO_INTERNAL bool pop(DataFiFo &fifo, TileDataCons &tile)
        {
            using T = typename TileDataCons::DType;
            constexpr int ProdM = TileDataProd::Rows;
            constexpr int ProdN = TileDataProd::Cols;
            constexpr int ConsM = TileDataCons::Rows;
            constexpr int ConsN = TileDataCons::Cols;
            constexpr int VEC_CORES = (VCRatio == VecCubeRatio::V2C1_VECS) ? 2 : 1;

            static_assert(TileDataCons::Loc == TileType::Vec || TileDataCons::Loc == TileType::Mat,
                          "Fix: TPOP has unsupported tile type!");
            if constexpr (TileDataCons::Loc == TileType::Vec) {
                static_assert((DataFiFo::fifoType == FIFOType::GM_FIFO) || (DataFiFo::fifoType == FIFOType::VEC_FIFO),
                              "Fix: TPOP has unsupported fifo type!");
                if constexpr (DataFiFo::fifoType == FIFOType::GM_FIFO) {
                    popVecTileFromGMFiFo<T, ProdM, ProdN, ConsM, ConsN>(fifo, tile);
                    return true;
                } else if constexpr (DataFiFo::fifoType == FIFOType::VEC_FIFO) {
                    return false;
                } else if constexpr (DataFiFo::fifoType == FIFOType::CTRL_FIFO) {
                    popCtrlFromCtrlFiFo(fifo);
                    return false;
                }
            } else if constexpr (TileDataCons::Loc == TileType::Mat) {
                static_assert((DataFiFo::fifoType == FIFOType::GM_FIFO) || (DataFiFo::fifoType == FIFOType::MAT_FIFO),
                              "Fix: TPOP has unsupported fifo type!");
                if constexpr (DataFiFo::fifoType == FIFOType::GM_FIFO) {
                    popMatTileFromGMFiFo<T, ConsM, ConsN, ProdN>(fifo, tile);
                    return true;
                } else if constexpr (DataFiFo::fifoType == FIFOType::MAT_FIFO) {
                    return false;
                }
            }
        }
    };

    DataFiFo fifo;
    Producer prod;
    Consumer cons;

    // Constructors for GM_FIFO base address initialization
    template <FIFOType T = FiFoType, typename std::enable_if_t<T == FIFOType::GM_FIFO, int> = 0>
    PTO_INTERNAL explicit TMPipe(__gm__ typename TileDataCons::DType *gmFiFoBase, uint32_t localFiFoBase)
        : fifo(gmFiFoBase, localFiFoBase), prod(), cons()
    {
        cons.free();
    }

    template <FIFOType T = FiFoType, typename std::enable_if_t<T != FIFOType::GM_FIFO, int> = 0>
    PTO_INTERNAL explicit TMPipe(uint32_t fifoBase) : fifo(fifoBase), prod(), cons()
    {
        cons.free();
    }

    // Destructor for TPipe
    PTO_INTERNAL ~TMPipe()
    {
        prod.allocate();
    }
};

template <typename TileData, typename Pipe>
PTO_INTERNAL void TPUSH_IMPL(TileData &tile, Pipe &pipe)
{
    // 1. Cross-Core: Wait for space
    bool isAllocate = pipe.prod.getAllocateStatus();
    if (isAllocate) {
        pipe.prod.allocate();
    }

    // 2. Address Calculation
    pipe.prod.push(pipe.fifo, tile);

    // 3； Cross-Core: Commit & Signal
    bool isRecord = pipe.prod.getRecordStatus();
    if (isRecord) {
        pipe.prod.record();
    }
}

} // namespace pto

#endif
