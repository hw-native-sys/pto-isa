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

#include <pto/common/fifo.hpp>
#include <pto/npu/a5/TStore.hpp>
#include <pto/npu/a5/TLoad.hpp>
#include <pto/npu/a5/custom/TInsertCustom.hpp>

namespace pto {

// Operation types for TSync - identifies the producer/consumer operation
enum class TSyncOpType : uint8_t
{
    TSTORE_C2GM,  // Store (Cube core operation via PIPE_FIX) - GM path
    TSTORE_V2GM,  // Store (Vector core operation via PIPE_MTE3) - GM path
    TMOV_C2UB,    // TMOV from L0C to UB (Cube core operation via PIPE_FIX) - UB path
    TINSERT_V2L1, // TINSERT from UB to L1 (Vector core operation via PIPE_MTE3) - UB path
                  // TINSERT uses copy_ubuf_to_cbuf which goes through MTE3 pipe
                  // Cube consumer waits on PIPE_MTE1 (L1 side receives via MTE1)
    TLOAD,        // Load operation (consumer operation)
    NONE
};

// -----------------------------------------------------------------------------
// Compile-time direction inference based on producer/consumer ops
// GM path:
//   TSTORE_C2GM (producer) + TLOAD (consumer) = Cube to Vector via PIPE_FIX
//   TSTORE_V2GM (producer) + TLOAD (consumer) = Vector to Cube via PIPE_MTE3
// UB path:
//   TMOV_C2UB (producer) + NONE(consumer) = Cube to Vector via PIPE_FIX
//   TINSERT_V2L1 (producer) + NONE(consumer) = Vector to Cube via PIPE_MTE3
//   TINSERT (UB->L1) uses MTE3 on Vec side, Cube waits on MTE1
// -----------------------------------------------------------------------------
template <TSyncOpType ProducerOp, TSyncOpType ConsumerOp>
struct TSyncTraits {
    // GM path: Cube produces via TSTORE_C2GM (PIPE_FIX) - consumer waits on PIPE_MTE2
    static constexpr bool is_cube_to_vec_gm = (ProducerOp == TSyncOpType::TSTORE_C2GM);
    // UB path: Cube produces via TMOV_C2UB (PIPE_FIX) - consumer waits on PIPE_V
    static constexpr bool is_cube_to_vec_ub = (ProducerOp == TSyncOpType::TMOV_C2UB);
    // Unified Cube-to-Vec detection
    static constexpr bool is_cube_to_vec = is_cube_to_vec_gm || is_cube_to_vec_ub;

    // GM path: Vector produces via TSTORE_V2GM (PIPE_MTE3)
    static constexpr bool is_vec_to_cube_gm = (ProducerOp == TSyncOpType::TSTORE_V2GM);
    // UB path: Vector produces via TINSERT_V2L1 (PIPE_MTE3) - Cube waits on PIPE_MTE1
    static constexpr bool is_vec_to_cube_ub = (ProducerOp == TSyncOpType::TINSERT_V2L1);
    // Unified Vec-to-Cube detection
    static constexpr bool is_vec_to_cube = is_vec_to_cube_gm || is_vec_to_cube_ub;

    static_assert(is_cube_to_vec || is_vec_to_cube,
                  "Producer must be TSTORE_C2GM, TMOV_C2UB (Cube) or TSTORE_V2GM, TINSERT_V2L1 (Vector)");
};

/**
 * Pipe: Manages Cross-Core Pipe Synchronization
 * @tparam ReadyFlag    Signal from Producer to Consumer (Data Ready)
 * @tparam ConsumedFlag Signal from Consumer to Producer (Space Released)
 * @tparam Depth        FIFO Depth (e.g., 2 for Double Buffering)
 * @tparam Period       Sync Period (Sync once every N tiles)
 * @tparam ProdRole     Logic role of Producer (CUBE/VECTOR) -> Deduce signal pipe
 * @tparam ConsRole     Logic role of Consumer (CUBE/VECTOR) -> Deduce signal pipe
 */
template <uint16_t FlagID, FIFOType fifoType, uint8_t FiFoDepth, uint8_t FiFoSyncT, typename TileDataProd,
          typename TileDataCons, TSyncOpType ProducerOp, TSyncOpType ConsumerOp,
          VecCubeRatio VCRatio = VecCubeRatio::V2C1_VECS>
struct TPipe {
    using Traits = TSyncTraits<ProducerOp, ConsumerOp>;
    static constexpr bool is_c2v = Traits::is_cube_to_vec;
    static constexpr bool is_c2v_gm = Traits::is_cube_to_vec_gm;
    static constexpr bool is_c2v_ub = Traits::is_cube_to_vec_ub;
    static constexpr bool is_v2c = Traits::is_vec_to_cube;
    static constexpr bool is_v2c_gm = Traits::is_vec_to_cube_gm;
    static constexpr bool is_v2c_ub = Traits::is_vec_to_cube_ub;
    static constexpr int VEC_CORE_ID_OFFSET = 16;

    using DataFiFo = std::conditional_t<(fifoType == FIFOType::GM_FIFO),
                                        DataFIFO<typename TileDataCons::DType, fifoType, FiFoDepth, FiFoSyncT>,
                                        DataFIFO<TileDataCons, fifoType, FiFoDepth, FiFoSyncT>>;

    // -------------------------------------------------------------------------
    // Producer Interface
    // -------------------------------------------------------------------------
    struct Producer {
        int tile_id;
        int sub_tile_id;
        bool isAllocate;
        bool isRecord;
        int entryOffset;

        PTO_INTERNAL Producer()
        {
            tile_id = -1;
            sub_tile_id = -1;
        }

        PTO_INTERNAL void setTileId(int t_id, int sub_t_id)
        {
            tile_id = t_id;
            sub_tile_id = sub_t_id;
        }

        PTO_INTERNAL int getTileId()
        {
            return tile_id;
        }

        PTO_INTERNAL int getSubTileId()
        {
            return sub_tile_id;
        }

        PTO_INTERNAL void setAllocateStatus(bool allocate)
        {
            isAllocate = allocate;
        }

        PTO_INTERNAL bool getAllocateStatus()
        {
            return isAllocate;
        }

        PTO_INTERNAL void setRecordStatus(bool record)
        {
            isRecord = record;
        }

        PTO_INTERNAL bool getRecordStatus()
        {
            return isRecord;
        }

        PTO_INTERNAL void setEntryOffset(int offset)
        {
            entryOffset = offset;
        }

        /**
         * alloc: Request space in FIFO
         * Logic:
         * 1. (iter >= Depth): Startup protection. Don't check flags when buffer is empty.
         * 2. (iter % Period == 0): Sparse sync. Only check flag periodically.
         */
        PTO_INTERNAL void allocate()
        {
            if constexpr (is_c2v) {
                // Cube producer waits for Vec consumer to free buffer
                // Vec signals on flag_id+1 only, but Cube must wait on BOTH
                // (because Vec0 signals flag_id+1, Vec1 signals flag_id+1+16 from Cube's view)
                wait_intra_block(PIPE_FIX, FlagID + 1);
                wait_intra_block(PIPE_FIX, FlagID + 1 + VEC_CORE_ID_OFFSET);
            } else { // is_v2c (both gm and ub)
                // Vec producer waits for Cube consumer to free buffer
                // Cube signals on BOTH, Vec waits on flag_id+1 only
                wait_intra_block(PIPE_MTE3, FlagID + 1);
            }
        }

        // Forward dependency: record (producer) and wait (consumer)
        /**
         * record - Producer signals that data is ready
         * Called by the producer after completing the operation (TSTORE_C2GM or TSTORE_V2GM)
         */
        PTO_INTERNAL void record()
        {
            if constexpr (is_c2v) {
                // Cube -> Vec: Cube sets BOTH flags on PIPE_FIX
                set_intra_block(PIPE_FIX, FlagID);
                set_intra_block(PIPE_FIX, FlagID + VEC_CORE_ID_OFFSET);
            } else { // is_v2c (both gm and ub)
                // Vec -> Cube: Vec sets flag_id only on PIPE_MTE3
                // Each Vec subblock executes this; hardware maps subblock 1's flag to flag_id+16
                set_intra_block(PIPE_MTE3, FlagID);
            }
        }

        template <typename T, int ProdM, int ProdN, int ConsM, int ConsN>
        PTO_INTERNAL void pushAcc2GMFiFo(DataFiFo &fifo, TileDataProd &tile)
        {
            // calculate base address in GM FIFO for this tile
            constexpr int kTileFactor = ConsN / ProdN;
            uint32_t buf_idx = static_cast<uint32_t>(tile_id % DataFiFo::fifoDepth);
            // entryBase is the base address for the entire tile in the FIFO;
            // entryOffset is the offset for this sub-tile, which is specified by the caller based on the sub-tile's
            // position within the full tile
            size_t entryBase = buf_idx * kTileFactor * ProdM * ProdN * sizeof(T);
            using GlobalData = GlobalTensor<T, pto::Shape<1, 1, 1, ProdM, ProdN>, pto::Stride<1, 1, 1, ProdN, 1>>;
            GlobalData globalTensor(fifo.fifoBase + entryBase + entryOffset);
            // TODO: store tile to GM FIFO, enable unit-flag one
            // TSTORE_IMPL<TileDataProd, GlobalData, AtomicType::AtomicNone, STPhase::Final>(globalTensor, tile);
            TSTORE_IMPL(globalTensor, tile);
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
                TASSIGN(vecTile, (uint64_t)fifo.tilePtr->data() + entryBase + entryOffset);
                TMOV_IMPL<TileDataVec, TileDataProd, AccToVecMode::DualModeSplitM>(vecTile, tile);
            } else if constexpr (isSplitN) {
                // split N between two vectors
                constexpr int kTileFactor = ConsN / ProdN;
                constexpr uint32_t VecN = ProdN / VEC_CORES / kTileFactor;
                using TileDataVec = Tile<TileType::Vec, T, ProdM, VecN, BLayout::RowMajor, ProdM, VecN>;
                TileDataVec vecTile;
                uint64_t entryBase = (tile_id % DataFiFo::fifoDepth) * ProdM * VecN * sizeof(T);
                TASSIGN(vecTile, (uint64_t)fifo.tilePtr->data() + entryBase + entryOffset);
                TMOV_IMPL<TileDataVec, TileDataProd, AccToVecMode::DualModeSplitN>(vecTile, tile);
            } else if constexpr (nonSplit) {
                // single vector core (1v:1v)
                TileDataCons vecTile;
                uint64_t entryBase = (tile_id % DataFiFo::fifoDepth) * ProdM * ProdN * sizeof(T);
                TASSIGN(vecTile, (uint64_t)fifo.tilePtr->data() + entryBase + entryOffset);
                TMOV_IMPL<TileDataCons, TileDataProd, AccToVecMode::SingleModeVec0>(vecTile, tile);
            } else {
                static_assert(isSplitM || isSplitN || nonSplit,
                              "Fix: TPUSH(pushAcc2VecFiFo) has unsupported split mode!");
            }
        }

        template <typename T, int ProdM, int ProdN, int ConsM, int ConsN>
        PTO_INTERNAL void pushVec2GMFiFo(DataFiFo &fifo, TileDataProd &tile)
        {
            static_assert((TileDataProd::Loc == TileType::Vec) && (DataFiFo::fifoType == FIFOType::GM_FIFO),
                          "Fix: TPUSH has unsupported fifo type!");
            // calculate base address in GM FIFO for this tile
            constexpr int kTileFactor = ProdN / ConsN;
            uint32_t buf_idx = static_cast<uint32_t>(tile_id % DataFiFo::fifoDepth);
            size_t entryBase = buf_idx * kTileFactor * ConsM * ConsN * sizeof(T);
            using GlobalDataSub = GlobalTensor<T, pto::Shape<1, 1, 1, ProdM, ConsN>, pto::Stride<1, 1, 1, ConsN, 1>>;
            using TileDataSub = Tile<TileType::Vec, T, ProdM, ProdN, BLayout::RowMajor, ProdM, ConsN>;
            TileDataSub subTile;
            __gm__ T *addr = fifo.fifoBase + entryBase + entryOffset;
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
                int subblock_base_rows = VecM * static_cast<size_t>(get_subblockid());
                int row_offset = subblock_base_rows + entryOffset;
                uint64_t entryBase = (tile_id % DataFiFo::fifoDepth) * ConsM * ConsN * sizeof(T);
                TileDataCons matTile;
                TASSIGN_IMPL(matTile, (uint64_t)fifo.tilePtr->data() + entryBase);
                constexpr bool isNZPlus1 = (ConsM / ProdM) != 2;
                if constexpr (isNZPlus1) { // NZ + 1 mode
                    TINSERT_CUSTOM<TInsertMode::NZ_PLUS_1>(matTile, tile, row_offset, 0);
                } else { // NZ mode
                    TINSERT_CUSTOM<TInsertMode::NZ>(matTile, tile, row_offset, 0);
                }
            } else if constexpr (isSplitN) {
                // split N between vectors
                int col_index = ProdN;
                uint64_t entryBase = (tile_id % DataFiFo::fifoDepth) * ConsM * ConsN * sizeof(T);
                TileDataCons matTile;
                TASSIGN_IMPL(matTile, (uint64_t)fifo.tilePtr->data() + entryBase);
                constexpr bool isNZPlus1 = (ConsM / ProdM) != 2;
                if constexpr (isNZPlus1) { // NZ+1 mode
                    TINSERT_CUSTOM<TInsertMode::NZ_PLUS_1>(matTile, tile, 0, col_index);
                } else {
                    TINSERT_CUSTOM<TInsertMode::NZ>(matTile, tile, 0, col_index);
                }
            } else if constexpr (nonSplit) {
                // single vector core
                TileDataCons matTile;
                uint64_t entryBase = (tile_id % DataFiFo::fifoDepth) * ConsM * ConsN * sizeof(T);
                TASSIGN_IMPL(matTile, (uint64_t)fifo.tilePtr->data() + entryBase);
                constexpr bool isNZPlus1 = (ProdM > ConsM);
                if constexpr (isNZPlus1) { // NZ+1 mode
                    TINSERT_CUSTOM<TInsertMode::NZ_PLUS_1>(matTile, tile, 0, 0);
                } else {
                    TINSERT_CUSTOM<TInsertMode::NZ>(matTile, tile, 0, 0);
                }
            } else {
                static_assert(isSplitM || isSplitN || nonSplit,
                              "Fix: TPUSH(pushVec2MatFiFo) has unsupported split mode!");
            }
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
                              "Fix: TPUSH has unsuported fifo type!");
                if constexpr (DataFiFo::fifoType == FIFOType::GM_FIFO) {
                    pushAcc2GMFiFo<T, ProdM, ProdN, ConsM, ConsN>(fifo, tile);
                } else if constexpr (DataFiFo::fifoType == FIFOType::VEC_FIFO) {
                    pushAcc2VecFiFo<T, ProdM, ProdN, ConsM, ConsN, VEC_CORES>(fifo, tile);
                }
            } else if constexpr (TileDataProd::Loc == TileType::Vec) {
                static_assert(DataFiFo::fifoType == FIFOType::GM_FIFO || DataFiFo::fifoType == FIFOType::MAT_FIFO,
                              "Fix: TPUSH has unsupported fifoType!");
                if constexpr (DataFiFo::fifoType == FIFOType::GM_FIFO) {
                    pushVec2GMFiFo<T, ProdM, ProdN, ConsM, ConsN>(fifo, tile);
                } else if constexpr (DataFiFo::fifoType == FIFOType::MAT_FIFO) {
                    pushVec2MatFiFo<T, ProdM, ProdN, ConsM, ConsN, VEC_CORES>(fifo, tile);
                }
            }
        } // end of store
    };

    // -------------------------------------------------------------------------
    // Consumer Interface
    // -------------------------------------------------------------------------
    struct Consumer {
        int tile_id;
        int sub_tile_id;
        bool isWait;
        bool isFree;
        int entryOffset;

        PTO_INTERNAL Consumer()
        {
            tile_id = -1;
            sub_tile_id = -1;
            entryOffset = 0;
        }

        PTO_INTERNAL void setTileId(int tid, int sub_tid)
        {
            tile_id = tid;
            sub_tile_id = sub_tid;
        }

        PTO_INTERNAL int getTileId()
        {
            return tile_id;
        }

        PTO_INTERNAL int getSubTileId()
        {
            return sub_tile_id;
        }

        PTO_INTERNAL void setWaitStatus(bool wait)
        {
            isWait = wait;
        }

        PTO_INTERNAL bool getWaitStatus()
        {
            return isWait;
        }

        PTO_INTERNAL void setFreeStatus(bool free)
        {
            isFree = free;
        }

        PTO_INTERNAL bool getFreeStatus()
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
        PTO_INTERNAL void wait()
        {
            if constexpr (is_c2v_gm) {
                // Cube -> Vec (GM path): Vec waits on PIPE_MTE2 (data loaded from GM)
                wait_intra_block(PIPE_MTE2, FlagID);
            } else if constexpr (is_c2v_ub) {
                // Cube -> Vec (UB path): Vec waits on PIPE_V before vector ops on UB data
                // Cube sets PIPE_FIX, Vec waits PIPE_V (Vec does vector ops, not TLOAD)
                wait_intra_block(PIPE_V, FlagID);
            } else if constexpr (is_v2c_gm) {
                // Vec -> Cube (GM path): Cube waits on PIPE_MTE2, BOTH flags
                wait_intra_block(PIPE_MTE2, FlagID);
                wait_intra_block(PIPE_MTE2, FlagID + VEC_CORE_ID_OFFSET);
            } else { // is_v2c_ub
                // Vec -> Cube (UB path - TINSERT): Cube waits on PIPE_MTE1, BOTH flags
                wait_intra_block(PIPE_MTE1, FlagID);
                wait_intra_block(PIPE_MTE1, FlagID + VEC_CORE_ID_OFFSET);
            }
        }

        /**
         * free: Release space in FIFO
         * Logic:
         * 1. (iter >= Depth - Period): Silence at start. Don't signal if Producer
         * is still enjoying the initial free buffer space.
         * 2. (is_sync_step): Accumulate free slots and signal in batches.
         */
        PTO_INTERNAL void free()
        {
            if constexpr (is_c2v_gm) {
                // Vec consumer frees buffer for Cube - signals on PIPE_MTE2, flag_id+1 only
                set_intra_block(PIPE_MTE2, FlagID + 1);
            } else if constexpr (is_c2v_ub) {
                // Vec consumer frees buffer for Cube - signals on PIPE_V, flag_id+1 only
                // Vec signals after vector ops complete (PIPE_V)
                set_intra_block(PIPE_V, FlagID + 1);
            } else { // is_v2c (both gm and ub)
                // Cube consumer frees buffer for Vec - signals BOTH flags on PIPE_MTE1
                set_intra_block(PIPE_MTE1, FlagID + 1);
                set_intra_block(PIPE_MTE1, FlagID + 1 + VEC_CORE_ID_OFFSET);
            }
        }

        template <typename T, int ProdM, int ProdN, int ConsM, int ConsN>
        PTO_INTERNAL void popVecTileFromGMFiFo(DataFiFo &fifo, TileDataCons &tile)
        {
            size_t buf_idx = static_cast<size_t>(tile_id) % fifo.fifoDepth;
            constexpr int kTileFactor = ConsN / ProdN;
            size_t entryBase = static_cast<size_t>(buf_idx) * kTileFactor * ProdM * ProdN * sizeof(T);

            __gm__ T *addr = fifo.fifoBase + entryBase + entryOffset;
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

        template <typename T, int ConsM, int ConsN, int ProdN>
        PTO_INTERNAL void popMatTileFromGMFiFo(DataFiFo &fifo, TileDataCons &tile)
        {
            uint32_t buf_idx = static_cast<uint32_t>(tile_id % fifo.fifoDepth);
            size_t entryBase = buf_idx * ConsM * ProdN * sizeof(T);
            using GlobaData = GlobalTensor<T, pto::Shape<1, 1, 1, ConsM, ConsN>, pto::Stride<1, 1, 1, ConsN, 1>>;
            GlobaData globalTensor(fifo.fifoBase + entryBase + entryOffset);
            TLOAD_IMPL(tile, globalTensor);
        }

        PTO_INTERNAL void pop(DataFiFo &fifo, TileDataCons &tile)
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
                } else if constexpr (DataFiFo::fifoType == FIFOType::VEC_FIFO) {
                    return;
                }
            } else if constexpr (TileDataCons::Loc == TileType::Mat) {
                static_assert((DataFiFo::fifoType == FIFOType::GM_FIFO) || (DataFiFo::fifoType == FIFOType::MAT_FIFO),
                              "Fix: TPOP has unsupported fifo type!");
                if constexpr (DataFiFo::fifoType == FIFOType::GM_FIFO) {
                    popMatTileFromGMFiFo<T, ConsM, ConsN, ProdN>(fifo, tile);
                } else if constexpr (DataFiFo::fifoType == FIFOType::MAT_FIFO) {
                    return;
                }
            }
        }
    };
};

/**
 * TPUSH: Push Tile to FIFO
 * * Flow:
 * 1. [Alloc]   Check GM space (Cross-Core)
 * 2. [Store]   Write data to GM
 * 3. [Commit]  Signal Consumer (Cross-Core)
 */
template <typename PipeProd, typename TileData, typename DataFiFo>
PTO_INTERNAL void TPUSH_IMPL(PipeProd &prod, TileData &tile, DataFiFo &fifo)
{
    // 1. Cross-Core: Wait for space
    bool isAllocate = prod.getAllocateStatus();
    if (isAllocate) {
        prod.allocate();
    }

    // 2. Address Calculation
    prod.push(fifo, tile);

    // 3； Cross-Core: Commit & Signal
    bool isRecord = prod.getRecordStatus();
    if (isRecord) {
        prod.record();
    }
}

template <typename PipeProd>
PTO_INTERNAL void TPUSHSTART_IMPL(PipeProd &prod)
{
    bool isAllocate = prod.getAllocateStatus();
    if (isAllocate) {
        prod.allocate();
    }
}

} // namespace pto

#endif