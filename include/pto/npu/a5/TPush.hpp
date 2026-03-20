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
#include <pto/npu/a5/TInsert.hpp>

namespace pto {

/**
 * Pipe: Manages Cross-Core Pipe Synchronization
 * @tparam FlagID      Signal from Producer to Consumer (Data Ready)
 * @tparam FiFoType     FIFO Type (e.g., GM_FIFO, VEC_FIFO, MAT_FIFO)
 * @tparam FiFoDepth    FIFO Depth (Number of entries in the FIFO)
 * @tparam FiFoSyncT    FIFO Sync Period (Sync once every N tiles)
 * @tparam TileDataProd Data type for the producer tile
 * @tparam TileDataCons Data type for the consumer tile
 * @tparam LocalFiFoDepth  Local FIFO Depth for GM FIFOs (ignored for non-GM FIFOs)
 * @tparam EN_UNIT_FLAG    Whether to enable unit flags (only for GM FIFOs)
 * @tparam VCRatio         Vector-to-Cube core ratio
 */
template <uint8_t FlagID, FIFOType FiFoType, uint8_t FiFoDepth, uint8_t FiFoSyncT, typename TileDataProd,
          typename TileDataCons, bool EN_UNIT_FLAG = false, uint8_t LocalFiFoDepth = 2,
          VecCubeRatio VCRatio = VecCubeRatio::V2C1_VECS>
struct TPipe {
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
                uint8_t waitVec0ID = FlagID + 1;
                uint8_t waitVec1ID = FlagID + 1 + VEC_CORE_ID_OFFSET;
                wait_intra_block(PIPE_FIX, waitVec0ID);
                wait_intra_block(PIPE_FIX, waitVec1ID);
#endif
            } else if constexpr (is_v2c_gm || is_v2c_mat) {
                // is_v2c (both gm and mat)
                // Vec producer waits for Cube consumer to free buffer
                // Cube signals on BOTH, Vec waits on flag_id+1 only
#ifdef __DAV_VEC__
                uint8_t waitCubeID = FlagID + 1;
                wait_intra_block(PIPE_MTE3, waitCubeID);
#endif
            } else {
                // is_v2c_ctrl
                // Control signals from Vec to Cube: Vec signals on flag_id, Cube waits on flag_id only
#ifdef __DAV_VEC__
                uint8_t waitCubeID = FlagID + 1;
                wait_intra_block(PIPE_S, waitCubeID);
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
            uint32_t buf_idx = static_cast<uint32_t>(tile_id % DataFiFo::fifoDepth);
            size_t entryBase = buf_idx * kTileFactor * ProdM * ProdN * sizeof(T);
            using GlobalData = GlobalTensor<T, pto::Shape<1, 1, 1, ProdM, ProdN>, pto::Stride<1, 1, 1, ProdN, 1>>;
            GlobalData globalTensor((__gm__ T *)((uint64_t)fifo.fifoBase + entryBase + entryOffset));
            // store tile to GM FIFO, enable unit-flag or diable unit-flag
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
                uint64_t fifoBase = (fifo.tilePtr != nullptr) ? (uint64_t)fifo.tilePtr->data() : fifo.fifoBase;
                uint64_t entryBase = (tile_id % DataFiFo::fifoDepth) * VecM * ProdN * sizeof(T);
                TASSIGN(vecTile, fifoBase + entryBase + entryOffset);
                TMOV_IMPL<TileDataVec, TileDataProd, AccToVecMode::DualModeSplitM>(vecTile, tile);
            } else if constexpr (isSplitN) {
                // split N between two vectors
                constexpr int kTileFactor = ConsN / ProdN;
                constexpr uint32_t VecN = ProdN / VEC_CORES / kTileFactor;
                using TileDataVec = Tile<TileType::Vec, T, ProdM, VecN, BLayout::RowMajor, ProdM, VecN>;
                TileDataVec vecTile;
                uint64_t fifoBase = (fifo.tilePtr != nullptr) ? (uint64_t)fifo.tilePtr->data() : fifo.fifoBase;
                uint64_t entryBase = (tile_id % DataFiFo::fifoDepth) * ProdM * VecN * sizeof(T);
                TASSIGN(vecTile, fifoBase + entryBase + entryOffset);
                TMOV_IMPL<TileDataVec, TileDataProd, AccToVecMode::DualModeSplitN>(vecTile, tile);
            } else if constexpr (nonSplit) {
                // single vector core (1v:1v)
                TileDataCons vecTile;
                uint64_t fifoBase = (fifo.tilePtr != nullptr) ? (uint64_t)fifo.tilePtr->data() : fifo.fifoBase;
                uint64_t entryBase = (tile_id % DataFiFo::fifoDepth) * ProdM * ProdN * sizeof(T);
                TASSIGN(vecTile, fifoBase + entryBase + entryOffset);
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
            uint32_t buf_idx = static_cast<uint32_t>(tile_id % DataFiFo::fifoDepth);
            size_t entryBase = buf_idx * kTileFactor * ConsM * ConsN * sizeof(T);
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
                int subblock_base_rows = VecM * static_cast<size_t>(get_subblockid());
                int row_offset = subblock_base_rows + entryOffset;
                uint64_t entryBase = (tile_id % DataFiFo::fifoDepth) * ConsM * ConsN * sizeof(T);
                TileDataCons matTile;
                uint64_t fifoBase = (fifo.tilePtr != nullptr) ? (uint64_t)fifo.tilePtr->data() : fifo.fifoBase;
                TASSIGN_IMPL(matTile, fifoBase + entryBase);
                constexpr bool isNZPlus1 = (ConsM / ProdM) != 2;
                if constexpr (isNZPlus1) { // NZ + 1 mode
                    TINSERT_IMPL<TInsertMode::NZ_PLUS_1>(matTile, tile, row_offset, 0);
                } else { // NZ mode
                    TINSERT_IMPL<TInsertMode::NZ>(matTile, tile, row_offset, 0);
                }
            } else if constexpr (isSplitN) {
                // split N between vectors
                int col_index = ProdN;
                uint64_t entryBase = (tile_id % DataFiFo::fifoDepth) * ConsM * ConsN * sizeof(T);
                TileDataCons matTile;
                uint64_t fifoBase = (fifo.tilePtr != nullptr) ? (uint64_t)fifo.tilePtr->data() : fifo.fifoBase;
                TASSIGN_IMPL(matTile, fifoBase + entryBase);
                constexpr bool isNZPlus1 = (ConsM / ProdM) != 2;
                if constexpr (isNZPlus1) { // NZ+1 mode for bank conflict optimization
                    TINSERT_IMPL<TInsertMode::NZ_PLUS_1>(matTile, tile, 0, col_index);
                } else {
                    TINSERT_IMPL<TInsertMode::NZ>(matTile, tile, 0, col_index);
                }
            } else if constexpr (nonSplit) {
                // single vector core
                TileDataCons matTile;
                uint64_t fifoBase = (fifo.tilePtr != nullptr) ? (uint64_t)fifo.tilePtr->data() : fifo.fifoBase;
                uint64_t entryBase = (tile_id % DataFiFo::fifoDepth) * ConsM * ConsN * sizeof(T);
                TASSIGN_IMPL(matTile, fifoBase + entryBase);
                constexpr bool isNZPlus1 = (ProdM > ConsM);
                if constexpr (isNZPlus1) { // NZ+1 mode
                    TINSERT_IMPL<TInsertMode::NZ_PLUS_1>(matTile, tile, 0, 0);
                } else {
                    TINSERT_IMPL<TInsertMode::NZ>(matTile, tile, 0, 0);
                }
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
            uint64_t fifoBase = (fifo.tilePtr != nullptr) ? (uint64_t)fifo.tilePtr->data() : fifo.fifoBase;
            uint32_t buf_idx = static_cast<uint32_t>(tile_id % DataFiFo::fifoDepth);
            uint64_t entryBase = buf_idx * sizeof(uint32_t);
            __ssbuf__ uint32_t *ctrlBuf = (__ssbuf__ uint32_t *)(fifoBase + entryBase + entryOffset);
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
                              "Fix: TPUSH has unsuported fifo type!");
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
            } else if constexpr (is_v2c_gm) { // is_v2c_mat
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
                uint8_t freeCubeID = FlagID + 1;
                set_intra_block(PIPE_MTE2, freeCubeID);
#endif
            } else if constexpr (is_c2v_ub) {
                // Vec consumer frees buffer for Cube - signals on PIPE_V, flag_id+1 only
                // Vec signals after vector ops complete (PIPE_V)
#ifdef __DAV_VEC__
                uint8_t freeCubeID = FlagID + 1;
                set_intra_block(PIPE_V, freeCubeID);
#endif
            } else if constexpr (is_v2c_gm || is_v2c_mat) { // is_v2c (both gm and mat)
                // Cube consumer frees buffer for Vec - signals BOTH flags on PIPE_MTE1
#ifdef __DAV_CUBE__
                uint8_t freeVec0ID = FlagID + 1;
                uint8_t freeVec1ID = FlagID + 1 + VEC_CORE_ID_OFFSET;
                set_intra_block(PIPE_MTE1, freeVec0ID);
                set_intra_block(PIPE_MTE1, freeVec1ID);
#endif
            } else { // is_v2c_ctrl
                     // Control signals from Vec to Cube: Vec signals on flag_id, Cube waits on flag_id only
#ifdef __DAV_CUBE__
                uint8_t freeVec0ID = FlagID + 1;
                uint8_t freeVec1ID = FlagID + 1 + VEC_CORE_ID_OFFSET;
                set_intra_block(PIPE_S, freeVec0ID);
                set_intra_block(PIPE_S, freeVec1ID);
#endif
            }
        }

        template <typename T, int ProdM, int ProdN, int ConsM, int ConsN>
        PTO_INTERNAL void popVecTileFromGMFiFo(DataFiFo &fifo, TileDataCons &tile)
        {
            size_t buf_idx = static_cast<size_t>(tile_id) % fifo.fifoDepth;
            constexpr int kTileFactor = ConsN / ProdN;
            size_t entryBase = static_cast<size_t>(buf_idx) * kTileFactor * ProdM * ProdN * sizeof(T);
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
            uint32_t buf_idx = static_cast<uint32_t>(tile_id % DataFiFo::fifoDepth);
            uint64_t fifoBase = (fifo.tilePtr != nullptr) ? (uint64_t)fifo.tilePtr->data() : fifo.fifoBase;
            size_t entryBase = buf_idx * ConsM * ConsN * sizeof(T);
            uint64_t localTileBase = fifoBase + entryBase + entryOffset;
            TASSIGN_IMPL(tile, localTileBase);
        }

        template <typename T, int ConsM, int ConsN, int ProdN>
        PTO_INTERNAL void popMatTileFromGMFiFo(DataFiFo &fifo, TileDataCons &tile)
        {
            uint32_t buf_idx = static_cast<uint32_t>(tile_id % fifo.fifoDepth);
            size_t entryBase = buf_idx * ConsM * ProdN * sizeof(T);
            using GlobaData = GlobalTensor<T, pto::Shape<1, 1, 1, ConsM, ConsN>, pto::Stride<1, 1, 1, ConsN, 1>>;
            GlobaData globalTensor((__gm__ T *)((uint64_t)fifo.fifoBase + entryBase + entryOffset));

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
            uint32_t buf_idx = static_cast<uint32_t>(tile_id % fifo.fifoDepth);
            size_t entryBase = buf_idx * sizeof(uint32_t);
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
                    popTileFromLocalFiFo<T, ProdM, ProdN, ConsM, ConsN>(fifo, tile);
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
                    popTileFromLocalFiFo<T, ProdM, ProdN, ConsM, ConsN>(fifo, tile);
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
    PTO_INTERNAL explicit TPipe(__gm__ typename TileDataCons::DType *gmFiFoBase, uint32_t localFiFoBase)
        : fifo(gmFiFoBase, localFiFoBase), prod(), cons()
    {
        cons.free();
    }

    // constructors for TILE-based FIFO initialization (for non-GM FIFOs)
    template <FIFOType T = FiFoType, typename std::enable_if_t<T != FIFOType::GM_FIFO, int> = 0>
    PTO_INTERNAL explicit TPipe(uint32_t fifoBase) : fifo(fifoBase), prod(), cons()
    {
        cons.free();
    }

    template <FIFOType T = FiFoType, typename std::enable_if_t<T != FIFOType::GM_FIFO, int> = 0>
    PTO_INTERNAL explicit TPipe(TileDataCons *tilePtr) : fifo(tilePtr), prod(), cons()
    {
        cons.free();
    }

    template <FIFOType T = FiFoType, typename std::enable_if_t<T != FIFOType::GM_FIFO, int> = 0>
    PTO_INTERNAL explicit TPipe(TileDataCons &tile) : fifo(tile), prod(), cons()
    {
        cons.free();
    }

    // Destructor for TPipe
    PTO_INTERNAL ~TPipe()
    {
        prod.allocate();
    }
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
    prod.tile_id++;

    // 3； Cross-Core: Commit & Signal
    bool isRecord = prod.getRecordStatus();
    if (isRecord) {
        prod.record();
    }
}

template <typename TileData, typename Pipe>
PTO_INTERNAL void TPUSH_IMPL(TileData &tile, Pipe &pipe)
{
    TPUSH_IMPL(pipe.prod, tile, pipe.fifo);
}

} // namespace pto

#endif
