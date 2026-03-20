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
#include <pto/npu/a2a3/TStore.hpp>
#include <pto/npu/a2a3/TLoad.hpp>

namespace pto {

enum TSyncCVMode : uint8_t
{
    CUBE_ALL_CORE_SYNC = 0,
    VEC_ALL_CORE_SYNC = 0,
    VEC_SUBCORES_SYNC = 1,
    CV_CORES_SYNC = 2
};

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
    static constexpr bool is_c2v =
        (FiFoType == FIFOType::GM_FIFO) && (TileDataProd::Loc == TileType::Acc) && (TileDataCons::Loc == TileType::Vec);
    static constexpr bool is_v2c =
        (FiFoType == FIFOType::GM_FIFO) && (TileDataProd::Loc == TileType::Vec) && (TileDataCons::Loc == TileType::Mat);

    using DataFiFo = DataFIFO<typename TileDataCons::DType, FiFoType, FiFoDepth, FiFoSyncT, LocalFiFoDepth>;

    PTO_INTERNAL static uint64_t getFFTSMsgCfg(TSyncCVMode mode, uint16_t flagID, uint16_t base_const = 0x1)
    {
        constexpr uint16_t FFTS_MODE_BIT_START = 4;
        constexpr uint16_t FFTS_FLAG_ID_BIT_START = 8;
        return ((base_const & 0xf) + ((mode & 0x3) << FFTS_MODE_BIT_START) +
                ((flagID & 0xf) << FFTS_FLAG_ID_BIT_START));
    }

    // -------------------------------------------------------------------------
    // Producer Interface
    // -------------------------------------------------------------------------
    struct Producer {
        int tile_id = 0;
        int sub_tile_id = 0;
        int entryOffset = 0;
        bool isAllocate = true;
        bool isRecord = true;

        PTO_INTERNAL Producer() = default;

        PTO_INTERNAL void setTileId(int t_id, int sub_t_id)
        {
            tile_id = t_id;
            sub_tile_id = sub_t_id;
        }

        PTO_INTERNAL void setAllocateStatus(bool allocate)
        {
            isAllocate = allocate;
        }

        PTO_INTERNAL void setRecordStatus(bool record)
        {
            isRecord = record;
        }

        PTO_INTERNAL void setEntryOffset(int offset)
        {
            entryOffset = offset;
        }

        PTO_INTERNAL int getTileId() const
        {
            return tile_id;
        }

        PTO_INTERNAL int getSubTileId() const
        {
            return sub_tile_id;
        }

        PTO_INTERNAL bool getAllocateStatus() const
        {
            return isAllocate;
        }

        PTO_INTERNAL bool getRecordStatus() const
        {
            return isRecord;
        }

        /**
         * alloc: Request space in FIFO
         * 1. (iter >= Depth): Startup protection. Don't check flags when buffer is empty.
         * 2. (iter % Period == 0): Sparse sync. Only check flag periodically.
         */
        PTO_INTERNAL void allocate() const
        {
            // Cube waits for Vector to free buffer
            if constexpr (is_c2v) {
#ifdef __DAV_CUBE__
                wait_flag_dev(FlagID + 1);
#endif
            } else {
                // Vector waits for Cube to free buffer
#ifdef __DAV_VEC__
                wait_flag_dev(FlagID + 1);
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
                // Cube produces, Vector consumes
                ffts_cross_core_sync(PIPE_FIX, getFFTSMsgCfg(TSyncCVMode::CV_CORES_SYNC, FlagID));
            } else { // is_v2c
                // Vector produces, Cube consumes
                ffts_cross_core_sync(PIPE_MTE3, getFFTSMsgCfg(TSyncCVMode::CV_CORES_SYNC, FlagID));
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
            // store tile to GM FIFO, enable unit-flag one
            if constexpr (EN_UNIT_FLAG) {
                TSTORE_IMPL<TileDataProd, GlobalData, AtomicType::AtomicNone, STPhase::Final>(globalTensor, tile);
            } else { // disable unit flag
                TSTORE_IMPL(globalTensor, tile);
            }
        } // end of Acc->GM

        template <typename T, int ProdM, int ProdN, int ConsM, int ConsN>
        PTO_INTERNAL void pushVec2GMFiFo(DataFiFo &fifo, TileDataProd &tile)
        {
            static_assert(DataFiFo::fifoType == FIFOType::GM_FIFO, "Fix: TPUSH has unsupported fifoType!");
            constexpr int kTileFactor = ProdN / ConsN;
            uint32_t bufIndex = static_cast<uint32_t>(tile_id % DataFiFo::fifoDepth);
            using GlobalDataSub = GlobalTensor<T, pto::Shape<1, 1, 1, ProdM, ConsN>, pto::Stride<1, 1, 1, ConsN, 1>>;
            size_t entryBase = bufIndex * kTileFactor * ConsM * ConsN * sizeof(T);
            __gm__ T *addr = (__gm__ T *)((uint64_t)fifo.fifoBase + entryBase + entryOffset);
            // store tile to GM FIFO in sub-tiles if needed (when Tile_S1 > Cube_S1)
            Tile<TileType::Vec, T, ProdM, ProdN, BLayout::RowMajor, ProdM, ConsN> subTile;
            for (int sub_col = 0; sub_col < kTileFactor; ++sub_col) {
                __gm__ T *addrSub = addr + sub_col * ConsM * ConsN;
                GlobalDataSub globalDataSub((__gm__ T *)(addrSub));
                uint64_t col_byte_offset = static_cast<uint64_t>(sub_col * ConsN * sizeof(T));
                TASSIGN_IMPL(subTile, (uint64_t)tile.data() + col_byte_offset);
                TSTORE_IMPL(globalDataSub, subTile);
            }
        }

        PTO_INTERNAL void pushVec2CtrlFiFo(DataFiFo &fifo, TileDataProd &tile)
        {
            static_assert(DataFiFo::fifoType == FIFOType::CTRL_FIFO, "Fix: TPUSH has unsupported fifo Type!");
            uint32_t bufIndex = static_cast<uint32_t>(tile_id % DataFiFo::fifoDepth);
            uint64_t entryBase = bufIndex * sizeof(uint32_t);
            __gm__ uint32_t *ctrlBuf = (__gm__ uint32_t *)(fifo.fifoBase + entryBase + entryOffset);
            set_flag(PIPE_V, PIPE_S, EVENT_ID0);
            wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
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

            static_assert(TileDataProd::Loc == TileType::Acc || TileDataProd::Loc == TileType::Vec,
                          "Fix: TPUSH has unsupported tile type!");
            if constexpr (TileDataProd::Loc == TileType::Acc) {
                pushAcc2GMFiFo<T, ProdM, ProdN, ConsM, ConsN>(fifo, tile);
            } else if constexpr (TileDataProd::Loc == TileType::Vec) {
                static_assert(DataFiFo::fifoType == FIFOType::GM_FIFO || DataFiFo::fifoType == FIFOType::CTRL_FIFO,
                              "Fix: TPUSH has unsupported fifo type!");
                if constexpr (DataFiFo::fifoType == FIFOType::GM_FIFO) {
                    pushVec2GMFiFo<T, ProdM, ProdN, ConsM, ConsN>(fifo, tile);
                } else if constexpr (DataFiFo::fifoType == FIFOType::CTRL_FIFO) {
                    pushVec2CtrlFiFo(fifo, tile);
                }
            }
        } // end of store
    };    // end of Producer

    // -------------------------------------------------------------------------
    // Consumer Interface
    // -------------------------------------------------------------------------
    struct Consumer {
        int tile_id = 0;
        int sub_tile_id = 0;
        int entryOffset = 0;
        bool isFree = true;
        bool isWait = true;

        PTO_INTERNAL Consumer() = default;

        PTO_INTERNAL void setTileId(int tid, int sub_tid)
        {
            tile_id = tid;
            sub_tile_id = sub_tid;
        }

        PTO_INTERNAL void setEntryOffset(int offset)
        {
            entryOffset = offset;
        }

        PTO_INTERNAL void setWaitStatus(bool wait)
        {
            isWait = wait;
        }

        PTO_INTERNAL void setFreeStatus(bool free)
        {
            isFree = free;
        }

        PTO_INTERNAL int getTileId() const
        {
            return tile_id;
        }

        PTO_INTERNAL int getSubTileId() const
        {
            return sub_tile_id;
        }

        PTO_INTERNAL bool getWaitStatus() const
        {
            return isWait;
        }

        PTO_INTERNAL bool getFreeStatus() const
        {
            return isFree;
        }

        /**
         * wait: Block until data is ready
         * Consumers strictly wait for data (no sparse optimization for safety).
         */
        PTO_INTERNAL void wait() const
        {
            // Vector waits for Cube
            // Or Cube waits for Vector
            wait_flag_dev(FlagID);
        }

        /**
         * free: Release space in FIFO
         * 1. (iter >= Depth - Period): Silence at start. Don't signal if Producer
         * is still enjoying the initial free buffer space.
         * 2. (is_sync_step): Accumulate free slots and signal in batches.
         */
        PTO_INTERNAL void free() const
        {
            // Vector frees buffer for Cube
            // Or Cube frees buffer for Vector
            if constexpr (is_c2v) {
#ifdef __DAV_VEC__
                // Vec consumer frees buffer for Cube
                ffts_cross_core_sync(PIPE_MTE2, getFFTSMsgCfg(TSyncCVMode::CV_CORES_SYNC, FlagID + 1));
#endif
            } else { // is_v2c
                     // cube consumer frees buffer for vec
#ifdef __DAV_CUBE__
                ffts_cross_core_sync(PIPE_MTE2, getFFTSMsgCfg(TSyncCVMode::CV_CORES_SYNC, FlagID + 1));
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
                uint64_t localTileBase = fifo.localFiFoBase + (static_cast<size_t>(tile_id) % fifo.localFiFoDepth) *
                                                                  ConsM * ConsN * sizeof(T);
                TASSIGN_IMPL(tile, localTileBase);
            }

            Tile<TileType::Vec, T, ConsM, ConsN, BLayout::RowMajor, ConsM, ProdN> tileSub;
            using GlobalDataSub = GlobalTensor<T, pto::Shape<1, 1, 1, ConsM, ProdN>, pto::Stride<1, 1, 1, ProdN, 1>>;
            for (int sub_col = 0; sub_col < kTileFactor; ++sub_col) {
                __gm__ T *addrSub = addr + sub_col * ProdM * ProdN;
                GlobalDataSub globalTensorSub(addrSub);
                uint64_t col_byte_offset = sub_col * ProdN * sizeof(T);
                TASSIGN_IMPL(tileSub, (uint64_t)tile.data() + col_byte_offset);
                TLOAD_IMPL(tileSub, globalTensorSub);
            }
        }

        template <typename T, int ConsM, int ConsN, int ProdN>
        PTO_INTERNAL void popMatTileFromGMFiFo(DataFiFo &fifo, TileDataCons &tile)
        {
            using GlobaData = GlobalTensor<T, pto::Shape<1, 1, 1, ConsM, ConsN>, pto::Stride<1, 1, 1, ConsN, 1>>;
            uint32_t bufIndex = static_cast<uint32_t>(tile_id % fifo.fifoDepth);
            size_t entryBase = bufIndex * ConsM * ProdN * sizeof(T);
            GlobaData globalTensor((__gm__ T *)((uint64_t)fifo.fifoBase + entryBase + entryOffset));

            if constexpr (DataFiFo::useLocalFiFo) {
                uint64_t tileBase = fifo.localFiFoBase +
                                    (static_cast<size_t>(tile_id) % fifo.localFiFoDepth) * ConsM * ConsN * sizeof(T);
                TASSIGN_IMPL(tile, tileBase);
            }
            TLOAD_IMPL(tile, globalTensor);
        }

        PTO_INTERNAL void popCtrlFromCtrlFiFo(DataFiFo &fifo)
        {
            uint32_t bufIndex = static_cast<uint32_t>(tile_id % fifo.fifoDepth);
            size_t entryBase = bufIndex * sizeof(uint32_t);
            uint64_t ctrlTileBase = fifo.fifoBase + entryBase + entryOffset;
            fifo.ctrlSignal = ((*(__gm__ uint32_t *)(ctrlTileBase)) == 1) ? true : false;
        }

        PTO_INTERNAL void pop(DataFiFo &fifo, TileDataCons &tile)
        {
            using T = typename TileDataCons::DType;
            constexpr int ConsM = TileDataCons::Rows;
            constexpr int ConsN = TileDataCons::Cols;
            constexpr int ProdM = TileDataProd::Rows;
            constexpr int ProdN = TileDataProd::Cols;
            constexpr int VEC_CORES = (VCRatio == VecCubeRatio::V2C1_VECS) ? 2 : 1;
            static_assert(DataFiFo::fifoType == FIFOType::GM_FIFO || DataFiFo::fifoType == FIFOType::CTRL_FIFO,
                          "Fix: TPOP has unsupported fifo type!");
            static_assert(TileDataCons::Loc == TileType::Vec || TileDataCons::Loc == TileType::Mat,
                          "Fix: TPOP has unsupported tile type!");
            if constexpr (DataFiFo::fifoType == FIFOType::GM_FIFO) {
                if constexpr (TileDataCons::Loc == TileType::Vec) {
                    popVecTileFromGMFiFo<T, ProdM, ProdN, ConsM, ConsN>(fifo, tile);
                } else if constexpr (TileDataCons::Loc == TileType::Mat) {
                    popMatTileFromGMFiFo<T, ConsM, ConsN, ProdN>(fifo, tile);
                }
            } else if constexpr (DataFiFo::fifoType == FIFOType::CTRL_FIFO) {
                popCtrlFromCtrlFiFo(fifo);
            }
        }
    };

    DataFiFo fifo;
    Producer prod;
    Consumer cons;

    template <FIFOType T = FiFoType, typename std::enable_if_t<T == FIFOType::GM_FIFO, int> = 0>
    PTO_INTERNAL explicit TPipe(__gm__ typename TileDataCons::DType *gmFiFoBase, uint32_t localFiFoBase)
        : fifo(gmFiFoBase, localFiFoBase), prod(), cons()
    {
        cons.free();
    }

    template <FIFOType T = FiFoType, typename std::enable_if_t<T == FIFOType::CTRL_FIFO, int> = 0>
    PTO_INTERNAL explicit TPipe(uint32_t fifoBase) : fifo(fifoBase), prod(), cons()
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
template <typename TileData, typename Pipe>
PTO_INTERNAL void TPUSH_IMPL(TileData &tile, Pipe &pipe)
{
    TPUSH_IMPL(pipe.prod, tile, pipe.fifo);
}

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

} // namespace pto

#endif
