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

// Operation types for TSync - identifies the producer/consumer operation
enum class TSyncOpType : uint8_t
{
    TSTORE_C2GM, // Store (Cube core operation)
    TSTORE_V2GM, // Store (Vector core operation)
    TLOAD        // Load operation (consumer operation)
};

// Compile-time direction inference based on producer/consumer ops
// TSTORE_C2GM (producer) + TLOAD (consumer) = Cube to Vector
// TSTORE_V2GM (producer) + TLOAD (consumer) = Vector to Cube
template <TSyncOpType ProducerOp, TSyncOpType ConsumerOp>
struct TSyncTraits {
    // Direction is inferred from producer operation:
    // TSTORE_C2GM -> Cube produces (C2V)
    // TSTORE_V2GM -> Vector produces (V2C)
    static constexpr bool is_cube_to_vec = (ProducerOp == TSyncOpType::TSTORE_C2GM);
    static constexpr bool is_vec_to_cube = (ProducerOp == TSyncOpType::TSTORE_V2GM);

    static_assert(ConsumerOp == TSyncOpType::TLOAD, "Consumer operation must be TLOAD");
    static_assert(is_cube_to_vec || is_vec_to_cube,
                  "Producer must be either TSTORE_C2GM (Cube) or TSTORE_V2GM (Vector)");
};

enum TSyncCVMode : uint8_t
{
    CUBE_ALL_CORE_SYNC = 0,
    VEC_ALL_CORE_SYNC = 0,
    VEC_SUBCORES_SYNC = 1,
    CV_CORES_SYNC = 2
};

/**
 * Pipe: Manages Cross-Core FIFO Synchronization
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
    // default to 2:1 ratio, can be set by user based on actual tile size and core count
    using Traits = TSyncTraits<ProducerOp, ConsumerOp>;
    static constexpr bool is_c2v = Traits::is_cube_to_vec;
    static constexpr bool is_v2c = Traits::is_vec_to_cube;

    using DataFiFo = DataFIFO<typename TileDataCons::DType, fifoType, FiFoDepth, FiFoSyncT>;

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

        PTO_INTERNAL uint16_t getFFTSMsg(TSyncCVMode mode, uint16_t flag_id, uint16_t base_const = 0x1)
        {
            constexpr uint16_t FFTS_MODE_BIT_START = 4;
            constexpr uint16_t FFTS_FLAG_ID_BIT_START = 6;
            return ((base_const & 0xf) + ((mode & 0x3) << FFTS_MODE_BIT_START) +
                    ((flag_id & 0xf) << FFTS_FLAG_ID_BIT_START));
        }

        /**
         * alloc: Request space in FIFO
         * Logic:
         * 1. (iter >= Depth): Startup protection. Don't check flags when buffer is empty.
         * 2. (iter % Period == 0): Sparse sync. Only check flag periodically.
         */
        PTO_INTERNAL void allocate()
        {
            // Cube waits for Vector to free buffer
            // Or Vector waits for Cube to free buffer
            wait_flag_dev(FlagID + 1);
        }

        // Forward dependency: record (producer) and wait (consumer)
        /**
         * record - Producer signals that data is ready
         * Called by the producer after completing the operation (TSTORE_C2GM or TSTORE_V2GM)
         */
        PTO_INTERNAL void record()
        {
            if constexpr (is_c2v) {
                // Cube produces, Vector consumes
                ffts_cross_core_sync(PIPE_FIX, getFFTSMsg(TSyncCVMode::CV_CORES_SYNC, FlagID));
            } else { // is_v2c
                // Vector produces, Cube consumes
                ffts_cross_core_sync(PIPE_MTE3, getFFTSMsg(TSyncCVMode::CV_CORES_SYNC, FlagID));
            }
        }

        template <typename T, int ProdM, int ProdN, int ConsN>
        PTO_INTERNAL void pushAcc2GMFiFo(DataFiFo &fifo, TileDataProd &tile)
        {
            // calculate base address in GM FIFO for this tile
            constexpr int kTileFactor = ConsN / ProdN;
            uint32_t buf_idx = static_cast<uint32_t>(tile_id % DataFiFo::fifoDepth);
            size_t entryBase = buf_idx * kTileFactor * ProdM * ProdN;
            using GlobalData = GlobalTensor<T, pto::Shape<1, 1, 1, ProdM, ProdN>, pto::Stride<1, 1, 1, ProdN, 1>>;
            GlobalData globalTensor(fifo.fifoBase + entryBase + entryOffset);
            // store tile to GM FIFO, enable unit-flag one
            TSTORE_IMPL<TileDataProd, GlobalData, AtomicType::AtomicNone, STPhase::Final>(globalTensor, tile);
        }

        template <typename T, int ProdM, int ProdN, int ConsM, int ConsN>
        PTO_INTERNAL void pushVec2GMFiFo(DataFiFo &fifo, TileDataProd &tile)
        {
            static_assert(DataFiFo::fifoType == FIFOType::GM_FIFO, "Fix: TPUSH: unsupported fifoType!");
            // calculate base address in GM FIFO for this tile
            constexpr int kTileFactor = ProdN / ConsN;
            uint32_t buf_idx = static_cast<uint32_t>(tile_id % DataFiFo::fifoDepth);
            size_t entryBase = buf_idx * kTileFactor * ConsM * ConsN;
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
                pushAcc2GMFiFo<T, ProdM, ProdN, ConsN>(fifo, tile);
            } else if constexpr (TileDataProd::Loc == TileType::Vec) {
                pushVec2GMFiFo<T, ProdM, ProdN, ConsM, ConsN>(fifo, tile);
            }
        } // end of store
    };    // end of Producer

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

        PTO_INTERNAL void setEntryOffset(int offset)
        {
            entryOffset = offset;
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

        /**
         * wait: Block until data is ready
         * Consumers strictly wait for data (no sparse optimization for safety).
         */
        PTO_INTERNAL void wait()
        {
            if constexpr (is_c2v) {
                // Vector waits for Cube
                wait_flag_dev(FlagID);
            } else { // is_v2c
                // Cube waits for Vector
                wait_flag_dev(FlagID);
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
            if constexpr (is_c2v) {
                // Vector frees buffer for Cube
                ffts_cross_core_sync(PIPE_MTE2, getFFTSMsg(TSyncCVMode::CV_CORES_SYNC, FlagID + 1));
            } else { // is_v2c
                // Cube frees buffer for Vector
                ffts_cross_core_sync(PIPE_MTE2, getFFTSMsg(TSyncCVMode::CV_CORES_SYNC, FlagID + 1));
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
            constexpr int ConsM = TileDataCons::Rows;
            constexpr int ConsN = TileDataCons::Cols;
            constexpr int ProdM = TileDataProd::Rows;
            constexpr int ProdN = TileDataProd::Cols;
            constexpr int VEC_CORES = (VCRatio == VecCubeRatio::V2C1_VECS) ? 2 : 1;
            static_assert(TileDataCons::Loc == TileType::Vec || TileDataCons::Loc == TileType::Mat,
                          "Fix: TPOP has unsupported tile type!");
            if constexpr (TileDataCons::Loc == TileType::Vec) {
                popVecTileFromGMFiFo<T, ProdM, ProdN, ConsM, ConsN>(fifo, tile);
            } else if constexpr (TileDataCons::Loc == TileType::Mat) {
                popMatTileFromGMFiFo<T, ConsM, ConsN, ProdN>(fifo, tile);
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