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

template <uint8_t FlagID, uint8_t DirType, uint32_t SlotSize, uint32_t SlotNum,
          uint32_t CompatLocalSlotNumOrIsNoSplit = 0, uint32_t LocalSlotNum = 2,
          bool EN_UNIT_FLAG = false>
struct TPipe {
    static constexpr uint8_t DIR_MASK = 0x7;
    static constexpr uint8_t DIR_TYPE = DIR_MASK & DirType;
    static constexpr bool is_c2v = (DIR_TYPE == Direction::DIR_C2V);           // 1
    static constexpr bool is_v2c = (DIR_TYPE == Direction::DIR_V2C);           // 2
    static constexpr bool is_both = (DIR_TYPE == Direction::DIR_BOTH);         // 3
    static constexpr bool is_v2c_ctrl = (DIR_TYPE == Direction::DIR_V2C_CTRL); // 4
    static constexpr bool IsNoSplit =
        CompatLocalSlotNumOrIsNoSplit <= 1 && static_cast<bool>(CompatLocalSlotNumOrIsNoSplit);
    static constexpr uint32_t EffectiveLocalSlotNum =
        CompatLocalSlotNumOrIsNoSplit <= 1 ? LocalSlotNum : CompatLocalSlotNumOrIsNoSplit;
    static_assert(is_c2v || is_v2c || is_both || is_v2c_ctrl,
                  "Fix: TPipe only supports C2V or V2C or Both or V2C_CTRL communication on A2A3.");

    using RingFiFo = RingFIFO<SlotSize, SlotNum, EffectiveLocalSlotNum>;

    PTO_INTERNAL static uint64_t getFFTSMsgCfg(TSyncCVMode mode, uint16_t flagID, uint16_t base_const = 0x1)
    {
        constexpr uint16_t FFTS_MODE_BIT_START = 4;
        constexpr uint16_t FFTS_FLAG_ID_BIT_START = 8;
        return ((base_const & 0xf) + ((mode & 0x3) << FFTS_MODE_BIT_START) +
                ((flagID & 0xf) << FFTS_FLAG_ID_BIT_START));
    }

    struct Producer {
        uint32_t tileIndex = 0;
        uint32_t subTileIndex = 0;
        bool isAllocate = true;
        bool isRecord = true;
        int entryOffset = 0;

        PTO_INTERNAL Producer() = default;

        PTO_INTERNAL void setAllocateStatus(bool allocate)
        {
            isAllocate = allocate;
        }

        PTO_INTERNAL void setRecordStatus(bool record)
        {
            isRecord = record;
        }

        PTO_INTERNAL void setTileId(int tIndex, int subIndex)
        {
            tileIndex = tIndex;
            subTileIndex = subIndex;
        }

        PTO_INTERNAL void setEntryOffset(int offset)
        {
            entryOffset = offset;
        }

        PTO_INTERNAL int getSubTileId() const
        {
            return subTileIndex;
        }

        PTO_INTERNAL int getTileId() const
        {
            return tileIndex;
        }

        PTO_INTERNAL bool getAllocateStatus() const
        {
            return isAllocate;
        }

        PTO_INTERNAL bool getRecordStatus() const
        {
            return isRecord;
        }

        PTO_INTERNAL void allocate() const
        {
            // Cube waits for Vector to free buffer
            if constexpr (is_c2v) {
#ifdef __DAV_CUBE__
                wait_flag_dev(FlagID + 1);
#endif
            } else if constexpr (is_v2c) {
                // Vector waits for Cube to free buffer
#ifdef __DAV_VEC__
                wait_flag_dev(FlagID + 1);
#endif
            } else if constexpr (is_both) {
                wait_flag_dev(FlagID + 1);
            }
        }

        PTO_INTERNAL void record() const
        {
            if constexpr (is_c2v) {
                // Cube produces, Vector consumes
#ifdef __DAV_CUBE__
                ffts_cross_core_sync(PIPE_FIX, getFFTSMsgCfg(TSyncCVMode::CV_CORES_SYNC, FlagID));
#endif
            } else if constexpr (is_v2c) {
                // Vector produces, Cube consumes
#ifdef __DAV_VEC__
                ffts_cross_core_sync(PIPE_MTE3, getFFTSMsgCfg(TSyncCVMode::CV_CORES_SYNC, FlagID));
#endif
            } else if constexpr (is_both) {
#ifdef __DAV_CUBE__
                ffts_cross_core_sync(PIPE_FIX, getFFTSMsgCfg(TSyncCVMode::CV_CORES_SYNC, FlagID));
#endif
#ifdef __DAV_VEC__
                ffts_cross_core_sync(PIPE_MTE3, getFFTSMsgCfg(TSyncCVMode::CV_CORES_SYNC, FlagID));
#endif
            }
        }

        template <typename TileProd>
        PTO_INTERNAL void pushAcc2GMFiFo(RingFiFo &fifo, TileProd &tile)
        {
            using T = typename TileProd::DType;
            constexpr int ProdM = TileProd::Rows;
            constexpr int ProdN = TileProd::Cols;
            size_t entryBase = (tileIndex % RingFiFo::SLOT_NUM) * RingFiFo::SLOT_SIZE;
            using GlobalData = GlobalTensor<T, pto::Shape<1, 1, 1, ProdM, ProdN>, pto::Stride<1, 1, 1, ProdN, 1>>;
            GlobalData globalTensor((__gm__ T *)((uint64_t)fifo.GM_SLOT_BUFFER + entryBase + entryOffset));
            // store tile to GM FIFO, enable unit-flag one
            if constexpr (EN_UNIT_FLAG) {
                TSTORE_IMPL<TileProd, GlobalData, AtomicType::AtomicNone, STPhase::Final>(globalTensor, tile);
            } else { // disable unit flag
                TSTORE_IMPL(globalTensor, tile);
            }
        }

        template <typename TileProd, TileSplitAxis Split>
        PTO_INTERNAL void pushVec2GMFiFo(RingFiFo &fifo, TileProd &tile)
        {
            using T = typename TileProd::DType;
            constexpr int splitNum = 2;
            constexpr int ProdM = TileProd::Rows;
            constexpr int ProdN = TileProd::Cols;
            constexpr int ConsM = (Split == TileSplitAxis::TILE_UP_DOWN) ? ProdM * splitNum : ProdM;
            constexpr int ConsN = (Split == TileSplitAxis::TILE_LEFT_RIGHT) ? ProdN * splitNum : ProdN;
            size_t entryBase = (tileIndex % RingFiFo::SLOT_NUM) * RingFiFo::SLOT_SIZE; // ConsM * ConsN * sizeof(T);
            constexpr int gmValidR = ProdM;
            constexpr int gmValidC = ProdN;
            constexpr int gmStrideR = ConsN;
            size_t subAIVOffset = 0;
            if constexpr (Split == TileSplitAxis::TILE_NO_SPLIT) {
                // TILE_NO_SPLIT : single writer, no offset needed
                subAIVOffset = 0;
            } else if constexpr (Split == TileSplitAxis::TILE_UP_DOWN) {
                // TILE_UP_DOWN  : Vec1 starts at the second row-block → offset = ProdM * ProdN * sizeof(T)
                subAIVOffset = get_subblockid() * ProdM * ProdN * sizeof(T);
            } else { // TILE_LEFT_RIGHT
                // TILE_LEFT_RIGHT: Vec1 starts at column ProdN within row 0 → offset = ProdN * sizeof(T)
                subAIVOffset = get_subblockid() * ProdN * sizeof(T);
            }
            using GlobalData =
                GlobalTensor<T, pto::Shape<1, 1, 1, gmValidR, gmValidC>, pto::Stride<1, 1, 1, gmStrideR, 1>>;
            __gm__ T *addr = (__gm__ T *)((uint64_t)fifo.GM_SLOT_BUFFER + entryBase + subAIVOffset + entryOffset);
            GlobalData globalData(addr);
            TSTORE_IMPL(globalData, tile);
        }

        template <typename TileProd>
        PTO_INTERNAL void pushVec2CtrlFiFo(RingFiFo &fifo, TileProd &tile)
        {
            size_t slotIndex = (tileIndex % RingFiFo::SLOT_NUM);
            uint64_t entryBase = slotIndex * sizeof(uint64_t);
            __gm__ uint64_t *ctrlBuf = (__gm__ uint64_t *)(fifo.CTRL_SLOT_BUFFER + entryBase + entryOffset);
            set_flag(PIPE_V, PIPE_S, EVENT_ID0);
            wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
            uint64_t ctrlSignal = *(tile.data());
            *(ctrlBuf) = ctrlSignal;
        }

        template <typename TileProd, TileSplitAxis Split>
        PTO_INTERNAL void push(RingFiFo &fifo, TileProd &tile)
        {
            static_assert(TileProd::Loc == TileType::Acc || TileProd::Loc == TileType::Vec,
                          "Fix: TPUSH has unsupported tile type!");
            if constexpr (is_c2v) {
#ifdef __DAV_CUBE__
                pushAcc2GMFiFo<TileProd>(fifo, tile);
#endif
            } else if constexpr (is_v2c) {
#ifdef __DAV_VEC__
                pushVec2GMFiFo<TileProd, Split>(fifo, tile);
#endif
            } else if constexpr (is_both) {
#ifdef __DAV_CUBE__
                pushAcc2GMFiFo<TileProd>(fifo, tile);
#endif
#ifdef __DAV_VEC__
                pushVec2GMFiFo<TileProd, Split>(fifo, tile);
#endif
            } else if constexpr (is_v2c_ctrl) {
                pushVec2CtrlFiFo<TileProd>(fifo, tile);
            }
        }
    }; // end of Producer

    struct Consumer {
        int tileIndex = 0;
        int subTileIndex = 0;
        bool isWait = true;
        bool isFree = true;
        int entryOffset = 0;

        PTO_INTERNAL Consumer() = default;

        PTO_INTERNAL void setTileId(int tid, int sub_tid)
        {
            tileIndex = tid;
            subTileIndex = sub_tid;
        }

        PTO_INTERNAL void setWaitStatus(bool wait)
        {
            isWait = wait;
        }

        PTO_INTERNAL void setFreeStatus(bool free)
        {
            isFree = free;
        }

        PTO_INTERNAL void setEntryOffset(int offset)
        {
            entryOffset = offset;
        }

        PTO_INTERNAL int getTileId() const
        {
            return tileIndex;
        }

        PTO_INTERNAL int getSubTileId() const
        {
            return subTileIndex;
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
            if constexpr (is_c2v) { // Vec consumer frees buffer for Cube
#ifdef __DAV_VEC__
                ffts_cross_core_sync(PIPE_MTE2, getFFTSMsgCfg(TSyncCVMode::CV_CORES_SYNC, FlagID + 1));
#endif
            } else if constexpr (is_v2c) { // cube consumer frees buffer for vec
#ifdef __DAV_CUBE__
                ffts_cross_core_sync(PIPE_MTE2, getFFTSMsgCfg(TSyncCVMode::CV_CORES_SYNC, FlagID + 1));
#endif
            } else if constexpr (is_both) {
                ffts_cross_core_sync(PIPE_MTE2, getFFTSMsgCfg(TSyncCVMode::CV_CORES_SYNC, FlagID + 1));
            }
        }

        template <typename TileCons, TileSplitAxis Split>
        PTO_INTERNAL void popVecTileFromGMFiFo(RingFiFo &fifo, TileCons &tile)
        {
            using T = typename TileCons::DType;
            constexpr int splitNum = 2;
            constexpr int ConsM = TileCons::Rows;
            constexpr int ConsN = TileCons::Cols;
            constexpr int ProdM = (Split == TileSplitAxis::TILE_UP_DOWN) ? ConsM * splitNum : ConsM;
            constexpr int ProdN = (Split == TileSplitAxis::TILE_LEFT_RIGHT) ? ConsN * splitNum : ConsN;

            // global tensor
            size_t entryBase = (static_cast<size_t>(tileIndex) % RingFiFo::SLOT_NUM) *
                               RingFiFo::SLOT_SIZE; // ProdM * ProdN * sizeof(T);
            constexpr int gmValidR = ConsM;
            constexpr int gmValidC = ConsN;
            constexpr int gmStrideR = ProdN;
            size_t subAIVOffset = 0;
            if constexpr (Split == TileSplitAxis::TILE_NO_SPLIT) {
                subAIVOffset = 0; // TILE_NO_SPLIT : single reader, no offset needed
            } else if constexpr (Split == TileSplitAxis::TILE_UP_DOWN) {
                // TILE_UP_DOWN  : Vec1 starts at the second row-block → offset = VEC_M * ProdN * sizeof(T)
                subAIVOffset = get_subblockid() * ConsM * ConsN * sizeof(T);
            } else { // TILE_LEFT_RIGHT
                // TILE_LEFT_RIGHT: Vec1 starts at column ConsN within row 0 → offset = ConsN * sizeof(T)
                subAIVOffset = get_subblockid() * ConsN * sizeof(T);
            }
            __gm__ T *addr = (__gm__ T *)((uint64_t)fifo.GM_SLOT_BUFFER + entryBase + subAIVOffset + entryOffset);
            using GlobalData =
                GlobalTensor<T, pto::Shape<1, 1, 1, gmValidR, gmValidC>, pto::Stride<1, 1, 1, gmStrideR, 1>>;
            GlobalData globalTensor(addr);

            // local vector tile
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
            constexpr int ConsM = TileCons::Rows;
            constexpr int ConsN = TileCons::Cols;
            size_t entryBase = (static_cast<size_t>(tileIndex) % RingFiFo::SLOT_NUM) *
                               RingFiFo::SLOT_SIZE; // ConsM * ConsN * sizeof(T);
            using GlobalData = GlobalTensor<T, pto::Shape<1, 1, 1, ConsM, ConsN>, pto::Stride<1, 1, 1, ConsN, 1>>;
            GlobalData globalTensor((__gm__ T *)((uint64_t)fifo.GM_SLOT_BUFFER + entryBase + entryOffset));

            uint64_t localTileBase =
                fifo.V2C_CONSUMER_BUF +
                (static_cast<size_t>(tileIndex) % RingFiFo::LOCAL_SLOT_NUM) * ConsM * ConsN * sizeof(T);
            TASSIGN_IMPL(tile, localTileBase);
            TLOAD_IMPL(tile, globalTensor);
        }

        PTO_INTERNAL void popCtrlFromCtrlFiFo(RingFiFo &fifo)
        {
            uint32_t slotIndex = (tileIndex % RingFiFo::SLOT_NUM);
            size_t entryBase = slotIndex * sizeof(uint32_t);
            uint64_t ctrlTileBase = fifo.CTRL_SLOT_BUFFER + entryBase + entryOffset;
            fifo.ctrlSignal = ((*(__gm__ uint32_t *)(ctrlTileBase)) == 1) ? true : false;
        }

        template <typename TileCons, TileSplitAxis Split>
        PTO_INTERNAL void pop(RingFiFo &fifo, TileCons &tile)
        {
            static_assert(TileCons::Loc == TileType::Vec || TileCons::Loc == TileType::Mat,
                          "Fix: TPOP has unsupported tile type!");
            if constexpr (TileCons::Loc == TileType::Vec) {
                popVecTileFromGMFiFo<TileCons, Split>(fifo, tile);
            } else if constexpr (TileCons::Loc == TileType::Mat) {
                popMatTileFromGMFiFo<TileCons, Split>(fifo, tile);
            } else {
                popCtrlFromCtrlFiFo(fifo);
            }
        }
    };

    RingFiFo fifo;
    Producer prod;
    Consumer cons;

    PTO_INTERNAL explicit TPipe(__gm__ void *GM_SLOT_BUFFER, uint32_t C2V_CONSUMER_BUF, uint32_t V2C_CONSUMER_BUF)
        : fifo(GM_SLOT_BUFFER, C2V_CONSUMER_BUF, V2C_CONSUMER_BUF), prod(), cons()
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
template <typename Pipe, typename TileProd, TileSplitAxis Split>
PTO_INTERNAL void TPUSH_IMPL(Pipe &pipe, TileProd &tile)
{
    // 1. Cross-Core: Wait for space
    bool isAllocate = pipe.prod.getAllocateStatus();
    if (isAllocate) {
        pipe.prod.allocate();
    }

    // 2. Address Calculation
    pipe.prod.template push<TileProd, Split>(pipe.fifo, tile);
    pipe.prod.tileIndex++;

    // 3. Cross-Core: Commit & Signal
    bool isRecord = pipe.prod.getRecordStatus();
    if (isRecord) {
        pipe.prod.record();
    }
}

//---------------------multiple pipe----------------------
template <uint8_t FlagID, FIFOType FiFoType, uint8_t FiFoDepth, uint8_t FiFoSyncT, typename TileDataProd,
          typename TileDataCons, bool EN_UNIT_FLAG = false, uint8_t LocalFiFoDepth = 2,
          VecCubeRatio VCRatio = VecCubeRatio::V2C1_VECS>
struct TMPipe {
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
            static_assert(DataFiFo::fifoType == FIFOType::CTRL_FIFO, "Fix: TPUSH has unsupported fifo type!");
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
            using GlobalData = GlobalTensor<T, pto::Shape<1, 1, 1, ConsM, ConsN>, pto::Stride<1, 1, 1, ConsN, 1>>;
            uint32_t bufIndex = static_cast<uint32_t>(tile_id % fifo.fifoDepth);
            size_t entryBase = bufIndex * ConsM * ProdN * sizeof(T);
            GlobalData globalTensor((__gm__ T *)((uint64_t)fifo.fifoBase + entryBase + entryOffset));

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
    PTO_INTERNAL explicit TMPipe(__gm__ typename TileDataCons::DType *gmFiFoBase, uint32_t localFiFoBase)
        : fifo(gmFiFoBase, localFiFoBase), prod(), cons()
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
    bool isAllocate = pipe.prod.getAllocateStatus();
    if (isAllocate) {
        pipe.prod.allocate();
    }

    // 2. Address Calculation
    pipe.prod.push(pipe.fifo, tile);
    pipe.prod.tile_id++;

    // 3. Cross-Core: Commit & Signal
    bool isRecord = pipe.prod.getRecordStatus();
    if (isRecord) {
        pipe.prod.record();
    }
}

} // namespace pto

#endif
