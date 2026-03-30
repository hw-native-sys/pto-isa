/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef TPUSH_HPP
#define TPUSH_HPP

#include <array>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <pto/common/fifo.hpp>

#include <pto/cpu/TAssign.hpp>
#include <pto/cpu/TLoad.hpp>
#include <pto/cpu/TMov.hpp>
#include <pto/cpu/TStore.hpp>
#include <pto/cpu/tile_offsets.hpp>

namespace pto {

namespace cpu_pipe {
template <TileSplitAxis Split>
PTO_INTERNAL constexpr uint32_t GetSplitCount()
{
    return (Split == TileSplitAxis::TILE_NO_SPLIT) ? 1u : 2u;
}

template <typename TileData>
PTO_INTERNAL void FillTile(TileData &tile, typename TileData::DType value)
{
    for (int r = 0; r < tile.GetValidRow(); ++r) {
        for (int c = 0; c < tile.GetValidCol(); ++c) {
            tile.data()[GetTileElementOffset<TileData>(r, c)] = value;
        }
    }
}

template <typename DstTileData, typename SrcTileData>
PTO_INTERNAL void CopyTileWindow(DstTileData &dst, SrcTileData &src, uint32_t rowOffset = 0, uint32_t colOffset = 0)
{
    for (int r = 0; r < dst.GetValidRow(); ++r) {
        for (int c = 0; c < dst.GetValidCol(); ++c) {
            dst.data()[GetTileElementOffset<DstTileData>(r, c)] =
                src.data()[GetTileElementOffset<SrcTileData>(r + rowOffset, c + colOffset)];
        }
    }
}

template <typename DstTileData, typename SrcTileData>
PTO_INTERNAL void InsertTileWindow(DstTileData &dst, SrcTileData &src, uint32_t rowOffset = 0, uint32_t colOffset = 0)
{
    for (int r = 0; r < src.GetValidRow(); ++r) {
        for (int c = 0; c < src.GetValidCol(); ++c) {
            dst.data()[GetTileElementOffset<DstTileData>(r + rowOffset, c + colOffset)] =
                src.data()[GetTileElementOffset<SrcTileData>(r, c)];
        }
    }
}

template <typename T, typename SrcTileData>
PTO_INTERNAL void CopyTileWindowToLinear(T *dst, uint32_t dstCols, SrcTileData &src, uint32_t dstRows,
                                         uint32_t srcRowOffset = 0, uint32_t srcColOffset = 0)
{
    for (uint32_t r = 0; r < dstRows; ++r) {
        for (uint32_t c = 0; c < dstCols; ++c) {
            dst[r * dstCols + c] = src.data()[GetTileElementOffset<SrcTileData>(r + srcRowOffset, c + srcColOffset)];
        }
    }
}

template <typename DstTileData, typename T>
PTO_INTERNAL void CopyLinearToTile(DstTileData &dst, const T *src, uint32_t srcCols)
{
    for (int r = 0; r < dst.GetValidRow(); ++r) {
        for (int c = 0; c < dst.GetValidCol(); ++c) {
            dst.data()[GetTileElementOffset<DstTileData>(r, c)] = src[r * srcCols + c];
        }
    }
}

template <TileSplitAxis Split, typename TileData>
PTO_INTERNAL uint32_t GetSplitRowOffset()
{
    if constexpr (Split == TileSplitAxis::TILE_UP_DOWN) {
        return static_cast<uint32_t>(get_subblockid()) * (TileData::Rows / 2);
    }
    return 0;
}

template <TileSplitAxis Split, typename TileData>
PTO_INTERNAL uint32_t GetSplitColOffset()
{
    if constexpr (Split == TileSplitAxis::TILE_LEFT_RIGHT) {
        return static_cast<uint32_t>(get_subblockid()) * (TileData::Cols / 2);
    }
    return 0;
}
} // namespace cpu_pipe

template <uint8_t FlagID, uint8_t DirType, uint32_t SlotSize, uint32_t SlotNum, uint32_t LocalSlotNum = 2,
          bool EN_UNIT_FLAG = false>
struct TPipe {
    static constexpr uint8_t DIR_MASK = 0x7;
    static constexpr uint8_t DIR_TYPE = DIR_MASK & DirType;
    static constexpr bool is_c2v = ((DIR_TYPE & Direction::DIR_C2V) == Direction::DIR_C2V);
    static constexpr bool is_v2c = ((DIR_TYPE & Direction::DIR_V2C) == Direction::DIR_V2C);
    static constexpr bool is_v2c_ctrl = ((DIR_TYPE & Direction::DIR_V2C_CTRL) == Direction::DIR_V2C_CTRL);
    static constexpr uint8_t VEC_CORE_ID_OFFSET = 16;
    using RingFiFo = RingFIFO<SlotSize, SlotNum, LocalSlotNum>;
    static constexpr uint32_t LOCAL_SPLIT_COPIES = is_c2v ? 2u : 1u;
    static constexpr uint32_t LOCAL_SLOT_STORAGE_SIZE = SlotSize * LOCAL_SPLIT_COPIES;

    struct SharedState {
        std::mutex mutex;
        std::condition_variable cv;
        int next_producer_slot = 0;
        int next_consumer_slot = 0;
        int occupied = 0;
        std::array<std::array<uint8_t, LOCAL_SLOT_STORAGE_SIZE>, SlotNum> local_slot_storage{};
        std::array<uint32_t, SlotNum> remaining_consumers{};
    };

    inline static SharedState shared_state{};

    PTO_INTERNAL static void reset_for_cpu_sim()
    {
        std::lock_guard<std::mutex> lock(shared_state.mutex);
        shared_state.next_producer_slot = 0;
        shared_state.next_consumer_slot = 0;
        shared_state.occupied = 0;
        for (auto &slot : shared_state.local_slot_storage) {
            slot.fill(0);
        }
        shared_state.remaining_consumers.fill(0);
        shared_state.cv.notify_all();
    }

    struct Producer {
        int tileIndex = 0;
        int subTileIndex = 0;
        bool isAllocate = true;
        bool isRecord = true;
        int entryOffset = 0;

        PTO_INTERNAL Producer() = default;

        PTO_INTERNAL void setTileId(int tIndex, int subIndex)
        {
            tileIndex = tIndex;
            subTileIndex = subIndex;
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

        template <TileSplitAxis Split = TileSplitAxis::TILE_UP_DOWN>
        PTO_INTERNAL void allocate()
        {
            (void)Split;
            std::unique_lock<std::mutex> lock(TPipe::shared_state.mutex);
            TPipe::shared_state.cv.wait(lock, []() { return TPipe::shared_state.occupied < RingFiFo::SLOT_NUM; });
            tileIndex = TPipe::shared_state.next_producer_slot;
            subTileIndex = static_cast<int>(get_subblockid());
        }

        template <TileSplitAxis Split = TileSplitAxis::TILE_UP_DOWN>
        PTO_INTERNAL void record()
        {
            (void)Split;
            {
                std::lock_guard<std::mutex> lock(TPipe::shared_state.mutex);
                if constexpr (TPipe::is_c2v && Split != TileSplitAxis::TILE_NO_SPLIT) {
                    TPipe::shared_state.remaining_consumers[static_cast<std::size_t>(tileIndex % RingFiFo::SLOT_NUM)] =
                        cpu_pipe::GetSplitCount<Split>();
                } else {
                    TPipe::shared_state.remaining_consumers[static_cast<std::size_t>(tileIndex % RingFiFo::SLOT_NUM)] = 1;
                }
                TPipe::shared_state.next_producer_slot = (tileIndex + 1) % RingFiFo::SLOT_NUM;
                ++TPipe::shared_state.occupied;
            }
            TPipe::shared_state.cv.notify_all();
        }
    };

    struct Consumer {
        int tileIndex = 0;
        int subTileIndex = 0;
        bool isWait = true;
        bool isFree = true;
        int entryOffset = 0;

        PTO_INTERNAL Consumer() = default;

        PTO_INTERNAL void setTileId(int tid, int subTid)
        {
            tileIndex = tid;
            subTileIndex = subTid;
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

        PTO_INTERNAL void setentryOffset(int offset)
        {
            entryOffset = offset;
        }

        template <TileSplitAxis Split = TileSplitAxis::TILE_UP_DOWN>
        PTO_INTERNAL void wait()
        {
            (void)Split;
            std::unique_lock<std::mutex> lock(TPipe::shared_state.mutex);
            TPipe::shared_state.cv.wait(lock, []() { return TPipe::shared_state.occupied > 0; });
            tileIndex = TPipe::shared_state.next_consumer_slot;
            subTileIndex = static_cast<int>(get_subblockid());
        }

        template <TileSplitAxis Split = TileSplitAxis::TILE_UP_DOWN>
        PTO_INTERNAL void free()
        {
            (void)Split;
            {
                std::lock_guard<std::mutex> lock(TPipe::shared_state.mutex);
                const auto slotIndex = static_cast<std::size_t>(tileIndex % RingFiFo::SLOT_NUM);
                auto &remaining = TPipe::shared_state.remaining_consumers[slotIndex];
                if (remaining > 1) {
                    --remaining;
                } else {
                    remaining = 0;
                    TPipe::shared_state.next_consumer_slot = (tileIndex + 1) % RingFiFo::SLOT_NUM;
                    --TPipe::shared_state.occupied;
                }
            }
            TPipe::shared_state.cv.notify_all();
        }
    };

    RingFiFo fifo;
    Producer prod;
    Consumer cons;

    PTO_INTERNAL explicit TPipe(__gm__ void *gmSlotBuffer, uint32_t c2vConsumerBuf, uint32_t v2cConsumerBuf)
        : fifo(gmSlotBuffer, c2vConsumerBuf, v2cConsumerBuf), prod(), cons()
    {}
};

template <typename Pipe, typename TileProd, TileSplitAxis Split>
PTO_INTERNAL void TPUSH_IMPL(Pipe &pipe, TileProd &tile)
{
    if (pipe.prod.getAllocateStatus()) {
        pipe.prod.template allocate<Split>();
    }

    const std::size_t slotIndex = static_cast<std::size_t>(pipe.prod.getTileId() % Pipe::RingFiFo::SLOT_NUM);
    const std::size_t entryBase =
        slotIndex * Pipe::RingFiFo::SLOT_SIZE + static_cast<std::size_t>(pipe.prod.entryOffset);

    if (pipe.fifo.GM_SLOT_BUFFER != nullptr) {
        using T = typename TileProd::DType;
        constexpr int rows = TileProd::Rows;
        constexpr int cols = TileProd::Cols;
        std::size_t subOffset = 0;
        if constexpr (Split != TileSplitAxis::TILE_NO_SPLIT) {
            subOffset = static_cast<std::size_t>(get_subblockid()) * rows * cols * sizeof(T);
        }
        using GlobalData = GlobalTensor<T, Shape<1, 1, 1, rows, cols>, Stride<1, 1, 1, cols, 1>>;
        auto *addr = reinterpret_cast<__gm__ T *>(reinterpret_cast<std::uintptr_t>(pipe.fifo.GM_SLOT_BUFFER) +
                                                  entryBase + subOffset);
        GlobalData globalData(addr);
        TSTORE_IMPL(globalData, tile);
    } else if constexpr (Pipe::is_c2v) {
        using T = typename TileProd::DType;
        constexpr int consRows =
            (Split == TileSplitAxis::TILE_UP_DOWN) ? (TileProd::Rows / 2) : static_cast<int>(TileProd::Rows);
        constexpr int consCols =
            (Split == TileSplitAxis::TILE_LEFT_RIGHT) ? (TileProd::Cols / 2) : static_cast<int>(TileProd::Cols);
        if constexpr (Split == TileSplitAxis::TILE_NO_SPLIT) {
            using SlotTile = Tile<TileType::Vec, T, consRows, consCols, BLayout::RowMajor, consRows, consCols>;

            SlotTile slotTile;
            TASSIGN(slotTile, static_cast<uint64_t>(pipe.fifo.C2V_CONSUMER_BUF + entryBase));
            cpu_pipe::CopyTileWindow(slotTile, tile, 0, 0);
        } else {
            auto &slotStorage = Pipe::shared_state.local_slot_storage[slotIndex];
            for (uint32_t splitIndex = 0; splitIndex < cpu_pipe::GetSplitCount<Split>(); ++splitIndex) {
                auto *slotPtr = reinterpret_cast<T *>(slotStorage.data() +
                                                      splitIndex * Pipe::RingFiFo::SLOT_SIZE + pipe.prod.entryOffset);
                const uint32_t rowOffset = (Split == TileSplitAxis::TILE_UP_DOWN) ? splitIndex * consRows : 0;
                const uint32_t colOffset = (Split == TileSplitAxis::TILE_LEFT_RIGHT) ? splitIndex * consCols : 0;
                cpu_pipe::CopyTileWindowToLinear(slotPtr, consCols, tile, consRows, rowOffset, colOffset);
            }
        }
    } else if constexpr (Pipe::is_v2c) {
        using T = typename TileProd::DType;
        constexpr int consRows =
            (Split == TileSplitAxis::TILE_UP_DOWN) ? (TileProd::Rows * 2) : static_cast<int>(TileProd::Rows);
        constexpr int consCols =
            (Split == TileSplitAxis::TILE_LEFT_RIGHT) ? (TileProd::Cols * 2) : static_cast<int>(TileProd::Cols);
        using SlotTile = Tile<TileType::Mat, T, consRows, consCols, BLayout::RowMajor, consRows, consCols>;

        SlotTile slotTile;
        TASSIGN(slotTile, static_cast<uint64_t>(pipe.fifo.V2C_CONSUMER_BUF + entryBase));
        cpu_pipe::FillTile(slotTile, static_cast<T>(0));
        cpu_pipe::InsertTileWindow(slotTile, tile, cpu_pipe::GetSplitRowOffset<Split, SlotTile>(),
                                   cpu_pipe::GetSplitColOffset<Split, SlotTile>());
    }

    if (pipe.prod.getRecordStatus()) {
        pipe.prod.template record<Split>();
    }
}

template <typename TileProd, typename Pipe>
PTO_INTERNAL void TPUSH_IMPL(TileProd &tile, Pipe &pipe)
{
    TPUSH_IMPL<Pipe, TileProd, TileSplitAxis::TILE_NO_SPLIT>(pipe, tile);
}

} // namespace pto

#endif
