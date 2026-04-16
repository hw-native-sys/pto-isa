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

#include <atomic>
#include <array>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <new>
#include <thread>
#include <type_traits>
#ifdef __CPU_SIM
#include <unordered_map>
#include <vector>
#endif
#include <format>
#include <pto/common/fifo.hpp>

#include <pto/cpu/TAssign.hpp>
#include <pto/cpu/TLoad.hpp>
#include <pto/cpu/TMov.hpp>
#include <pto/cpu/TStore.hpp>
#include <pto/cpu/tile_offsets.hpp>

namespace pto {

namespace cpu_pipe {
enum class TransferDir : uint8_t
{
    None = 0,
    C2V = 1,
    V2C = 2,
};

template <typename TileProd>
PTO_INTERNAL constexpr bool IsC2VProducerTile()
{
    return TileProd::Loc == TileType::Acc;
}

template <typename TileProd>
PTO_INTERNAL constexpr bool IsV2CProducerTile()
{
    return TileProd::Loc == TileType::Vec;
}

template <typename TileCons>
PTO_INTERNAL constexpr bool IsC2VConsumerTile()
{
    return TileCons::Loc == TileType::Vec;
}

template <typename Pipe>
PTO_INTERNAL constexpr TransferDir GetPipeTransferDir()
{
    if constexpr (Pipe::is_c2v && !Pipe::is_v2c) {
        return TransferDir::C2V;
    }
    if constexpr (Pipe::is_v2c && !Pipe::is_c2v) {
        return TransferDir::V2C;
    }
    return TransferDir::None;
}

template <typename Pipe, typename TileProd>
PTO_INTERNAL constexpr TransferDir GetProducerTransferDir()
{
    constexpr auto pipeDir = GetPipeTransferDir<Pipe>();
    if constexpr (pipeDir != TransferDir::None) {
        return pipeDir;
    }
    if constexpr (IsC2VProducerTile<TileProd>()) {
        return TransferDir::C2V;
    }
    return TransferDir::V2C;
}

template <typename Pipe, typename TileCons>
PTO_INTERNAL constexpr TransferDir GetConsumerTransferDir()
{
    constexpr auto pipeDir = GetPipeTransferDir<Pipe>();
    if constexpr (pipeDir != TransferDir::None) {
        return pipeDir;
    }
    if constexpr (IsC2VConsumerTile<TileCons>()) {
        return TransferDir::C2V;
    }
    return TransferDir::V2C;
}

template <TileSplitAxis Split>
PTO_INTERNAL constexpr uint32_t GetSplitCount()
{
    return (Split == TileSplitAxis::TILE_NO_SPLIT) ? 1u : 2u;
}

template <TileSplitAxis Split>
PTO_INTERNAL uint32_t GetSplitLaneId()
{
    constexpr uint32_t splitCount = GetSplitCount<Split>();
    const uint32_t subblockId = get_subblockid();
    return (subblockId < splitCount) ? subblockId : (splitCount - 1);
}

template <TileSplitAxis Split>
PTO_INTERNAL constexpr uint32_t GetSplitLaneMask(uint32_t laneId)
{
    return 1u << laneId;
}

template <TileSplitAxis Split>
PTO_INTERNAL constexpr uint32_t GetAllSplitLaneMask()
{
    return (1u << GetSplitCount<Split>()) - 1u;
}

template <TileSplitAxis Split>
PTO_INTERNAL uint32_t GetActiveSplitCount()
{
    constexpr uint32_t splitCount = GetSplitCount<Split>();
    if ((cpu_sim::injected_subblock_id_hook != nullptr) || (cpu_sim::injected_pipe_shared_state_hook != nullptr) ||
        (cpu_sim::ResolveSubblockIdHook() != nullptr) || (cpu_sim::ResolvePipeSharedStateHook() != nullptr)) {
        return splitCount;
    }
    const uint32_t subblockDim = get_subblockdim();
    if (subblockDim != 0) {
        return (subblockDim < splitCount) ? subblockDim : splitCount;
    }
    return 1u;
}

template <TileSplitAxis Split>
PTO_INTERNAL uint32_t GetActiveSplitLaneMask()
{
    return (1u << GetActiveSplitCount<Split>()) - 1u;
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

template <typename T>
PTO_INTERNAL void FillLinearRegion(T *dst, uint32_t dstCols, T value, uint32_t rowStart, uint32_t rowCount,
                                   uint32_t colStart, uint32_t colCount)
{
    for (uint32_t r = rowStart; r < rowStart + rowCount; ++r) {
        for (uint32_t c = colStart; c < colStart + colCount; ++c) {
            dst[r * dstCols + c] = value;
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
PTO_INTERNAL void InsertTileWindowToLinear(T *dst, uint32_t dstCols, SrcTileData &src, uint32_t rowOffset = 0,
                                           uint32_t colOffset = 0)
{
    for (int r = 0; r < src.GetValidRow(); ++r) {
        for (int c = 0; c < src.GetValidCol(); ++c) {
            dst[(r + rowOffset) * dstCols + (c + colOffset)] = src.data()[GetTileElementOffset<SrcTileData>(r, c)];
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

#ifdef __CPU_SIM
template <typename TileData>
PTO_INTERNAL void EnsureTileStorage(TileData &tile)
{
    using TilePtr = std::remove_reference_t<decltype(tile.data())>;
    static_assert(std::is_pointer_v<TilePtr>, "CPU-sim tile backing helper requires pointer-backed tile storage.");

    if (tile.data() != nullptr) {
        return;
    }

    static thread_local std::unordered_map<const void *, std::vector<typename TileData::DType>> buffers;
    auto &buffer = buffers[static_cast<const void *>(&tile)];
    const auto numel = static_cast<std::size_t>(TileData::Rows * TileData::Cols);
    if (buffer.size() != numel) {
        buffer.resize(numel);
    }
    tile.data() = buffer.data();
}
#else
template <typename TileData>
PTO_INTERNAL void EnsureTileStorage(TileData &tile)
{
    (void)tile;
}
#endif

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
        std::array<cpu_pipe::TransferDir, SlotNum> transfer_dirs{};
        std::array<uint32_t, SlotNum> remaining_consumers{};
        std::array<uint32_t, SlotNum> consumers_claimed{};
        std::array<uint32_t, SlotNum> producers_allocated{};
        std::array<uint32_t, SlotNum> producers_done{};
    };

    struct SharedStateStorage {
        std::atomic<uint32_t> init_state{0};
        alignas(SharedState) unsigned char payload[sizeof(SharedState)]{};
    };

    PTO_INTERNAL static void EnsureSharedStateInitialized(SharedStateStorage &storage)
    {
        uint32_t expected = 0;
        if (storage.init_state.compare_exchange_strong(expected, 1, std::memory_order_acq_rel)) {
            new (storage.payload) SharedState();
            storage.init_state.store(2, std::memory_order_release);
            return;
        }
        while (storage.init_state.load(std::memory_order_acquire) != 2) {
            std::this_thread::yield();
        }
    }

    PTO_INTERNAL static SharedState &GetSharedState()
    {
        constexpr uint64_t pipeKey = (static_cast<uint64_t>(FlagID) << 56) | (static_cast<uint64_t>(DirType) << 48) |
                                     (static_cast<uint64_t>(SlotNum) << 40) |
                                     (static_cast<uint64_t>(LocalSlotNum) << 32) | static_cast<uint64_t>(SlotSize);
        if (cpu_sim::injected_pipe_shared_state_hook != nullptr) {
            auto *storage = reinterpret_cast<SharedStateStorage *>(
                cpu_sim::injected_pipe_shared_state_hook(pipeKey, sizeof(SharedStateStorage)));
            if (storage != nullptr) {
                EnsureSharedStateInitialized(*storage);
                return *std::launder(reinterpret_cast<SharedState *>(storage->payload));
            }
        }
        if (auto hook = cpu_sim::ResolvePipeSharedStateHook(); hook != nullptr) {
            auto *storage = reinterpret_cast<SharedStateStorage *>(hook(pipeKey, sizeof(SharedStateStorage)));
            if (storage != nullptr) {
                EnsureSharedStateInitialized(*storage);
                return *std::launder(reinterpret_cast<SharedState *>(storage->payload));
            }
        }
        if (auto hook = cpu_sim::ResolveSharedStorageHook(); hook != nullptr) {
            char key[128] = {};
            std::format_to(key, "pto-pipe-%llu-%u-%u-%u-%u-%u-%u", static_cast<unsigned long long>(get_task_cookie()),
                           get_block_idx(), FlagID, DirType, SlotSize, SlotNum, LocalSlotNum);
            auto *storage = reinterpret_cast<SharedStateStorage *>(hook(key, sizeof(SharedStateStorage)));
            EnsureSharedStateInitialized(*storage);
            return *std::launder(reinterpret_cast<SharedState *>(storage->payload));
        }

        static SharedStateStorage storage{};
        EnsureSharedStateInitialized(storage);
        return *std::launder(reinterpret_cast<SharedState *>(storage.payload));
    }

    PTO_INTERNAL static void reset_for_cpu_sim()
    {
        auto &shared_state = GetSharedState();
        std::lock_guard<std::mutex> lock(shared_state.mutex);
        shared_state.next_producer_slot = 0;
        shared_state.next_consumer_slot = 0;
        shared_state.occupied = 0;
        for (auto &slot : shared_state.local_slot_storage) {
            slot.fill(0);
        }
        shared_state.remaining_consumers.fill(0);
        shared_state.consumers_claimed.fill(0);
        shared_state.producers_allocated.fill(0);
        shared_state.producers_done.fill(0);
        shared_state.transfer_dirs.fill(cpu_pipe::TransferDir::None);
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

        template <typename TileProd, TileSplitAxis Split = TileSplitAxis::TILE_UP_DOWN>
        PTO_INTERNAL void allocate()
        {
            (void)Split;
            auto &shared_state = TPipe::GetSharedState();
            std::unique_lock<std::mutex> lock(shared_state.mutex);
            if constexpr (TPipe::is_v2c && cpu_pipe::IsV2CProducerTile<TileProd>() &&
                          Split != TileSplitAxis::TILE_NO_SPLIT) {
                const uint32_t laneId = cpu_pipe::GetSplitLaneId<Split>();
                const uint32_t laneMask = cpu_pipe::GetSplitLaneMask<Split>(laneId);
                shared_state.cv.wait(lock, [&shared_state, laneMask]() {
                    return shared_state.occupied < RingFiFo::SLOT_NUM &&
                           (shared_state
                                .producers_allocated[static_cast<std::size_t>(shared_state.next_producer_slot)] &
                            laneMask) == 0;
                });
                tileIndex = shared_state.next_producer_slot;
                shared_state.producers_allocated[static_cast<std::size_t>(tileIndex % RingFiFo::SLOT_NUM)] |= laneMask;
                subTileIndex = static_cast<int>(laneId);
                return;
            } else {
                shared_state.cv.wait(lock, [&shared_state]() { return shared_state.occupied < RingFiFo::SLOT_NUM; });
            }
            tileIndex = shared_state.next_producer_slot;
            subTileIndex = static_cast<int>(get_subblockid());
        }

        template <typename TileProd, TileSplitAxis Split = TileSplitAxis::TILE_UP_DOWN>
        PTO_INTERNAL void record()
        {
            (void)Split;
            auto &shared_state = TPipe::GetSharedState();
            {
                std::lock_guard<std::mutex> lock(shared_state.mutex);
                const auto slotIdx = static_cast<std::size_t>(tileIndex % RingFiFo::SLOT_NUM);
                shared_state.transfer_dirs[slotIdx] = cpu_pipe::GetProducerTransferDir<TPipe, TileProd>();
                if constexpr (TPipe::is_c2v && cpu_pipe::IsC2VProducerTile<TileProd>() &&
                              Split != TileSplitAxis::TILE_NO_SPLIT) {
                    shared_state.remaining_consumers[slotIdx] = cpu_pipe::GetSplitCount<Split>();
                } else {
                    shared_state.remaining_consumers[slotIdx] = 1;
                }
                if constexpr (TPipe::is_v2c && cpu_pipe::IsV2CProducerTile<TileProd>() &&
                              Split != TileSplitAxis::TILE_NO_SPLIT) {
                    const uint32_t laneMask = cpu_pipe::GetSplitLaneMask<Split>(static_cast<uint32_t>(subTileIndex));
                    shared_state.producers_done[slotIdx] |= laneMask;
                    if (shared_state.producers_done[slotIdx] != cpu_pipe::GetActiveSplitLaneMask<Split>()) {
                        return;
                    }
                    shared_state.producers_allocated[slotIdx] = 0;
                    shared_state.producers_done[slotIdx] = 0;
                }
                shared_state.next_producer_slot = (tileIndex + 1) % RingFiFo::SLOT_NUM;
                ++shared_state.occupied;
            }
            shared_state.cv.notify_all();
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

        template <typename TileCons, TileSplitAxis Split = TileSplitAxis::TILE_UP_DOWN>
        PTO_INTERNAL void wait()
        {
            (void)Split;
            auto &shared_state = TPipe::GetSharedState();
            std::unique_lock<std::mutex> lock(shared_state.mutex);
            constexpr auto expectedDir = cpu_pipe::GetConsumerTransferDir<TPipe, TileCons>();
            if constexpr (TPipe::is_c2v && cpu_pipe::IsC2VConsumerTile<TileCons>() &&
                          Split != TileSplitAxis::TILE_NO_SPLIT) {
                const uint32_t laneId = cpu_pipe::GetSplitLaneId<Split>();
                const uint32_t laneMask = cpu_pipe::GetSplitLaneMask<Split>(laneId);
                shared_state.cv.wait(lock, [&shared_state, laneMask, expectedDir]() {
                    return shared_state.occupied > 0 &&
                           shared_state.transfer_dirs[static_cast<std::size_t>(shared_state.next_consumer_slot)] ==
                               expectedDir &&
                           (shared_state.consumers_claimed[static_cast<std::size_t>(shared_state.next_consumer_slot)] &
                            laneMask) == 0;
                });
                tileIndex = shared_state.next_consumer_slot;
                shared_state.consumers_claimed[static_cast<std::size_t>(tileIndex % RingFiFo::SLOT_NUM)] |= laneMask;
                subTileIndex = static_cast<int>(laneId);
                return;
            }
            shared_state.cv.wait(lock, [&shared_state, expectedDir]() {
                return shared_state.occupied > 0 &&
                       shared_state.transfer_dirs[static_cast<std::size_t>(shared_state.next_consumer_slot)] ==
                           expectedDir;
            });
            tileIndex = shared_state.next_consumer_slot;
            subTileIndex = static_cast<int>(get_subblockid());
        }

        template <TileSplitAxis Split = TileSplitAxis::TILE_UP_DOWN>
        PTO_INTERNAL void free()
        {
            (void)Split;
            auto &shared_state = TPipe::GetSharedState();
            {
                std::lock_guard<std::mutex> lock(shared_state.mutex);
                const auto slotIndex = static_cast<std::size_t>(tileIndex % RingFiFo::SLOT_NUM);
                auto &remaining = shared_state.remaining_consumers[slotIndex];
                if (remaining > 1) {
                    --remaining;
                } else {
                    remaining = 0;
                    shared_state.consumers_claimed[slotIndex] = 0;
                    shared_state.transfer_dirs[slotIndex] = cpu_pipe::TransferDir::None;
                    shared_state.next_consumer_slot = (tileIndex + 1) % RingFiFo::SLOT_NUM;
                    --shared_state.occupied;
                }
            }
            shared_state.cv.notify_all();
        }

        template <typename TileCons, TileSplitAxis Split>
        PTO_INTERNAL void popTileFromVecFiFo(RingFiFo &fifo, TileCons &tile)
        {
            const std::size_t slotIndex = static_cast<std::size_t>(tileIndex % RingFiFo::SLOT_NUM);
            const std::size_t entryBase = slotIndex * RingFiFo::SLOT_SIZE + static_cast<std::size_t>(entryOffset);
            uint64_t localTileBase = fifo.C2V_CONSUMER_BUF + entryBase;
            TASSIGN_IMPL(tile, localTileBase);
        }

        template <typename TileCons, TileSplitAxis Split>
        PTO_INTERNAL void popTileFromVecFiFoSplit(RingFiFo &fifo, TileCons &tile)
        {
            using T = typename TileCons::DType;
            constexpr uint32_t splitCount = cpu_pipe::GetSplitCount<Split>();
            const uint32_t splitIndex = (get_subblockid() < splitCount) ? get_subblockid() : (splitCount - 1);
            const std::size_t slotIndex = static_cast<std::size_t>(tileIndex % RingFiFo::SLOT_NUM);
            const auto &slotStorage = TPipe::GetSharedState().local_slot_storage[slotIndex];
            const auto *slotPtr =
                reinterpret_cast<const T *>(slotStorage.data() + splitIndex * RingFiFo::SLOT_SIZE + entryOffset);
            cpu_pipe::EnsureTileStorage(tile);
            cpu_pipe::CopyLinearToTile(tile, slotPtr, static_cast<uint32_t>(tile.GetValidCol()));
        }

        template <typename TileCons, TileSplitAxis Split>
        PTO_INTERNAL void popTileFromMatFiFo(RingFiFo &fifo, TileCons &tile)
        {
            using T = typename TileCons::DType;
            constexpr int rows = TileCons::Rows;
            constexpr int cols = TileCons::Cols;
            using SlotTile = Tile<TileType::Mat, T, rows, cols, BLayout::RowMajor, rows, cols>;
            const std::size_t slotIndex = static_cast<std::size_t>(tileIndex % RingFiFo::SLOT_NUM);
            const std::size_t entryBase = slotIndex * RingFiFo::SLOT_SIZE + static_cast<std::size_t>(entryOffset);
            const auto &slotStorage = TPipe::GetSharedState().local_slot_storage[slotIndex];
            const auto *slotPtr = reinterpret_cast<const T *>(slotStorage.data() + entryOffset);

            SlotTile slotTile;
            TASSIGN_IMPL(slotTile, fifo.V2C_CONSUMER_BUF + entryBase);
            cpu_pipe::CopyLinearToTile(slotTile, slotPtr, static_cast<uint32_t>(slotTile.GetValidCol()));
            cpu_pipe::EnsureTileStorage(tile);
            TMOV_IMPL(tile, slotTile);
        }

        template <typename TileCons, TileSplitAxis Split>
        PTO_INTERNAL void popTileFromGMFiFo(RingFiFo &fifo, TileCons &tile)
        {
            using T = typename TileCons::DType;
            const std::size_t slotIndex = static_cast<std::size_t>(tileIndex % RingFiFo::SLOT_NUM);
            const std::size_t entryBase = slotIndex * RingFiFo::SLOT_SIZE + static_cast<std::size_t>(entryOffset);
            cpu_pipe::EnsureTileStorage(tile);
            if constexpr (TPipe::is_c2v && TileCons::Loc == TileType::Vec) {
                constexpr int splitNum = 2;
                constexpr int consRows = TileCons::Rows;
                constexpr int consCols = TileCons::Cols;
                constexpr int prodCols = (Split == TileSplitAxis::TILE_LEFT_RIGHT) ? consCols * splitNum : consCols;
                constexpr int gmValidR = consRows;
                constexpr int gmValidC = consCols;
                constexpr int gmStrideR = prodCols;
                std::size_t subOffset = 0;
                if constexpr (Split == TileSplitAxis::TILE_UP_DOWN) {
                    subOffset = static_cast<std::size_t>(get_subblockid()) * consRows * prodCols * sizeof(T);
                } else if constexpr (Split == TileSplitAxis::TILE_LEFT_RIGHT) {
                    subOffset = static_cast<std::size_t>(get_subblockid()) * consCols * sizeof(T);
                }
                using GlobalData = GlobalTensor<T, Shape<1, 1, 1, gmValidR, gmValidC>, Stride<1, 1, 1, gmStrideR, 1>>;
                auto *addr = reinterpret_cast<__gm__ T *>(reinterpret_cast<std::uintptr_t>(fifo.GM_SLOT_BUFFER) +
                                                          entryBase + subOffset);
                GlobalData globalData(addr);
                TLOAD_IMPL(tile, globalData);
                return;
            }

            constexpr int rows = TileCons::Rows;
            constexpr int cols = TileCons::Cols;
            using GlobalData = GlobalTensor<T, Shape<1, 1, 1, rows, cols>, Stride<1, 1, 1, cols, 1>>;
            auto *addr =
                reinterpret_cast<__gm__ T *>(reinterpret_cast<std::uintptr_t>(fifo.GM_SLOT_BUFFER) + entryBase);
            GlobalData globalData(addr);
            TLOAD_IMPL(tile, globalData);
        }

        template <typename TileCons, TileSplitAxis Split>
        PTO_INTERNAL bool pop(RingFiFo &fifo, TileCons &tile)
        {
            if (fifo.GM_SLOT_BUFFER != nullptr) {
                popTileFromGMFiFo<TileCons, Split>(fifo, tile);
                return true;
            } else if constexpr (TPipe::is_c2v) {
                if constexpr (Split == TileSplitAxis::TILE_NO_SPLIT) {
                    popTileFromVecFiFo<TileCons, Split>(fifo, tile);
                } else {
                    popTileFromVecFiFoSplit<TileCons, Split>(fifo, tile);
                }
                return false;
            } else if constexpr (TPipe::is_v2c) {
                popTileFromMatFiFo<TileCons, Split>(fifo, tile);
                return false;
            }
            return false;
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
PTO_INTERNAL void TPush_c2v(Pipe &pipe, TileProd &tile, size_t entryBase, size_t slotIndex)
{
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
        auto &slotStorage = Pipe::GetSharedState().local_slot_storage[slotIndex];
        for (uint32_t splitIndex = 0; splitIndex < cpu_pipe::GetSplitCount<Split>(); ++splitIndex) {
            auto *slotPtr = reinterpret_cast<T *>(slotStorage.data() + splitIndex * Pipe::RingFiFo::SLOT_SIZE +
                                                  pipe.prod.entryOffset);
            const uint32_t rowOffset = (Split == TileSplitAxis::TILE_UP_DOWN) ? splitIndex * consRows : 0;
            const uint32_t colOffset = (Split == TileSplitAxis::TILE_LEFT_RIGHT) ? splitIndex * consCols : 0;
            cpu_pipe::CopyTileWindowToLinear(slotPtr, consCols, tile, consRows, rowOffset, colOffset);
        }
    }
}

template <typename Pipe, typename TileProd, TileSplitAxis Split>
PTO_INTERNAL void TPush_v2c(Pipe &pipe, TileProd &tile, size_t entryBase)
{
    using T = typename TileProd::DType;
    constexpr int consRows =
        (Split == TileSplitAxis::TILE_UP_DOWN) ? (TileProd::Rows * 2) : static_cast<int>(TileProd::Rows);
    constexpr int consCols =
        (Split == TileSplitAxis::TILE_LEFT_RIGHT) ? (TileProd::Cols * 2) : static_cast<int>(TileProd::Cols);
    using SlotTile = Tile<TileType::Mat, T, consRows, consCols, BLayout::RowMajor, consRows, consCols>;
    (void)entryBase;
    const std::size_t slotIndex = static_cast<std::size_t>(pipe.prod.getTileId() % Pipe::RingFiFo::SLOT_NUM);
    auto &slotStorage = Pipe::GetSharedState().local_slot_storage[slotIndex];
    auto *slotPtr = reinterpret_cast<T *>(slotStorage.data() + pipe.prod.entryOffset);
    const uint32_t rowOff = cpu_pipe::GetSplitRowOffset<Split, SlotTile>();
    const uint32_t colOff = cpu_pipe::GetSplitColOffset<Split, SlotTile>();
    cpu_pipe::FillLinearRegion(slotPtr, static_cast<uint32_t>(consCols), static_cast<T>(0), rowOff,
                               static_cast<uint32_t>(TileProd::Rows), colOff, static_cast<uint32_t>(TileProd::Cols));
    cpu_pipe::InsertTileWindowToLinear(slotPtr, static_cast<uint32_t>(consCols), tile, rowOff, colOff);
}

template <typename Pipe, typename TileProd, TileSplitAxis Split>
PTO_INTERNAL void TPUSH_IMPL(Pipe &pipe, TileProd &tile)
{
    if (pipe.prod.getAllocateStatus()) {
        pipe.prod.template allocate<TileProd, Split>();
    }
    const std::size_t slotIndex = static_cast<std::size_t>(pipe.prod.getTileId() % Pipe::RingFiFo::SLOT_NUM);
    const std::size_t entryBase =
        slotIndex * Pipe::RingFiFo::SLOT_SIZE + static_cast<std::size_t>(pipe.prod.entryOffset);
    if (pipe.fifo.GM_SLOT_BUFFER != nullptr) {
        using T = typename TileProd::DType;
        constexpr int rows = TileProd::Rows;
        constexpr int cols = TileProd::Cols;
        if constexpr (Pipe::is_v2c && TileProd::Loc == TileType::Vec) {
            constexpr int gmStrideR = (Split == TileSplitAxis::TILE_LEFT_RIGHT) ? (cols * 2) : cols;
            std::size_t subOffset = 0;
            if constexpr (Split == TileSplitAxis::TILE_UP_DOWN) {
                subOffset = static_cast<std::size_t>(get_subblockid()) * rows * cols * sizeof(T);
            } else if constexpr (Split == TileSplitAxis::TILE_LEFT_RIGHT) {
                subOffset = static_cast<std::size_t>(get_subblockid()) * cols * sizeof(T);
            }
            using GlobalData = GlobalTensor<T, Shape<1, 1, 1, rows, cols>, Stride<1, 1, 1, gmStrideR, 1>>;
            auto *addr = reinterpret_cast<__gm__ T *>(reinterpret_cast<std::uintptr_t>(pipe.fifo.GM_SLOT_BUFFER) +
                                                      entryBase + subOffset);
            GlobalData globalData(addr);
            TSTORE_IMPL(globalData, tile);
        } else {
            using GlobalData = GlobalTensor<T, Shape<1, 1, 1, rows, cols>, Stride<1, 1, 1, cols, 1>>;
            auto *addr =
                reinterpret_cast<__gm__ T *>(reinterpret_cast<std::uintptr_t>(pipe.fifo.GM_SLOT_BUFFER) + entryBase);
            GlobalData globalData(addr);
            TSTORE_IMPL(globalData, tile);
        }
    } else if constexpr (Pipe::is_c2v) {
        TPush_c2v<Pipe, TileProd, Split>(pipe, tile, entryBase, slotIndex);
    } else if constexpr (Pipe::is_v2c) {
        TPush_v2c<Pipe, TileProd, Split>(pipe, tile, entryBase);
    }
    if (pipe.prod.getRecordStatus()) {
        pipe.prod.template record<TileProd, Split>();
    }
}

template <typename TileProd, typename Pipe>
PTO_INTERNAL void TPUSH_IMPL(TileProd &tile, Pipe &pipe)
{
    TPUSH_IMPL<Pipe, TileProd, TileSplitAxis::TILE_NO_SPLIT>(pipe, tile);
}

} // namespace pto

#endif
