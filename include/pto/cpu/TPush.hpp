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
#include <cstring>
#include <mutex>
#include <new>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <type_traits>
#include <unordered_map>
#include <vector>
#include <pto/common/fifo.hpp>
#include <pto/common/fixpipe.hpp>

#include <pto/cpu/TAssign.hpp>
#include <pto/cpu/TLoad.hpp>
#include <pto/cpu/TMov.hpp>
#include <pto/cpu/TStore.hpp>
#include <pto/cpu/tile_offsets.hpp>

namespace pto {

namespace cpu_pipe {

constexpr uint16_t PSTATE_INITIALIZED = 2;

enum class TransferDir : uint8_t {
    None = 0,
    C2V = 1,
    V2C = 2,
};

template <typename TileProd>
PTO_INTERNAL constexpr bool IsC2VProducerTile()
{
    using CleanTileProd = std::remove_cv_t<std::remove_reference_t<TileProd>>;
    if constexpr (is_global_data_v<CleanTileProd>) {
        return false;
    } else {
        return CleanTileProd::Loc == TileType::Acc || CleanTileProd::Loc == TileType::Mat;
    }
}

template <typename TileProd>
PTO_INTERNAL constexpr bool IsV2CProducerTile()
{
    using CleanTileProd = std::remove_cv_t<std::remove_reference_t<TileProd>>;
    if constexpr (is_global_data_v<CleanTileProd>) {
        return false;
    } else {
        return CleanTileProd::Loc == TileType::Vec;
    }
}

template <typename TileCons>
PTO_INTERNAL constexpr bool IsC2VConsumerTile()
{
    using CleanTileCons = std::remove_cv_t<std::remove_reference_t<TileCons>>;
    if constexpr (is_global_data_v<CleanTileCons>) {
        return false;
    } else {
        return CleanTileCons::Loc == TileType::Vec;
    }
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

template <typename TileData>
PTO_INTERNAL constexpr uint32_t GetThreadSubblockDim()
{
    using CleanTileData = std::remove_cv_t<std::remove_reference_t<TileData>>;
    if constexpr (is_global_data_v<CleanTileData>) {
        return 1u;
    } else {
        static_assert(
            is_tile_data_v<CleanTileData> || is_conv_tile_v<CleanTileData>,
            "GetThreadSubblockDim requires a Tile or ConvTile type.");
        constexpr uint32_t kVecSubblockDim = 2u;
        constexpr uint32_t kDefaultSubblockDim = 1u;
        return (CleanTileData::Loc == TileType::Vec) ? kVecSubblockDim : kDefaultSubblockDim;
    }
}

template <typename TileData, TileSplitAxis Split>
PTO_INTERNAL constexpr uint32_t GetActiveSplitCount()
{
    constexpr uint32_t splitCount = GetSplitCount<Split>();
    constexpr uint32_t subblockDim = GetThreadSubblockDim<TileData>();
    return (subblockDim < splitCount) ? subblockDim : splitCount;
}

template <typename TileData, TileSplitAxis Split>
PTO_INTERNAL constexpr uint32_t GetActiveSplitLaneMask()
{
    return (1u << GetActiveSplitCount<TileData, Split>()) - 1u;
}

template <typename TileData, TileSplitAxis Split>
PTO_INTERNAL bool IsInactiveNoSplitVecLane()
{
    using CleanTileData = std::remove_cv_t<std::remove_reference_t<TileData>>;
    if constexpr (is_global_data_v<CleanTileData>) {
        return false;
    } else {
        static_assert(
            is_tile_data_v<CleanTileData> || is_conv_tile_v<CleanTileData>,
            "IsInactiveNoSplitVecLane requires a Tile or ConvTile type.");
        if constexpr (Split != TileSplitAxis::TILE_NO_SPLIT) {
            return false;
        }
        if constexpr (CleanTileData::Loc != TileType::Vec) {
            return false;
        }
        return get_subblockid() != 0;
    }
}

template <typename Pipe, TileSplitAxis Split>
PTO_INTERNAL bool IsInactiveNoSplitVecConsumerLane()
{
    if constexpr (Split != TileSplitAxis::TILE_NO_SPLIT) {
        return false;
    }
    if constexpr (!Pipe::is_c2v) {
        return false;
    }
    return get_subblockid() != 0;
}

// Compile-time predicate for the C2V NO_SPLIT broadcast consumer/free path. Whether the second
// vector lane actually participates is decided at runtime by IsDualLaneC2VActive() below; this
// predicate only selects the code path that honours that decision.
template <typename Pipe, typename TileCons, TileSplitAxis Split>
PTO_INTERNAL constexpr bool ShouldNoSplitC2VConsumerLaneParticipate()
{
    return Split == TileSplitAxis::TILE_NO_SPLIT && Pipe::is_c2v && Pipe::is_no_split && IsC2VConsumerTile<TileCons>();
}

template <typename Pipe, TileSplitAxis Split>
PTO_INTERNAL constexpr bool ShouldNoSplitC2VConsumerLaneFree()
{
    return Split == TileSplitAxis::TILE_NO_SPLIT && Pipe::is_c2v && Pipe::is_no_split;
}

// Runtime gate for the dual-lane (two-vector-subblock) C2V NO_SPLIT protocol. It is active only
// when the runtime reports two vector subblocks; otherwise the pipe runs as a single consumer
// (the second lane is treated as inactive). The consumer count is taken from get_subblockdim()
// rather than assumed, so producer and consumer agree on however many lanes are actually present.
PTO_INTERNAL bool IsDualLaneC2VActive() { return get_subblockdim() >= 2u; }

template <typename Pipe, typename TileProd, TileSplitAxis Split>
PTO_INTERNAL constexpr uint32_t GetRequiredConsumerCount()
{
    if constexpr (
        Pipe::is_c2v && Pipe::is_no_split && IsC2VProducerTile<TileProd>() && Split == TileSplitAxis::TILE_NO_SPLIT) {
        // Commit the slot for a single consumer by default; the producer does not observe how many
        // vector subblocks consume it. When the dual-lane protocol is active, the first consumer
        // lane raises remaining_consumers to the actual subblock count in wait() before any free.
        return 1u;
    } else if constexpr (Pipe::is_c2v && IsC2VProducerTile<TileProd>()) {
        return GetSplitCount<Split>();
    } else {
        return 1u;
    }
}

template <std::size_t SlotNum, typename RejectSlotFn>
PTO_INTERNAL bool FindNextTransferSlot(
    const std::array<TransferDir, SlotNum>& transferDirs, int start, TransferDir expectedDir, int& slotIndex,
    RejectSlotFn rejectSlot)
{
    for (std::size_t offset = 0; offset < SlotNum; ++offset) {
        const int candidate = static_cast<int>((static_cast<std::size_t>(start) + offset) % SlotNum);
        if (transferDirs[static_cast<std::size_t>(candidate)] == expectedDir && !rejectSlot(candidate)) {
            slotIndex = candidate;
            return true;
        }
    }
    return false;
}

// A DIR_BOTH pipe shares one producer cursor while each consumer keeps its own, so cursors cannot
// order the directions; select by producer commit order instead.
template <std::size_t SlotNum, typename RejectSlotFn>
PTO_INTERNAL bool FindOldestTransferSlot(
    const std::array<TransferDir, SlotNum>& transferDirs, const std::array<uint64_t, SlotNum>& commitSeq,
    TransferDir expectedDir, int& slotIndex, RejectSlotFn rejectSlot)
{
    bool found = false;
    uint64_t bestSeq = 0;
    for (std::size_t i = 0; i < SlotNum; ++i) {
        if (transferDirs[i] != expectedDir || commitSeq[i] == 0) {
            continue;
        }
        if (rejectSlot(static_cast<int>(i))) {
            continue;
        }
        if (!found || commitSeq[i] < bestSeq) {
            found = true;
            bestSeq = commitSeq[i];
            slotIndex = static_cast<int>(i);
        }
    }
    return found;
}

template <std::size_t SlotNum>
PTO_INTERNAL bool FindNextTransferSlot(
    const std::array<TransferDir, SlotNum>& transferDirs, int start, TransferDir expectedDir, int& slotIndex)
{
    return FindNextTransferSlot(transferDirs, start, expectedDir, slotIndex, [](int) { return false; });
}

template <std::size_t SlotNum>
PTO_INTERNAL void PushPendingSlot(std::array<int, SlotNum>& slots, int& count, int slotIndex)
{
    if (count < 0 || static_cast<std::size_t>(count) >= SlotNum) {
        return;
    }
    slots[static_cast<std::size_t>(count)] = slotIndex;
    ++count;
}

template <std::size_t SlotNum>
PTO_INTERNAL bool PopPendingSlot(std::array<int, SlotNum>& slots, int& count, int& slotIndex)
{
    if (count <= 0 || static_cast<std::size_t>(count) > SlotNum) {
        return false;
    }
    slotIndex = slots[0];
    for (int i = 1; i < count; ++i) {
        slots[static_cast<std::size_t>(i - 1)] = slots[static_cast<std::size_t>(i)];
    }
    --count;
    return true;
}

// `popped_slots` interleaves both directions on a DIR_BOTH pipe, so a consumer drops its own slot
// by value rather than the head.
template <std::size_t SlotNum>
PTO_INTERNAL bool ErasePendingSlot(std::array<int, SlotNum>& slots, int& count, int slotIndex)
{
    if (count <= 0 || static_cast<std::size_t>(count) > SlotNum) {
        return false;
    }
    int found = -1;
    for (int i = 0; i < count; ++i) {
        if (slots[static_cast<std::size_t>(i)] == slotIndex) {
            found = i;
            break;
        }
    }
    if (found < 0) {
        return false;
    }
    for (int i = found + 1; i < count; ++i) {
        slots[static_cast<std::size_t>(i - 1)] = slots[static_cast<std::size_t>(i)];
    }
    --count;
    return true;
}

template <typename TileData>
PTO_INTERNAL void FillTile(TileData& tile, typename TileData::DType value)
{
    for (int r = 0; r < tile.GetValidRow(); ++r) {
        for (int c = 0; c < tile.GetValidCol(); ++c) {
            tile.data()[GetTileElementOffset<TileData>(r, c)] = value;
        }
    }
}

template <typename T, std::size_t StorageSize>
PTO_INTERNAL std::size_t GetCheckedByteStorageOffset(
    const std::array<uint8_t, StorageSize>& storage, std::size_t slotIndex, std::size_t baseByteOffset,
    std::size_t regionByteEnd, std::size_t elementIndex, const char* operation)
{
    (void)storage;
    static_assert(std::is_trivially_copyable_v<T>, "CPU TPipe byte storage requires a trivially copyable type.");
    if (baseByteOffset > regionByteEnd || regionByteEnd > StorageSize ||
        elementIndex >= (regionByteEnd - baseByteOffset) / sizeof(T)) {
        std::ostringstream message;
        message << "CPU TPipe " << operation << " exceeds local slot storage: slot=" << slotIndex
                << ", base_byte_offset=" << baseByteOffset << ", element_index=" << elementIndex
                << ", element_size=" << sizeof(T) << ", region_byte_end=" << regionByteEnd
                << ", storage_size=" << StorageSize;
        throw std::out_of_range(message.str());
    }
    return baseByteOffset + elementIndex * sizeof(T);
}

template <typename T, std::size_t StorageSize>
PTO_INTERNAL T LoadByteStorageElement(
    const std::array<uint8_t, StorageSize>& storage, std::size_t slotIndex, std::size_t baseByteOffset,
    std::size_t regionByteEnd, std::size_t elementIndex, const char* operation)
{
    const std::size_t byteOffset =
        GetCheckedByteStorageOffset<T>(storage, slotIndex, baseByteOffset, regionByteEnd, elementIndex, operation);
    T value;
    std::memcpy(&value, storage.data() + byteOffset, sizeof(T));
    return value;
}

template <typename T, std::size_t StorageSize>
PTO_INTERNAL void StoreByteStorageElement(
    std::array<uint8_t, StorageSize>& storage, std::size_t slotIndex, std::size_t baseByteOffset,
    std::size_t regionByteEnd, std::size_t elementIndex, T value, const char* operation)
{
    const std::size_t byteOffset =
        GetCheckedByteStorageOffset<T>(storage, slotIndex, baseByteOffset, regionByteEnd, elementIndex, operation);
    std::memcpy(storage.data() + byteOffset, &value, sizeof(T));
}

template <typename T, std::size_t StorageSize>
PTO_INTERNAL void FillLinearRegion(
    std::array<uint8_t, StorageSize>& dst, std::size_t slotIndex, std::size_t baseByteOffset, std::size_t regionByteEnd,
    uint32_t dstCols, T value, uint32_t rowStart, uint32_t rowCount, uint32_t colStart, uint32_t colCount)
{
    for (uint32_t r = rowStart; r < rowStart + rowCount; ++r) {
        for (uint32_t c = colStart; c < colStart + colCount; ++c) {
            StoreByteStorageElement(
                dst, slotIndex, baseByteOffset, regionByteEnd, static_cast<std::size_t>(r) * dstCols + c, value,
                "FillLinearRegion");
        }
    }
}

template <typename DstTileData, typename SrcTileData>
PTO_INTERNAL void CopyTileWindow(DstTileData& dst, SrcTileData& src, uint32_t rowOffset = 0, uint32_t colOffset = 0)
{
    for (int r = 0; r < dst.GetValidRow(); ++r) {
        for (int c = 0; c < dst.GetValidCol(); ++c) {
            dst.data()[GetTileElementOffset<DstTileData>(r, c)] =
                src.data()[GetTileElementOffset<SrcTileData>(r + rowOffset, c + colOffset)];
        }
    }
}

template <typename DstTileData, typename SrcTileData>
PTO_INTERNAL void InsertTileWindow(DstTileData& dst, SrcTileData& src, uint32_t rowOffset = 0, uint32_t colOffset = 0)
{
    for (int r = 0; r < src.GetValidRow(); ++r) {
        for (int c = 0; c < src.GetValidCol(); ++c) {
            dst.data()[GetTileElementOffset<DstTileData>(r + rowOffset, c + colOffset)] =
                src.data()[GetTileElementOffset<SrcTileData>(r, c)];
        }
    }
}

template <typename DstT, typename SrcTileData, QuantMode_t quantMode, ReluPreMode reluPreMode, std::size_t StorageSize>
PTO_INTERNAL void CopyTileWindowToLinear(
    std::array<uint8_t, StorageSize>& dst, std::size_t slotIndex, std::size_t baseByteOffset, std::size_t regionByteEnd,
    uint32_t dstRows, uint32_t dstCols, SrcTileData& src, uint32_t srcRowOffset, uint32_t srcColOffset,
    const std::vector<uint64_t>& scalars = {})
{
    using SrcT = typename SrcTileData::DType;
    constexpr bool use_relu = reluPreMode == ReluPreMode::NormalRelu;
    for (uint32_t r = 0; r < dstRows; ++r) {
        for (uint32_t c = 0; c < dstCols; ++c) {
            SrcT val = src.data()[GetTileElementOffset<SrcTileData>(r + srcRowOffset, c + srcColOffset)];
            StoreByteStorageElement(
                dst, slotIndex, baseByteOffset, regionByteEnd, static_cast<std::size_t>(r) * dstCols + c,
                ConvertStoreValue<DstT, SrcT, quantMode, use_relu>(val, scalars[c]), "CopyTileWindowToLinear");
        }
    }
}

template <typename DstTileData, typename T, std::size_t StorageSize>
PTO_INTERNAL void CopyLinearToTile(
    DstTileData& dst, const std::array<uint8_t, StorageSize>& src, std::size_t slotIndex, std::size_t baseByteOffset,
    std::size_t regionByteEnd, uint32_t srcCols)
{
    for (int r = 0; r < dst.GetValidRow(); ++r) {
        for (int c = 0; c < dst.GetValidCol(); ++c) {
            dst.data()[GetTileElementOffset<DstTileData>(r, c)] = LoadByteStorageElement<T>(
                src, slotIndex, baseByteOffset, regionByteEnd,
                static_cast<std::size_t>(r) * srcCols + static_cast<std::size_t>(c), "CopyLinearToTile");
        }
    }
}

template <typename DstT, typename SrcTileData, std::size_t StorageSize>
PTO_INTERNAL void InsertTileWindowToLinear(
    std::array<uint8_t, StorageSize>& dst, std::size_t slotIndex, std::size_t baseByteOffset, std::size_t regionByteEnd,
    uint32_t dstCols, SrcTileData& src, uint32_t dstRowOffset, uint32_t dstColOffset)
{
    for (int r = 0; r < src.GetValidRow(); ++r) {
        for (int c = 0; c < src.GetValidCol(); ++c) {
            const std::size_t elementIndex =
                (static_cast<uint32_t>(r) + dstRowOffset) * dstCols + static_cast<uint32_t>(c) + dstColOffset;
            StoreByteStorageElement(
                dst, slotIndex, baseByteOffset, regionByteEnd, elementIndex,
                static_cast<DstT>(src.data()[GetTileElementOffset<SrcTileData>(r, c)]), "InsertTileWindowToLinear");
        }
    }
}

#ifdef __CPU_SIM
template <typename TileData>
PTO_INTERNAL void EnsureTileStorage(TileData& tile)
{
    using TilePtr = std::remove_reference_t<decltype(tile.data())>;
    static_assert(std::is_pointer_v<TilePtr>, "CPU-sim tile backing helper requires pointer-backed tile storage.");

    if (tile.data() != nullptr) {
        return;
    }

    static thread_local std::unordered_map<const void*, std::vector<typename TileData::DType>> buffers;
    auto& buffer = buffers[static_cast<const void*>(&tile)];
    const auto numel = static_cast<std::size_t>(TileData::Rows * TileData::Cols);
    if (buffer.size() != numel) {
        buffer.resize(numel);
    }
    tile.data() = buffer.data();
}
#else
template <typename TileData>
PTO_INTERNAL void EnsureTileStorage(TileData& tile)
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

template <
    uint8_t FlagID, uint8_t DirType, uint32_t SlotSize, uint32_t SlotNum, uint32_t LocalSlotNum = 2,
    bool IsNoSplit = false, bool EN_UNIT_FLAG = false>
struct TPipe {
    static constexpr uint8_t DIR_MASK = 0x7;
    static constexpr uint8_t DIR_TYPE = DIR_MASK & DirType;
    static constexpr bool is_c2v = ((DIR_TYPE & Direction::DIR_C2V) == Direction::DIR_C2V);
    static constexpr bool is_v2c = ((DIR_TYPE & Direction::DIR_V2C) == Direction::DIR_V2C);
    static constexpr bool is_v2c_ctrl = ((DIR_TYPE & Direction::DIR_V2C_CTRL) == Direction::DIR_V2C_CTRL);
    static constexpr bool is_no_split = IsNoSplit;
    static constexpr uint8_t VEC_CORE_ID_OFFSET = 16;
    using RingFiFo = RingFIFO<SlotSize, SlotNum, LocalSlotNum>;
    static constexpr uint32_t LOCAL_SPLIT_COPIES = is_c2v ? 2u : 1u;
    static constexpr uint32_t LOCAL_SLOT_STORAGE_SIZE = SlotSize * LOCAL_SPLIT_COPIES;

    struct SharedState {
        std::mutex mutex;
        std::condition_variable cv;
        int next_producer_slot = 0;
        // Separate cursors keep DIR_BOTH traffic from blocking across directions.
        int next_c2v_consumer_slot = 0;
        int next_v2c_consumer_slot = 0;
        std::array<int, 2> next_consumer_slots_by_lane{};
        int occupied = 0;
        int popped_not_freed = 0;
        std::array<int, 2> popped_not_freed_by_lane{};
        std::array<std::array<int, SlotNum>, 2> popped_slots_by_lane{};
        std::array<int, SlotNum> popped_slots{};
        std::array<std::array<uint8_t, LOCAL_SLOT_STORAGE_SIZE>, SlotNum> local_slot_storage{};
        std::array<cpu_pipe::TransferDir, SlotNum> transfer_dirs{};
        // Producer commit order per slot, 0 when the slot holds nothing.
        std::array<uint64_t, SlotNum> commit_seq{};
        uint64_t next_commit_seq = 1;
        std::array<uint32_t, SlotNum> remaining_consumers{};
        std::array<uint32_t, SlotNum> consumers_claimed{};
        std::array<uint32_t, SlotNum> producers_allocated{};
        std::array<uint32_t, SlotNum> producers_done{};
        // Per-slot reserved flag: set when a producer claims the slot (allocate),
        // cleared when the slot is fully consumed (free). Lets allocate() gate on
        // the specific cursor slot being free, which keeps DIR_BOTH / two-lane C2V
        // traffic correct when slots are released out of producer order.
        std::array<int, SlotNum> slot_busy{};
    };

    struct SharedStateStorage {
        std::atomic<uint32_t> init_state{0};
        alignas(SharedState) unsigned char payload[sizeof(SharedState)]{};
    };

    PTO_INTERNAL static void EnsureSharedStateInitialized(SharedStateStorage& storage)
    {
        uint32_t expected = 0;
        if (storage.init_state.compare_exchange_strong(expected, 1, std::memory_order_acq_rel)) {
            new (storage.payload) SharedState();
            storage.init_state.store(cpu_pipe::PSTATE_INITIALIZED, std::memory_order_release);
            return;
        }
        while (storage.init_state.load(std::memory_order_acquire) != cpu_pipe::PSTATE_INITIALIZED) {
            std::this_thread::yield();
        }
    }

    PTO_INTERNAL static SharedState& GetSharedState()
    {
        constexpr uint64_t pipeKey = (static_cast<uint64_t>(FlagID) << 56) | (static_cast<uint64_t>(DirType) << 48) |
                                     (static_cast<uint64_t>(SlotNum) << 40) |
                                     (static_cast<uint64_t>(LocalSlotNum) << 32) | static_cast<uint64_t>(SlotSize);
        if (cpu_sim::injected_pipe_shared_state_hook != nullptr) {
            auto* storage = reinterpret_cast<SharedStateStorage*>(
                cpu_sim::injected_pipe_shared_state_hook(pipeKey, sizeof(SharedStateStorage)));
            if (storage != nullptr) {
                EnsureSharedStateInitialized(*storage);
                return *std::launder(reinterpret_cast<SharedState*>(storage->payload));
            }
        }
        if (auto hook = cpu_sim::ResolvePipeSharedStateHook(); hook != nullptr) {
            auto* storage = reinterpret_cast<SharedStateStorage*>(hook(pipeKey, sizeof(SharedStateStorage)));
            if (storage != nullptr) {
                EnsureSharedStateInitialized(*storage);
                return *std::launder(reinterpret_cast<SharedState*>(storage->payload));
            }
        }
        if (auto hook = cpu_sim::ResolveSharedStorageHook(); hook != nullptr) {
            std::stringstream ss;
            ss << "pto-pipe-" << static_cast<unsigned long long>(get_task_cookie()) << "-" << get_block_idx() << "-"
               << static_cast<uint32_t>(FlagID) << "-" << static_cast<uint32_t>(DirType) << "-" << SlotSize << "-"
               << SlotNum << "-" << LocalSlotNum;
            auto* storage = reinterpret_cast<SharedStateStorage*>(hook(ss.str(), sizeof(SharedStateStorage)));
            EnsureSharedStateInitialized(*storage);
            return *std::launder(reinterpret_cast<SharedState*>(storage->payload));
        }

        static SharedStateStorage storage{};
        EnsureSharedStateInitialized(storage);
        return *std::launder(reinterpret_cast<SharedState*>(storage.payload));
    }

    PTO_INTERNAL static void reset_for_cpu_sim()
    {
        auto& shared_state = GetSharedState();
        std::lock_guard<std::mutex> lock(shared_state.mutex);
        shared_state.next_producer_slot = 0;
        shared_state.next_c2v_consumer_slot = 0;
        shared_state.next_v2c_consumer_slot = 0;
        shared_state.next_consumer_slots_by_lane.fill(0);
        shared_state.occupied = 0;
        shared_state.popped_not_freed = 0;
        shared_state.popped_slots.fill(0);
        shared_state.popped_not_freed_by_lane.fill(0);
        for (auto& lane : shared_state.popped_slots_by_lane) {
            lane.fill(0);
        }
        for (auto& slot : shared_state.local_slot_storage) {
            slot.fill(0);
        }
        shared_state.remaining_consumers.fill(0);
        shared_state.consumers_claimed.fill(0);
        shared_state.producers_allocated.fill(0);
        shared_state.producers_done.fill(0);
        shared_state.slot_busy.fill(0);
        shared_state.transfer_dirs.fill(cpu_pipe::TransferDir::None);
        shared_state.commit_seq.fill(0);
        shared_state.next_commit_seq = 1;
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

        PTO_INTERNAL int getTileId() const { return tileIndex; }

        PTO_INTERNAL int getSubTileId() const { return subTileIndex; }

        PTO_INTERNAL void setAllocateStatus(bool allocate) { isAllocate = allocate; }

        PTO_INTERNAL bool getAllocateStatus() const { return isAllocate; }

        PTO_INTERNAL void setRecordStatus(bool record) { isRecord = record; }

        PTO_INTERNAL bool getRecordStatus() const { return isRecord; }

        PTO_INTERNAL void setEntryOffset(int offset) { entryOffset = offset; }

        template <typename TileProd, TileSplitAxis Split = TileSplitAxis::TILE_UP_DOWN>
        PTO_INTERNAL void allocate()
        {
            (void)Split;
            auto& shared_state = TPipe::GetSharedState();
            std::unique_lock<std::mutex> lock(shared_state.mutex);
            if constexpr (
                TPipe::is_v2c && cpu_pipe::IsV2CProducerTile<TileProd>() && Split != TileSplitAxis::TILE_NO_SPLIT) {
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
                shared_state.slot_busy[static_cast<std::size_t>(tileIndex % RingFiFo::SLOT_NUM)] = 1;
                subTileIndex = static_cast<int>(laneId);
                return;
            } else {
                // Claim the slot at next_producer_slot only once it is actually free:
                // unused (slot_busy clear, no pending transfer) and the ring has room.
                // With DIR_BOTH or two-lane C2V, slots are released out of production
                // order, so the cursor slot may still hold unconsumed data even while
                // other slots are free.
                shared_state.cv.wait(lock, [&shared_state]() {
                    const auto slot = static_cast<std::size_t>(shared_state.next_producer_slot);
                    return shared_state.occupied < RingFiFo::SLOT_NUM && shared_state.slot_busy[slot] == 0 &&
                           shared_state.transfer_dirs[slot] == cpu_pipe::TransferDir::None;
                });
            }
            tileIndex = shared_state.next_producer_slot;
            shared_state.slot_busy[static_cast<std::size_t>(tileIndex % RingFiFo::SLOT_NUM)] = 1;
            subTileIndex = static_cast<int>(get_subblockid());
        }

        template <typename TileProd, TileSplitAxis Split = TileSplitAxis::TILE_UP_DOWN>
        PTO_INTERNAL void record()
        {
            (void)Split;
            auto& shared_state = TPipe::GetSharedState();
            {
                std::lock_guard<std::mutex> lock(shared_state.mutex);
                const auto slotIdx = static_cast<std::size_t>(tileIndex % RingFiFo::SLOT_NUM);
                if constexpr (
                    TPipe::is_v2c && cpu_pipe::IsV2CProducerTile<TileProd>() && Split != TileSplitAxis::TILE_NO_SPLIT) {
                    const uint32_t laneMask = cpu_pipe::GetSplitLaneMask<Split>(static_cast<uint32_t>(subTileIndex));
                    shared_state.producers_done[slotIdx] |= laneMask;
                    if (shared_state.producers_done[slotIdx] != cpu_pipe::GetActiveSplitLaneMask<TileProd, Split>()) {
                        return;
                    }
                    shared_state.producers_allocated[slotIdx] = 0;
                    shared_state.producers_done[slotIdx] = 0;
                }
                // Mark the slot consumable only once fully committed: for a split V2C producer
                // that is after both AIV lanes have written, so the consumer never reads a
                // half-written slot.
                shared_state.transfer_dirs[slotIdx] = cpu_pipe::GetProducerTransferDir<TPipe, TileProd>();
                shared_state.remaining_consumers[slotIdx] =
                    cpu_pipe::GetRequiredConsumerCount<TPipe, TileProd, Split>();
                shared_state.commit_seq[slotIdx] = shared_state.next_commit_seq++;
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
        // CPU-sim only: this consumer's pops awaiting their TFREE, oldest first; free() releases the
        // head. Kept per consumer because the shared `popped_slots` head can belong to the other
        // direction on a DIR_BOTH pipe.
        std::array<int, RingFiFo::SLOT_NUM> pendingSlots{};
        int pendingSlotCount = 0;

        PTO_INTERNAL Consumer() = default;

        PTO_INTERNAL void setTileId(int tid, int subTid)
        {
            tileIndex = tid;
            subTileIndex = subTid;
        }

        PTO_INTERNAL int getTileId() const { return tileIndex; }

        PTO_INTERNAL int getSubTileId() const { return subTileIndex; }

        PTO_INTERNAL void setWaitStatus(bool wait) { isWait = wait; }

        PTO_INTERNAL bool getWaitStatus() const { return isWait; }

        PTO_INTERNAL void setFreeStatus(bool free) { isFree = free; }

        PTO_INTERNAL bool getFreeStatus() const { return isFree; }

        PTO_INTERNAL void setentryOffset(int offset) { entryOffset = offset; }

        template <typename TileCons, TileSplitAxis Split = TileSplitAxis::TILE_UP_DOWN>
        PTO_INTERNAL void wait()
        {
            (void)Split;
            auto& shared_state = TPipe::GetSharedState();
            std::unique_lock<std::mutex> lock(shared_state.mutex);
            // Not cleared here: earlier pops on this consumer stay counted until their own TFREE.
            constexpr auto expectedDir = cpu_pipe::GetConsumerTransferDir<TPipe, TileCons>();
            constexpr bool kBothDir = TPipe::is_c2v && TPipe::is_v2c;
            if constexpr (
                TPipe::is_c2v && cpu_pipe::IsC2VConsumerTile<TileCons>() && Split != TileSplitAxis::TILE_NO_SPLIT) {
                const uint32_t laneId = cpu_pipe::GetSplitLaneId<Split>();
                const uint32_t laneMask = cpu_pipe::GetSplitLaneMask<Split>(laneId);
                int foundSlot = 0;
                shared_state.cv.wait(lock, [&shared_state, laneMask, expectedDir, &foundSlot]() {
                    return shared_state.occupied > 0 &&
                           cpu_pipe::FindNextTransferSlot(
                               shared_state.transfer_dirs, shared_state.next_c2v_consumer_slot, expectedDir, foundSlot,
                               [&shared_state, laneMask](int candidate) {
                                   return (shared_state.consumers_claimed[static_cast<std::size_t>(candidate)] &
                                           laneMask) != 0;
                               });
                });
                tileIndex = foundSlot;
                shared_state.consumers_claimed[static_cast<std::size_t>(tileIndex % RingFiFo::SLOT_NUM)] |= laneMask;
                subTileIndex = static_cast<int>(laneId);
                return;
            }
            // A pure V2C consumer (the cube) pops one full combined tile regardless of how the
            // producers split it, so it uses the same pending-slot tracking as no-split mode to
            // keep overlapping pops on distinct slots when the ring is reused.
            if constexpr (
                Split == TileSplitAxis::TILE_NO_SPLIT || (TPipe::is_v2c && !TPipe::is_c2v) ||
                (kBothDir && expectedDir == cpu_pipe::TransferDir::V2C)) {
                if constexpr (cpu_pipe::ShouldNoSplitC2VConsumerLaneParticipate<TPipe, TileCons, Split>()) {
                    if (cpu_pipe::IsDualLaneC2VActive()) {
                        const uint32_t laneId = cpu_pipe::GetSplitLaneId<TileSplitAxis::TILE_UP_DOWN>();
                        const uint32_t laneMask = cpu_pipe::GetSplitLaneMask<TileSplitAxis::TILE_UP_DOWN>(laneId);
                        auto& laneNext = shared_state.next_consumer_slots_by_lane[laneId];
                        auto& lanePopped = shared_state.popped_not_freed_by_lane[laneId];
                        int foundSlot = 0;
                        shared_state.cv.wait(lock, [&shared_state, laneMask, expectedDir, &foundSlot]() {
                            return cpu_pipe::FindOldestTransferSlot(
                                shared_state.transfer_dirs, shared_state.commit_seq, expectedDir, foundSlot,
                                [&shared_state, laneMask](int candidate) {
                                    return (shared_state.consumers_claimed[static_cast<std::size_t>(candidate)] &
                                            laneMask) != 0;
                                });
                        });
                        tileIndex = foundSlot;
                        const auto dualLaneSlot = static_cast<std::size_t>(tileIndex % RingFiFo::SLOT_NUM);
                        // The first lane to claim this slot's occupancy sets the consumer count to the
                        // actual number of vector subblocks, replacing the producer's default of 1.
                        if (shared_state.consumers_claimed[dualLaneSlot] == 0u) {
                            shared_state.remaining_consumers[dualLaneSlot] = get_subblockdim();
                        }
                        shared_state.consumers_claimed[dualLaneSlot] |= laneMask;
                        subTileIndex = static_cast<int>(laneId);
                        laneNext = (tileIndex + 1) % RingFiFo::SLOT_NUM;
                        cpu_pipe::PushPendingSlot(shared_state.popped_slots_by_lane[laneId], lanePopped, tileIndex);
                        return;
                    }
                    // get_subblockdim()<2: single consumer — fall through to the shared-cursor path below.
                }
                int foundSlot = 0;
                int& consumerCursor = (expectedDir == cpu_pipe::TransferDir::V2C) ?
                                          shared_state.next_v2c_consumer_slot :
                                          shared_state.next_c2v_consumer_slot;
                (void)consumerCursor;
                shared_state.cv.wait(lock, [&shared_state, expectedDir, &foundSlot]() {
                    return cpu_pipe::FindOldestTransferSlot(
                        shared_state.transfer_dirs, shared_state.commit_seq, expectedDir, foundSlot,
                        [&shared_state](int candidate) {
                            for (int i = 0; i < shared_state.popped_not_freed; ++i) {
                                if (shared_state.popped_slots[static_cast<std::size_t>(i)] == candidate) {
                                    return true;
                                }
                            }
                            return false;
                        });
                });
                tileIndex = foundSlot;
                subTileIndex = static_cast<int>(get_subblockid());
                consumerCursor = (tileIndex + 1) % RingFiFo::SLOT_NUM;
                cpu_pipe::PushPendingSlot(shared_state.popped_slots, shared_state.popped_not_freed, tileIndex);
                cpu_pipe::PushPendingSlot(pendingSlots, pendingSlotCount, tileIndex);
                return;
            }
            shared_state.cv.wait(lock, [&shared_state, expectedDir]() {
                return shared_state.occupied > 0 &&
                       shared_state.transfer_dirs[static_cast<std::size_t>(shared_state.next_c2v_consumer_slot)] ==
                           expectedDir;
            });
            tileIndex = shared_state.next_c2v_consumer_slot;
            subTileIndex = static_cast<int>(get_subblockid());
        }

        template <TileSplitAxis Split = TileSplitAxis::TILE_UP_DOWN>
        PTO_INTERNAL void free()
        {
            (void)Split;
            auto& shared_state = TPipe::GetSharedState();
            {
                std::lock_guard<std::mutex> lock(shared_state.mutex);
                if constexpr (cpu_pipe::ShouldNoSplitC2VConsumerLaneFree<TPipe, Split>()) {
                    if (cpu_pipe::IsDualLaneC2VActive()) {
                        const uint32_t laneId = cpu_pipe::GetSplitLaneId<TileSplitAxis::TILE_UP_DOWN>();
                        cpu_pipe::PopPendingSlot(
                            shared_state.popped_slots_by_lane[laneId], shared_state.popped_not_freed_by_lane[laneId],
                            tileIndex);
                    }
                }
                // Releases this consumer's oldest outstanding pop into tileIndex.
                const bool wasPendingSlotTracked = pendingSlotCount > 0;
                if (wasPendingSlotTracked) {
                    cpu_pipe::PopPendingSlot(pendingSlots, pendingSlotCount, tileIndex);
                    cpu_pipe::ErasePendingSlot(shared_state.popped_slots, shared_state.popped_not_freed, tileIndex);
                }
                const auto slotIndex = static_cast<std::size_t>(tileIndex % RingFiFo::SLOT_NUM);
                auto& remaining = shared_state.remaining_consumers[slotIndex];
                if (remaining > 1) {
                    --remaining;
                } else {
                    remaining = 0;
                    shared_state.consumers_claimed[slotIndex] = 0;
                    shared_state.transfer_dirs[slotIndex] = cpu_pipe::TransferDir::None;
                    shared_state.commit_seq[slotIndex] = 0;
                    shared_state.slot_busy[slotIndex] = 0;
                    // Only a split consumer that advances the shared cursor in free() does so here;
                    // a pending-tracked V2C consumer already advanced it in wait(), and a NO_SPLIT
                    // consumer advances its own per-lane/search cursor in wait() and must not touch
                    // the C2V cursor here (it is shared with the other direction).
                    bool advanceConsumerCursor = false;
                    if constexpr (TPipe::is_c2v && TPipe::is_v2c) {
                        advanceConsumerCursor = (Split != TileSplitAxis::TILE_NO_SPLIT) && !wasPendingSlotTracked;
                    } else if constexpr (Split != TileSplitAxis::TILE_NO_SPLIT && !(TPipe::is_v2c && !TPipe::is_c2v)) {
                        advanceConsumerCursor = true;
                    }
                    if (advanceConsumerCursor) {
                        shared_state.next_c2v_consumer_slot = (tileIndex + 1) % RingFiFo::SLOT_NUM;
                    }
                    --shared_state.occupied;
                }
            }
            shared_state.cv.notify_all();
        }

        template <typename TileCons, TileSplitAxis Split>
        PTO_INTERNAL void popTileFromVecFiFo(RingFiFo& fifo, TileCons& tile)
        {
            const std::size_t slotIndex = static_cast<std::size_t>(tileIndex % RingFiFo::SLOT_NUM);
            const std::size_t entryBase = slotIndex * RingFiFo::SLOT_SIZE + static_cast<std::size_t>(entryOffset);
            (void)fifo;
            (void)entryBase;

            using T = typename TileCons::DType;
            const auto& slotStorage = TPipe::GetSharedState().local_slot_storage[slotIndex];
            cpu_pipe::EnsureTileStorage(tile);
            cpu_pipe::CopyLinearToTile<TileCons, T>(
                tile, slotStorage, slotIndex, static_cast<std::size_t>(entryOffset), RingFiFo::SLOT_SIZE,
                static_cast<uint32_t>(TileCons::Cols));
        }

        template <typename TileCons, TileSplitAxis Split>
        PTO_INTERNAL void popTileFromVecFiFoSplit(RingFiFo& fifo, TileCons& tile)
        {
            using T = typename TileCons::DType;
            const uint32_t splitIndex = cpu_pipe::GetSplitLaneId<Split>();
            const std::size_t slotIndex = static_cast<std::size_t>(tileIndex % RingFiFo::SLOT_NUM);
            const auto& slotStorage = TPipe::GetSharedState().local_slot_storage[slotIndex];
            cpu_pipe::EnsureTileStorage(tile);
            cpu_pipe::CopyLinearToTile<TileCons, T>(
                tile, slotStorage, slotIndex,
                static_cast<std::size_t>(splitIndex) * RingFiFo::SLOT_SIZE + static_cast<std::size_t>(entryOffset),
                (static_cast<std::size_t>(splitIndex) + 1) * RingFiFo::SLOT_SIZE,
                static_cast<uint32_t>(TileCons::Cols));
        }

        template <typename TileCons, TileSplitAxis Split>
        PTO_INTERNAL void popTileFromMatFiFo(RingFiFo& fifo, TileCons& tile)
        {
            using T = typename TileCons::DType;
            constexpr int rows = TileCons::Rows;
            constexpr int cols = TileCons::Cols;
            using SlotTile = Tile<TileType::Mat, T, rows, cols, BLayout::RowMajor, rows, cols>;
            const std::size_t slotIndex = static_cast<std::size_t>(tileIndex % RingFiFo::SLOT_NUM);
            const std::size_t entryBase = slotIndex * RingFiFo::SLOT_SIZE + static_cast<std::size_t>(entryOffset);
            const auto& slotStorage = TPipe::GetSharedState().local_slot_storage[slotIndex];

            SlotTile slotTile;
            TASSIGN_IMPL(slotTile, fifo.V2C_CONSUMER_BUF + entryBase);
            cpu_pipe::CopyLinearToTile<SlotTile, T>(
                slotTile, slotStorage, slotIndex, static_cast<std::size_t>(entryOffset), RingFiFo::SLOT_SIZE,
                static_cast<uint32_t>(slotTile.GetValidCol()));
            cpu_pipe::EnsureTileStorage(tile);
            TMOV_IMPL(tile, slotTile);
        }

        template <typename TileCons>
        PTO_INTERNAL void popTileFromLocalFiFo(TileCons& tile)
        {
            using T = typename TileCons::DType;
            const std::size_t slotIndex = static_cast<std::size_t>(tileIndex % RingFiFo::SLOT_NUM);
            const auto& slotStorage = TPipe::GetSharedState().local_slot_storage[slotIndex];
            cpu_pipe::EnsureTileStorage(tile);
            cpu_pipe::CopyLinearToTile<TileCons, T>(
                tile, slotStorage, slotIndex, static_cast<std::size_t>(entryOffset), RingFiFo::SLOT_SIZE,
                static_cast<uint32_t>(TileCons::Cols));
        }

        template <typename TileCons, TileSplitAxis Split>
        PTO_INTERNAL bool pop(RingFiFo& fifo, TileCons& tile)
        {
            // CPU TileData uses host FIFO storage, not the target-side GM workspace.
            if constexpr (TPipe::is_c2v && cpu_pipe::IsC2VConsumerTile<TileCons>()) {
                if constexpr (Split == TileSplitAxis::TILE_NO_SPLIT) {
                    popTileFromVecFiFo<TileCons, Split>(fifo, tile);
                } else {
                    popTileFromVecFiFoSplit<TileCons, Split>(fifo, tile);
                }
                return false;
            } else if constexpr (TPipe::is_v2c) {
                if constexpr (!is_global_data_v<TileCons>) {
                    if constexpr (TileCons::Loc == TileType::Mat) {
                        popTileFromMatFiFo<TileCons, Split>(fifo, tile);
                        return false;
                    }
                }
            }
            popTileFromLocalFiFo(tile);
            return false;
        }
    };

    RingFiFo fifo;
    Producer prod;
    Consumer cons;

    PTO_INTERNAL explicit TPipe(__gm__ void* gmSlotBuffer, uint32_t c2vConsumerBuf, uint32_t v2cConsumerBuf)
        : fifo(gmSlotBuffer, c2vConsumerBuf, v2cConsumerBuf), prod(), cons()
    {}
};

// Fixpipe conversion helpers.
template <typename TileProd, typename TConfig>
using FixpipeConsType = FixpipeConsDType_t<TConfig::QuantPre, typename TileProd::DType>;

template <typename TileProd, typename T, LayoutMode_t LayoutMode>
using FixpipeVecTile = std::conditional_t<
    LayoutMode == LayoutMode_t::NZ2ND,
    Tile<TileType::Vec, T, TileProd::Rows, TileProd::Cols, BLayout::RowMajor, TileProd::Rows, TileProd::Cols>,
    std::conditional_t<
        LayoutMode == LayoutMode_t::NZ2DN,
        Tile<TileType::Vec, T, TileProd::Rows, TileProd::Cols, BLayout::ColMajor, TileProd::Rows, TileProd::Cols>,
        Tile<
            TileType::Vec, T, TileProd::Rows, TileProd::Cols, BLayout::ColMajor, TileProd::Rows, TileProd::Cols,
            SLayout::RowMajor>>>;

template <QuantMode_t QuantPre>
void InitializeQuantScalars(std::vector<uint64_t>& scalars)
{
    if constexpr (is_vector_quant_v<QuantPre>) {
        uint64_t addr = GET_QUANT_VECTOR_IMPL();
        uint64_t* ptr = reinterpret_cast<uint64_t*>(addr);
        for (size_t i = 0; i < scalars.size(); i++) {
            scalars[i] = ptr[i];
        }
    } else {
        uint64_t scalar = GET_QUANT_SCALAR_IMPL();
        if (scalar == 0u) {
            float mockScalar = 1.0f;
            scalar = static_cast<uint64_t>(*reinterpret_cast<int32_t*>(&mockScalar));
        }
        for (size_t i = 0; i < scalars.size(); i++) {
            scalars[i] = scalar;
        }
    }
}

template <typename Pipe, typename TileProd, typename TConfig, TileSplitAxis Split>
PTO_INTERNAL void TPush_c2v(Pipe& pipe, TileProd& tile, size_t slotIndex)
{
    using DstT = FixpipeConsType<TileProd, TConfig>;
    using TileCons = FixpipeVecTile<TileProd, DstT, TConfig::LayoutMode>;
    constexpr QuantMode_t QuantPre = TConfig::QuantPre;
    constexpr ReluPreMode ReluMode = TConfig::ReluMode;

    const uint32_t consRows = [&tile]() -> uint32_t {
        if constexpr (Split == TileSplitAxis::TILE_NO_SPLIT) {
            return static_cast<uint32_t>(tile.GetValidRow());
        } else if constexpr (Split == TileSplitAxis::TILE_UP_DOWN) {
            return static_cast<uint32_t>(TileProd::Rows / 2);
        } else {
            return static_cast<uint32_t>(TileProd::Rows);
        }
    }();
    const uint32_t consCols = [&tile]() -> uint32_t {
        if constexpr (Split == TileSplitAxis::TILE_NO_SPLIT) {
            return static_cast<uint32_t>(tile.GetValidCol());
        } else if constexpr (Split == TileSplitAxis::TILE_LEFT_RIGHT) {
            return static_cast<uint32_t>(TileProd::Cols / 2);
        } else {
            return static_cast<uint32_t>(TileProd::Cols);
        }
    }();

    std::vector<uint64_t> scalars(static_cast<std::size_t>(consCols), 0);
    if constexpr (QuantPre != QuantMode_t::NoQuant) {
        InitializeQuantScalars<QuantPre>(scalars);
    }

    auto& slotStorage = Pipe::GetSharedState().local_slot_storage[slotIndex];
    for (uint32_t splitIndex = 0; splitIndex < cpu_pipe::GetSplitCount<Split>(); ++splitIndex) {
        const std::size_t baseByteOffset = static_cast<std::size_t>(splitIndex) * Pipe::RingFiFo::SLOT_SIZE +
                                           static_cast<std::size_t>(pipe.prod.entryOffset);
        const uint32_t rowOffset = (Split == TileSplitAxis::TILE_UP_DOWN) ? splitIndex * consRows : 0;
        const uint32_t colOffset = (Split == TileSplitAxis::TILE_LEFT_RIGHT) ? splitIndex * consCols : 0;
        cpu_pipe::CopyTileWindowToLinear<DstT, TileProd, QuantPre, ReluMode>(
            slotStorage, slotIndex, baseByteOffset,
            (static_cast<std::size_t>(splitIndex) + 1) * Pipe::RingFiFo::SLOT_SIZE, consRows, consCols, tile, rowOffset,
            colOffset, scalars);
    }
}

template <typename Pipe, typename TileProd, typename TConfig, TileSplitAxis Split>
PTO_INTERNAL void TPush_v2c(Pipe& pipe, TileProd& tile, size_t slotIndex)
{
    using DstT = FixpipeConsType<TileProd, TConfig>;

    constexpr int slotRows =
        (Split == TileSplitAxis::TILE_UP_DOWN) ? (TileProd::Rows * 2) : static_cast<int>(TileProd::Rows);
    constexpr int slotCols =
        (Split == TileSplitAxis::TILE_LEFT_RIGHT) ? (TileProd::Cols * 2) : static_cast<int>(TileProd::Cols);
    using SlotTile = Tile<TileType::Mat, DstT, slotRows, slotCols, BLayout::RowMajor, slotRows, slotCols>;

    // Lay the payload out with the pushed window, that is the valid shape. TileProd::Cols
    // would stride it by the parent width when a narrower view is pushed. Mirrors TPush_c2v.
    const uint32_t consRows = [&tile]() -> uint32_t {
        if constexpr (Split == TileSplitAxis::TILE_NO_SPLIT) {
            return static_cast<uint32_t>(tile.GetValidRow());
        } else if constexpr (Split == TileSplitAxis::TILE_UP_DOWN) {
            return static_cast<uint32_t>(TileProd::Rows * 2);
        } else {
            return static_cast<uint32_t>(TileProd::Rows);
        }
    }();
    const uint32_t consCols = [&tile]() -> uint32_t {
        if constexpr (Split == TileSplitAxis::TILE_NO_SPLIT) {
            return static_cast<uint32_t>(tile.GetValidCol());
        } else if constexpr (Split == TileSplitAxis::TILE_LEFT_RIGHT) {
            return static_cast<uint32_t>(TileProd::Cols * 2);
        } else {
            return static_cast<uint32_t>(TileProd::Cols);
        }
    }();

    auto& slotStorage = Pipe::GetSharedState().local_slot_storage[slotIndex];
    const std::size_t baseByteOffset = static_cast<std::size_t>(pipe.prod.entryOffset);
    if constexpr (Split == TileSplitAxis::TILE_NO_SPLIT) {
        cpu_pipe::FillLinearRegion(
            slotStorage, slotIndex, baseByteOffset, Pipe::RingFiFo::SLOT_SIZE, consCols, static_cast<DstT>(0), 0,
            consRows, 0, consCols);
    }
    cpu_pipe::InsertTileWindowToLinear<DstT>(
        slotStorage, slotIndex, baseByteOffset, Pipe::RingFiFo::SLOT_SIZE, consCols, tile,
        cpu_pipe::GetSplitRowOffset<Split, SlotTile>(), cpu_pipe::GetSplitColOffset<Split, SlotTile>());
}

template <typename Pipe, typename TileProd, typename TConfig, TileSplitAxis Split>
PTO_INTERNAL void TPush_impl(Pipe& pipe, TileProd& tile)
{
    if (cpu_pipe::IsInactiveNoSplitVecLane<TileProd, Split>()) {
        return;
    }
    if (pipe.prod.getAllocateStatus()) {
        pipe.prod.template allocate<TileProd, Split>();
    }
    const std::size_t slotIndex = static_cast<std::size_t>(pipe.prod.getTileId() % Pipe::RingFiFo::SLOT_NUM);

    // Keep CPU TileData payloads in host FIFO storage, not the NPU GM workspace.
    if constexpr (Pipe::is_v2c && TileProd::Loc == TileType::Vec) {
        TPush_v2c<Pipe, TileProd, TConfig, Split>(pipe, tile, slotIndex);
    } else if constexpr (Pipe::is_c2v) {
        TPush_c2v<Pipe, TileProd, TConfig, Split>(pipe, tile, slotIndex);
    }
    if (pipe.prod.getRecordStatus()) {
        pipe.prod.template record<TileProd, Split>();
    }
}

template <typename Pipe, typename TileProd, TileSplitAxis Split>
PTO_INTERNAL void TPUSH_IMPL(Pipe& pipe, TileProd& tile)
{
    using FixpipeConfig = FixpipeParams<LayoutMode_t::NZ2ND, QuantMode_t::NoQuant>;
    TPush_impl<Pipe, TileProd, FixpipeConfig, Split>(pipe, tile);
}

template <typename TileProd, typename Pipe>
PTO_INTERNAL void TPUSH_IMPL(TileProd& tile, Pipe& pipe)
{
    TPUSH_IMPL<Pipe, TileProd, TileSplitAxis::TILE_NO_SPLIT>(pipe, tile);
}

template <typename Pipe, typename TileProd, typename TConfig>
PTO_INTERNAL void TPUSH_IMPL(Pipe& pipe, TileProd& tile)
{
    TPush_impl<Pipe, TileProd, TConfig, TileSplitAxis::TILE_NO_SPLIT>(pipe, tile);
}

} // namespace pto

#endif
