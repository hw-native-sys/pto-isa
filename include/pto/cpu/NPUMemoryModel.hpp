// --------------------------------------------------------------------------------
// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// --------------------------------------------------------------------------------

/**
 * NPU Memory Model for CPU Simulation
 *
 * Provides UB, L1, L0A, L0B, L0C memory buffers sized per NPU architecture.
 * TASSIGN maps tiles to offsets within these buffers based on TileType.
 *
 * Each thread gets its own independent NPUMemoryModel instance via
 * thread_local storage, accurately modeling the hardware where each
 * AICore has physically separate UB/L0 memory.
 *
 * Memory mapping:
 *   - Vec tiles   → UB (Unified Buffer)
 *   - Mat tiles   → L1
 *   - Left tiles  → L0A
 *   - Right tiles → L0B
 *   - Acc tiles   → L0C
 */
#ifndef PTO_NPU_MEMORY_MODEL_HPP
#define PTO_NPU_MEMORY_MODEL_HPP

#include <algorithm>
#include <charconv>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include <pto/common/pto_tile.hpp>

namespace pto {

enum class NPUArch { A2A3, A5 };

class NPUMemoryModel {
private:
    enum MemoryRegion {
        REG,    // Registers - simulates NPU registers
        UB,     // Unified Buffer - for Vec tiles
        L1,     // L1 Buffer - for Mat tiles
        L0A,    // L0A Buffer - for Left tiles
        L0B,    // L0B Buffer - for Right tiles
        L0C,    // L0C Buffer - for Acc tiles
        L0A_MX, // L0A Scale Buffer - for Left MX scale tiles
        L0B_MX, // L0B Scale Buffer - for Right MX scale tiles
        _MAX_REGIONS
    };

public:
    // Architecture-specific memory sizes (bytes)
    using ArchMemorySizes = std::size_t[MemoryRegion::_MAX_REGIONS];

private:
    static NPUArch GetDefaultArch()
    {
        if (defaultArch_.has_value()) {
            return *defaultArch_;
        }
        const char* value = std::getenv("PTO_CPU_SIM_ARCH");
        if (value == nullptr || *value == '\0') {
            return NPUArch::A2A3;
        }
        const std::string archName(value);
        if (archName == "a5" || archName == "A5") {
            return NPUArch::A5;
        }
        if (archName == "a2a3" || archName == "A2A3") {
            return NPUArch::A2A3;
        }
        throw std::invalid_argument("PTO_CPU_SIM_ARCH must be a2a3 or a5");
    }

    // Memory sizes by architecture
    // A2/A3:
    // https://www.hiascend.com/doc_center/source/zh/canncommercial/80RC3/devguide/appdevg/sdpdevg/atlasprogramming_12_0003.html
    static inline constexpr ArchMemorySizes kA2A3MemorySizes = {
        16 * 8,     // REGISTERS 16 with 8 byte
        192 * 1024, // UB:  192 KB
        512 * 1024, // L1:  512 KB
        64 * 1024,  // L0A: 64 KB
        64 * 1024,  // L0B: 64 KB
        128 * 1024, // L0C: 128 KB
        4 * 1024,   // L0A_MX: 4 KB
        4 * 1024,   // L0B: 4 KB
    };

    static inline constexpr ArchMemorySizes kA5MemorySizes = {
        16 * 8,     // REGISTERS 16 with 8 byte
        256 * 1024, // UB:  256 KB
        512 * 1024, // L1:  512 KB
        64 * 1024,  // L0A: 64 KB (placeholder - verify actual A5 spec)
        64 * 1024,  // L0B: 64 KB
        256 * 1024, // L0C: 256 KB
        4 * 1024,   // L0A_MX: 4 KB
        4 * 1024,   // L0B: 4 KB
    };

    static std::size_t ReadSizeOverride(const char* name, std::size_t fallback)
    {
        const char* value = std::getenv(name);
        if (value == nullptr || *value == '\0') {
            return fallback;
        }
        const std::string strValue(value);
        std::size_t capacityBytes = 0;
        const auto result = std::from_chars(strValue.data(), strValue.data() + strValue.size(), capacityBytes);
        if (result.ec != std::errc{} || result.ptr != strValue.data() + strValue.size() || capacityBytes == 0) {
            return fallback;
        }
        return capacityBytes;
    }

    void ApplySizeOverrides()
    {
        sizes_[MemoryRegion::UB] = ReadSizeOverride("PTO_CPU_SIM_UB_BYTES", sizes_[MemoryRegion::UB]);
        sizes_[MemoryRegion::L1] = ReadSizeOverride("PTO_CPU_SIM_L1_BYTES", sizes_[MemoryRegion::L1]);
        sizes_[MemoryRegion::L0A] = ReadSizeOverride("PTO_CPU_SIM_L0A_BYTES", sizes_[MemoryRegion::L0A]);
        sizes_[MemoryRegion::L0B] = ReadSizeOverride("PTO_CPU_SIM_L0B_BYTES", sizes_[MemoryRegion::L0B]);
        sizes_[MemoryRegion::L0C] = ReadSizeOverride("PTO_CPU_SIM_L0C_BYTES", sizes_[MemoryRegion::L0C]);
    }

public:
    // Each thread gets its own NPUMemoryModel instance, accurately modeling
    // the hardware where each AICore has physically separate memory.
    static NPUMemoryModel& Instance()
    {
        thread_local NPUMemoryModel instance;
        return instance;
    }

    // Set the default architecture for all threads.
    // Call once before any thread uses Instance().
    static void SetDefaultArch(NPUArch arch) { defaultArch_ = arch; }

    // Initialize with specific architecture (call once per thread at startup)
    void Initialize(NPUArch arch)
    {
        switch (arch) {
            case NPUArch::A2A3:
                // Yes, we know about memcpy, but CI does not allow it
                for (size_t i = 0; i < std::size(kA2A3MemorySizes); i++) {
                    sizes_[i] = kA2A3MemorySizes[i];
                }
                break;
            case NPUArch::A5:
                for (size_t i = 0; i < std::size(kA5MemorySizes); i++) {
                    sizes_[i] = kA5MemorySizes[i];
                }
                break;
        }

        ApplySizeOverrides();

        for (int i = 0; i < MemoryRegion::_MAX_REGIONS; i++) {
            buffers_[i].resize(sizes_[i], 0);
        }
        arch_ = arch;
        initialized_ = true;
    }

    // Auto-initialize with default architecture if not already done
    void EnsureInitialized()
    {
        if (!initialized_) {
            Initialize(GetDefaultArch());
        }
    }

    template <typename T>
    std::size_t GetNPUAddr(T* ptr)
    {
        const auto addr = reinterpret_cast<std::uintptr_t>(ptr);
        for (int region = 0; region < MemoryRegion::_MAX_REGIONS; ++region) {
            const auto start = reinterpret_cast<std::uintptr_t>(buffers_[region].data());
            const auto end = start + buffers_[region].size();
            if (addr >= start && addr < end) {
                return addr - start;
            }
        }
        throw std::invalid_argument("Cannot get proper NPU addr out of pointer");
    }

    // Get pointer to memory at offset within a region
    template <typename TileDef>
    TileDef::DType* GetPointer(std::size_t byteOffset)
    {
        static_assert(is_tile_data_v<TileDef> || is_conv_tile_v<TileDef>);
        if constexpr (TileDef::Loc == TileType::Mat) {
            return GetPointer<TileDef>(byteOffset, MemoryRegion::L1);
        } else if constexpr (TileDef::Loc == TileType::Left) {
            return GetPointer<TileDef>(byteOffset, MemoryRegion::L0A);
        } else if constexpr (TileDef::Loc == TileType::Right) {
            return GetPointer<TileDef>(byteOffset, MemoryRegion::L0B);
        } else if constexpr (TileDef::Loc == TileType::Acc) {
            return GetPointer<TileDef>(byteOffset, MemoryRegion::L0C);
        } else if constexpr (TileDef::Loc == TileType::ScaleLeft) {
            return GetPointer<TileDef>(byteOffset, MemoryRegion::L0A_MX);
        } else if constexpr (TileDef::Loc == TileType::ScaleRight) {
            return GetPointer<TileDef>(byteOffset, MemoryRegion::L0B_MX);
        } else {
            return GetPointer<TileDef>(byteOffset, MemoryRegion::UB); // For Vec and unknown types
        }
    }

    // Track separately allocated tile storage without extending its owner's lifetime.
    // The backing vector must not be resized while registered.
    template <typename T>
    void RegisterHostStorage(const std::shared_ptr<std::vector<T>>& storage)
    {
        if (storage && !storage->empty()) {
            // Expired ranges can overlap later allocations at reused heap addresses.
            std::erase_if(hostStorageRanges_, [](const auto& entry) { return entry.second.owner.expired(); });
            const auto baseAddress = reinterpret_cast<std::uintptr_t>(storage->data());
            hostStorageRanges_[baseAddress] = {baseAddress, storage->size() * sizeof(T), storage};
        }
    }

    // PTOAS-generated CPU-sim kernels may TASSIGN either:
    // - a byte offset within the simulated NPU region, or
    // - an already-materialized host pointer into a region or registered host storage
    //   (used when creating another tile view over the same backing storage).
    template <typename TileDef>
    typename TileDef::DType* ResolveAssignedAddress(std::uintptr_t addr)
    {
        static_assert(is_tile_data_v<TileDef> || is_conv_tile_v<TileDef>);
        EnsureInitialized();

        if (auto* direct = TryResolveExistingPointer<TileDef>(addr)) {
            return direct;
        }
        return GetPointer<TileDef>(static_cast<std::size_t>(addr));
    }

    // Get raw buffer bases (for debugging/direct access)
    char* GetREGBase()
    {
        EnsureInitialized();
        return buffers_[MemoryRegion::REG].data();
    }
    char* GetUBBase()
    {
        EnsureInitialized();
        return buffers_[MemoryRegion::UB].data();
    }
    char* GetL1Base()
    {
        EnsureInitialized();
        return buffers_[MemoryRegion::L1].data();
    }
    char* GetL0ABase()
    {
        EnsureInitialized();
        return buffers_[MemoryRegion::L0A].data();
    }
    char* GetL0BBase()
    {
        EnsureInitialized();
        return buffers_[MemoryRegion::L0B].data();
    }
    char* GetL0CBase()
    {
        EnsureInitialized();
        return buffers_[MemoryRegion::L0C].data();
    }

    const NPUMemoryModel::ArchMemorySizes& GetSizes() const { return sizes_; }
    NPUArch GetArch() const { return initialized_ ? arch_ : GetDefaultArch(); }
    bool IsInitialized() const { return initialized_; }

    // Returns true when rawAddr points into this thread's simulated on-chip
    // memory or live registered host storage, including interior pointers for tile views.
    bool ContainsAddress(std::uintptr_t rawAddr) const
    {
        if (FindHostStorageRange(rawAddr) != nullptr) {
            return true;
        }
        if (!initialized_) {
            return false;
        }
        for (const auto& buf : buffers_) {
            if (buf.empty()) {
                continue;
            }
            const auto begin = reinterpret_cast<std::uintptr_t>(buf.data());
            const auto end = begin + buf.size();
            if (rawAddr >= begin && rawAddr < end) {
                return true;
            }
        }
        return false;
    }

    // Clear all memory (zero-fill)
    void Clear()
    {
        if (initialized_) {
            for (auto& buf : buffers_) {
                std::fill(buf.begin(), buf.end(), 0);
            }
        }
    }

    // Reset to uninitialized state
    void Reset()
    {
        for (auto& buf : buffers_) {
            buf.clear();
        }
        initialized_ = false;
    }

private:
    struct HostStorageRange {
        std::uintptr_t baseAddress;
        std::size_t sizeBytes;
        std::weak_ptr<void> owner;
    };

    const HostStorageRange* FindHostStorageRange(std::uintptr_t addr) const
    {
        auto rangeIt = hostStorageRanges_.upper_bound(addr);
        if (rangeIt == hostStorageRanges_.begin()) {
            return nullptr;
        }
        --rangeIt;
        if (addr - rangeIt->first < rangeIt->second.sizeBytes && !rangeIt->second.owner.expired()) {
            return &rangeIt->second;
        }
        return nullptr;
    }

    template <typename TileDef>
    static constexpr std::size_t GetTileSizeInBytes()
    {
        if constexpr (is_tile_data_v<TileDef>) {
            return TileDef::GetSizeInBytes();
        } else {
            return static_cast<std::size_t>(TileDef::bufferSize);
        }
    }

    template <typename TileDef>
    typename TileDef::DType* TryResolveExistingPointer(std::uintptr_t addr)
    {
        for (int region = 0; region < MemoryRegion::_MAX_REGIONS; ++region) {
            auto* base = buffers_[region].data();
            const auto start = reinterpret_cast<std::uintptr_t>(base);
            const auto end = start + buffers_[region].size();
            if (addr >= start && addr < end) {
                return GetPointer<TileDef>(addr - start, static_cast<MemoryRegion>(region));
            }
        }
        if (const auto* hostRange = FindHostStorageRange(addr)) {
            const auto byteOffset = addr - hostRange->baseAddress;
            PTO_CPU_ASSERT(
                GetTileSizeInBytes<TileDef>() <= hostRange->sizeBytes - byteOffset,
                "Tile assignment exceeds host storage capacity.");
            return reinterpret_cast<typename TileDef::DType*>(addr);
        }
        return nullptr;
    }

    template <typename TileDef>
    inline typename TileDef::DType* GetPointer(std::size_t byteOffset, MemoryRegion region)
    {
        EnsureInitialized();

        constexpr std::size_t tileBytes = GetTileSizeInBytes<TileDef>();
        PTO_CPU_ASSERT(
            byteOffset <= sizes_[region] && tileBytes <= sizes_[region] - byteOffset,
            "Tile assignment exceeds memory region capacity.");
        return reinterpret_cast<typename TileDef::DType*>(buffers_[region].data() + byteOffset);
    }

    NPUMemoryModel() = default;
    NPUMemoryModel(const NPUMemoryModel&) = delete;
    NPUMemoryModel(const NPUMemoryModel&&) = delete;

    // Explicit selection takes precedence over the environment for subsequent auto-initialization.
    static inline std::optional<NPUArch> defaultArch_;

    // Per-thread memory buffers (thread_local instance owns these)
    std::vector<char> buffers_[MemoryRegion::_MAX_REGIONS];

    // Host storage is independent of the simulated regions and survives Reset/Initialize.
    // Weak owners avoid stale addresses and thread-local destruction-order dependencies.
    std::map<std::uintptr_t, HostStorageRange> hostStorageRanges_;

    ArchMemorySizes sizes_ = {};
    NPUArch arch_ = NPUArch::A2A3;
    bool initialized_ = false;
};

constexpr uint32_t SHIFT_MX_ADDR = 4;
template <typename T>
uint64_t GetScaleAddr(T* dst)
{
    uintptr_t addr = NPUMemoryModel::Instance().GetNPUAddr(dst);
    return addr >> SHIFT_MX_ADDR;
}

} // namespace pto

#endif // PTO_NPU_MEMORY_MODEL_HPP
