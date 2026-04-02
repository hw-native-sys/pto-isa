// --------------------------------------------------------------------------------
// Copyright (c) 2025 Huawei Technologies Co., Ltd.
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

#include <cstddef>
#include <algorithm>
#include <cstdint>
#include <vector>
#include <pto/common/pto_tile.hpp>

namespace pto {

enum class NPUArch
{
    A2A3,
    A5
};

class NPUMemoryModel {
private:
    enum MemoryRegion
    {
        UB,  // Unified Buffer - for Vec tiles
        L1,  // L1 Buffer - for Mat tiles
        L0A, // L0A Buffer - for Left tiles
        L0B, // L0B Buffer - for Right tiles
        L0C, // L0C Buffer - for Acc tiles
        _MAX_REGIONS
    };

public:
    // Architecture-specific memory sizes (bytes)
    using ArchMemorySizes = std::size_t[MemoryRegion::_MAX_REGIONS];

private:
    // Memory sizes by architecture
    // A2/A3:
    // https://www.hiascend.com/doc_center/source/zh/canncommercial/80RC3/devguide/appdevg/sdpdevg/atlasprogramming_12_0003.html
    static inline constexpr ArchMemorySizes kA2A3MemorySizes = {
        192 * 1024, // UB:  192 KB
        512 * 1024, // L1:  512 KB
        64 * 1024,  // L0A: 64 KB
        64 * 1024,  // L0B: 64 KB
        128 * 1024  // L0C: 128 KB
    };

    static inline constexpr ArchMemorySizes kA5MemorySizes = {
        256 * 1024, // UB:  256 KB
        512 * 1024, // L1:  512 KB
        64 * 1024,  // L0A: 64 KB (placeholder - verify actual A5 spec)
        64 * 1024,  // L0B: 64 KB
        256 * 1024  // L0C: 256 KB
    };

public:
    // Each thread gets its own NPUMemoryModel instance, accurately modeling
    // the hardware where each AICore has physically separate memory.
    static NPUMemoryModel &Instance()
    {
        thread_local NPUMemoryModel instance;
        return instance;
    }

    // Set the default architecture for all threads.
    // Call once before any thread uses Instance().
    static void SetDefaultArch(NPUArch arch)
    {
        defaultArch_ = arch;
    }

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
            Initialize(defaultArch_);
        }
    }

    // Get pointer to memory at offset within a region
    template <typename TileDef>
    TileDef::DType *GetPointer(std::size_t byteOffset)
    {
        static_assert(is_tile_data_v<TileDef>);

        if constexpr (TileDef::Loc == TileType::Mat) {
            return GetPointer<typename TileDef::DType, MemoryRegion::L1>(byteOffset, TileDef::Numel);
        } else if constexpr (TileDef::Loc == TileType::Left) {
            return GetPointer<typename TileDef::DType, MemoryRegion::L0A>(byteOffset, TileDef::Numel);
        } else if constexpr (TileDef::Loc == TileType::Right) {
            return GetPointer<typename TileDef::DType, MemoryRegion::L0B>(byteOffset, TileDef::Numel);
        } else if constexpr (TileDef::Loc == TileType::Acc) {
            return GetPointer<typename TileDef::DType, MemoryRegion::L0C>(byteOffset, TileDef::Numel);
        } else {
            return GetPointer<typename TileDef::DType, MemoryRegion::UB>(byteOffset,
                                                                         TileDef::Numel); // For Vec and unknown types
        }
    }

    // PTOAS-generated CPU-sim kernels may TASSIGN either:
    // - a byte offset within the simulated NPU region, or
    // - an already-materialized host pointer to a tile in that region
    //   (used when creating another tile view over the same backing storage).
    template <typename TileDef>
    typename TileDef::DType *ResolveAssignedAddress(std::uintptr_t addr)
    {
        static_assert(is_tile_data_v<TileDef>);
        EnsureInitialized();

        if (auto *direct = TryResolveExistingPointer<typename TileDef::DType>(addr)) {
            return direct;
        }
        return GetPointer<TileDef>(static_cast<std::size_t>(addr));
    }

    // Get raw buffer bases (for debugging/direct access)
    char *GetUBBase()
    {
        EnsureInitialized();
        return buffers_[MemoryRegion::UB].data();
    }
    char *GetL1Base()
    {
        EnsureInitialized();
        return buffers_[MemoryRegion::L1].data();
    }
    char *GetL0ABase()
    {
        EnsureInitialized();
        return buffers_[MemoryRegion::L0A].data();
    }
    char *GetL0BBase()
    {
        EnsureInitialized();
        return buffers_[MemoryRegion::L0B].data();
    }
    char *GetL0CBase()
    {
        EnsureInitialized();
        return buffers_[MemoryRegion::L0C].data();
    }

    const NPUMemoryModel::ArchMemorySizes &GetSizes() const
    {
        return sizes_;
    }
    NPUArch GetArch() const
    {
        return arch_;
    }
    bool IsInitialized() const
    {
        return initialized_;
    }

    // Clear all memory (zero-fill)
    void Clear()
    {
        if (initialized_) {
            for (auto &buf : buffers_) {
                std::fill(buf.begin(), buf.end(), 0);
            }
        }
    }

    // Reset to uninitialized state
    void Reset()
    {
        for (auto &buf : buffers_) {
            buf.clear();
        }
        initialized_ = false;
    }

private:
    template <typename T>
    T *TryResolveExistingPointer(std::uintptr_t addr)
    {
        for (int region = 0; region < MemoryRegion::_MAX_REGIONS; ++region) {
            auto *base = buffers_[region].data();
            const auto start = reinterpret_cast<std::uintptr_t>(base);
            const auto end = start + buffers_[region].size();
            if (addr >= start && addr < end) {
                return reinterpret_cast<T *>(addr);
            }
        }
        return nullptr;
    }

    template <typename T, MemoryRegion region>
    inline T *GetPointer(std::size_t byteOffset, size_t numel)
    {
        EnsureInitialized();

        assert(byteOffset + numel * sizeof(T) <= sizes_[region]);
        return reinterpret_cast<T *>(buffers_[region].data() + byteOffset);
    }

    NPUMemoryModel() = default;
    NPUMemoryModel(const NPUMemoryModel &) = delete;
    NPUMemoryModel(const NPUMemoryModel &&) = delete;

    // Shared default architecture — set once, read by all threads during auto-init
    static inline NPUArch defaultArch_ = NPUArch::A2A3;

    // Per-thread memory buffers (thread_local instance owns these)
    std::vector<char> buffers_[MemoryRegion::_MAX_REGIONS];

    ArchMemorySizes sizes_ = {};
    NPUArch arch_ = NPUArch::A2A3;
    bool initialized_ = false;
};

} // namespace pto

#endif // PTO_NPU_MEMORY_MODEL_HPP
