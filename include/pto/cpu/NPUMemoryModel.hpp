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

namespace pto {

// Architecture-specific memory sizes (bytes)
struct ArchMemorySizes {
    std::size_t ubSize;  // Unified Buffer (Vec tiles)
    std::size_t l1Size;  // L1 Buffer (Mat tiles)
    std::size_t l0aSize; // L0A Buffer (Left tiles)
    std::size_t l0bSize; // L0B Buffer (Right tiles)
    std::size_t l0cSize; // L0C Buffer (Acc tiles)
};

// Memory sizes by architecture
// A2/A3:
// https://www.hiascend.com/doc_center/source/zh/canncommercial/80RC3/devguide/appdevg/sdpdevg/atlasprogramming_12_0003.html
inline constexpr ArchMemorySizes kA2A3MemorySizes = {
    192 * 1024, // UB:  192 KB
    512 * 1024, // L1:  512 KB
    64 * 1024,  // L0A: 64 KB
    64 * 1024,  // L0B: 64 KB
    128 * 1024  // L0C: 128 KB
};

inline constexpr ArchMemorySizes kA5MemorySizes = {
    256 * 1024, // UB:  256 KB
    256 * 1024, // L1:  256 KB
    64 * 1024,  // L0A: 64 KB (placeholder - verify actual A5 spec)
    64 * 1024,  // L0B: 64 KB
    256 * 1024  // L0C: 256 KB
};

enum class NPUArch
{
    A2A3,
    A5
};

enum class MemoryRegion
{
    UB,  // Unified Buffer - for Vec tiles
    L1,  // L1 Buffer - for Mat tiles
    L0A, // L0A Buffer - for Left tiles
    L0B, // L0B Buffer - for Right tiles
    L0C  // L0C Buffer - for Acc tiles
};

class NPUMemoryModel {
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
                sizes_ = kA2A3MemorySizes;
                break;
            case NPUArch::A5:
                sizes_ = kA5MemorySizes;
                break;
        }

        ubBuffer_.resize(sizes_.ubSize, 0);
        l1Buffer_.resize(sizes_.l1Size, 0);
        l0aBuffer_.resize(sizes_.l0aSize, 0);
        l0bBuffer_.resize(sizes_.l0bSize, 0);
        l0cBuffer_.resize(sizes_.l0cSize, 0);

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
    template <typename T>
    T *GetPointer(MemoryRegion region, std::size_t byteOffset)
    {
        EnsureInitialized();

        char *base = nullptr;
        std::size_t regionSize = 0;

        switch (region) {
            case MemoryRegion::UB:
                base = ubBuffer_.data();
                regionSize = sizes_.ubSize;
                break;
            case MemoryRegion::L1:
                base = l1Buffer_.data();
                regionSize = sizes_.l1Size;
                break;
            case MemoryRegion::L0A:
                base = l0aBuffer_.data();
                regionSize = sizes_.l0aSize;
                break;
            case MemoryRegion::L0B:
                base = l0bBuffer_.data();
                regionSize = sizes_.l0bSize;
                break;
            case MemoryRegion::L0C:
                base = l0cBuffer_.data();
                regionSize = sizes_.l0cSize;
                break;
        }

        return reinterpret_cast<T *>(base + byteOffset);
    }

    // Get raw buffer bases (for debugging/direct access)
    char *GetUBBase()
    {
        EnsureInitialized();
        return ubBuffer_.data();
    }
    char *GetL1Base()
    {
        EnsureInitialized();
        return l1Buffer_.data();
    }
    char *GetL0ABase()
    {
        EnsureInitialized();
        return l0aBuffer_.data();
    }
    char *GetL0BBase()
    {
        EnsureInitialized();
        return l0bBuffer_.data();
    }
    char *GetL0CBase()
    {
        EnsureInitialized();
        return l0cBuffer_.data();
    }

    const ArchMemorySizes &GetSizes() const
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
            std::fill(ubBuffer_.begin(), ubBuffer_.end(), 0);
            std::fill(l1Buffer_.begin(), l1Buffer_.end(), 0);
            std::fill(l0aBuffer_.begin(), l0aBuffer_.end(), 0);
            std::fill(l0bBuffer_.begin(), l0bBuffer_.end(), 0);
            std::fill(l0cBuffer_.begin(), l0cBuffer_.end(), 0);
        }
    }

    // Reset to uninitialized state
    void Reset()
    {
        ubBuffer_.clear();
        l1Buffer_.clear();
        l0aBuffer_.clear();
        l0bBuffer_.clear();
        l0cBuffer_.clear();
        initialized_ = false;
    }

private:
    NPUMemoryModel() = default;

    // Shared default architecture — set once, read by all threads during auto-init
    static inline NPUArch defaultArch_ = NPUArch::A2A3;

    // Per-thread memory buffers (thread_local instance owns these)
    std::vector<char> ubBuffer_;
    std::vector<char> l1Buffer_;
    std::vector<char> l0aBuffer_;
    std::vector<char> l0bBuffer_;
    std::vector<char> l0cBuffer_;

    ArchMemorySizes sizes_{};
    NPUArch arch_ = NPUArch::A2A3;
    bool initialized_ = false;
};

} // namespace pto

#endif // PTO_NPU_MEMORY_MODEL_HPP
