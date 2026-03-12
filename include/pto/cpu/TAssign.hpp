/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef TTILE_ASSIGN
#define TTILE_ASSIGN
#include <cstdint>
#include <pto/common/pto_tile.hpp>

#ifdef __CPU_SIM
#include <pto/cpu/NPUMemoryModel.hpp>
#endif

namespace pto {

template <typename T, typename AddrType>
PTO_INTERNAL void TASSIGN_IMPL(T &obj, AddrType addr)
{
    if constexpr (is_tile_data_v<T>) {
#ifdef __CPU_SIM
        using DType = typename T::DType;
        constexpr TileType tileType = T::Loc;

        // Determine memory region based on tile type
        // - Vec   → UB  (Unified Buffer)
        // - Mat   → L1
        // - Left  → L0A
        // - Right → L0B
        // - Acc   → L0C
        MemoryRegion region;
        if constexpr (tileType == TileType::Vec) {
            region = MemoryRegion::UB;
        } else if constexpr (tileType == TileType::Mat) {
            region = MemoryRegion::L1;
        } else if constexpr (tileType == TileType::Left) {
            region = MemoryRegion::L0A;
        } else if constexpr (tileType == TileType::Right) {
            region = MemoryRegion::L0B;
        } else if constexpr (tileType == TileType::Acc) {
            region = MemoryRegion::L0C;
        } else {
            region = MemoryRegion::UB; // Default for unknown types
        }

        std::size_t byteOffset = static_cast<std::size_t>(addr);
        obj.data() = NPUMemoryModel::Instance().GetPointer<DType>(region, byteOffset);
#else
        // No-op on real NPU - address binding is handled by hardware
        return;
#endif
    } else {
        static_assert(is_global_data_v<T>, "Only Tile and GlobalTensor data types are supported.");
        static_assert(std::is_pointer_v<AddrType>, "GlobalTensor can only be assigned with address of pointer type.");
        static_assert(std::is_same_v<std::remove_cv_t<std::remove_pointer_t<AddrType>>, typename T::DType>,
                      "GlobalTensor can only be assigned with pointer of same data type.");
        obj.SetAddr(addr);
    }
}

#ifdef __CPU_SIM
// Initialize NPU memory model with specific architecture
// Sets the default arch for all threads, and initializes the calling thread's instance.
// Other threads auto-initialize via EnsureInitialized() on first use.
inline void NPU_MEMORY_INIT(NPUArch arch = NPUArch::A2A3)
{
    NPUMemoryModel::SetDefaultArch(arch);
    NPUMemoryModel::Instance().Initialize(arch);
}

// Clear all NPU memory (useful between test iterations)
inline void NPU_MEMORY_CLEAR()
{
    NPUMemoryModel::Instance().Clear();
}
#endif

} // namespace pto
#endif
