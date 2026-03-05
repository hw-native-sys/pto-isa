/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_COMMON_TASSIGN_CHECK_HPP
#define PTO_COMMON_TASSIGN_CHECK_HPP

#include <cstddef>
#include <pto/common/pto_tile.hpp>

// CPUSIM does not model on-chip buffer capacities, so skip all static checks.
#ifdef __CPU_SIM

namespace pto {
namespace detail {

template <typename TileT, std::size_t Addr>
struct tassign_static_check {
};

} // namespace detail
} // namespace pto

#else // NPU targets — full static checks

#include <pto/common/buffer_limits.hpp>
#include <pto/common/memory.hpp>

namespace pto {
namespace detail {

// =============================================================================
// TileStorageBytes: compute the byte footprint of a Tile or ConvTile.
// =============================================================================

template <typename T, typename = void>
struct TileStorageBytes;

// Regular Tile — Rows * Cols * sizeof(DType)
template <typename T>
struct TileStorageBytes<T, std::enable_if_t<is_tile_data_v<T> && !is_conv_tile_v<T> > > {
    static constexpr std::size_t value =
        static_cast<std::size_t>(T::Rows) * static_cast<std::size_t>(T::Cols) * sizeof(typename T::DType);
};

// ConvTile — uses bufferSize member
template <typename T>
struct TileStorageBytes<T, std::enable_if_t<is_conv_tile_v<T> > > {
    static constexpr std::size_t value = static_cast<std::size_t>(T::bufferSize) * sizeof(typename T::DType);
};

template <typename T>
inline constexpr std::size_t tile_storage_bytes_v = TileStorageBytes<T>::value;

// =============================================================================
// BufferTraits: per-TileType capacity and alignment.
// =============================================================================

template <TileType Loc>
struct BufferTraits;

template <>
struct BufferTraits<TileType::Vec> {
    static constexpr std::size_t capacity = PTO_UBUF_SIZE_BYTES;
    static constexpr std::size_t alignment = PTO_UBUF_ALIGN_BYTES;
    static constexpr const char *name = "UB";
};

template <>
struct BufferTraits<TileType::Mat> {
    static constexpr std::size_t capacity = PTO_CBUF_SIZE_BYTES;
    static constexpr std::size_t alignment = PTO_CBUF_ALIGN_BYTES;
    static constexpr const char *name = "L1";
};

template <>
struct BufferTraits<TileType::Left> {
    static constexpr std::size_t capacity = PTO_L0A_SIZE_BYTES;
    static constexpr std::size_t alignment = PTO_L0A_ALIGN_BYTES;
    static constexpr const char *name = "L0A";
};

template <>
struct BufferTraits<TileType::Right> {
    static constexpr std::size_t capacity = PTO_L0B_SIZE_BYTES;
    static constexpr std::size_t alignment = PTO_L0B_ALIGN_BYTES;
    static constexpr const char *name = "L0B";
};

template <>
struct BufferTraits<TileType::Acc> {
    static constexpr std::size_t capacity = PTO_L0C_SIZE_BYTES;
    static constexpr std::size_t alignment = PTO_L0C_ALIGN_BYTES;
    static constexpr const char *name = "L0C";
};

template <>
struct BufferTraits<TileType::Bias> {
    static constexpr std::size_t capacity = PTO_BIAS_SIZE_BYTES;
    static constexpr std::size_t alignment = PTO_BIAS_ALIGN_BYTES;
    static constexpr const char *name = "Bias";
};

template <>
struct BufferTraits<TileType::Scaling> {
    static constexpr std::size_t capacity = PTO_FBUF_SIZE_BYTES;
    static constexpr std::size_t alignment = PTO_FBUF_ALIGN_BYTES;
    static constexpr const char *name = "FBuffer";
};

template <>
struct BufferTraits<TileType::ScaleLeft> {
    static constexpr std::size_t capacity = PTO_SCALELEFT_SIZE_BYTES;
    static constexpr std::size_t alignment = PTO_L0A_ALIGN_BYTES;
    static constexpr const char *name = "L0A(Scale)";
};

template <>
struct BufferTraits<TileType::ScaleRight> {
    static constexpr std::size_t capacity = PTO_SCALERIGHT_SIZE_BYTES;
    static constexpr std::size_t alignment = PTO_L0B_ALIGN_BYTES;
    static constexpr const char *name = "L0B(Scale)";
};

// =============================================================================
// tassign_static_check: compile-time address validation for TASSIGN<Addr>(tile).
//
// Checks performed (SA-0351 .. SA-0354):
//   1. Memory space exists on this architecture (capacity != 0).
//   2. Tile storage size does not exceed memory space capacity.
//   3. addr + tile_size does not exceed memory space capacity.
//   4. addr is properly aligned.
// =============================================================================

template <typename TileT, std::size_t Addr>
struct tassign_static_check {
    using Traits = BufferTraits<TileT::Loc>;

    static constexpr std::size_t tile_bytes = tile_storage_bytes_v<TileT>;
    static constexpr std::size_t capacity = Traits::capacity;
    static constexpr std::size_t alignment = Traits::alignment;
    static constexpr std::size_t end_addr = Addr + tile_bytes;

    // SA-0351: Memory space is not available on this architecture.
    // FIX-A12: Use a TileType whose memory space exists on the target
    //          architecture, or switch to a platform that supports this
    //          memory space (e.g. ScaleLeft/ScaleRight require A5).
    static_assert(capacity > 0,
                  "[SA-0351] TASSIGN: memory space is not available on this architecture "
                  "(capacity is 0). (Fix: FIX-A12)");

    // SA-0352: Tile storage size exceeds memory space capacity.
    // FIX-A12: Reduce the tile dimensions (Rows/Cols) or element type size,
    //          or override the capacity via -DPTO_xxx_SIZE_BYTES=<value>.
    static_assert(tile_bytes <= capacity,
                  "[SA-0352] TASSIGN: Tile storage size exceeds memory space capacity. "
                  "Reduce tile dimensions or element size. (Fix: FIX-A12)");

    // SA-0353: addr + tile_size exceeds memory space capacity (out of bounds).
    // FIX-A12: Choose a smaller Addr so that Addr + tile_size <= capacity,
    //          or reduce the tile size.
    static_assert(capacity == 0 || end_addr <= capacity,
                  "[SA-0353] TASSIGN: addr + tile_size exceeds memory space capacity "
                  "(out of bounds). Use a smaller address or reduce tile size. (Fix: FIX-A12)");

    // SA-0354: addr is not properly aligned for the target memory space.
    // FIX-A12: Choose an Addr that is a multiple of the alignment requirement
    //          (see include/pto/common/buffer_limits.hpp for values).
    static_assert(alignment == 0 || (Addr % alignment) == 0,
                  "[SA-0354] TASSIGN: addr is not properly aligned for the target memory space. "
                  "Addr must be a multiple of the alignment (e.g. 32 bytes). (Fix: FIX-A12)");
};

} // namespace detail
} // namespace pto

#endif // __CPU_SIM

#endif // PTO_COMMON_TASSIGN_CHECK_HPP
