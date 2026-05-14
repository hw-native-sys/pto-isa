/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

// GridPipe: neighbor-core FIFO communication primitives.
//
// This is the proposal-level abstraction described in
// distributed-ffn-grid-tpush-tpop_zh-SPR_WFE.md.  On LPU WSE silicon the
// operations defined here are expected to lower to "SPR write + WFE wait +
// existing TMOV" sequences (see design doc section 5.3).  On A2/A3 we provide
// a mock backend that emulates the SPR+WFE semantics with HCCL shared windows
// and GM atomic flags; see include/pto/npu/a2a3/GridTPush.hpp.
//
// Compiler-visible static constraints are enforced via static_assert inside
// the intrinsic overloads in pto_instr.hpp.

#ifndef PTO_GRID_PIPE_HPP
#define PTO_GRID_PIPE_HPP

#include <cstdint>
#include <type_traits>

#include <pto/common/arch_macro.hpp>
#include <pto/common/type.hpp> // for AICORE (callable from both host and aicore contexts)

// Forward declaration: provided by the target backend (cpu_stub.hpp on
// CPU sim builds, CCE intrinsic / runtime header on A2/A3 NPU builds).
// GetGridCoord below uses this; we declare it here so grid_pipe.hpp can be
// included before any backend headers without triggering an undeclared name.
uint32_t get_block_idx();

namespace pto {

// ---------------------------------------------------------------------------
// 2D mesh coordinates and shape (design doc section 2).
// ---------------------------------------------------------------------------
struct GridShape {
    int gridRows = 0; // N
    int gridCols = 0; // M
};

struct GridCoord {
    int row = 0; // 0 .. gridRows-1
    int col = 0; // 0 .. gridCols-1
};

// ---------------------------------------------------------------------------
// Direction enum (design doc section 3.1).  Strongly-typed to avoid clashing
// with the cluster-local pto::Direction enum used by TPipe.
// ---------------------------------------------------------------------------
enum class GridDirection : uint8_t
{
    SOURCE = 0, // GM/Host/Runtime injection.  Only valid for TPOP.
    NORTH = 1,  // row -> row-1
    EAST = 2,   // col -> col+1
    WEST = 3,   // col -> col-1
};

inline constexpr int kGridDirectionCount = 4;

AICORE constexpr int GridDirectionIndex(GridDirection d)
{
    return static_cast<int>(d);
}

// ---------------------------------------------------------------------------
// GridPipe<TileT, SlotBytes, SlotCount>
//
// One instance describes the FIFO state for a single logical channel that the
// current core uses; each (core, direction) pair has its own ring buffer with
// independent prod/cons indices, ready/free signals, and slot region.  Fields
// are populated by the runtime during InitGridPipe and are read by the lower
// half (compiler-generated intrinsic expansions).
// ---------------------------------------------------------------------------
template <typename TileT_, int SlotBytes_, int SlotCount_>
struct GridPipe {
    static_assert(SlotCount_ > 0, "GridPipe requires SlotCount > 0");
    static_assert(SlotBytes_ > 0, "GridPipe requires SlotBytes > 0");

    using TileType = TileT_;
    static constexpr int SlotBytes = SlotBytes_;
    static constexpr int SlotCount = SlotCount_;

    // Shape + coord cached from runtime (design doc 2.1 / 2.2).
    GridShape shape{};
    GridCoord coord{};

    // Per-direction state.  Index by GridDirectionIndex(dir).
    __gm__ uint8_t *slotBase[kGridDirectionCount] = {nullptr};
    __gm__ uint32_t *readyFlags[kGridDirectionCount] = {nullptr};
    __gm__ uint32_t *freeFlags[kGridDirectionCount] = {nullptr};
    uint32_t prodIndex[kGridDirectionCount] = {0};
    uint32_t consIndex[kGridDirectionCount] = {0};

    // Opaque runtime pointer used by the A2/A3 backend to resolve cross-rank
    // addresses (HCCL device context).  Other targets may reinterpret.
    __gm__ void *runtimeCtx = nullptr;

    // Stable logical id used for runtime telemetry / mock SPR_PIPE_ID_<DIR>.
    uint32_t pipeId = 0;
};

// ---------------------------------------------------------------------------
// SFINAE marker: lets pto_instr.hpp's TPUSH/TPOP grid overloads disambiguate
// against the existing TPipe overloads without ambiguity.
// ---------------------------------------------------------------------------
template <typename T>
struct is_grid_pipe : std::false_type {};

template <typename TileT, int SlotBytes, int SlotCount>
struct is_grid_pipe<GridPipe<TileT, SlotBytes, SlotCount>> : std::true_type {};

template <typename T>
inline constexpr bool is_grid_pipe_v = is_grid_pipe<std::remove_reference_t<T>>::value;

// ---------------------------------------------------------------------------
// Coordinate bootstrap (design doc 2.1).  Row-major mapping from launcher's
// block_idx to (row, col).  AICORE-qualified because it calls get_block_idx(),
// which is a device intrinsic and has no host implementation.
// ---------------------------------------------------------------------------
AICORE inline GridCoord GetGridCoord(GridShape shape)
{
    int blockIdx = static_cast<int>(get_block_idx());
    return GridCoord{blockIdx / shape.gridCols, blockIdx % shape.gridCols};
}

AICORE inline int RankFromCoord(GridCoord coord, GridShape shape)
{
    return coord.row * shape.gridCols + coord.col;
}

// ---------------------------------------------------------------------------
// Compile-time / run-time direction validity (design doc 2.3).
// ---------------------------------------------------------------------------
AICORE constexpr bool CanPush(GridDirection dir, GridCoord c, GridShape s)
{
    switch (dir) {
        case GridDirection::NORTH:
            return c.row > 0;
        case GridDirection::EAST:
            return c.col + 1 < s.gridCols;
        case GridDirection::WEST:
            return c.col > 0;
        case GridDirection::SOURCE:
            return false; // Never legal to push to SOURCE.
    }
    return false;
}

AICORE constexpr bool CanPop(GridDirection dir, GridCoord c, GridShape s)
{
    switch (dir) {
        case GridDirection::NORTH:
            return c.row + 1 < s.gridRows;
        case GridDirection::EAST:
            return c.col > 0;
        case GridDirection::WEST:
            return c.col + 1 < s.gridCols;
        case GridDirection::SOURCE:
            return true;
    }
    return false;
}

AICORE constexpr GridCoord NeighborForPush(GridDirection dir, GridCoord c)
{
    switch (dir) {
        case GridDirection::NORTH:
            return {c.row - 1, c.col};
        case GridDirection::EAST:
            return {c.row, c.col + 1};
        case GridDirection::WEST:
            return {c.row, c.col - 1};
        case GridDirection::SOURCE:
            return c; // Unused; static_assert blocks TPUSH<SOURCE>.
    }
    return c;
}

AICORE constexpr GridCoord NeighborForPop(GridDirection dir, GridCoord c)
{
    switch (dir) {
        case GridDirection::NORTH:
            return {c.row + 1, c.col};
        case GridDirection::EAST:
            return {c.row, c.col - 1};
        case GridDirection::WEST:
            return {c.row, c.col + 1};
        case GridDirection::SOURCE:
            return c; // Bound by runtime to source queue.
    }
    return c;
}

inline constexpr int kInvalidRank = -1;

AICORE constexpr int NeighborRankForPush(GridDirection dir, GridCoord c, GridShape s)
{
    if (!CanPush(dir, c, s)) {
        return kInvalidRank;
    }
    GridCoord n = NeighborForPush(dir, c);
    return n.row * s.gridCols + n.col;
}

AICORE constexpr int NeighborRankForPop(GridDirection dir, GridCoord c, GridShape s)
{
    if (!CanPop(dir, c, s)) {
        return kInvalidRank;
    }
    GridCoord n = NeighborForPop(dir, c);
    return n.row * s.gridCols + n.col;
}

} // namespace pto

#endif // PTO_GRID_PIPE_HPP
