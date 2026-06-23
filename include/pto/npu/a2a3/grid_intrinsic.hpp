/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

// Grid TPUSH/TPOP intrinsic layer (A2/A3 backend).
//
// This header is the consolidation of four former include/pto/common headers:
//   * grid_pipe.hpp             -> Section 1: GridPipe mesh model + resolvers
//   * grid_pipe_mock_spr.hpp    -> Section 2: A2/A3 SPR+WFE mock (GM atomic flags)
//   * grid_counter_intrinsic.hpp-> Section 3: neighbor-counter intrinsic (mtspr/wfe)
//   * grid_sram_intrinsic.hpp   -> Section 4: neighbor SRAM addressing + transfer
//
// The GridPipe TPUSH/TPOP overloads in pto/common/pto_instr.hpp and the A2/A3
// backends in GridTPush.hpp / GridTPop.hpp both pull this single header in.
// Compiler-visible static constraints are still enforced via static_assert
// inside the intrinsic overloads in pto_instr.hpp.

#ifndef PTO_A2A3_GRID_INTRINSIC_HPP
#define PTO_A2A3_GRID_INTRINSIC_HPP

#include <cstdint>
#include <type_traits>

#include <pto/common/arch_macro.hpp>
#include <pto/common/type.hpp> // for AICORE (callable from both host and aicore contexts)

// ===========================================================================
// Section 1: GridPipe -- neighbor-core FIFO communication primitives.
//            (formerly include/pto/common/grid_pipe.hpp)
//
// This is the proposal-level abstraction described in
// distributed-ffn-grid-tpush-tpop_zh-SPR_WFE.md.  On LPU WSE silicon the
// operations defined here are expected to lower to "SPR write + WFE wait +
// existing TMOV" sequences (see design doc section 5.3).  On A2/A3 we provide
// a mock backend that emulates the SPR+WFE semantics with HCCL shared windows
// and GM atomic flags; see Section 2 below and GridTPush.hpp.
// ===========================================================================

// Forward declaration: provided by the target backend (cpu_stub.hpp on
// CPU sim builds, CCE intrinsic / runtime header on A2/A3 NPU builds).
// GetGridCoord below uses this; we declare it here so this header can be
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
    SOUTH = 4,  // row -> row+1
};

inline constexpr int kGridDirectionCount = 5;

AICORE constexpr int GridDirectionIndex(GridDirection d)
{
    return static_cast<int>(d);
}

// ---------------------------------------------------------------------------
// Broadcast span (single-source row/column multicast).  A ROW broadcast fans a
// payload to every other cell on the source's row; a COL broadcast fans along
// the source's column.  Each span is two opposite 1-D arms -- ROW = EAST+WEST,
// COL = NORTH+SOUTH (see SpanArmA / SpanArmB).  Because each arm is 1-D, the
// (direction, distance) pair is a complete fan-in-1 lane key, so a single-source
// broadcast needs no Scheme-B slot/flag expansion: a receiver east of the source
// drains it with the ordinary TPOP<EAST, dist>, a receiver west with
// TPOP<WEST, dist>.  Used by the GridPipe TPUSH<GridSpan> broadcast overload.
// ---------------------------------------------------------------------------
enum class GridSpan : uint8_t
{
    ROW = 0, // fan along the source's row:    EAST arm + WEST arm
    COL = 1, // fan along the source's column: NORTH arm + SOUTH arm
};

// The two opposite GridDirections a span decomposes into.  constexpr so they
// fold into the non-type template arguments of the per-arm broadcast helpers.
AICORE constexpr GridDirection SpanArmA(GridSpan s)
{
    return s == GridSpan::ROW ? GridDirection::EAST : GridDirection::NORTH;
}

AICORE constexpr GridDirection SpanArmB(GridSpan s)
{
    return s == GridSpan::ROW ? GridDirection::WEST : GridDirection::SOUTH;
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
        case GridDirection::SOUTH:
            return c.row + 1 < s.gridRows;
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
        case GridDirection::SOUTH:
            return c.row > 0;
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
        case GridDirection::SOUTH:
            return {c.row + 1, c.col};
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
        case GridDirection::SOUTH:
            return {c.row - 1, c.col};
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

// ---------------------------------------------------------------------------
// Multi-hop (routed K-hop unicast) generalisation of the neighbor resolvers.
//
// Scheme A: a K-hop *unicast* push keeps the receiver's per-direction slot/flag
// state at fan-in 1, so distance enters only the *target rank* (and the
// doorbell reach), never the buffer count.  A K-hop push is therefore the
// 1-hop expansion with "+1/-1" replaced by "+K/-K"; nothing in the GridPipe
// window layout changes.  The GridKHopSelfCheck() static_assert below pins
// k == 1 to the existing CanPush/NeighborRankForPush/CanPop/NeighborRankForPop
// behaviour so the default-distance (= 1) path stays byte-identical.
//
// Precondition (caller's responsibility): within one direction and phase, at
// most one (source, distance) pair targets a given receiver, i.e. fan-in <= 1.
// Concurrent multi-source receive (gather/multicast) is out of scope here and
// needs the fan-in-indexed channel layout (Scheme B).
// ---------------------------------------------------------------------------
AICORE constexpr bool CanPushK(GridDirection dir, GridCoord c, GridShape s, int k)
{
    switch (dir) {
        case GridDirection::NORTH:
            return c.row - k >= 0;
        case GridDirection::EAST:
            return c.col + k < s.gridCols;
        case GridDirection::WEST:
            return c.col - k >= 0;
        case GridDirection::SOUTH:
            return c.row + k < s.gridRows;
        case GridDirection::SOURCE:
            return false; // Never legal to push to SOURCE.
    }
    return false;
}

AICORE constexpr GridCoord NeighborForPushK(GridDirection dir, GridCoord c, int k)
{
    switch (dir) {
        case GridDirection::NORTH:
            return {c.row - k, c.col};
        case GridDirection::EAST:
            return {c.row, c.col + k};
        case GridDirection::WEST:
            return {c.row, c.col - k};
        case GridDirection::SOUTH:
            return {c.row + k, c.col};
        case GridDirection::SOURCE:
            return c; // Unused; TPUSH<SOURCE> is blocked by static_assert.
    }
    return c;
}

AICORE constexpr int RankForPushK(GridDirection dir, GridCoord c, GridShape s, int k)
{
    if (!CanPushK(dir, c, s, k)) {
        return kInvalidRank;
    }
    GridCoord n = NeighborForPushK(dir, c, k);
    return n.row * s.gridCols + n.col;
}

// Consumer side: the producer that fed a `dir` channel sits k hops in the
// *opposite* direction (an EAST channel is fed from the WEST, etc).  Used by
// TPOP to route the free-credit doorbell back to the K-hop producer.
AICORE constexpr bool CanPopK(GridDirection dir, GridCoord c, GridShape s, int k)
{
    switch (dir) {
        case GridDirection::NORTH:
            return c.row + k < s.gridRows; // upstream to the south
        case GridDirection::EAST:
            return c.col - k >= 0;         // upstream to the west
        case GridDirection::WEST:
            return c.col + k < s.gridCols; // upstream to the east
        case GridDirection::SOUTH:
            return c.row - k >= 0;         // upstream to the north
        case GridDirection::SOURCE:
            return true;                   // SOURCE pop is bound to the runtime queue, distance-free.
    }
    return false;
}

AICORE constexpr GridCoord NeighborForPopK(GridDirection dir, GridCoord c, int k)
{
    switch (dir) {
        case GridDirection::NORTH:
            return {c.row + k, c.col};
        case GridDirection::EAST:
            return {c.row, c.col - k};
        case GridDirection::WEST:
            return {c.row, c.col + k};
        case GridDirection::SOUTH:
            return {c.row - k, c.col};
        case GridDirection::SOURCE:
            return c; // Bound by runtime to source queue.
    }
    return c;
}

AICORE constexpr int RankForPopK(GridDirection dir, GridCoord c, GridShape s, int k)
{
    if (!CanPopK(dir, c, s, k)) {
        return kInvalidRank;
    }
    GridCoord n = NeighborForPopK(dir, c, k);
    return n.row * s.gridCols + n.col;
}

// Compile-time pin: the K-hop resolvers must collapse to the 1-hop neighbor
// resolvers at k == 1, so existing TPUSH<DIR>/TPOP<DIR> behaviour (Dist == 1)
// is preserved bit-for-bit.  A representative 2-hop case is also checked.
AICORE constexpr bool GridKHopSelfCheck()
{
    GridShape s{4, 4};
    GridCoord c{2, 2};
    bool ok = true;
    // k == 1 reproduces the 1-hop resolvers for every real direction.
    ok = ok && (CanPushK(GridDirection::NORTH, c, s, 1) == CanPush(GridDirection::NORTH, c, s));
    ok = ok && (CanPushK(GridDirection::EAST, c, s, 1) == CanPush(GridDirection::EAST, c, s));
    ok = ok && (CanPushK(GridDirection::WEST, c, s, 1) == CanPush(GridDirection::WEST, c, s));
    ok = ok && (CanPushK(GridDirection::SOUTH, c, s, 1) == CanPush(GridDirection::SOUTH, c, s));
    ok = ok && (RankForPushK(GridDirection::NORTH, c, s, 1) == NeighborRankForPush(GridDirection::NORTH, c, s));
    ok = ok && (RankForPushK(GridDirection::EAST, c, s, 1) == NeighborRankForPush(GridDirection::EAST, c, s));
    ok = ok && (RankForPushK(GridDirection::WEST, c, s, 1) == NeighborRankForPush(GridDirection::WEST, c, s));
    ok = ok && (RankForPushK(GridDirection::SOUTH, c, s, 1) == NeighborRankForPush(GridDirection::SOUTH, c, s));
    ok = ok && (CanPopK(GridDirection::NORTH, c, s, 1) == CanPop(GridDirection::NORTH, c, s));
    ok = ok && (CanPopK(GridDirection::EAST, c, s, 1) == CanPop(GridDirection::EAST, c, s));
    ok = ok && (CanPopK(GridDirection::WEST, c, s, 1) == CanPop(GridDirection::WEST, c, s));
    ok = ok && (CanPopK(GridDirection::SOUTH, c, s, 1) == CanPop(GridDirection::SOUTH, c, s));
    ok = ok && (RankForPopK(GridDirection::NORTH, c, s, 1) == NeighborRankForPop(GridDirection::NORTH, c, s));
    ok = ok && (RankForPopK(GridDirection::EAST, c, s, 1) == NeighborRankForPop(GridDirection::EAST, c, s));
    ok = ok && (RankForPopK(GridDirection::WEST, c, s, 1) == NeighborRankForPop(GridDirection::WEST, c, s));
    ok = ok && (RankForPopK(GridDirection::SOUTH, c, s, 1) == NeighborRankForPop(GridDirection::SOUTH, c, s));
    // Representative 2-hop case on a 1x4 row: col0 --EAST,2--> col2, popped at col2.
    GridShape row{1, 4};
    ok = ok && CanPushK(GridDirection::EAST, GridCoord{0, 0}, row, 2);
    ok = ok && (RankForPushK(GridDirection::EAST, GridCoord{0, 0}, row, 2) == 2);
    ok = ok && !CanPushK(GridDirection::EAST, GridCoord{0, 2}, row, 2); // 2+2 == 4 out of range
    ok = ok && CanPopK(GridDirection::EAST, GridCoord{0, 2}, row, 2);
    ok = ok && (RankForPopK(GridDirection::EAST, GridCoord{0, 2}, row, 2) == 0);
    return ok;
}
static_assert(GridKHopSelfCheck(), "GridPipe K-hop resolver self-test failed");

} // namespace pto

// ===========================================================================
// Section 2: Mock layer that stands in for LPU WSE SPR + WFE primitives.
//            (formerly include/pto/common/grid_pipe_mock_spr.hpp)
//
// Design doc section 5 defines the LPU WSE GridPipe lowering as:
//   - mtspr SPR_RDY_<DIR>    -> cross-core SPR write, wakes neighbor WFE
//   - mtspr SPR_FREE_<DIR>   -> cross-core SPR write, wakes producer WFE
//   - mfspr SPR_RDY_<DIR>    -> read local mirror of ready flag
//   - wfe   SPR_RDY_<DIR>, N -> block until ready >= N
//   - wfe   SPR_FREE_<DIR>,N -> block until free  >= N
//
// On A2/A3 silicon there is no native cross-core mesh SPR + event line.  The
// mocks below emulate the contract via GM atomic flags + spin-wait, so the
// GridPipe semantics can be exercised on real A2/A3 boards while staying
// trivially swappable for a real LPU WSE SPR backend later.
//
// Each MOCK_* function carries a comment naming the corresponding LPU WSE
// pseudo-asm line from design doc section 5.3, so `grep -n "LPU WSE:"` returns
// the substitution sites.
// ===========================================================================

namespace pto {
namespace grid_mock {

#ifndef PTO_GRID_MOCK_WFE_MAX_SPINS
#define PTO_GRID_MOCK_WFE_MAX_SPINS 100000000U
#endif

inline constexpr uint32_t kDefaultWfeMaxSpins = PTO_GRID_MOCK_WFE_MAX_SPINS;
inline constexpr uint32_t kFaultFlagWordOffset = 2 * kGridDirectionCount;

// MOCK: design doc 5.3 producer step (4), consumer step (4).
//
// LPU WSE: mtspr SPR_RDY_<DIR>, newValue      (cross-core, wakes neighbor WFE)
// LPU WSE: mtspr SPR_FREE_<DIR>, newValue     (cross-core, wakes producer WFE)
//
// A2/A3 mock: producer / consumer holds a pointer into the *neighbor's* GM
// window (resolved by HcclRemotePtr at runtime); we write the new monotonic
// counter into that remote location.  Pairing read happens via MockWfe* below.
//
// volatile cast prevents the compiler from caching the write.  Cross-rank
// visibility on A2/A3 requires an explicit dsb(DSB_DDR) + dcci pair around the
// store: AICORE caches are not coherent between cores, so without the dcci the
// pairing MockWfe spin on the remote rank may never observe the write.  This
// matches the SetLocalSummaryReady pattern in ready_queue.hpp (allgather_gemm).
inline AICORE void MockMtsprCounter(__gm__ uint32_t *remoteFlag, uint32_t newValue)
{
    if (remoteFlag != nullptr) {
        volatile __gm__ uint32_t *ptr = reinterpret_cast<volatile __gm__ uint32_t *>(remoteFlag);
        // Match the canonical TNotify Set pattern: pre-invalidate, store,
        // post-invalidate, dsb(DSB_DDR).  Compiler barriers prevent reordering.
        __asm__ __volatile__("" ::: "memory");
        dcci(reinterpret_cast<__gm__ void *>(const_cast<__gm__ uint32_t *>(ptr)), SINGLE_CACHE_LINE);
        __asm__ __volatile__("" ::: "memory");
        *ptr = newValue;
        __asm__ __volatile__("" ::: "memory");
        dcci(reinterpret_cast<__gm__ void *>(const_cast<__gm__ uint32_t *>(ptr)), SINGLE_CACHE_LINE);
        __asm__ __volatile__("" ::: "memory");
        dsb(DSB_DDR);
    }
}

inline AICORE void MockMtsprReady(__gm__ uint32_t *remoteFlag, uint32_t newValue)
{
    MockMtsprCounter(remoteFlag, newValue);
}

inline AICORE void MockMtsprFree(__gm__ uint32_t *remoteFlag, uint32_t newValue)
{
    MockMtsprCounter(remoteFlag, newValue);
}

// MOCK: design doc 5.3 producer step (1), consumer step (1).
//
// LPU WSE: mfspr r_ready, SPR_RDY_<DIR>
// LPU WSE: mfspr r_free,  SPR_FREE_<DIR>
//
// A2/A3 mock: volatile read of the local mirror of the flag.
inline AICORE uint32_t MockMfspr(__gm__ uint32_t *localFlag)
{
    if (localFlag == nullptr) {
        return 0;
    }
    return *reinterpret_cast<volatile __gm__ uint32_t *>(localFlag);
}

// MOCK: design doc 5.3 producer step (1) wait, consumer step (1) wait.
//
// LPU WSE: wfe SPR_RDY_<DIR>,  threshold      (block until SPR >= threshold)
// LPU WSE: wfe SPR_FREE_<DIR>, threshold
//
// A2/A3 mock: spin-poll the GM flag with `dcci` each iteration to invalidate
// the local cache line so the next load fetches from DDR.  Without the dcci,
// the AICORE may cache the original 0 indefinitely and never observe the
// producer's write (cross-core caches are not auto-coherent on A2/A3).
inline AICORE bool MockTryWfeCounter(__gm__ uint32_t *localFlag, uint32_t threshold,
                                     uint32_t maxSpins = kDefaultWfeMaxSpins)
{
    if (localFlag == nullptr) {
        return true;
    }
    volatile __gm__ uint32_t *p = reinterpret_cast<volatile __gm__ uint32_t *>(localFlag);
    uint32_t spin = 0;
    constexpr uint32_t kFenceInterval = 64;
    while (true) {
        __asm__ __volatile__("" ::: "memory");
        dcci(reinterpret_cast<__gm__ void *>(const_cast<__gm__ uint32_t *>(p)), SINGLE_CACHE_LINE);
        __asm__ __volatile__("" ::: "memory");
        if (*p >= threshold) {
            return true;
        }
        if (maxSpins != 0 && spin >= maxSpins) {
            return false;
        }
        if ((++spin % kFenceInterval) == 0) {
            pipe_barrier(PIPE_ALL);
        }
    }
}

inline AICORE bool MockTryWfeReady(__gm__ uint32_t *localFlag, uint32_t threshold,
                                   uint32_t maxSpins = kDefaultWfeMaxSpins)
{
    return MockTryWfeCounter(localFlag, threshold, maxSpins);
}

inline AICORE bool MockTryWfeFree(__gm__ uint32_t *localFlag, uint32_t threshold,
                                  uint32_t maxSpins = kDefaultWfeMaxSpins)
{
    return MockTryWfeCounter(localFlag, threshold, maxSpins);
}

inline AICORE void MockWfeReady(__gm__ uint32_t *localFlag, uint32_t threshold)
{
    (void)MockTryWfeReady(localFlag, threshold, 0);
}

inline AICORE void MockWfeFree(__gm__ uint32_t *localFlag, uint32_t threshold)
{
    (void)MockTryWfeFree(localFlag, threshold, 0);
}

inline AICORE void MockSetFault(__gm__ uint32_t *faultFlag, uint32_t faultCode)
{
    if (faultFlag != nullptr) {
        volatile __gm__ uint32_t *ptr = reinterpret_cast<volatile __gm__ uint32_t *>(faultFlag);
        __asm__ __volatile__("" ::: "memory");
        *ptr = faultCode;
        __asm__ __volatile__("" ::: "memory");
        dcci(reinterpret_cast<__gm__ void *>(const_cast<__gm__ uint32_t *>(ptr)), SINGLE_CACHE_LINE);
        dsb(DSB_DDR);
    }
}

// MOCK: design doc 5.4 SPR_BOUNDARY_MASK fault.
//
// LPU WSE: writing SPR_RDY_<DIR> when SPR_BOUNDARY_MASK has that direction
// disabled triggers a hardware fault / squash.
//
// A2/A3 mock: explicit early-exit + sentinel write so the host can detect the
// out-of-bound attempt.  Real boards will raise a fault; here we trap softly
// by writing a sentinel and aborting the kernel branch.  The host launcher
// inspects a "fault sentinel" GM word after each kernel and fails the run.
inline AICORE void MockBoundaryFault(__gm__ uint32_t *faultSentinel, uint32_t faultCode)
{
    if (faultSentinel != nullptr) {
        *reinterpret_cast<volatile __gm__ uint32_t *>(faultSentinel) = faultCode;
    }
    // Best-effort halt of the current kernel branch.  Real silicon will fault
    // here; on A2/A3 we just stop emitting further GridPipe ops in this branch.
}

// Fault codes mirror SPR_BOUNDARY_MASK fields (design doc section 5.2).
inline constexpr uint32_t kFaultPushNorth = 0x101;
inline constexpr uint32_t kFaultPushEast = 0x102;
inline constexpr uint32_t kFaultPushWest = 0x103;
inline constexpr uint32_t kFaultPushSouth = 0x104;
inline constexpr uint32_t kFaultPushSource = 0x105; // Always illegal.
inline constexpr uint32_t kFaultPopNorth = 0x201;
inline constexpr uint32_t kFaultPopEast = 0x202;
inline constexpr uint32_t kFaultPopWest = 0x203;
inline constexpr uint32_t kFaultPopSouth = 0x204;
// TPOP tried to drain a slot outside this core's own SRAM segment.  The NoC
// fabric has no remote-read path, so this can only happen via a mis-wired mock;
// the GmSramArena guard in GRID_TRY_TPOP_IMPL traps it here (design: NoC is
// write-only, TPOP is local-only).
inline constexpr uint32_t kFaultPopNonLocal = 0x205;
inline constexpr uint32_t kFaultWaitReadyTimeout = 0x301;
inline constexpr uint32_t kFaultWaitFreeTimeout = 0x302;

// Direction-keyed fault code lookup.  Explicit switch avoids relying on the
// numeric layout of GridDirection so renumbering the enum cannot silently
// remap fault codes.
AICORE constexpr uint32_t PushFaultCode(GridDirection dir)
{
    switch (dir) {
        case GridDirection::NORTH:
            return kFaultPushNorth;
        case GridDirection::EAST:
            return kFaultPushEast;
        case GridDirection::WEST:
            return kFaultPushWest;
        case GridDirection::SOUTH:
            return kFaultPushSouth;
        case GridDirection::SOURCE:
            return kFaultPushSource;
    }
    return kFaultPushSource;
}

AICORE constexpr uint32_t PopFaultCode(GridDirection dir)
{
    switch (dir) {
        case GridDirection::NORTH:
            return kFaultPopNorth;
        case GridDirection::EAST:
            return kFaultPopEast;
        case GridDirection::WEST:
            return kFaultPopWest;
        case GridDirection::SOUTH:
            return kFaultPopSouth;
        case GridDirection::SOURCE:
            return 0; // SOURCE pop is legal; never raises a boundary fault.
    }
    return 0;
}

} // namespace grid_mock
} // namespace pto

// ===========================================================================
// Section 3: CCE-intrinsic-style API for neighbor-core monotonic counters.
//            (formerly include/pto/common/grid_counter_intrinsic.hpp)
//
// These two functions intentionally sit at the same layer as hardware-adapter
// intrinsics such as dcci: callers pass the concrete backend operand, while the
// function body is the only place that knows whether the target is native
// hardware or today's GM-counter mock.
//
// When hardware support is available, define PTO_GRID_COUNTER_NATIVE_INTRINSIC
// and provide compiler builtins with the same contract.  Call sites do not need
// to change.
// ===========================================================================

namespace pto {

enum class NeighborCounterKind : uint8_t
{
    Ready = 0,
    Free = 1,
};

// Backend operand for the neighbor-counter intrinsic.
//
// Native hardware: `addr` is ignored and the compiler lowers (kind, dir, value)
// to SPR/WFE instructions.
// Current mock: `addr` points to the GM counter that represents either the
// peer-visible counter for set or the local mirror counter for wait.
struct NeighborCounterOperand {
    __gm__ uint32_t *addr = nullptr;
};

// Set a neighbor-visible monotonic counter `dist` hops away along `dir`.
//
// `dist` is the routed-unicast hop count; dist == 1 is the original adjacent
// doorbell (fully backward compatible).  For dist > 1 the counter write is
// routed `dist` hops in the kind-implied direction -- ready flows downstream
// along `dir`, free flows upstream against `dir` -- so a producer can ring a
// consumer K hops away (and vice versa for the free credit).
//
// Hardware contract (dist == 1):
//   mtspr_neighbor_counter(Ready, EAST, 1, value) ~= mtspr SPR_RDY_EAST, value
//   mtspr_neighbor_counter(Free,  EAST, 1, value) ~= mtspr SPR_FREE_EAST, value
// Native lowering for dist > 1 must provide a routed remote-notify that lands on
// the target core's same-named SPR; the direction-only SPR doorbell alone is
// adjacency-scoped.
//
// Memory ordering: release.  Earlier payload writes must become visible before
// the peer can observe this counter update.
AICORE inline void mtspr_neighbor_counter(NeighborCounterKind kind, uint32_t dir, uint32_t dist, uint32_t value,
                                          NeighborCounterOperand operand = {})
{
#if defined(PTO_GRID_COUNTER_NATIVE_INTRINSIC)
    (void)operand;
    __builtin_pto_mtspr_neighbor_counter(static_cast<uint32_t>(kind), dir, dist, value);
#else
    // Mock target is fully encoded in operand.addr (the caller resolves the
    // K-hop rank via RemoteCounterPtr), so the mock store needs neither dir nor
    // dist -- they exist only to drive the native routed-notify above.
    (void)dir;
    (void)dist;
    if (kind == NeighborCounterKind::Ready) {
        grid_mock::MockMtsprReady(operand.addr, value);
    } else {
        grid_mock::MockMtsprFree(operand.addr, value);
    }
#endif
}

// Wait until a neighbor-produced counter mirror reaches threshold.
//
// Hardware contract:
//   wfe_neighbor_counter(Ready, EAST, n) ~= wfe SPR_RDY_EAST, n
//   wfe_neighbor_counter(Free,  EAST, n) ~= wfe SPR_FREE_EAST, n
//
// Memory ordering: acquire.  Operations after the wait must not be reordered
// before the counter condition has been satisfied.
AICORE inline bool wfe_neighbor_counter(NeighborCounterKind kind, uint32_t dir, uint32_t threshold,
                                        NeighborCounterOperand operand = {}, uint32_t maxSpins = 0)
{
#if defined(PTO_GRID_COUNTER_NATIVE_INTRINSIC)
    (void)operand;
    (void)maxSpins;
    __builtin_pto_wfe_neighbor_counter(static_cast<uint32_t>(kind), dir, threshold);
    return true;
#else
    (void)dir;
    if (kind == NeighborCounterKind::Ready) {
        return grid_mock::MockTryWfeReady(operand.addr, threshold, maxSpins);
    }
    return grid_mock::MockTryWfeFree(operand.addr, threshold, maxSpins);
#endif
}

} // namespace pto

// ===========================================================================
// Section 4: CCE-intrinsic-style API for neighbor-core SRAM addressing and
//            transfer.  (formerly include/pto/common/grid_sram_intrinsic.hpp)
//
// Grid TPUSH/TPOP calls this layer so the same protocol can use either native
// __builtin_pto_* instructions or the current A2/A3 mock lowering.
//
// Grid TPUSH/TPOP payload movement targets neighbor-visible on-chip storage,
// not a global-memory object in the hardware contract. Hardware exposes this
// path as local SRAM <-> neighbor SRAM without distinguishing UB and CBUF in
// the cross-core intrinsic names. Current A2/A3 demos emulate those neighbor
// windows with local GM windows; the mock address operand is therefore kept as
// backend context while the public API exposes address-register style values.
// ===========================================================================

namespace pto {

// Address-register model for a neighbor-visible SRAM location.
// Native hardware: value is the encoded SRAM address register payload.
// Current mock: value is the integer form of the backing fake-window pointer.
struct neighbor_sram_addr {
    uint64_t value = 0;
};

// Backend operand used only by the mock lowering. Native hardware ignores it
// and derives the neighbor SRAM mapping from dir/peer configuration registers.
struct NeighborSramOperand {
    __gm__ void *runtimeCtx = nullptr;
};

// ---------------------------------------------------------------------------
// GmSramArena: explicit GM address-segment model of future-hardware per-core
// SRAM.
//
// Real silicon gives every core a private on-chip SRAM that the NoC fabric can
// only *write* into from a neighbor (TPUSH = cross-hop write), never *read* out
// of remotely (TPOP only drains the local core's own SRAM).  Until that
// hardware exists we model the SRAM as a contiguous GM arena cut into equal,
// per-core address segments.  Core `c` owns segment `c`:
//
//   [base + c*segBytes, base + (c+1)*segBytes)
//
// The NoC contract this encodes:
//   * a core may WRITE across segments  (TPUSH pushes into a neighbor segment),
//   * a core may only READ its own segment (TPOP pops from local SRAM).
//
// The arena is the single source of truth for "which core owns this address",
// so the mock TPOP path can reject a cross-segment read instead of silently
// servicing it through the GM-backed fake window (which physically *can* read
// any address, unlike the fabric it stands in for).
struct GmSramArena {
    uint64_t base = 0;     // segment 0 base == contiguous arena base
    uint64_t segBytes = 0; // bytes per per-core segment (== HCCL winSize in the demo)
    uint32_t numSegs = 0;  // number of cores / segments

    AICORE constexpr uint64_t SegmentBase(int seg) const
    {
        return base + static_cast<uint64_t>(seg) * segBytes;
    }

    // Index of the segment that owns `addr`, or -1 if `addr` is outside the arena.
    AICORE constexpr int SegmentOf(uint64_t addr) const
    {
        if (numSegs == 0 || segBytes == 0 || addr < base) {
            return -1;
        }
        uint64_t idx = (addr - base) / segBytes;
        return idx < numSegs ? static_cast<int>(idx) : -1;
    }

    // True iff [addr, addr+bytes) lies entirely within segment `seg`.  This is
    // exactly the "may core `seg` read this slot?" test used by the TPOP guard.
    AICORE constexpr bool InSegment(int seg, uint64_t addr, uint64_t bytes) const
    {
        if (seg < 0 || static_cast<uint32_t>(seg) >= numSegs) {
            return false;
        }
        uint64_t lo = SegmentBase(seg);
        uint64_t hi = lo + segBytes;
        return addr >= lo && (addr + bytes) <= hi && (addr + bytes) >= addr; // last term traps wrap-around
    }
};

// Compile-time self-test of the segment classifier.  It is built into every
// A2/A3 kernel that pulls in this header (GridTPush.hpp -> pto_instr_impl.hpp),
// so a regression in the segment math fails the build rather than silently
// mis-routing a TPOP.  It also doubles as executable documentation of the rule.
AICORE constexpr bool GmSramArenaSelfCheck()
{
    GmSramArena arena{0x1000, 0x100, 4};          // 4 cores, 0x100-byte segments, based at 0x1000
    bool ok = true;
    ok = ok && (arena.SegmentOf(0x1000) == 0);    // first byte of core 0
    ok = ok && (arena.SegmentOf(0x11FF) == 1);    // last byte of core 1
    ok = ok && (arena.SegmentOf(0x1200) == 2);    // first byte of core 2
    ok = ok && (arena.SegmentOf(0x0FFF) == -1);   // below the arena
    ok = ok && (arena.SegmentOf(0x1400) == -1);   // past the arena ([0x1000,0x1400))
    ok = ok && arena.InSegment(1, 0x1100, 0x40);  // wholly inside core 1 -> local
    ok = ok && !arena.InSegment(1, 0x11F0, 0x40); // spills past core 1 -> not local
    ok = ok && !arena.InSegment(1, 0x1200, 0x10); // core 1 reading core 2 -> not local
    return ok;
}
static_assert(GmSramArenaSelfCheck(), "GmSramArena segment classifier self-test failed");

namespace grid_sram_mock {

AICORE uint64_t MockGetNeighborSramAddr(__gm__ void *runtimeCtx, uint64_t localSramAddr, int32_t peerRank);
AICORE void MockCopySramToNeighborSram(neighbor_sram_addr dst, neighbor_sram_addr src, uint32_t bytes, uint64_t config);
AICORE void MockCopyNeighborSramToSram(neighbor_sram_addr dst, neighbor_sram_addr src, uint32_t bytes, uint64_t config);
// Read-locality predicate for the TPOP guard.  Returns true iff
// [slotAddr, slotAddr+bytes) lies inside callerRank's own GmSramArena segment.
// The demo owns the definition because it knows the runtime window layout.
AICORE bool MockSramPopIsLocal(__gm__ void *runtimeCtx, uint64_t slotAddr, uint32_t bytes, int32_t callerRank);

} // namespace grid_sram_mock

// Resolve a local SRAM slot address to the same slot in a neighbor core.
//
// Parameter order follows CCE data/address instructions: output register first,
// source address register next, then topology/control operands.
AICORE inline void get_neighbor_sram_addr(neighbor_sram_addr &dst, neighbor_sram_addr src, uint32_t dir,
                                          int32_t peerRank, NeighborSramOperand operand = {})
{
#if defined(PTO_GRID_SRAM_NATIVE_INTRINSIC)
    (void)operand;
    dst.value = __builtin_pto_get_neighbor_sram_addr(src.value, dir, peerRank);
#else
    (void)dir;
    dst.value = grid_sram_mock::MockGetNeighborSramAddr(operand.runtimeCtx, src.value, peerRank);
#endif
}

// Cross-core payload writes. Source and destination are SRAM address-register
// values; the intrinsic name intentionally does not expose UB/CBUF variants.
AICORE inline void copy_sram_to_neighbor_sram(neighbor_sram_addr dst, neighbor_sram_addr src, uint32_t bytes,
                                              uint64_t config)
{
#if defined(PTO_GRID_SRAM_NATIVE_INTRINSIC)
    __builtin_pto_copy_sram_to_neighbor_sram(dst.value, src.value, bytes, config);
#else
    // A2/A3 validation keeps the intrinsic call shape and only changes the
    // final lowering to a GM-backed fake window.
    grid_sram_mock::MockCopySramToNeighborSram(dst, src, bytes, config);
#endif
}

// Cross-core payload read interface. The native lowering is intentionally left
// as an interface placeholder until hardware/compiler support is available.
AICORE inline void copy_neighbor_sram_to_sram(neighbor_sram_addr dst, neighbor_sram_addr src, uint32_t bytes,
                                              uint64_t config)
{
#if defined(PTO_GRID_SRAM_NATIVE_INTRINSIC)
    (void)dst;
    (void)src;
    (void)bytes;
    (void)config;
#else
    // Keep TPOP visible as intrinsic-style SRAM transfer in the A2/A3 mock.
    grid_sram_mock::MockCopyNeighborSramToSram(dst, src, bytes, config);
#endif
}

// TPOP read-locality guard.  Returns true iff `slot` is the calling core's own
// SRAM segment, i.e. the read is legal under the NoC "TPOP only pops local
// SRAM" rule.
//
// Native hardware physically cannot read a neighbor's SRAM (the fabric has no
// remote-read path), so the read address is always local by construction and
// the native lowering is a no-op that returns true.  The A2/A3 mock backs SRAM
// with a GM-mapped fake window that *can* be read at any address, so we must
// check explicitly: the mock validates `slot` against callerRank's GmSramArena
// segment and reports a cross-segment read instead of silently servicing it.
AICORE inline bool sram_pop_is_local(neighbor_sram_addr slot, uint32_t bytes, int32_t callerRank,
                                     NeighborSramOperand operand = {})
{
#if defined(PTO_GRID_SRAM_NATIVE_INTRINSIC)
    (void)slot;
    (void)bytes;
    (void)callerRank;
    (void)operand;
    return true;
#else
    return grid_sram_mock::MockSramPopIsLocal(operand.runtimeCtx, slot.value, bytes, callerRank);
#endif
}

} // namespace pto

#endif // PTO_A2A3_GRID_INTRINSIC_HPP
