/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under
the terms and conditions of CANN Open Software License Agreement Version 2.0
(the "License"). Please refer to the License for details. You may not use this
file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON AN "AS
IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A
PARTICULAR PURPOSE. See LICENSE in the root of the software repository for the
full text of the License.
*/

// Single-PE FFN cell, AllGather split variant — high-fidelity WSE pseudo-kernel.
//
// This is NOT a mixed Cube/Vec A2/A3 kernel: it models a single WSE processing
// element whose matmul unit and vector unit share one L1 (the "cube L1" and the
// "vector UB" are fused into a single L1 address space).  The cell body is one
// straight-line instruction stream — no __DAV_CUBE__/__DAV_VEC__ split.  The
// instruction names and directions mirror the fabric:
//
//   Batcher   -> L1        : TLOAD            (stream this cell's input X in)
//   DRAM      -> L1        : TLOAD            (weight load)
//   L1        -> L0A/L0B   : TMOV             (feed the matmul unit)
//   L0C       -> L1(UB)    : TMOV(pipe, acc)  (fixpipe drain of the matmul
//                                              accumulator into a vector tile;
//                                              the C2V FIFO is the ISA's only
//                                              Acc->Vec path, see tpipe_tmov_inl)
//   L1        -> L1' (NoC)  : hidden shard exchange rides the inter-core NoC that
//                            wires every cell's L1 together — it never parks in
//                            Batcher mem.  Publish = expose the shard in this
//                            cell's NoC-visible L1 SRAM window; comm::TAllGather
//                            reads peer cells' L1 SRAM over the NoC into local L1.
//   L0C       -> L1        : TMOV             (drain down accumulator to L1)
//   L1        -> Batcher   : TSTORE           (write the final result back)
//
// Two memory domains, kept distinct:
//   * Batcher mem — host I/O only: streams input X into each cell's L1 and
//     collects the final [T, Hc] result out of L1.  Nothing intermediate lives
//     here.
//   * NoC-interconnected L1 — the cells' L1 SRAMs are wired together by the
//     on-chip NoC bus; the row-local hidden AllGather moves L1<->L1 across cells
//     over that bus, with no Batcher round-trip.
//
// Row role: gridRows = data-parallel token shards, gridCols = model-parallel FFN
// intermediate shards.  Each cell (row, col) computes gate/up/down for its own
// [T, Fi] intermediate slice.  The down projection needs the full row-concat
// hidden [T, F=Fi*cols], so the cell AllGathers the other columns' [T, Fi]
// shards from their L1 SRAM windows over the NoC, interleaved with down.
//
// Fine-grained AllGather<->Matmul overlap: the original variant did one big
// AllGather to build [T, F], then one down GEMM.  Here the gather is split into
// FFN_AG_CHUNKS feature-chunks; each gathered chunk is immediately consumed by an
// accumulating down GEMM over that K-band, and the next chunk's gather (MTE2) is
// launched before the current chunk's GEMM (M) so communication hides under
// compute.  down projection is sharded by output-H columns, so there is no
// post-down reduce — each cell writes its own [T, Hc] output shard.
//
// SCOPE — this is high-fidelity pseudo-code, not a buildable dav-c220 object.
// dav-c220 (Ascend 910B) splits matmul (cube/AIC) and vector (AIV) into two
// sub-cores with *disjoint* instruction sets and compiles the kernel once per
// sub-core; the stock mixed kernels therefore gate every op behind
// __DAV_CUBE__/__DAV_VEC__ so each sub-core only ever instantiates its own
// intrinsics.  A unified WSE PE has no such split, so this single stream
// deliberately keeps TMATMUL and the vlrelu/vmul/vconv vector ops side by side.
// It consequently does NOT compile for dav-c220 (each sub-core pass rejects the
// other sub-core's intrinsics with "does not support the given target feature").
// That is by design — the goal is a faithful WSE instruction stream, not an A3
// binary.  Instruction names, signatures, tile types and data flow are all real
// PTO ISA (e.g. the only legal Acc->Vec path, the C2V fixpipe, is spelled as the
// directional TMOV(pipe, acc)/TMOV(vec, pipe) from tpipe_tmov_inl.hpp).

#include <cstddef>
#include <cstdint>

#include <pto/pto-inst.hpp>
#include <pto/common/fifo.hpp>

// Collective plumbing.  We pull only comm_types (CollEngine / ParallelGroup /
// Signal) plus the two signal *_IMPL headers used for the cross-cell arrival
// barrier.  We deliberately do NOT include pto/comm/pto_comm_inst.hpp: the
// unified comm::TAllGather this demo needs (feature-axis concat straight into an
// L1 matmul tile) is defined in-file below, and the aggregate header trips the
// PIPE_FIX redefinition in a mixed-arch TU.
#include <pto/comm/comm_types.hpp>
#include <pto/comm/a2a3/TNotify.hpp>
#include <pto/comm/a2a3/TWait.hpp>

#include "ffn_config.hpp"
#include "tpipe_tmov_inl.hpp" // directional TMOV(pipe,tile)/TMOV(tile,pipe) fixpipe

#ifdef __CCE_AICORE__
using namespace pto;

// ---------------------------------------------------------------------------
// comm::TAllGather — feature-axis AllGather, one fine-grained chunk per call.
//
// Signature style follows the library collectives in pto/comm/pto_comm_inst.hpp
// (TGATHER / TREDUCE): a leading CollEngine template selector, a ParallelGroup
// of per-rank source views, and a staging tile destination.  Two things differ,
// both dictated by the FFN down projection:
//
//   * Concatenation axis.  comm::TGATHER stacks rank r along DIM_3 (rows), which
//     would give [cols*T, Fi]; the down GEMM instead wants the columns laid out
//     along the feature axis as [T, F].  TAllGather therefore places rank r at
//     feature offset r*shardCols.  (See 2026-06-02 GAP analysis: TGATHER's row
//     stacking forces an extra TLOAD+TINSERT re-layout; TAllGather folds it in.)
//
//   * Chunking.  Instead of gathering the whole [T, F] in one shot, the caller
//     asks for a single [T, chunkCols] band starting at global feature offset
//     `featOffset`.  The band is TLOAD-ed over the NoC straight from the owning
//     peer cell's L1 SRAM window into `dstL1Mat` (a matmul L1 tile — TLOAD
//     reformats the peer's row-major shard into the tile's ND2NZ block layout),
//     ready for TMOV -> L0A.  The source views live in peer L1 SRAM (resolved by
//     the NoC), never in Batcher mem.  Splitting the gather this way lets chunk
//     k+1's TLOAD (MTE2) overlap chunk k's GEMM (M).
//
// AIV engine only: the A2/A3 CCU path does not exist (mirrors the TGATHER stub).
// The [T, chunkCols] chunk shape is taken from the destination tile's static
// ValidRow/ValidCol so the ND2NZ TLOAD sees a fully static GlobalTensor.
// ---------------------------------------------------------------------------
namespace pto {
namespace comm {

template <CollEngine engine = CollEngine::AIV, typename ParallelGroupType, typename DstL1MatTile,
          typename... WaitEvents>
PTO_INST RecordEvent TAllGather(ParallelGroupType &parallelGroup, DstL1MatTile &dstL1Mat, int featOffset, int chunkCols,
                                int shardCols, WaitEvents &...events)
{
    static_assert(engine == CollEngine::AIV, "TAllGather<CollEngine::CCU> requires A5 hardware; AIV only on A2/A3.");
    WaitAllEvents(events...);

    using GlobalSrc = typename ParallelGroupTraits<ParallelGroupType>::GlobalDataType;
    using T = typename GlobalSrc::RawDType;
    static_assert(std::is_same_v<T, typename DstL1MatTile::DType>,
                  "TAllGather: staging tile element type must match the shard element type");

    // Per-shard static geometry — the source shard is [nRows, shardCols_] dense.
    constexpr int nRows = GlobalSrc::staticShape[3];
    constexpr int shardCols_ = GlobalSrc::staticShape[4];
    constexpr int chunkCols_ = DstL1MatTile::ValidCol; // fine-grained band width
    PTO_ASSERT(chunkCols == chunkCols_ && shardCols == shardCols_, "TAllGather: chunk/shard dims must match the tiles");

    // Each fine-grained chunk is sized so it lands wholly inside one column's
    // shard (F/FFN_AG_CHUNKS divides Fi for the shipped configs).  That keeps the
    // gather a single peer read; a band straddling two shards would need a
    // TINSERT seam, which the aligned FFN tiling never hits.
    const int srcRank = featOffset / shardCols_;
    const int inShard = featOffset - srcRank * shardCols_;
    PTO_ASSERT(inShard + chunkCols_ <= shardCols_, "TAllGather: fine-grained chunk must lie within one rank's shard");

    // Static column-band view of peer `srcRank`'s L1 SRAM shard (read over the
    // NoC): only the base pointer varies at runtime (data() + inShard);
    // shape/stride are static so the Mat-tile ND2NZ TLOAD is legal.
    using BandShape = Shape<1, 1, 1, nRows, chunkCols_>;
    using BandStride = Stride<nRows * shardCols_, nRows * shardCols_, nRows * shardCols_, shardCols_, 1>;
    using BandView = GlobalTensor<T, BandShape, BandStride, GlobalSrc::layout>;

    BandView band(parallelGroup[srcRank].data() + inShard);
    TLOAD(dstL1Mat, band); // peer L1 SRAM (NoC) -> local L1 matmul tile
    return {};
}

} // namespace comm
} // namespace pto

// ---------------------------------------------------------------------------
// Per-cell NoC-visible L1 SRAM window (NOT Batcher mem).  Each cell exposes a
// slice of its L1 on the inter-core NoC so peers can read it; the host reserves
// FFN_GRID_WINDOW_BYTES per cell for it.  On this single device the cells' SRAM
// windows form one contiguous arena, so a peer's L1 window is plain NoC-address
// arithmetic: window(cell) = base + cell * FFN_GRID_WINDOW_BYTES.
//   [0 .. 3]                      int32 "shard published" ready flag (NoC)
//   [FFN_GRID_FLAGS_BYTES ..]     fp16 [T, Fi] hidden shard (NoC-readable L1)
// ---------------------------------------------------------------------------
constexpr int kWinFlagOffset = 0;
constexpr int kWinHiddenOffset = FFN_GRID_FLAGS_BYTES;

// Fine-grained AllGather<->down-Matmul pipeline depth.  The single [T, F]
// AllGather is issued as FFN_AG_CHUNKS feature-chunks, each followed by one
// accumulating down GEMM over its K-band, so the collective overlaps compute.
constexpr int FFN_AG_CHUNKS = 4;
static_assert((FFN_FFN_TOTAL_TILE % FFN_AG_CHUNKS) == 0, "AllGather chunking requires F divisible by FFN_AG_CHUNKS.");
constexpr int FFN_AG_CHUNK_TILE = FFN_FFN_TOTAL_TILE / FFN_AG_CHUNKS; // features per gather pass
static_assert((FFN_FFN_TILE % FFN_AG_CHUNK_TILE) == 0 || (FFN_AG_CHUNK_TILE % FFN_FFN_TILE) == 0,
              "Each fine-grained chunk must nest inside one column shard.");

using GateF32Tile = Tile<TileType::Vec, float, FFN_TOKEN_TILE, FFN_FFN_TILE, BLayout::RowMajor>;
using UpF32Tile = GateF32Tile;
using HiddenF32Tile = GateF32Tile;
using HiddenF16Tile = Tile<TileType::Vec, half, FFN_TOKEN_TILE, FFN_FFN_TILE, BLayout::RowMajor>;

// Per-cell published hidden shard [T, Fi] living in a peer's NoC-visible L1 SRAM window.
using ShapeShard = Shape<1, 1, 1, FFN_TOKEN_TILE, FFN_FFN_TILE>;
using StrideShard = Stride<FFN_TOKEN_TILE * FFN_FFN_TILE, FFN_TOKEN_TILE * FFN_FFN_TILE, FFN_TOKEN_TILE * FFN_FFN_TILE,
                           FFN_FFN_TILE, 1>;
using GHiddenShardF16 = GlobalTensor<half, ShapeShard, StrideShard, Layout::ND>;

// Output shard [T, Hc] written to Batcher mem (models the next layer's batcher
// input; there is no L1 -> DRAM path in this cell).
using ShapeYShard = Shape<1, 1, 1, FFN_TOKEN_TILE, FFN_MODEL_SHARD_TILE>;
using StrideYShard = Stride<FFN_TOKEN_TILE * FFN_MODEL_TILE, FFN_TOKEN_TILE * FFN_MODEL_TILE,
                            FFN_TOKEN_TILE * FFN_MODEL_TILE, FFN_MODEL_TILE, 1>;
using GYShardF32 = GlobalTensor<float, ShapeYShard, StrideYShard, Layout::ND>;

constexpr int AlignUp(int value, int align)
{
    return ((value + align - 1) / align) * align;
}

// -------- unified L1 address map (matmul tiles + vector tiles share L1) -------
constexpr int kL1AlignBytes = 0x1000;
constexpr int kL1X = 0x00000;
constexpr int kL1WGate = AlignUp(kL1X + FFN_X_BYTES, kL1AlignBytes);
constexpr int kL1WUp = AlignUp(kL1WGate + FFN_W_GATE_BYTES, kL1AlignBytes);
constexpr int kL1WDown = AlignUp(kL1WUp + FFN_W_UP_BYTES, kL1AlignBytes);
constexpr int kL1GateF32 = AlignUp(kL1WDown + FFN_W_DOWN_BYTES, kL1AlignBytes);
constexpr int kL1UpF32 = AlignUp(kL1GateF32 + FFN_GATE_PARTIAL_BYTES, kL1AlignBytes);
constexpr int kL1HiddenF32 = AlignUp(kL1UpF32 + FFN_UP_PARTIAL_BYTES, kL1AlignBytes);
constexpr int kL1HiddenF16 = AlignUp(kL1HiddenF32 + FFN_GATE_PARTIAL_BYTES, kL1AlignBytes);
constexpr int kL1Chunk0 = AlignUp(kL1HiddenF16 + FFN_HIDDEN_BYTES, kL1AlignBytes);
constexpr int kL1ChunkBytes = FFN_TOKEN_TILE * FFN_AG_CHUNK_TILE * 2; // fp16 [T, chunk]
constexpr int kL1Chunk1 = AlignUp(kL1Chunk0 + kL1ChunkBytes, kL1AlignBytes);
constexpr int kL1Down = AlignUp(kL1Chunk1 + kL1ChunkBytes, kL1AlignBytes); // down result staged in L1

#endif // __CCE_AICORE__

__global__ AICORE void DistributedFfnGridAllGatherMixedKernel(__gm__ uint8_t *fftsAddr,
                                                              __gm__ uint8_t *gatherPipeWindow, __gm__ uint8_t *x,
                                                              __gm__ uint8_t *wGate, __gm__ uint8_t *wUp,
                                                              __gm__ uint8_t *wDown, __gm__ uint8_t *gatePartial,
                                                              __gm__ uint8_t *upPartial, __gm__ uint8_t *hiddenIn,
                                                              __gm__ uint8_t *downPartial, __gm__ uint8_t *yOutput,
                                                              __gm__ uint8_t *hcclCtxRaw, int gridRows, int gridCols)
{
#ifdef __CCE_AICORE__
    static_assert((FFN_MODEL_TILE % FFN_GRID_COLS) == 0, "AllGather split requires H divisible by gridCols.");

    set_ffts_base_addr(reinterpret_cast<uint64_t>(fftsAddr));

    int blockIdx = get_block_idx();
    int totalBlocks = gridRows * gridCols;
    if (blockIdx < 0 || blockIdx >= totalBlocks) {
        return;
    }
    int row = blockIdx / gridCols;
    int col = blockIdx - row * gridCols;

    constexpr int validM = FFN_TOKEN_TILE;            // T
    constexpr int validK = FFN_MODEL_TILE;            // H
    constexpr int validN = FFN_FFN_TILE;              // Fi
    constexpr int validF = FFN_FFN_TOTAL_TILE;        // F = Fi * cols
    constexpr int validChunk = FFN_AG_CHUNK_TILE;     // F / FFN_AG_CHUNKS
    constexpr int validHShard = FFN_MODEL_SHARD_TILE; // Hc = H / cols
    constexpr int blockAlign = C0_SIZE_BYTE / static_cast<int>(sizeof(half));
    constexpr int M = ((validM + 15) / 16) * 16;
    constexpr int K = ((validK + blockAlign - 1) / blockAlign) * blockAlign;
    constexpr int N = ((validN + blockAlign - 1) / blockAlign) * blockAlign;
    constexpr int KChunk = ((validChunk + blockAlign - 1) / blockAlign) * blockAlign;
    constexpr int KDown = ((validF + blockAlign - 1) / blockAlign) * blockAlign;
    constexpr int NDown = ((validHShard + blockAlign - 1) / blockAlign) * blockAlign;

    // -------- global (DRAM) views for the per-cell weights + activation --------
    using GX = GlobalTensor<half, Shape<1, 1, 1, validM, validK>,
                            Stride<validM * validK, validM * validK, validM * validK, validK, 1>>;
    using GW = GlobalTensor<half, Shape<1, 1, 1, validK, validN>,
                            Stride<validK * validN, validK * validN, validK * validN, validN, 1>>;
    using GWDown =
        GlobalTensor<half, Shape<1, 1, 1, validF, validHShard>,
                     Stride<validF * validHShard, validF * validHShard, validF * validHShard, validHShard, 1>>;

    // -------- L1 matmul tiles --------
    using TileA = Tile<TileType::Mat, half, M, K, BLayout::ColMajor, validM, validK, SLayout::RowMajor, 512>;
    using TileB = Tile<TileType::Mat, half, K, N, BLayout::ColMajor, validK, validN, SLayout::RowMajor, 512>;
    using HiddenChunkMat =
        Tile<TileType::Mat, half, M, KChunk, BLayout::ColMajor, validM, validChunk, SLayout::RowMajor, 512>;
    using WDownMat =
        Tile<TileType::Mat, half, KDown, NDown, BLayout::ColMajor, validF, validHShard, SLayout::RowMajor, 512>;
    // down result staged in L1 (Acc -> Mat drain, then TSTORE -> Batcher mem).
    using DownMat =
        Tile<TileType::Mat, float, M, NDown, BLayout::ColMajor, validM, validHShard, SLayout::RowMajor, 512>;
    // -------- L0 operand / accumulator tiles --------
    using TL = TileLeft<half, M, K, validM, validK>;
    using TR = TileRight<half, K, N, validK, validN>;
    using TC = TileAcc<float, M, N, validM, validN>;
    using TLDown = TileLeft<half, M, KChunk, validM, validChunk>;
    using TRDown = TileRight<half, KChunk, NDown, validChunk, validHShard>;
    using TCDown = TileAcc<float, M, NDown, validM, validHShard>;

    TileA xMat;
    TileB wGateMat;
    TileB wUpMat;
    WDownMat wDownMat;
    HiddenChunkMat hiddenChunkMat[2];
    DownMat downMat;
    TASSIGN(xMat, kL1X);
    TASSIGN(wGateMat, kL1WGate);
    TASSIGN(wUpMat, kL1WUp);
    TASSIGN(wDownMat, kL1WDown);
    TASSIGN(hiddenChunkMat[0], kL1Chunk0);
    TASSIGN(hiddenChunkMat[1], kL1Chunk1);
    TASSIGN(downMat, kL1Down);

    TL aT;
    TR bT;
    TC cT;
    TLDown aDownT;
    TRDown bDownT;
    TCDown cDownT;
    TASSIGN(aT, 0x0);
    TASSIGN(bT, 0x0);
    TASSIGN(cT, 0x0);
    TASSIGN(aDownT, 0x0);
    TASSIGN(bDownT, 0x0);
    TASSIGN(cDownT, 0x0);

    GateF32Tile gateF32;
    UpF32Tile upF32;
    HiddenF32Tile hiddenF32;
    HiddenF16Tile hiddenF16;
    TASSIGN(gateF32, kL1GateF32);
    TASSIGN(upF32, kL1UpF32);
    TASSIGN(hiddenF32, kL1HiddenF32);
    TASSIGN(hiddenF16, kL1HiddenF16);

    // -------- per-cell input slices --------
    __gm__ uint8_t *xBlock = x + blockIdx * FFN_X_BYTES;              // Batcher-mem input X
    __gm__ uint8_t *wGateBlock = wGate + blockIdx * FFN_W_GATE_BYTES; // DRAM weights
    __gm__ uint8_t *wUpBlock = wUp + blockIdx * FFN_W_UP_BYTES;
    __gm__ uint8_t *wDownBlock = wDown + blockIdx * FFN_W_DOWN_BYTES;
    __gm__ uint8_t *gateBlock = gatePartial + blockIdx * FFN_GATE_PARTIAL_BYTES;
    __gm__ uint8_t *upBlock = upPartial + blockIdx * FFN_UP_PARTIAL_BYTES;

    // -------- per-cell NoC-visible L1 SRAM windows (this cell + row peers) -----
    __gm__ uint8_t *myL1Window = gatherPipeWindow + blockIdx * FFN_GRID_WINDOW_BYTES;
    __gm__ uint8_t *myHiddenSlot = myL1Window + kWinHiddenOffset;
    comm::Signal myReadyFlag(reinterpret_cast<__gm__ int32_t *>(myL1Window + kWinFlagOffset));

    // Final layer output shard, one [T, Hc] tile per column (no reduce),
    // written back to Batcher mem.
    __gm__ uint8_t *yBlock =
        yOutput + row * FFN_Y_OUTPUT_BYTES + col * FFN_MODEL_SHARD_TILE * static_cast<int>(sizeof(float));

    // gate/up matmul -> vector fixpipe: the ISA's only Acc->Vec path is the C2V
    // FIFO, so the accumulator drain is a directional TMOV(pipe, acc) push
    // followed by a TMOV(vecTile, pipe) pop.  Backed by the per-cell partial GM.
    using GatePipe = TPipe<0, Direction::DIR_C2V, FFN_GATE_PARTIAL_BYTES, 1>;
    using UpPipe = TPipe<2, Direction::DIR_C2V, FFN_UP_PARTIAL_BYTES, 1>;
    GatePipe gatePipe(reinterpret_cast<__gm__ void *>(gateBlock), kL1GateF32, 0);
    UpPipe upPipe(reinterpret_cast<__gm__ void *>(upBlock), kL1UpF32, 0);

    // ======================================================================
    // 1. Load inputs:  input X from Batcher mem -> L1; weights from DRAM -> L1.
    // ======================================================================
    GX xG(reinterpret_cast<__gm__ half *>(xBlock));
    GW wGateG(reinterpret_cast<__gm__ half *>(wGateBlock));
    GW wUpG(reinterpret_cast<__gm__ half *>(wUpBlock));
    GWDown wDownG(reinterpret_cast<__gm__ half *>(wDownBlock));

    TLOAD(xMat, xG);         // Batcher mem -> L1
    TLOAD(wGateMat, wGateG); // DRAM -> L1
    TLOAD(wUpMat, wUpG);     // DRAM -> L1
    TLOAD(wDownMat, wDownG); // DRAM -> L1

#ifndef __PTO_AUTO__
    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
#endif

    // ======================================================================
    // 2. gate = x @ W_gate    (L1 -> L0A/L0B TMOV, TMATMUL, L0C -> L1 fixpipe)
    // ======================================================================
    TMOV(aT, xMat);     // L1 -> L0A
    TMOV(bT, wGateMat); // L1 -> L0B

#ifndef __PTO_AUTO__
    set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
    wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
#endif

    TMATMUL(cT, aT, bT);

#ifndef __PTO_AUTO__
    set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
#endif

    TMOV(gatePipe, cT); // L0C -> C2V fixpipe (push)

#ifndef __PTO_AUTO__
    pipe_barrier(PIPE_ALL);
#endif
    dsb(DSB_DDR);

    // ======================================================================
    // 3. up = x @ W_up
    // ======================================================================
    TMOV(aT, xMat);
    TMOV(bT, wUpMat);

#ifndef __PTO_AUTO__
    set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
    wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
#endif

    TMATMUL(cT, aT, bT);

#ifndef __PTO_AUTO__
    set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
#endif

    TMOV(upPipe, cT); // L0C -> C2V fixpipe (push)

#ifndef __PTO_AUTO__
    pipe_barrier(PIPE_ALL);
#endif
    dsb(DSB_DDR);

    // ======================================================================
    // 4. activation on L1 (vector unit):  hidden = PReLU(gate) * up  -> fp16
    // ======================================================================
    TMOV(gateF32, gatePipe); // C2V fixpipe (pop) -> vector tile
    TMOV(upF32, upPipe);     // C2V fixpipe (pop) -> vector tile

#ifndef __PTO_AUTO__
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
#endif

    TLRELU(gateF32, gateF32, FFN_PRELU_ALPHA);
#ifndef __PTO_AUTO__
    pipe_barrier(PIPE_V);
#endif
    TMUL(hiddenF32, gateF32, upF32);
#ifndef __PTO_AUTO__
    pipe_barrier(PIPE_V);
#endif
    TCVT(hiddenF16, hiddenF32, RoundMode::CAST_RINT);

#ifndef __PTO_AUTO__
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
#endif

    // ======================================================================
    // 5. Publish this cell's hidden shard into its NoC-visible L1 SRAM window
    //    so peer cells can read it over the inter-core NoC (this is an
    //    L1->L1(SRAM) move, NOT a Batcher-mem store).  Then raise the "shard
    //    ready" flag on the NoC so row peers may gather it.
    // ======================================================================
    GHiddenShardF16 myHiddenG(reinterpret_cast<__gm__ half *>(myHiddenSlot));
    TSTORE(myHiddenG, hiddenF16); // L1 -> this cell's NoC-visible L1 SRAM window

#ifndef __PTO_AUTO__
    pipe_barrier(PIPE_ALL);
#endif
    dsb(DSB_DDR);
    comm::TNOTIFY_IMPL(myReadyFlag, 1, comm::NotifyOp::Set);

    // ======================================================================
    // 6. Row-local arrival barrier: wait for every column in this row to have
    //    published its shard into its L1 SRAM window before the AllGather reads
    //    it over the NoC.  (comm::TWAIT spins on the peer L1 window flag — an
    //    on-chip cross-cell barrier, no host stream sync needed.)
    // ======================================================================
    for (int c = 0; c < gridCols; ++c) {
        __gm__ uint8_t *peerL1Window = gatherPipeWindow + (row * gridCols + c) * FFN_GRID_WINDOW_BYTES;
        comm::Signal peerReady(reinterpret_cast<__gm__ int32_t *>(peerL1Window + kWinFlagOffset));
        comm::TWAIT_IMPL(peerReady, 1, comm::WaitCmp::EQ);
    }

    // ParallelGroup over the row's gridCols hidden shards, each living in a peer
    // cell's NoC-visible L1 SRAM window (read over the NoC, not Batcher mem).
    GHiddenShardF16 rowShards[FFN_GRID_COLS];
    for (int c = 0; c < gridCols; ++c) {
        __gm__ uint8_t *peerHidden =
            gatherPipeWindow + (row * gridCols + c) * FFN_GRID_WINDOW_BYTES + kWinHiddenOffset;
        rowShards[c] = GHiddenShardF16(reinterpret_cast<__gm__ half *>(peerHidden));
    }
    comm::ParallelGroup<GHiddenShardF16> rowGroup(rowShards, gridCols, /*rootIdx=*/0);

    // ======================================================================
    // 7. Fine-grained AllGather  <->  down-Matmul pipeline.
    //    down: yShard[T, Hc] = hidden_full[T, F] @ W_down[F, Hc]
    //    F (the contraction axis) is walked in FFN_AG_CHUNKS bands.  Each band is
    //    AllGathered from the peer cells' L1 SRAM over the NoC into a
    //    double-buffered L1 tile (peer L1 -> NoC -> local L1) and accumulated by
    //    a down GEMM over that K-band.  The next band's gather (MTE2) is issued
    //    before the current band's GEMM (M) so the NoC collective hides under the
    //    matmul.
    // ======================================================================
    // Prime: gather chunk 0 into buffer 0.
    comm::TAllGather<comm::CollEngine::AIV>(rowGroup, hiddenChunkMat[0], /*featOffset=*/0, validChunk, validN);
#ifndef __PTO_AUTO__
    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
#endif

    for (int j = 0; j < FFN_AG_CHUNKS; ++j) {
        const int buf = j & 1;
        const event_t bufEvt = (buf == 0) ? EVENT_ID0 : EVENT_ID1;

        // Launch the next chunk's gather so its TLOAD (MTE2) overlaps this
        // chunk's GEMM (M) below.
        if (j + 1 < FFN_AG_CHUNKS) {
            const int nb = (j + 1) & 1;
            const event_t nbEvt = (nb == 0) ? EVENT_ID0 : EVENT_ID1;
            comm::TAllGather<comm::CollEngine::AIV>(rowGroup, hiddenChunkMat[nb], (j + 1) * validChunk, validChunk,
                                                    validN);
#ifndef __PTO_AUTO__
            set_flag(PIPE_MTE2, PIPE_MTE1, nbEvt);
#endif
        }

#ifndef __PTO_AUTO__
        wait_flag(PIPE_MTE2, PIPE_MTE1, bufEvt);
#endif
        TMOV(aDownT, hiddenChunkMat[buf]);             // hidden K-band  L1 -> L0A
        TEXTRACT(bDownT, wDownMat, j * validChunk, 0); // W_down K-band  L1 -> L0B

#ifndef __PTO_AUTO__
        set_flag(PIPE_MTE1, PIPE_M, bufEvt);
        wait_flag(PIPE_MTE1, PIPE_M, bufEvt);
#endif

        if (j == 0) {
            TMATMUL(cDownT, aDownT, bDownT);
        } else {
            TMATMUL_ACC(cDownT, cDownT, aDownT, bDownT);
        }
    }

#ifndef __PTO_AUTO__
    set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
#endif

    // ======================================================================
    // 8. Drain the down accumulator L0C -> L1 (TMOV), then write the L1 result
    //    back to Batcher mem (TSTORE).  Batcher mem carries the final [T, Hc]
    //    output out; there is no L1 -> DRAM path.
    // ======================================================================
    TMOV(downMat, cDownT); // L0C -> L1

#ifndef __PTO_AUTO__
    set_flag(PIPE_FIX, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_FIX, PIPE_MTE3, EVENT_ID0);
#endif

    GYShardF32 yG(reinterpret_cast<__gm__ float *>(yBlock));
    TSTORE(yG, downMat); // L1 -> Batcher mem

#ifndef __PTO_AUTO__
    pipe_barrier(PIPE_ALL);
#endif
    dsb(DSB_DDR);

    // The unified-L1 stream no longer bounces hidden/down through GM staging.
    (void)hiddenIn;
    (void)downPartial;
    (void)hcclCtxRaw;
#else
    (void)fftsAddr;
    (void)gatherPipeWindow;
    (void)x;
    (void)wGate;
    (void)wUp;
    (void)wDown;
    (void)gatePartial;
    (void)upPartial;
    (void)hiddenIn;
    (void)downPartial;
    (void)yOutput;
    (void)hcclCtxRaw;
    (void)gridRows;
    (void)gridCols;
#endif
}

void launchDistributedFfnGridAllGatherMixedKernel(uint8_t *ffts, uint8_t *gatherPipeWindow, uint8_t *x, uint8_t *wGate,
                                                  uint8_t *wUp, uint8_t *wDown, uint8_t *gatePartial,
                                                  uint8_t *upPartial, uint8_t *hiddenIn, uint8_t *downPartial,
                                                  uint8_t *yOutput, uint8_t *hcclCtx, int gridRows, int gridCols,
                                                  void *stream)
{
    int totalBlocks = gridRows * gridCols;
    if (totalBlocks <= 0) {
        return;
    }
    DistributedFfnGridAllGatherMixedKernel<<<totalBlocks, nullptr, stream>>>(
        ffts, gatherPipeWindow, x, wGate, wUp, wDown, gatePartial, upPartial, hiddenIn, downPartial, yOutput, hcclCtx,
        gridRows, gridCols);
}
