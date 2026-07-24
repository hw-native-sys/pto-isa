/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software; you can redistribute it and/or modify it under
the terms and conditions of CANN Open Software License Agreement Version 2.0
(the "License"). Please refer to the License for details. You may not use this
software except in compliance with the License. THIS SOFTWARE IS PROVIDED ON AN
"AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A
PARTICULAR PURPOSE. See LICENSE in the software repository for the full text of
the License.
*/

// Pure 1D N-cut 32-cell WSE FFN emulation -- ReduceSum variant (Option B).
//
// Same pure-1D-N-cut topology as the AllGather variant: the intermediate I is
// split across all 32 cells (I_shard=96), x is broadcast (M=8 on every cell).
// ReduceSum differs in how the down projection is parallelised: instead of
// gathering hidden and H-sharding down (AllGather), each cell computes a FULL-H
// down PARTIAL from its local hidden (W_down is cut along I = the down GEMM's
// K-axis, a row shard [I_shard, H]), and the 32 partials are REDUCED:
//   partial_c = hidden_c @ W_down[I_c, :]   -> [8, H] (partial sum over I_c)
//   y = sum over all 32 cells of partial_c   = hidden_full @ W_down
// The 32-way sum is two fan-in-1 chain phases (symmetric to AllGather's two-phase
// gather): EAST 8-way (row) then SOUTH 4-way (col).  Because a partial is
// [8, H] fp32 = 224 KB > 192 KB UB, the reduce is H-chunked: it carries
// [8, H_base=1024] tiles (32 KB) at a time, looping 7 H-segments.
//
// Real shapes M=8, H=7168, I=3072.  Real K (7168 gate/up) and N (7168 down)
// exceed L1/L0C, so the cube GEMMs are tiled: gate/up K-tiled (baseK=64); down
// K-tiled (baseK=32, A=hidden from L1) + N-tiled (baseN=1024=H_base, B=W_down
// row-stride = H).  The host runs 3 phase-dispatched launches (A compute,
// B EAST-reduce waves, C SOUTH-reduce wave); 32 > 24 phys cores so the reduce is
// waved by whole rows / a single col.

#include <cstddef>
#include <cstdint>
#include <pto/pto-inst.hpp>

#include <pto/comm/comm_types.hpp> // pto::comm::ReduceOp
#include <pto/common/fifo.hpp>
#include <pto/npu/a2a3/grid_intrinsic.hpp>
#include <pto/npu/a2a3/grid_pipe_runtime.hpp>

#include "common.hpp"
#include "ffn_config.hpp"
#include "gridpipe_payload_inl.hpp"
#include "tpipe_tmov_inl.hpp"

#ifdef __CCE_AICORE__
using namespace pto;

#ifdef __DAV_CUBE__
constexpr bool DAV_CUBE = true;
#else
constexpr bool DAV_CUBE = false;
#endif
#ifdef __DAV_VEC__
constexpr bool DAV_VEC = true;
#else
constexpr bool DAV_VEC = false;
#endif

constexpr int kT = FFN_NCUT_T;                 // 8
constexpr int kBaseM = FFN_NCUT_BASE_M_ALIGN;  // 16 (one cube M tile, kT valid)
constexpr int kH = FFN_NCUT_H;                 // 7168
constexpr int kIShard = FFN_NCUT_I_SHARD;      // 96
constexpr int kHBase = FFN_RS_DOWN_N_BASE;     // 1024 (down N-tile == reduce H-chunk)
constexpr int kHSegs = kH / kHBase;            // 7
constexpr int kDownKBase = FFN_RS_DOWN_K_BASE; // 32 (down K-tile; 96/32 = 3)

// C2V/V2C working tiles for phase A.
using GatePipe = TPipe<0, Direction::DIR_C2V, FFN_NCUT_GATE_PARTIAL_BYTES, 1>;
using UpPipe = TPipe<2, Direction::DIR_C2V, FFN_NCUT_GATE_PARTIAL_BYTES, 1>;
using HiddenPipe = TPipe<4, Direction::DIR_V2C, FFN_NCUT_HIDDEN_SHARD_BYTES, 1>;

// EAST/SOUTH reduce tile = one H-segment [8, H_base] fp32.
using ReduceSegTile = Tile<TileType::Vec, float, kT, kHBase, BLayout::RowMajor>;
using FfnReducePipe = GridPipe<ReduceSegTile, FFN_RS_REDUCE_TILE_BYTES, FFN_RS_REDUCE_SLOT_COUNT>;

using GateAccTile = TileAcc<float, kBaseM, kIShard, kT, kIShard>; // [16,96] (gate/up)

// K-tiled cube GEMM, BOTH operands from GM.  bRowStride lets it serve gate/up
// (B=[Kfull,N], rowStride=N) and the down N-tile (caller passes rowStride=H and a
// column offset).  cTile[16,N] (fp32 acc) = aGM[kT,Kfull] @ bGM[Kfull,bRowStride].
template <int Kfull, int N, int BRowStride, int BaseK>
AICORE inline void RsCubeGemmFill(TileAcc<float, kBaseM, N, kT, N> &cTile, __gm__ half *aGM, __gm__ half *bGM, int bColOff)
{
    constexpr int kLoop = Kfull / BaseK;
    using TileMatA = Tile<TileType::Mat, half, kBaseM, BaseK, BLayout::ColMajor, kT, BaseK, SLayout::RowMajor>;
    using TileMatB = Tile<TileType::Mat, half, BaseK, N, BLayout::ColMajor, BaseK, N, SLayout::RowMajor>;
    using LeftTile = TileLeft<half, kBaseM, BaseK, kT, BaseK>;
    using RightTile = TileRight<half, BaseK, N, BaseK, N>;
    TileMatA aPanel;
    TileMatB bPanel;
    LeftTile aTile;
    RightTile bTile;
    TASSIGN(aPanel, 0x0);
    TASSIGN(bPanel, 0x0 + kBaseM * BaseK * static_cast<int>(sizeof(half)));
    TASSIGN(aTile, 0x0);
    TASSIGN(bTile, 0x0);

    using GlobalA = GlobalTensor<half, Shape<1, 1, 1, kT, BaseK>,
                                 Stride<kT * Kfull, kT * Kfull, kT * Kfull, Kfull, 1>>;
    using GlobalB = GlobalTensor<half, Shape<1, 1, 1, BaseK, N>,
                                 Stride<BaseK * BRowStride, BaseK * BRowStride, BaseK * BRowStride, BRowStride, 1>>;
    for (int k = 0; k < kLoop; ++k) {
        GlobalA aGm(aGM + k * BaseK);
        GlobalB bGm(bGM + k * BaseK * BRowStride + bColOff);
        TLOAD(aPanel, aGm);
        TLOAD(bPanel, bGm);
#ifndef __PTO_AUTO__
        set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
#endif
        TMOV(aTile, aPanel);
        TMOV(bTile, bPanel);
#ifndef __PTO_AUTO__
        set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
        wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
#endif
        if (k == 0) {
            TMATMUL(cTile, aTile, bTile);
        } else {
            TMATMUL_ACC(cTile, cTile, aTile, bTile);
        }
#ifndef __PTO_AUTO__
        pipe_barrier(PIPE_ALL);
#endif
    }
}

#endif // __CCE_AICORE__

__global__ AICORE void DistributedFfnGridTreduceReduceSumMixedKernel(
    __gm__ uint8_t *fftsAddr, __gm__ uint8_t *reduceWindow, __gm__ uint8_t *xFull, __gm__ uint8_t *wGateShards,
    __gm__ uint8_t *wUpShards, __gm__ uint8_t *wDownShards, __gm__ uint8_t *partialBuf, __gm__ uint8_t *rowPartialBuf,
    __gm__ uint8_t *yFull, __gm__ uint8_t *gatePartialBuf, __gm__ uint8_t *upPartialBuf, __gm__ uint8_t *hiddenBuf,
    __gm__ uint8_t *hcclCtxRaw, int phase, int rowStart, int colStart, int waveCols, int gridRows, int gridCols)
{
#ifdef __CCE_AICORE__
    set_ffts_base_addr(reinterpret_cast<uint64_t>(fftsAddr));

    const int blk = get_block_idx();
    const int row = rowStart + blk / waveCols;
    const int col = colStart + blk % waveCols;
    const int cell = row * gridCols + col;
    if (cell < 0 || cell >= gridRows * gridCols) {
        return;
    }

    using pto::GridDirection;
    using pto::comm::ReduceOp;

    // =========================== phase A: gate + up + SwiGLU + down ===========================
    if (phase == 0) {
        __gm__ uint8_t *xBlock = xFull; // broadcast
        __gm__ uint8_t *wGateBlock = wGateShards + cell * FFN_NCUT_W_GATE_SHARD_BYTES;
        __gm__ uint8_t *wUpBlock = wUpShards + cell * FFN_NCUT_W_UP_SHARD_BYTES;
        __gm__ uint8_t *wDownBlock = wDownShards + cell * FFN_RS_W_DOWN_SHARD_BYTES;
        __gm__ uint8_t *partialBlock = partialBuf + cell * FFN_RS_PARTIAL_BYTES;
        __gm__ uint8_t *gatePartialBlock = gatePartialBuf + cell * FFN_NCUT_GATE_PARTIAL_BYTES;
        __gm__ uint8_t *upPartialBlock = upPartialBuf + cell * FFN_NCUT_GATE_PARTIAL_BYTES;
        __gm__ uint8_t *hiddenBlock = hiddenBuf + cell * FFN_NCUT_HIDDEN_SHARD_BYTES;

        if constexpr (DAV_CUBE) {
            GateAccTile cGate;
            TASSIGN(cGate, 0x0);
            GatePipe gatePipe(reinterpret_cast<__gm__ void *>(gatePartialBlock), 0x0000, 0);
            UpPipe upPipe(reinterpret_cast<__gm__ void *>(upPartialBlock), 0x1000, 0);

            // gate = x @ W_gate[:, I_c]  (K-tiled over H=7168)
            RsCubeGemmFill<kH, kIShard, kIShard, FFN_NCUT_K_BASE>(cGate, reinterpret_cast<__gm__ half *>(xBlock),
                                                                  reinterpret_cast<__gm__ half *>(wGateBlock), 0);
#ifndef __PTO_AUTO__
            set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
            wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
#endif
            TMOV(gatePipe, cGate);
#ifndef __PTO_AUTO__
            pipe_barrier(PIPE_ALL);
#endif
            dsb(DSB_DDR);

            // up = x @ W_up[:, I_c]
            RsCubeGemmFill<kH, kIShard, kIShard, FFN_NCUT_K_BASE>(cGate, reinterpret_cast<__gm__ half *>(xBlock),
                                                                  reinterpret_cast<__gm__ half *>(wUpBlock), 0);
#ifndef __PTO_AUTO__
            set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
            wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
#endif
            TMOV(upPipe, cGate);
#ifndef __PTO_AUTO__
            pipe_barrier(PIPE_ALL);
#endif
            dsb(DSB_DDR);

            // hidden (V2C pop) -> L1 hiddenMat, then down (K+N-tiled) -> partial_c in GM.
            using TileHidden = Tile<TileType::Mat, half, kBaseM, kIShard, BLayout::ColMajor, kT, kIShard, SLayout::RowMajor>;
            TileHidden hiddenMat;
            constexpr int kL1Hidden = 0x0; // reuse the gate/up helper's transient aPanel space (gate/up is done)
            TASSIGN(hiddenMat, kL1Hidden);
            HiddenPipe hiddenPipe(reinterpret_cast<__gm__ void *>(hiddenBlock), 0, kL1Hidden);
            TMOV(hiddenMat, hiddenPipe);
#ifndef __PTO_AUTO__
            set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
            wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
            pipe_barrier(PIPE_ALL);
#endif
            dsb(DSB_DDR);

            // down: A = hidden (L1), B = W_down[I_c, :] [I_shard, H] (GM, row-stride H).
            // K-tiled (baseK=32) over I_shard, N-tiled (baseN=H_base) over H; assemble
            // partial_c [8,H] in GM one N-tile ([8,H_base]) at a time.
            using DownLeft = TileLeft<half, kBaseM, kDownKBase, kT, kDownKBase>;
            using DownRight = TileRight<half, kDownKBase, kHBase, kDownKBase, kHBase>;
            using DownAcc = TileAcc<float, kBaseM, kHBase, kT, kHBase>;
            using DownBPanel = Tile<TileType::Mat, half, kDownKBase, kHBase, BLayout::ColMajor, kDownKBase, kHBase, SLayout::RowMajor>;
            using DownBGlobal = GlobalTensor<half, Shape<1, 1, 1, kDownKBase, kHBase>,
                                             Stride<kDownKBase * kH, kDownKBase * kH, kDownKBase * kH, kH, 1>>;
            // partialBuf holds each cell's [T,H] down partial in SEGMENT-MAJOR form:
            // segment nTile ([T,kHBase]) is stored CONTIGUOUSLY (T-stride kHBase) at
            // offset nTile*(T*kHBase), so the phase-B group fan-in
            // (reduce_group_to_ubuf) can read every row-mate's segment as one
            // contiguous byte range.  partialBuf / rowPartialBuf are INTERMEDIATE;
            // only the final y output keeps the strided [T,H] golden layout below.
            using GPartialSeg = GlobalTensor<float, Shape<1, 1, 1, kT, kHBase>,
                                             Stride<kT * kHBase, kT * kHBase, kT * kHBase, kHBase, 1>>;
            DownLeft aDown;
            DownRight bDown;
            DownAcc cDown;
            DownBPanel bPanel;
            TASSIGN(aDown, 0x0);
            TASSIGN(bDown, 0x0);
            TASSIGN(cDown, 0x0);
            TASSIGN(bPanel, 0x0 + kBaseM * kIShard * static_cast<int>(sizeof(half)));
            for (int nTile = 0; nTile < kHSegs; ++nTile) {
                const int nOff = nTile * kHBase;
                for (int kk = 0; kk < (kIShard / kDownKBase); ++kk) {
                    TEXTRACT(aDown, hiddenMat, 0, kk * kDownKBase); // L1 hidden K-slice -> L0A
                    DownBGlobal bGm(reinterpret_cast<__gm__ half *>(wDownBlock) + kk * kDownKBase * kH + nOff);
                    TLOAD(bPanel, bGm); // GM W_down [K-slice, N-tile] -> L1
#ifndef __PTO_AUTO__
                    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
                    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
#endif
                    TMOV(bDown, bPanel); // L1 -> L0B
#ifndef __PTO_AUTO__
                    set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
                    wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
#endif
                    if (kk == 0) {
                        TMATMUL(cDown, aDown, bDown);
                    } else {
                        TMATMUL_ACC(cDown, cDown, aDown, bDown);
                    }
#ifndef __PTO_AUTO__
                    pipe_barrier(PIPE_ALL);
#endif
                }
#ifndef __PTO_AUTO__
                set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
                wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
#endif
                GPartialSeg segGm(reinterpret_cast<__gm__ float *>(partialBlock) +
                                  nTile * (kT * kHBase)); // segment-major: contiguous [T,kHBase]
                TSTORE(segGm, cDown);
#ifndef __PTO_AUTO__
                pipe_barrier(PIPE_ALL);
#endif
                dsb(DSB_DDR);
            }
        }

        if constexpr (DAV_VEC) {
            using F32Tile = Tile<TileType::Vec, float, kT, kIShard, BLayout::RowMajor>;
            using F16Tile = Tile<TileType::Vec, half, kT, kIShard, BLayout::RowMajor>;
            F32Tile gateF32;
            F32Tile upF32;
            F32Tile hiddenF32;
            F16Tile hiddenF16;
            F32Tile siluDenom;
            F32Tile siluTmp;
            TASSIGN(gateF32, 0x0000);
            TASSIGN(upF32, 0x1000);
            TASSIGN(hiddenF32, 0x2000);
            TASSIGN(hiddenF16, 0x3000);
            TASSIGN(siluDenom, 0x4000);
            TASSIGN(siluTmp, 0x5000);

            GatePipe gatePipe(reinterpret_cast<__gm__ void *>(gatePartialBlock), 0x0000, 0);
            UpPipe upPipe(reinterpret_cast<__gm__ void *>(upPartialBlock), 0x1000, 0);
            HiddenPipe hiddenPipe(reinterpret_cast<__gm__ void *>(hiddenBlock), 0, 0x0);

            TMOV(gateF32, gatePipe);
            TMOV(upF32, upPipe);
#ifndef __PTO_AUTO__
            set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
            wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
#endif
            // SwiGLU = SiLU(clamp(gate)) * up, fp32 ("SiLU + clamp(max=10)").
            TMAXS(gateF32, gateF32, FFN_SILU_CLAMP_MIN);
#ifndef __PTO_AUTO__
            pipe_barrier(PIPE_V);
#endif
            TMINS(gateF32, gateF32, FFN_SILU_CLAMP_MAX);
#ifndef __PTO_AUTO__
            pipe_barrier(PIPE_V);
#endif
            TMULS(siluDenom, gateF32, -1.0f);
#ifndef __PTO_AUTO__
            pipe_barrier(PIPE_V);
#endif
            TEXP(siluDenom, siluDenom);
#ifndef __PTO_AUTO__
            pipe_barrier(PIPE_V);
#endif
            TADDS(siluDenom, siluDenom, 1.0f);
#ifndef __PTO_AUTO__
            pipe_barrier(PIPE_V);
#endif
            TDIV(siluTmp, gateF32, siluDenom);
#ifndef __PTO_AUTO__
            pipe_barrier(PIPE_V);
#endif
            TMUL(hiddenF32, siluTmp, upF32);
#ifndef __PTO_AUTO__
            pipe_barrier(PIPE_V);
#endif
            TCVT(hiddenF16, hiddenF32, RoundMode::CAST_RINT);
#ifndef __PTO_AUTO__
            set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
            wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
            pipe_barrier(PIPE_V);
#endif
            TMOV(hiddenPipe, hiddenF16); // V2C push -> cube down
#ifndef __PTO_AUTO__
            pipe_barrier(PIPE_ALL);
#endif
            dsb(DSB_DDR);
        }
        return;
    }

    // =========================== phase B: EAST 8-way reduce (row, H-chunked) ===========================
    // Group fan-in (reduce_group_to_ubuf).  Each cell wrote its full-H partial to
    // partialBuf in phase A; the host stream barrier makes every row-mate's
    // partial visible here.  The row SINK (col == gridCols-1) folds all 8
    // row-mates' segment-h partials directly out of partialBuf with one
    // N->1 reduce intrinsic.  Member k == cell (row, k), so members are visited
    // col0..col7 -- the SAME order the EAST relay accumulated in, hence the FP
    // sum is bit-identical (IEEE-754 add is commutative).  Non-sink cells have
    // nothing to do (their partial is already in GM).
    if (phase == 1) {
        if constexpr (DAV_VEC) {
            if (col + 1 == gridCols) {
                __gm__ uint8_t *rowPartialBlock = rowPartialBuf + row * FFN_RS_ROW_PARTIAL_BYTES;
                ReduceSegTile seg;      // reduce accumulator / result
                ReduceSegTile scratch;  // in-core combine scratch
                TASSIGN(seg, 0x0000);
                TASSIGN(scratch, 0x8000);
                // rowPartialBuf is SEGMENT-MAJOR (matches partialBuf): segment h is a
                // contiguous [T,kHBase] block at offset h*(T*kHBase), T-stride kHBase.
                using GSegContig = GlobalTensor<float, Shape<1, 1, 1, kT, kHBase>,
                                                Stride<kT * kHBase, kT * kHBase, kT * kHBase, kHBase, 1>>;
                for (int h = 0; h < kHSegs; ++h) {
                    const int hSegOff = h * (kT * kHBase); // segment-major offset (floats)
                    // Contribution arena = cell (row, 0..gridCols-1)'s segment-h
                    // partials.  Cell (row, k) sits at partialBuf + (row*gridCols+k)*PPB
                    // + hSegOff; consecutive row members are uniformly spaced by PPB.
                    __gm__ const float *base =
                        reinterpret_cast<__gm__ const float *>(
                            partialBuf + static_cast<int64_t>(row) * gridCols * FFN_RS_PARTIAL_BYTES) +
                        hSegOff;
                    GRID_TREDUCE_GROUP_IMPL<GridGroup::ROW, ReduceOp::Sum, float>(seg, scratch, base,
                                                                                  FFN_RS_REDUCE_TILE_BYTES, gridCols,
                                                                                  GridRect{}, FFN_RS_PARTIAL_BYTES);
#ifndef __PTO_AUTO__
                    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
                    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
#endif
                    GSegContig outGm(reinterpret_cast<__gm__ float *>(rowPartialBlock) + hSegOff);
                    TSTORE(outGm, seg);
#ifndef __PTO_AUTO__
                    pipe_barrier(PIPE_ALL);
#endif
                    dsb(DSB_DDR);
                }
            }
        }
        return;
    }

    // =========================== phase C: SOUTH 4-way reduce (col 7, H-chunked) ===========================
    // Group fan-in (reduce_group_to_ubuf).  Phase B left one row partial per row
    // in rowPartialBuf; the column SINK (row == gridRows-1) folds all 4 rows'
    // segment-h partials out of rowPartialBuf with one N->1 reduce intrinsic.
    // Member k == row k at rowPartialBuf + k*RPPB (uniform stride RPPB); the
    // row0..row3 order matches the SOUTH relay accumulation (bit-identical).
    if (phase == 2) {
        if constexpr (DAV_VEC) {
            if (row + 1 == gridRows) {
                ReduceSegTile seg;
                ReduceSegTile scratch;
                TASSIGN(seg, 0x0000);
                TASSIGN(scratch, 0x8000);
                // rowPartialBuf is SEGMENT-MAJOR (segment h contiguous at h*(T*kHBase));
                // yFull keeps the strided [T,H] golden layout (segment h at column
                // offset h*kHBase, T-stride H), so the read and the store use different
                // offsets here.
                using GYSeg = GlobalTensor<float, Shape<1, 1, 1, kT, kHBase>,
                                           Stride<kT * kH, kT * kH, kT * kH, kH, 1>>;
                for (int h = 0; h < kHSegs; ++h) {
                    const int hSegOff = h * (kT * kHBase); // segment-major offset into rowPartialBuf
                    const int hStridedOff = h * kHBase;    // strided column offset into yFull [T,H]
                    __gm__ const float *base = reinterpret_cast<__gm__ const float *>(rowPartialBuf) + hSegOff;
                    GRID_TREDUCE_GROUP_IMPL<GridGroup::COL, ReduceOp::Sum, float>(seg, scratch, base,
                                                                                  FFN_RS_REDUCE_TILE_BYTES, gridRows,
                                                                                  GridRect{}, FFN_RS_ROW_PARTIAL_BYTES);
#ifndef __PTO_AUTO__
                    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
                    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
#endif
                    GYSeg outGm(reinterpret_cast<__gm__ float *>(yFull) + hStridedOff);
                    TSTORE(outGm, seg);
#ifndef __PTO_AUTO__
                    pipe_barrier(PIPE_ALL);
#endif
                    dsb(DSB_DDR);
                }
            }
        }
        return;
    }
#else
    (void)fftsAddr;
    (void)reduceWindow;
    (void)xFull;
    (void)wGateShards;
    (void)wUpShards;
    (void)wDownShards;
    (void)partialBuf;
    (void)rowPartialBuf;
    (void)yFull;
    (void)gatePartialBuf;
    (void)upPartialBuf;
    (void)hiddenBuf;
    (void)hcclCtxRaw;
    (void)phase;
    (void)rowStart;
    (void)colStart;
    (void)waveCols;
    (void)gridRows;
    (void)gridCols;
#endif
}

void launchDistributedFfnGridTreduceReduceSumMixedKernel(uint8_t *ffts, uint8_t *reduceWindow, uint8_t *xFull, uint8_t *wGateShards,
                                         uint8_t *wUpShards, uint8_t *wDownShards, uint8_t *partialBuf,
                                         uint8_t *rowPartialBuf, uint8_t *yFull, uint8_t *gatePartialBuf,
                                         uint8_t *upPartialBuf, uint8_t *hiddenBuf, uint8_t *hcclCtx, int phase,
                                         int rowStart, int colStart, int waveCols, int gridRows, int gridCols,
                                         int blockCount, void *stream)
{
    if (blockCount <= 0) {
        return;
    }
    DistributedFfnGridTreduceReduceSumMixedKernel<<<blockCount, nullptr, stream>>>(
        ffts, reduceWindow, xFull, wGateShards, wUpShards, wDownShards, partialBuf, rowPartialBuf, yFull, gatePartialBuf,
        upPartialBuf, hiddenBuf, hcclCtx, phase, rowStart, colStart, waveCols, gridRows, gridCols);
}
