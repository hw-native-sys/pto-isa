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

// M3 Session D-6: cube .so producer for the full FFN matmul pipeline.
//
// Single-device multi-block FFN cube kernel.  Each block owns one logical
// grid cell (row, col).  Phase 0 computes gate/up partials for that cell, the
// vec kernel computes hidden, and phase 1 computes the cell's down partial.
//
// gridRows*gridCols is the launched block count on one device.

#include <cstddef>
#include <cstdint>
#include <pto/pto-inst.hpp>

#include <pto/common/fifo.hpp>

#include "common.hpp"
#include "ffn_config.hpp"

#ifdef __CCE_AICORE__
using namespace pto;

template <uint8_t FlagId, uint8_t DirType, uint32_t SlotSize, uint32_t SlotNum = 1, uint32_t LocalSlotNum = 1>
struct FfnSerialTPipe {
  static constexpr uint8_t DIR_MASK = 0x7;
  static constexpr uint8_t DIR_TYPE = DIR_MASK & DirType;
  static constexpr bool is_c2v = (DIR_TYPE == Direction::DIR_C2V);
  static constexpr bool is_v2c = (DIR_TYPE == Direction::DIR_V2C);
  static constexpr uint32_t SyncPeriod = 1;
  using RingFiFo = RingFIFO<SlotSize, SlotNum, LocalSlotNum>;

  PTO_INTERNAL static bool shouldWaitFree(uint32_t) { return false; }
  PTO_INTERNAL static bool shouldNotifyFree(uint32_t) { return false; }

  struct Producer {
    uint32_t tileIndex = 0;
    int entryOffset = 0;

    PTO_INTERNAL void setAllocateStatus(bool) {}
    PTO_INTERNAL void setRecordStatus(bool) {}
    PTO_INTERNAL void setTileId(int tIndex, int) { tileIndex = static_cast<uint32_t>(tIndex); }
    PTO_INTERNAL void setEntryOffset(int offset) { entryOffset = offset; }
    PTO_INTERNAL bool getAllocateStatus() const { return false; }
    PTO_INTERNAL bool getRecordStatus() const { return false; }
    PTO_INTERNAL void allocate() const {}
    PTO_INTERNAL void record() const {}

    template <typename TileProd, TileSplitAxis Split>
    PTO_INTERNAL void push(RingFiFo &fifo, TileProd &tile) {
      static_assert(Split == TileSplitAxis::TILE_NO_SPLIT, "distributed_ffn_grid uses full-tile TPipe slots");
      static_assert((is_c2v && TileProd::Loc == TileType::Acc) || (is_v2c && TileProd::Loc == TileType::Vec),
                    "TPipe direction does not match producer tile type");
      using T = typename TileProd::DType;
      constexpr int rows = TileProd::Rows;
      constexpr int cols = TileProd::Cols;
      size_t entryBase = (tileIndex % RingFiFo::SLOT_NUM) * RingFiFo::SLOT_SIZE;
      using GlobalData = GlobalTensor<T, Shape<1, 1, 1, rows, cols>, Stride<1, 1, 1, cols, 1>>;
      GlobalData globalTensor(reinterpret_cast<__gm__ T *>(reinterpret_cast<uint64_t>(fifo.GM_SLOT_BUFFER) +
                                                           entryBase + entryOffset));
      TSTORE_IMPL(globalTensor, tile);
    }
  };

  struct Consumer {
    uint32_t tileIndex = 0;
    int entryOffset = 0;

    PTO_INTERNAL void setWaitStatus(bool) {}
    PTO_INTERNAL void setFreeStatus(bool) {}
    PTO_INTERNAL void setTileId(int tIndex, int) { tileIndex = static_cast<uint32_t>(tIndex); }
    PTO_INTERNAL void setEntryOffset(int offset) { entryOffset = offset; }
    PTO_INTERNAL bool getWaitStatus() const { return false; }
    PTO_INTERNAL bool getFreeStatus() const { return false; }
    PTO_INTERNAL void wait() const {}
    PTO_INTERNAL void free() const {}

    template <typename TileCons, TileSplitAxis Split>
    PTO_INTERNAL void pop(RingFiFo &fifo, TileCons &tile) {
      static_assert(Split == TileSplitAxis::TILE_NO_SPLIT, "distributed_ffn_grid uses full-tile TPipe slots");
      static_assert((is_c2v && TileCons::Loc == TileType::Vec) || (is_v2c && TileCons::Loc == TileType::Mat),
                    "TPipe direction does not match consumer tile type");
      using T = typename TileCons::DType;
      constexpr int rows = TileCons::Rows;
      constexpr int cols = TileCons::Cols;
      size_t entryBase = (tileIndex % RingFiFo::SLOT_NUM) * RingFiFo::SLOT_SIZE;
      uint64_t localBase = TileCons::Loc == TileType::Vec ? fifo.C2V_CONSUMER_BUF : fifo.V2C_CONSUMER_BUF;
      localBase += (tileIndex % RingFiFo::LOCAL_SLOT_NUM) * rows * cols * sizeof(T);
      TASSIGN_IMPL(tile, localBase);
      using GlobalData = GlobalTensor<T, Shape<1, 1, 1, rows, cols>, Stride<1, 1, 1, cols, 1>>;
      GlobalData globalTensor(reinterpret_cast<__gm__ T *>(reinterpret_cast<uint64_t>(fifo.GM_SLOT_BUFFER) +
                                                           entryBase + entryOffset));
      TLOAD_IMPL(tile, globalTensor);
    }
  };

  RingFiFo fifo;
  Producer prod;
  Consumer cons;

  PTO_INTERNAL explicit FfnSerialTPipe(__gm__ void *gmSlotBuffer, uint32_t c2vConsumerBuf,
                                      uint32_t v2cConsumerBuf)
      : fifo(gmSlotBuffer, c2vConsumerBuf, v2cConsumerBuf), prod(), cons() {}
};

#endif

__global__ AICORE void DistributedFfnGridComputeKernel(
    __gm__ uint8_t *x, __gm__ uint8_t *wGate, __gm__ uint8_t *wUp,
    __gm__ uint8_t *wDown, __gm__ uint8_t *gatePartial,
    __gm__ uint8_t *upPartial, __gm__ uint8_t *hiddenIn,
    __gm__ uint8_t *downPartial, __gm__ uint8_t *yOutput,
    __gm__ uint8_t *chunkFlags, int gridRows, int gridCols, int myRankId,
    int phase) {
#ifdef __CCE_AICORE__
  (void)myRankId;
  (void)chunkFlags;
  (void)yOutput;

  int blockIdx = get_block_idx();
  int totalBlocks = gridRows * gridCols;
  if (blockIdx < 0 || blockIdx >= totalBlocks) {
    return;
  }

  constexpr int validM = FFN_TOKEN_TILE;  // 16
  constexpr int validK = FFN_MODEL_TILE;  // 64
  constexpr int validN = FFN_FFN_TILE;    // 64
  constexpr int blockAlign = C0_SIZE_BYTE / static_cast<int>(sizeof(half));
  constexpr int M = ((validM + 15) / 16) * 16;
  constexpr int K = ((validK + blockAlign - 1) / blockAlign) * blockAlign;
  constexpr int N = ((validN + blockAlign - 1) / blockAlign) * blockAlign;

  using GX = GlobalTensor<
      half, Shape<1, 1, 1, validM, validK>,
      Stride<validM * validK, validM * validK, validM * validK, validK, 1>>;
  using GW = GlobalTensor<
      half, Shape<1, 1, 1, validK, validN>,
      Stride<validK * validN, validK * validN, validK * validN, validN, 1>>;
  using TileA = Tile<TileType::Mat, half, M, K, BLayout::ColMajor, validM,
                     validK, SLayout::RowMajor, 512>;
  using TileB = Tile<TileType::Mat, half, K, N, BLayout::ColMajor, validK,
                     validN, SLayout::RowMajor, 512>;
  using TL = TileLeft<half, M, K, validM, validK>;
  using TR = TileRight<half, K, N, validK, validN>;
  using TC = TileAcc<float, M, N, validM, validN>;

  using GatePipe = FfnSerialTPipe<0, static_cast<uint8_t>(Direction::DIR_C2V), FFN_GATE_PARTIAL_BYTES>;
  using UpPipe = FfnSerialTPipe<2, static_cast<uint8_t>(Direction::DIR_C2V), FFN_UP_PARTIAL_BYTES>;
  using HiddenPipe = FfnSerialTPipe<4, static_cast<uint8_t>(Direction::DIR_V2C), FFN_HIDDEN_BYTES>;
  using DownPipe = FfnSerialTPipe<6, static_cast<uint8_t>(Direction::DIR_C2V), FFN_DOWN_PARTIAL_BYTES>;

  constexpr int kL1X = 0x00000;
  constexpr int kL1Hidden = 0x10000;
  constexpr int kL1WGate = 0x20000;
  constexpr int kL1WUp = 0x24000;
  constexpr int kL1WDown = 0x28000;

  TileA xMat;
  TileA hiddenMat;
  TileB wGateMat;
  TileB wUpMat;
  TileB wDownMat;
  TASSIGN(xMat, kL1X);
  TASSIGN(hiddenMat, kL1Hidden);
  TASSIGN(wGateMat, kL1WGate);
  TASSIGN(wUpMat, kL1WUp);
  TASSIGN(wDownMat, kL1WDown);

  TL aT;
  TR bT;
  TC cT;
  TASSIGN(aT, 0x0);
  TASSIGN(bT, 0x0);
  TASSIGN(cT, 0x0);

  constexpr int xTileBytes = FFN_X_BYTES;
  constexpr int wGateTileBytes = FFN_W_GATE_BYTES;
  constexpr int wUpTileBytes = FFN_W_UP_BYTES;
  constexpr int wDownTileBytes = FFN_W_DOWN_BYTES;
  constexpr int partialTileBytes = FFN_GATE_PARTIAL_BYTES;
  constexpr int hiddenTileBytes = FFN_HIDDEN_BYTES;
  constexpr int downTileBytes = FFN_DOWN_PARTIAL_BYTES;
  __gm__ uint8_t *xBlock = x + blockIdx * xTileBytes;
  __gm__ uint8_t *wGateBlock = wGate + blockIdx * wGateTileBytes;
  __gm__ uint8_t *wUpBlock = wUp + blockIdx * wUpTileBytes;
  __gm__ uint8_t *wDownBlock = wDown + blockIdx * wDownTileBytes;
  __gm__ uint8_t *gateBlock = gatePartial + blockIdx * partialTileBytes;
  __gm__ uint8_t *upBlock = upPartial + blockIdx * partialTileBytes;
  __gm__ uint8_t *hiddenBlock = hiddenIn + blockIdx * hiddenTileBytes;
  __gm__ uint8_t *downBlock = downPartial + blockIdx * downTileBytes;

  GX xG(reinterpret_cast<__gm__ half *>(xBlock));
  GW wGateG(reinterpret_cast<__gm__ half *>(wGateBlock));
  GW wUpG(reinterpret_cast<__gm__ half *>(wUpBlock));
  GW wDownG(reinterpret_cast<__gm__ half *>(wDownBlock));

  TLOAD(xMat, xG);
  TLOAD(wGateMat, wGateG);
  TLOAD(wUpMat, wUpG);
  TLOAD(wDownMat, wDownG);

#ifndef __PTO_AUTO__
  set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
  wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
#endif

  if (phase == 0) {
    GatePipe gatePipe(reinterpret_cast<__gm__ void *>(gateBlock), 0, 0);
    UpPipe upPipe(reinterpret_cast<__gm__ void *>(upBlock), 0, 0);

    // -------- gate: gatePartial = x @ W_gate --------
    TMOV(aT, xMat);
    TMOV(bT, wGateMat);

#ifndef __PTO_AUTO__
    set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
    wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
#endif

    TMATMUL(cT, aT, bT);

#ifndef __PTO_AUTO__
    set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
#endif

    TPUSH<GatePipe, TC, TileSplitAxis::TILE_NO_SPLIT>(gatePipe, cT);

#ifndef __PTO_AUTO__
    pipe_barrier(PIPE_ALL);
#endif
    dsb(DSB_DDR);
    // -------- up: upPartial = x @ W_up --------
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

    TPUSH<UpPipe, TC, TileSplitAxis::TILE_NO_SPLIT>(upPipe, cT);

#ifndef __PTO_AUTO__
    pipe_barrier(PIPE_ALL);
#endif
    dsb(DSB_DDR);
    return;
  }

  if (phase != 1) {
    return;
  }

  HiddenPipe hiddenPipe(reinterpret_cast<__gm__ void *>(hiddenBlock), 0, kL1Hidden);
  TPOP<HiddenPipe, TileA, TileSplitAxis::TILE_NO_SPLIT>(hiddenPipe, hiddenMat);
#ifndef __PTO_AUTO__
  set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
  wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
  pipe_barrier(PIPE_ALL);
#endif
  dsb(DSB_DDR);

  {
    // -------- down: downPartial = hidden @ W_down --------
    // Hidden is [T, Fi] fp16, W_down is [Fi, H] fp16, output is [T, H] fp32.
    // With FFN_FFN_TILE == FFN_MODEL_TILE in the default 1x2 config the
    // GX/GW/GO/TileA/TileB/TileC type aliases are byte-equivalent to the
    // gate/up step, so we reuse them.  If the two tile dims diverge in a
    // future config, split the typedefs.
    TMOV(aT, hiddenMat);
    TMOV(bT, wDownMat);

#ifndef __PTO_AUTO__
    set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
    wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
#endif

    TMATMUL(cT, aT, bT);

#ifndef __PTO_AUTO__
    set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
#endif

    DownPipe downPipe(reinterpret_cast<__gm__ void *>(downBlock), 0, 0);
    TPUSH<DownPipe, TC, TileSplitAxis::TILE_NO_SPLIT>(downPipe, cT);

#ifndef __PTO_AUTO__
    pipe_barrier(PIPE_ALL);
#endif
    dsb(DSB_DDR);
  }
#else
  (void)x;
  (void)wGate;
  (void)wUp;
  (void)wDown;
  (void)gatePartial;
  (void)upPartial;
  (void)hiddenIn;
  (void)downPartial;
  (void)yOutput;
  (void)chunkFlags;
  (void)gridRows;
  (void)gridCols;
  (void)myRankId;
  (void)phase;
#endif
}

void launchDistributedFfnGridComputeKernel(
    uint8_t *x, uint8_t *wGate, uint8_t *wUp, uint8_t *wDown,
    uint8_t *gatePartial, uint8_t *upPartial, uint8_t *hiddenIn,
    uint8_t *downPartial, uint8_t *yOutput, uint8_t *chunkFlags, int gridRows,
    int gridCols, int myRankId, int phase, void *stream) {
  int totalBlocks = gridRows * gridCols;
  if (totalBlocks <= 0) {
    return;
  }
  DistributedFfnGridComputeKernel<<<totalBlocks, nullptr, stream>>>(
      x, wGate, wUp, wDown, gatePartial, upPartial, hiddenIn, downPartial,
      yOutput, chunkFlags, gridRows, gridCols, myRankId, phase);
}
