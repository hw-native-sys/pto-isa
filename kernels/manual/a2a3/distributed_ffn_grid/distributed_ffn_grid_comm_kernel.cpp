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

// Single-device multi-block FFN vec + local GridPipe reduce kernel.
//
// Each block owns one logical grid cell (row, col).  Phase 0 computes
// hidden = fp16(PReLU(gate) * up).  Phase 1 performs one active column of the
// EAST reduce using local same-device GridPipe windows.

#include <cstddef>
#include <cstdint>
#include <pto/pto-inst.hpp>

#include <pto/common/fifo.hpp>
#include <pto/common/grid_pipe.hpp>
#include <pto/npu/a2a3/grid_pipe_runtime.hpp>

#include "common.hpp"
#include "ffn_config.hpp"
#include "gridpipe_payload_inl.hpp"

#ifdef __CCE_AICORE__

using GateF32Tile = pto::Tile<pto::TileType::Vec, float, FFN_TOKEN_TILE,
                              FFN_FFN_TILE, pto::BLayout::RowMajor>;
using UpF32Tile = GateF32Tile;
using HiddenF32Tile = GateF32Tile;
using HiddenF16Tile = pto::Tile<pto::TileType::Vec, half, FFN_TOKEN_TILE,
                                FFN_FFN_TILE, pto::BLayout::RowMajor>;
using DownF32Tile = pto::Tile<pto::TileType::Vec, float, FFN_TOKEN_TILE,
                              FFN_MODEL_TILE, pto::BLayout::RowMajor>;
using FfnGridPipe = pto::GridPipe<DownF32Tile, FFN_SLOT_BYTES, FFN_SLOT_COUNT>;

using ShapeTFi = pto::Shape<1, 1, 1, FFN_TOKEN_TILE, FFN_FFN_TILE>;
using StrideTFi =
    pto::Stride<FFN_TOKEN_TILE * FFN_FFN_TILE, FFN_TOKEN_TILE * FFN_FFN_TILE,
                FFN_TOKEN_TILE * FFN_FFN_TILE, FFN_FFN_TILE, 1>;

using ShapeTH = pto::Shape<1, 1, 1, FFN_TOKEN_TILE, FFN_MODEL_TILE>;
using StrideTH =
    pto::Stride<FFN_TOKEN_TILE * FFN_MODEL_TILE,
                FFN_TOKEN_TILE * FFN_MODEL_TILE,
                FFN_TOKEN_TILE * FFN_MODEL_TILE, FFN_MODEL_TILE, 1>;
using GTHF32 = pto::GlobalTensor<float, ShapeTH, StrideTH, pto::Layout::ND>;

template <uint8_t FlagId, uint8_t DirType, uint32_t SlotSize, uint32_t SlotNum = 1, uint32_t LocalSlotNum = 1>
struct FfnSerialTPipe {
  static constexpr uint8_t DIR_MASK = 0x7;
  static constexpr uint8_t DIR_TYPE = DIR_MASK & DirType;
  static constexpr bool is_c2v = (DIR_TYPE == pto::Direction::DIR_C2V);
  static constexpr bool is_v2c = (DIR_TYPE == pto::Direction::DIR_V2C);
  static constexpr uint32_t SyncPeriod = 1;
  using RingFiFo = pto::RingFIFO<SlotSize, SlotNum, LocalSlotNum>;

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

    template <typename TileProd, pto::TileSplitAxis Split>
    PTO_INTERNAL void push(RingFiFo &fifo, TileProd &tile) {
      static_assert(Split == pto::TileSplitAxis::TILE_NO_SPLIT,
                    "distributed_ffn_grid uses full-tile TPipe slots");
      static_assert((is_c2v && TileProd::Loc == pto::TileType::Acc) ||
                        (is_v2c && TileProd::Loc == pto::TileType::Vec),
                    "TPipe direction does not match producer tile type");
      using T = typename TileProd::DType;
      constexpr int rows = TileProd::Rows;
      constexpr int cols = TileProd::Cols;
      size_t entryBase = (tileIndex % RingFiFo::SLOT_NUM) * RingFiFo::SLOT_SIZE;
      using GlobalData = pto::GlobalTensor<T, pto::Shape<1, 1, 1, rows, cols>, pto::Stride<1, 1, 1, cols, 1>>;
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

    template <typename TileCons, pto::TileSplitAxis Split>
    PTO_INTERNAL void pop(RingFiFo &fifo, TileCons &tile) {
      static_assert(Split == pto::TileSplitAxis::TILE_NO_SPLIT,
                    "distributed_ffn_grid uses full-tile TPipe slots");
      static_assert((is_c2v && TileCons::Loc == pto::TileType::Vec) ||
                        (is_v2c && TileCons::Loc == pto::TileType::Mat),
                    "TPipe direction does not match consumer tile type");
      using T = typename TileCons::DType;
      constexpr int rows = TileCons::Rows;
      constexpr int cols = TileCons::Cols;
      size_t entryBase = (tileIndex % RingFiFo::SLOT_NUM) * RingFiFo::SLOT_SIZE;
      uint64_t localBase = TileCons::Loc == pto::TileType::Vec ? fifo.C2V_CONSUMER_BUF : fifo.V2C_CONSUMER_BUF;
      localBase += (tileIndex % RingFiFo::LOCAL_SLOT_NUM) * rows * cols * sizeof(T);
      TASSIGN_IMPL(tile, localBase);
      using GlobalData = pto::GlobalTensor<T, pto::Shape<1, 1, 1, rows, cols>, pto::Stride<1, 1, 1, cols, 1>>;
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

using GatePipe = FfnSerialTPipe<0, static_cast<uint8_t>(pto::Direction::DIR_C2V), FFN_GATE_PARTIAL_BYTES>;
using UpPipe = FfnSerialTPipe<2, static_cast<uint8_t>(pto::Direction::DIR_C2V), FFN_UP_PARTIAL_BYTES>;
using HiddenPipe = FfnSerialTPipe<4, static_cast<uint8_t>(pto::Direction::DIR_V2C), FFN_HIDDEN_BYTES>;
using DownPipe = FfnSerialTPipe<6, static_cast<uint8_t>(pto::Direction::DIR_C2V), FFN_DOWN_PARTIAL_BYTES>;

constexpr int kUbGateF32 = 0x0000;
constexpr int kUbUpF32 = 0x1000;
constexpr int kUbHiddenF32 = 0x2000;
constexpr int kUbHiddenF16 = 0x3000;
constexpr int kUbDownF32 = 0x4000;
constexpr int kUbAddF32 = 0x5000;

#endif  // __CCE_AICORE__

__global__ AICORE void DistributedFfnGridCommKernel(
    __gm__ uint8_t *reducePipeWindow, __gm__ uint8_t *gatePartial,
    __gm__ uint8_t *upPartial, __gm__ uint8_t *hiddenOut,
    __gm__ uint8_t *downPartial, __gm__ uint8_t *yOutput,
    __gm__ uint8_t *hcclCtxRaw, __gm__ uint8_t *chunkFlags, int gridRows,
    int gridCols, int phase) {
#ifdef __CCE_AICORE__
  (void)chunkFlags;

  int blockIdx = get_block_idx();
  int totalBlocks = gridRows * gridCols;
  if (blockIdx < 0 || blockIdx >= totalBlocks) {
    return;
  }
  int row = blockIdx / gridCols;
  int col = blockIdx - row * gridCols;

  if (phase == 1) {
    DownF32Tile downF32;
    DownF32Tile addF32;
    TASSIGN(downF32, kUbDownF32);
    TASSIGN(addF32, kUbAddF32);

    constexpr int downTileBytes = FFN_DOWN_PARTIAL_BYTES;
    constexpr int yTileBytes = FFN_Y_OUTPUT_BYTES;
    __gm__ uint8_t *downBlock = downPartial + blockIdx * downTileBytes;
    __gm__ uint8_t *yBlock = yOutput + row * yTileBytes;

    DownPipe downPipe(reinterpret_cast<__gm__ void *>(downBlock), kUbDownF32, 0);
    pto::TPOP<DownPipe, DownF32Tile, pto::TileSplitAxis::TILE_NO_SPLIT>(downPipe, downF32);

#ifndef __PTO_AUTO__
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
#endif

    FfnGridPipe reducePipe;
    pto::GridShape shape{gridRows, gridCols};
    pto::GridCoord coord{row, col};
    __gm__ uint8_t *window = reducePipeWindow + blockIdx * FFN_GRID_WINDOW_BYTES;
    pto::a2a3_grid::InitGridPipeFromWindow(
        reducePipe, shape, coord, window, reinterpret_cast<__gm__ void *>(hcclCtxRaw),
        /*pipeId=*/0);

    using pto::GridDirection;
    if (col > 0) {
      pto::TPOP<GridDirection::EAST>(reducePipe, addF32);
#ifndef __PTO_AUTO__
      pipe_barrier(PIPE_ALL);
#endif
      dsb(DSB_DDR);
      TADD(downF32, downF32, addF32);
#ifndef __PTO_AUTO__
      pipe_barrier(PIPE_V);
#endif
    }

    if (col + 1 < gridCols) {
#ifndef __PTO_AUTO__
      pipe_barrier(PIPE_ALL);
#endif
      dsb(DSB_DDR);
      pto::TPUSH<GridDirection::EAST>(reducePipe, downF32);
    } else {
#ifndef __PTO_AUTO__
      set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
      wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
#endif
      GTHF32 yG(reinterpret_cast<__gm__ float *>(yBlock));
      TSTORE(yG, downF32);
    }

#ifndef __PTO_AUTO__
    pipe_barrier(PIPE_ALL);
#endif
    dsb(DSB_DDR);
    return;
  }

  if (phase != 0) {
    return;
  }

  GateF32Tile gateF32;
  UpF32Tile upF32;
  HiddenF32Tile hiddenF32;
  HiddenF16Tile hiddenF16;
  TASSIGN(gateF32, kUbGateF32);
  TASSIGN(upF32, kUbUpF32);
  TASSIGN(hiddenF32, kUbHiddenF32);
  TASSIGN(hiddenF16, kUbHiddenF16);

  constexpr int partialTileBytes = FFN_GATE_PARTIAL_BYTES;
  constexpr int hiddenTileBytes = FFN_HIDDEN_BYTES;
  __gm__ uint8_t *gateBlock = gatePartial + blockIdx * partialTileBytes;
  __gm__ uint8_t *upBlock = upPartial + blockIdx * partialTileBytes;
  __gm__ uint8_t *hiddenBlock = hiddenOut + blockIdx * hiddenTileBytes;

  GatePipe gatePipe(reinterpret_cast<__gm__ void *>(gateBlock), kUbGateF32, 0);
  UpPipe upPipe(reinterpret_cast<__gm__ void *>(upBlock), kUbUpF32, 0);
  pto::TPOP<GatePipe, GateF32Tile, pto::TileSplitAxis::TILE_NO_SPLIT>(gatePipe, gateF32);
  pto::TPOP<UpPipe, UpF32Tile, pto::TileSplitAxis::TILE_NO_SPLIT>(upPipe, upF32);

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

  TCVT(hiddenF16, hiddenF32, pto::RoundMode::CAST_RINT);

#ifndef __PTO_AUTO__
  set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
  wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
#endif

  HiddenPipe hiddenPipe(reinterpret_cast<__gm__ void *>(hiddenBlock), 0, 0);
  pto::TPUSH<HiddenPipe, HiddenF16Tile, pto::TileSplitAxis::TILE_NO_SPLIT>(hiddenPipe, hiddenF16);

#ifndef __PTO_AUTO__
  pipe_barrier(PIPE_ALL);
#endif
  dsb(DSB_DDR);
#else
  (void)reducePipeWindow;
  (void)gatePartial;
  (void)upPartial;
  (void)hiddenOut;
  (void)downPartial;
  (void)yOutput;
  (void)hcclCtxRaw;
  (void)chunkFlags;
  (void)gridRows;
  (void)gridCols;
  (void)phase;
#endif
}

void launchDistributedFfnGridCommKernel(uint8_t *reducePipeWindow,
                                        uint8_t *gatePartial,
                                        uint8_t *upPartial, uint8_t *hiddenOut,
                                        uint8_t *downPartial, uint8_t *yOutput,
                                        uint8_t *hcclCtx, uint8_t *chunkFlags,
                                        int gridRows, int gridCols,
                                        int phase, void *stream) {
  int totalBlocks = gridRows * gridCols;
  if (totalBlocks <= 0) {
    return;
  }
  DistributedFfnGridCommKernel<<<totalBlocks, nullptr, stream>>>(
      reducePipeWindow, gatePartial, upPartial, hiddenOut, downPartial, yOutput,
      hcclCtx, chunkFlags, gridRows, gridCols, phase);
}
