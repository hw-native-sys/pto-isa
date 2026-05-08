/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#include <pto/pto-inst.hpp>
#include <atomic>
#include <chrono>
#include <gtest/gtest.h>
#include <pto/common/fifo.hpp>
#include <thread>
#include <vector>
#include <barrier>
#include "test_common.h"

using namespace std;
using namespace pto;
using namespace PtoTestCommon;

namespace {
using T = float;

class TPUSH_A5Test : public testing::Test {
protected:
    void SetUp() override
    {}
    void TearDown() override
    {}
};

std::string GetGoldenDir()
{
    const testing::TestInfo *testInfo = testing::UnitTest::GetInstance()->current_test_info();
    const std::string caseName = testInfo->name();
    std::string suiteName = testInfo->test_suite_name();
    std::string fullPath = "../" + suiteName + "." + caseName;
    return fullPath;
}

static __aicore__ void main_incore_0_aic_BI_LEFTRIGHT(__gm__ float* v1, __gm__ float* v2, __gm__ float* v3, int32_t v4) {
  unsigned v5 = 0;
  __gm__ void * v6 = nullptr;
  const int32_t v7 = 8;
  const int32_t v8 = 64;
  const int32_t v9 = 1;
  const int32_t v10 = 512;
  const int32_t v11 = 32;
  const int64_t v12 = 0;
  const int64_t v13 = 32768;
  const int32_t v14 = 0;
  using T = float;

  auto v15 = TPipe<0, Direction::DIR_BOTH, 8192, 4, 2, false>(v6, v14, v14);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
  set_flag(PIPE_MTE1, PIPE_S, EVENT_ID0);
  set_flag(PIPE_MTE1, PIPE_S, EVENT_ID1);
  set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
  set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
  for (size_t v16 = (size_t) v14; v16 < ((size_t) v7); v16 += (size_t) v9) {
    Tile<TileType::Mat, float, 64, 64, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v17 = Tile<TileType::Mat, float, 64, 64, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null>(v8, v8);
    TASSIGN(v17, v13);
    Tile<TileType::Mat, float, 64, 64, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v18 = Tile<TileType::Mat, float, 64, 64, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null>(v8, v8);
    __cbuf__ float* v19 = v17.data();
    uint64_t v20 = reinterpret_cast<uint64_t>(v19);
    TASSIGN(v18, v20);
    pto::Shape<1, 1, 1, 64, 64> v21 = pto::Shape<1, 1, 1, 64, 64>();
    pto::Stride<32768, 32768, 32768, 512, 1> v22 = pto::Stride<32768, 32768, 32768, 512, 1>();
    GlobalTensor<float, pto::Shape<1, 1, 1, 64, 64>, pto::Stride<32768, 32768, 32768, 512, 1>, pto::Layout::ND> v23 = GlobalTensor<float, pto::Shape<1, 1, 1, 64, 64>, pto::Stride<32768, 32768, 32768, 512, 1>, pto::Layout::ND>(v3 + (v5 + v5 * (unsigned) v10 + (unsigned) ((int32_t) (uint32_t) ((int32_t) (uint32_t) ((int32_t) (uint32_t) v4 * (uint32_t) v7) + (uint32_t) ((int32_t) v16)) * (uint32_t) v8) * (unsigned) v9), v21, v22);
    wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
    TLOAD(v18, v23);
    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    Tile<TileType::Mat, float, 32, 64, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v24 = Tile<TileType::Mat, float, 32, 64, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null>(v11, v8);
    wait_flag(PIPE_MTE1, PIPE_S, EVENT_ID0);
    TPOP<TPipe<0, Direction::DIR_BOTH, 8192, 4, 2, false>, Tile<TileType::Mat, float, 32, 64, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null>, TileSplitAxis::TILE_LEFT_RIGHT>(v15, v24);
    set_flag(PIPE_S, PIPE_MTE1, EVENT_ID0);
    Tile<TileType::Left, float, 32, 64, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v25 = Tile<TileType::Left, float, 32, 64, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null>(v11, v8);
    TASSIGN(v25, v12);
    Tile<TileType::Left, float, 32, 64, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v26 = Tile<TileType::Left, float, 32, 64, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null>(v11, v8);
    __ca__ float* v27 = v25.data();
    uint64_t v28 = reinterpret_cast<uint64_t>(v27);
    TASSIGN(v26, v28);
    wait_flag(PIPE_S, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
    TMOV(v26, v24);
    set_flag(PIPE_MTE1, PIPE_S, EVENT_ID0);
    TFREE<TPipe<0, Direction::DIR_BOTH, 8192, 4, 2, false>, TileSplitAxis::TILE_LEFT_RIGHT>(v15);
    Tile<TileType::Right, float, 64, 64, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null> v29 = Tile<TileType::Right, float, 64, 64, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null>(v8, v8);
    TASSIGN(v29, v12);
    Tile<TileType::Right, float, 64, 64, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null> v30 = Tile<TileType::Right, float, 64, 64, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null>(v8, v8);
    __cb__ float* v31 = v29.data();
    uint64_t v32 = reinterpret_cast<uint64_t>(v31);
    TASSIGN(v30, v32);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    TMOV(v30, v18);
    set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
    set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
    Tile<TileType::Acc, float, 32, 64, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null, CompactMode::Null> v33 = Tile<TileType::Acc, float, 32, 64, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null, CompactMode::Null>(v11, v8);
    TASSIGN(v33, v12);
    Tile<TileType::Acc, float, 32, 64, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null, CompactMode::Null> v34 = Tile<TileType::Acc, float, 32, 64, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null, CompactMode::Null>(v11, v8);
    __cc__ float* v35 = v33.data();
    uint64_t v36 = reinterpret_cast<uint64_t>(v35);
    TASSIGN(v34, v36);
    wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
    wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
    TMATMUL(v34, v26, v30);
    set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
    set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    TPUSH<TPipe<0, Direction::DIR_BOTH, 8192, 4, 2, false>, Tile<TileType::Acc, float, 32, 64, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null, CompactMode::Null>, TileSplitAxis::TILE_LEFT_RIGHT>(v15, v34);
    set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
  }
  wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
  wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
  wait_flag(PIPE_MTE1, PIPE_S, EVENT_ID0);
  wait_flag(PIPE_MTE1, PIPE_S, EVENT_ID1);
  wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
  wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);

  return;
}

static __aicore__ void main_incore_0_aiv_BI_LEFTRIGHT(__gm__ float* v1, __gm__ float* v2, __gm__ float* v3, int32_t v4) {
  unsigned v5 = 0;
  __gm__ void * v6 = nullptr;
  const float v7 = 1.0f;
  const int32_t v8 = 8;
  const int32_t v9 = 64;
  const int32_t v10 = 1;
  const int32_t v11 = 512;
  const int32_t v12 = 32;
  const int64_t v13 = 45056;
  const int64_t v14 = 40960;
  const int64_t v15 = 36864;
  const int64_t v16 = 32768;
  const int32_t v17 = 0;
  using T = float;

  set_mask_norm();
  set_vector_mask(-1, -1);
  int64_t v18 = get_subblockid();
  auto v19 = TPipe<0, Direction::DIR_BOTH, 8192, 4, 2, false>(v6, v17, v17);
  Tile<TileType::Vec, float, 32, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v20 = Tile<TileType::Vec, float, 32, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v12, v12);
  TASSIGN(v20, v16);
  Tile<TileType::Vec, float, 32, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v21 = Tile<TileType::Vec, float, 32, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v12, v12);
  __ubuf__ float* v22 = v20.data();
  uint64_t v23 = reinterpret_cast<uint64_t>(v22);
  TASSIGN(v21, v23);
  int32_t v24 = (int32_t) ((uint32_t) ((int32_t) (int64_t) v18) * (uint32_t) v12);
  pto::Shape<1, 1, 1, 32, 32> v25 = pto::Shape<1, 1, 1, 32, 32>();
  pto::Stride<2048, 2048, 2048, 64, 1> v26 = pto::Stride<2048, 2048, 2048, 64, 1>();
  GlobalTensor<float, pto::Shape<1, 1, 1, 32, 32>, pto::Stride<2048, 2048, 2048, 64, 1>, pto::Layout::ND> v27 = GlobalTensor<float, pto::Shape<1, 1, 1, 32, 32>, pto::Stride<2048, 2048, 2048, 64, 1>, pto::Layout::ND>(v2 + (v5 + v5 * (unsigned) v9 + (unsigned) v24 * (unsigned) v10), v25, v26);
  set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
  set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
  set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
  set_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
  set_flag(PIPE_V, PIPE_S, EVENT_ID0);
  set_flag(PIPE_V, PIPE_S, EVENT_ID1);
  TLOAD(v21, v27);
  set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
  wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
  for (size_t v28 = (size_t) v17; v28 < ((size_t) v8); v28 += (size_t) v10) {
    Tile<TileType::Vec, float, 32, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v29 = Tile<TileType::Vec, float, 32, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v12, v12);
    TASSIGN(v29, v15);
    Tile<TileType::Vec, float, 32, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v30 = Tile<TileType::Vec, float, 32, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v12, v12);
    __ubuf__ float* v31 = v29.data();
    uint64_t v32 = reinterpret_cast<uint64_t>(v31);
    TASSIGN(v30, v32);
    pto::Shape<1, 1, 1, 32, 32> v33 = pto::Shape<1, 1, 1, 32, 32>();
    pto::Stride<16384, 16384, 16384, 512, 1> v34 = pto::Stride<16384, 16384, 16384, 512, 1>();
    GlobalTensor<float, pto::Shape<1, 1, 1, 32, 32>, pto::Stride<16384, 16384, 16384, 512, 1>, pto::Layout::ND> v35 = GlobalTensor<float, pto::Shape<1, 1, 1, 32, 32>, pto::Stride<16384, 16384, 16384, 512, 1>, pto::Layout::ND>(v1 + (v5 + v5 * (unsigned) v11 + (unsigned) ((int32_t) (uint32_t) ((int32_t) (uint32_t) ((int32_t) (uint32_t) ((int32_t) (uint32_t) v4 * (uint32_t) v8) + (uint32_t) ((int32_t) v28)) * (uint32_t) v9) + (uint32_t) v24) * (unsigned) v10), v33, v34);
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    TLOAD(v30, v35);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
    Tile<TileType::Vec, float, 32, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v36 = Tile<TileType::Vec, float, 32, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v12, v12);
    TASSIGN(v36, v14);
    Tile<TileType::Vec, float, 32, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v37 = Tile<TileType::Vec, float, 32, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v12, v12);
    __ubuf__ float* v38 = v36.data();
    uint64_t v39 = reinterpret_cast<uint64_t>(v38);
    TASSIGN(v37, v39);
    TADDS(v37, v21, v7);
    Tile<TileType::Vec, float, 32, 32, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v40 = Tile<TileType::Vec, float, 32, 32, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null>(v12, v12);
    TASSIGN(v40, v13);
    Tile<TileType::Vec, float, 32, 32, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v41 = Tile<TileType::Vec, float, 32, 32, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null>(v12, v12);
    __ubuf__ float* v42 = v40.data();
    uint64_t v43 = reinterpret_cast<uint64_t>(v42);
    TASSIGN(v41, v43);
    wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
    TMOV(v41, v37);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    TPUSH<TPipe<0, Direction::DIR_BOTH, 8192, 4, 2, false>, Tile<TileType::Vec, float, 32, 32, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null>, TileSplitAxis::TILE_LEFT_RIGHT>(v19, v41);
    set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
    Tile<TileType::Vec, float, 32, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v44 = Tile<TileType::Vec, float, 32, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v12, v12);
    wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
    TPOP<TPipe<0, Direction::DIR_BOTH, 8192, 4, 2, false>, Tile<TileType::Vec, float, 32, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>, TileSplitAxis::TILE_LEFT_RIGHT>(v19, v44);
    set_flag(PIPE_S, PIPE_V, EVENT_ID0);
    Tile<TileType::Vec, float, 32, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v45 = Tile<TileType::Vec, float, 32, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v12, v12);
    TASSIGN(v45, v15);
    Tile<TileType::Vec, float, 32, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v46 = Tile<TileType::Vec, float, 32, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v12, v12);
    __ubuf__ float* v47 = v45.data();
    uint64_t v48 = reinterpret_cast<uint64_t>(v47);
    TASSIGN(v46, v48);
    wait_flag(PIPE_S, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
    TADD(v46, v30, v44);
    set_flag(PIPE_V, PIPE_S, EVENT_ID0);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
    TFREE<TPipe<0, Direction::DIR_BOTH, 8192, 4, 2, false>, TileSplitAxis::TILE_LEFT_RIGHT>(v19);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
    pipe_barrier(PIPE_MTE3);
    TSTORE(v35, v46);
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
  }
  wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
  wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
  wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
  wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
  wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
  wait_flag(PIPE_V, PIPE_S, EVENT_ID1);

  return;
}

static __aicore__ void main_incore_0_aic_BI_NOSPLIT(__gm__ float* v1, __gm__ float* v2, __gm__ float* v3, int32_t v4) {
  unsigned v5 = 0;
  __gm__ void * v6 = nullptr;
  const int32_t v7 = 8;
  const int32_t v8 = 64;
  const int32_t v9 = 1;
  const int32_t v10 = 512;
  const int32_t v11 = 32;
  const int64_t v12 = 0;
  const int64_t v13 = 32768;
  const int32_t v14 = 0;
  using T = float;

  auto v15 = TPipe<0, Direction::DIR_BOTH, 8192, 4, 2, true>(v6, v14, v14);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
  set_flag(PIPE_MTE1, PIPE_S, EVENT_ID0);
  set_flag(PIPE_MTE1, PIPE_S, EVENT_ID1);
  set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
  set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
  for (size_t v16 = (size_t) v14; v16 < ((size_t) v7); v16 += (size_t) v9) {
    Tile<TileType::Mat, float, 64, 64, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v17 = Tile<TileType::Mat, float, 64, 64, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null>(v8, v8);
    TASSIGN(v17, v13);
    Tile<TileType::Mat, float, 64, 64, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v18 = Tile<TileType::Mat, float, 64, 64, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null>(v8, v8);
    __cbuf__ float* v19 = v17.data();
    uint64_t v20 = reinterpret_cast<uint64_t>(v19);
    TASSIGN(v18, v20);
    pto::Shape<1, 1, 1, 64, 64> v21 = pto::Shape<1, 1, 1, 64, 64>();
    pto::Stride<32768, 32768, 32768, 512, 1> v22 = pto::Stride<32768, 32768, 32768, 512, 1>();
    GlobalTensor<float, pto::Shape<1, 1, 1, 64, 64>, pto::Stride<32768, 32768, 32768, 512, 1>, pto::Layout::ND> v23 = GlobalTensor<float, pto::Shape<1, 1, 1, 64, 64>, pto::Stride<32768, 32768, 32768, 512, 1>, pto::Layout::ND>(v3 + (v5 + v5 * (unsigned) v10 + (unsigned) ((int32_t) (uint32_t) ((int32_t) (uint32_t) ((int32_t) (uint32_t) v4 * (uint32_t) v7) + (uint32_t) ((int32_t) v16)) * (uint32_t) v8) * (unsigned) v9), v21, v22);
    wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
    TLOAD(v18, v23);
    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    Tile<TileType::Mat, float, 32, 64, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v24 = Tile<TileType::Mat, float, 32, 64, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null>(v11, v8);
    wait_flag(PIPE_MTE1, PIPE_S, EVENT_ID0);
    TPOP<TPipe<0, Direction::DIR_BOTH, 8192, 4, 2, true>, Tile<TileType::Mat, float, 32, 64, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null>, TileSplitAxis::TILE_NO_SPLIT>(v15, v24);
    set_flag(PIPE_S, PIPE_MTE1, EVENT_ID0);
    Tile<TileType::Left, float, 32, 64, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v25 = Tile<TileType::Left, float, 32, 64, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null>(v11, v8);
    TASSIGN(v25, v12);
    Tile<TileType::Left, float, 32, 64, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v26 = Tile<TileType::Left, float, 32, 64, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null>(v11, v8);
    __ca__ float* v27 = v25.data();
    uint64_t v28 = reinterpret_cast<uint64_t>(v27);
    TASSIGN(v26, v28);
    wait_flag(PIPE_S, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
    TMOV(v26, v24);
    set_flag(PIPE_MTE1, PIPE_S, EVENT_ID0);
    TFREE<TPipe<0, Direction::DIR_BOTH, 8192, 4, 2, true>, TileSplitAxis::TILE_NO_SPLIT>(v15);
    Tile<TileType::Right, float, 64, 64, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null> v29 = Tile<TileType::Right, float, 64, 64, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null>(v8, v8);
    TASSIGN(v29, v12);
    Tile<TileType::Right, float, 64, 64, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null> v30 = Tile<TileType::Right, float, 64, 64, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null>(v8, v8);
    __cb__ float* v31 = v29.data();
    uint64_t v32 = reinterpret_cast<uint64_t>(v31);
    TASSIGN(v30, v32);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    TMOV(v30, v18);
    set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
    set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
    Tile<TileType::Acc, float, 32, 64, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null, CompactMode::Null> v33 = Tile<TileType::Acc, float, 32, 64, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null, CompactMode::Null>(v11, v8);
    TASSIGN(v33, v12);
    Tile<TileType::Acc, float, 32, 64, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null, CompactMode::Null> v34 = Tile<TileType::Acc, float, 32, 64, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null, CompactMode::Null>(v11, v8);
    __cc__ float* v35 = v33.data();
    uint64_t v36 = reinterpret_cast<uint64_t>(v35);
    TASSIGN(v34, v36);
    wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
    wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
    TMATMUL(v34, v26, v30);
    set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
    set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    TPUSH<TPipe<0, Direction::DIR_BOTH, 8192, 4, 2, true>, Tile<TileType::Acc, float, 32, 64, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null, CompactMode::Null>, TileSplitAxis::TILE_NO_SPLIT>(v15, v34);
    set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
  }
  wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
  wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
  wait_flag(PIPE_MTE1, PIPE_S, EVENT_ID0);
  wait_flag(PIPE_MTE1, PIPE_S, EVENT_ID1);
  wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
  wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);

  return;
}

static __aicore__ void main_incore_0_aiv_BI_NOSPLIT(__gm__ float* v1, __gm__ float* v2, __gm__ float* v3, int32_t v4) {
  unsigned v5 = 0;
  __gm__ void * v6 = nullptr;
  const float v7 = 1.0f;
  const int32_t v8 = 8;
  const int32_t v9 = 64;
  const int32_t v10 = 1;
  const int32_t v11 = 512;
  const int32_t v12 = 32;
  const int64_t v13 = 57344;
  const int64_t v14 = 49152;
  const int64_t v15 = 40960;
  const int64_t v16 = 32768;
  const int32_t v17 = 0;
  using T = float;

  set_mask_norm();
  set_vector_mask(-1, -1);
  if (get_subblockid() == 0) {
  auto v18 = TPipe<0, Direction::DIR_BOTH, 8192, 4, 2, true>(v6, v17, v17);
  Tile<TileType::Vec, float, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v19 = Tile<TileType::Vec, float, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v12, v9);
  TASSIGN(v19, v16);
  Tile<TileType::Vec, float, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v20 = Tile<TileType::Vec, float, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v12, v9);
  __ubuf__ float* v21 = v19.data();
  uint64_t v22 = reinterpret_cast<uint64_t>(v21);
  TASSIGN(v20, v22);
  pto::Shape<1, 1, 1, 32, 64> v23 = pto::Shape<1, 1, 1, 32, 64>();
  pto::Stride<2048, 2048, 2048, 64, 1> v24 = pto::Stride<2048, 2048, 2048, 64, 1>();
  GlobalTensor<float, pto::Shape<1, 1, 1, 32, 64>, pto::Stride<2048, 2048, 2048, 64, 1>, pto::Layout::ND> v25 = GlobalTensor<float, pto::Shape<1, 1, 1, 32, 64>, pto::Stride<2048, 2048, 2048, 64, 1>, pto::Layout::ND>(v2 + (v5 + v5 * (unsigned) v9 + v5 * (unsigned) v10), v23, v24);
  set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
  set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
  set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
  set_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
  set_flag(PIPE_V, PIPE_S, EVENT_ID0);
  set_flag(PIPE_V, PIPE_S, EVENT_ID1);
  TLOAD(v20, v25);
  set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
  wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
  for (size_t v26 = (size_t) v17; v26 < ((size_t) v8); v26 += (size_t) v10) {
    Tile<TileType::Vec, float, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v27 = Tile<TileType::Vec, float, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v12, v9);
    TASSIGN(v27, v15);
    Tile<TileType::Vec, float, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v28 = Tile<TileType::Vec, float, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v12, v9);
    __ubuf__ float* v29 = v27.data();
    uint64_t v30 = reinterpret_cast<uint64_t>(v29);
    TASSIGN(v28, v30);
    pto::Shape<1, 1, 1, 32, 64> v31 = pto::Shape<1, 1, 1, 32, 64>();
    pto::Stride<16384, 16384, 16384, 512, 1> v32 = pto::Stride<16384, 16384, 16384, 512, 1>();
    GlobalTensor<float, pto::Shape<1, 1, 1, 32, 64>, pto::Stride<16384, 16384, 16384, 512, 1>, pto::Layout::ND> v33 = GlobalTensor<float, pto::Shape<1, 1, 1, 32, 64>, pto::Stride<16384, 16384, 16384, 512, 1>, pto::Layout::ND>(v1 + (v5 + v5 * (unsigned) v11 + (unsigned) ((int32_t) (uint32_t) ((int32_t) (uint32_t) ((int32_t) (uint32_t) v4 * (uint32_t) v8) + (uint32_t) ((int32_t) v26)) * (uint32_t) v9) * (unsigned) v10), v31, v32);
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    TLOAD(v28, v33);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
    Tile<TileType::Vec, float, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v34 = Tile<TileType::Vec, float, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v12, v9);
    TASSIGN(v34, v14);
    Tile<TileType::Vec, float, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v35 = Tile<TileType::Vec, float, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v12, v9);
    __ubuf__ float* v36 = v34.data();
    uint64_t v37 = reinterpret_cast<uint64_t>(v36);
    TASSIGN(v35, v37);
    TADDS(v35, v20, v7);
    Tile<TileType::Vec, float, 32, 64, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v38 = Tile<TileType::Vec, float, 32, 64, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null>(v12, v9);
    TASSIGN(v38, v13);
    Tile<TileType::Vec, float, 32, 64, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v39 = Tile<TileType::Vec, float, 32, 64, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null>(v12, v9);
    __ubuf__ float* v40 = v38.data();
    uint64_t v41 = reinterpret_cast<uint64_t>(v40);
    TASSIGN(v39, v41);
    wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
    TMOV(v39, v35);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    TPUSH<TPipe<0, Direction::DIR_BOTH, 8192, 4, 2, true>, Tile<TileType::Vec, float, 32, 64, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null>, TileSplitAxis::TILE_NO_SPLIT>(v18, v39);
    set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
    Tile<TileType::Vec, float, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v42 = Tile<TileType::Vec, float, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v12, v9);
    wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
    TPOP<TPipe<0, Direction::DIR_BOTH, 8192, 4, 2, true>, Tile<TileType::Vec, float, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>, TileSplitAxis::TILE_NO_SPLIT>(v18, v42);
    set_flag(PIPE_S, PIPE_V, EVENT_ID0);
    Tile<TileType::Vec, float, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v43 = Tile<TileType::Vec, float, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v12, v9);
    TASSIGN(v43, v15);
    Tile<TileType::Vec, float, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v44 = Tile<TileType::Vec, float, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v12, v9);
    __ubuf__ float* v45 = v43.data();
    uint64_t v46 = reinterpret_cast<uint64_t>(v45);
    TASSIGN(v44, v46);
    wait_flag(PIPE_S, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
    TADD(v44, v28, v42);
    set_flag(PIPE_V, PIPE_S, EVENT_ID0);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
    TFREE<TPipe<0, Direction::DIR_BOTH, 8192, 4, 2, true>, TileSplitAxis::TILE_NO_SPLIT>(v18);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
    pipe_barrier(PIPE_MTE3);
    TSTORE(v33, v44);
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
  }
  wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
  wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
  wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
  wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
  wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
  wait_flag(PIPE_V, PIPE_S, EVENT_ID1);
  }

  return;
}

static __aicore__ void main_incore_0_aic_BI_TOPDOWN(__gm__ float* v1, __gm__ float* v2, __gm__ float* v3, int32_t v4) {
  unsigned v5 = 0;
  __gm__ void * v6 = nullptr;
  const int32_t v7 = 8;
  const int32_t v8 = 64;
  const int32_t v9 = 1;
  const int32_t v10 = 512;
  const int32_t v11 = 32;
  const int64_t v12 = 0;
  const int64_t v13 = 32768;
  const int32_t v14 = 0;
  using T = float;

  auto v15 = TPipe<0, Direction::DIR_BOTH, 8192, 4, 2, false>(v6, v14, v14);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
  set_flag(PIPE_MTE1, PIPE_S, EVENT_ID0);
  set_flag(PIPE_MTE1, PIPE_S, EVENT_ID1);
  set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
  set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
  for (size_t v16 = (size_t) v14; v16 < ((size_t) v7); v16 += (size_t) v9) {
    Tile<TileType::Mat, float, 64, 64, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v17 = Tile<TileType::Mat, float, 64, 64, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null>(v8, v8);
    TASSIGN(v17, v13);
    Tile<TileType::Mat, float, 64, 64, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v18 = Tile<TileType::Mat, float, 64, 64, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null>(v8, v8);
    __cbuf__ float* v19 = v17.data();
    uint64_t v20 = reinterpret_cast<uint64_t>(v19);
    TASSIGN(v18, v20);
    pto::Shape<1, 1, 1, 64, 64> v21 = pto::Shape<1, 1, 1, 64, 64>();
    pto::Stride<32768, 32768, 32768, 512, 1> v22 = pto::Stride<32768, 32768, 32768, 512, 1>();
    GlobalTensor<float, pto::Shape<1, 1, 1, 64, 64>, pto::Stride<32768, 32768, 32768, 512, 1>, pto::Layout::ND> v23 = GlobalTensor<float, pto::Shape<1, 1, 1, 64, 64>, pto::Stride<32768, 32768, 32768, 512, 1>, pto::Layout::ND>(v3 + (v5 + v5 * (unsigned) v10 + (unsigned) ((int32_t) (uint32_t) ((int32_t) (uint32_t) ((int32_t) (uint32_t) v4 * (uint32_t) v7) + (uint32_t) ((int32_t) v16)) * (uint32_t) v8) * (unsigned) v9), v21, v22);
    wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
    TLOAD(v18, v23);
    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    Tile<TileType::Mat, float, 32, 64, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v24 = Tile<TileType::Mat, float, 32, 64, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null>(v11, v8);
    wait_flag(PIPE_MTE1, PIPE_S, EVENT_ID0);
    TPOP<TPipe<0, Direction::DIR_BOTH, 8192, 4, 2, false>, Tile<TileType::Mat, float, 32, 64, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null>, TileSplitAxis::TILE_UP_DOWN>(v15, v24);
    set_flag(PIPE_S, PIPE_MTE1, EVENT_ID0);
    Tile<TileType::Left, float, 32, 64, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v25 = Tile<TileType::Left, float, 32, 64, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null>(v11, v8);
    TASSIGN(v25, v12);
    Tile<TileType::Left, float, 32, 64, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v26 = Tile<TileType::Left, float, 32, 64, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null>(v11, v8);
    __ca__ float* v27 = v25.data();
    uint64_t v28 = reinterpret_cast<uint64_t>(v27);
    TASSIGN(v26, v28);
    wait_flag(PIPE_S, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
    TMOV(v26, v24);
    set_flag(PIPE_MTE1, PIPE_S, EVENT_ID0);
    TFREE<TPipe<0, Direction::DIR_BOTH, 8192, 4, 2, false>, TileSplitAxis::TILE_UP_DOWN>(v15);
    Tile<TileType::Right, float, 64, 64, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null> v29 = Tile<TileType::Right, float, 64, 64, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null>(v8, v8);
    TASSIGN(v29, v12);
    Tile<TileType::Right, float, 64, 64, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null> v30 = Tile<TileType::Right, float, 64, 64, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null>(v8, v8);
    __cb__ float* v31 = v29.data();
    uint64_t v32 = reinterpret_cast<uint64_t>(v31);
    TASSIGN(v30, v32);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    TMOV(v30, v18);
    set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
    set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
    Tile<TileType::Acc, float, 32, 64, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null, CompactMode::Null> v33 = Tile<TileType::Acc, float, 32, 64, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null, CompactMode::Null>(v11, v8);
    TASSIGN(v33, v12);
    Tile<TileType::Acc, float, 32, 64, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null, CompactMode::Null> v34 = Tile<TileType::Acc, float, 32, 64, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null, CompactMode::Null>(v11, v8);
    __cc__ float* v35 = v33.data();
    uint64_t v36 = reinterpret_cast<uint64_t>(v35);
    TASSIGN(v34, v36);
    wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
    wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
    TMATMUL(v34, v26, v30);
    set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
    set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    TPUSH<TPipe<0, Direction::DIR_BOTH, 8192, 4, 2, false>, Tile<TileType::Acc, float, 32, 64, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null, CompactMode::Null>, TileSplitAxis::TILE_UP_DOWN>(v15, v34);
    set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
  }
  wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
  wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
  wait_flag(PIPE_MTE1, PIPE_S, EVENT_ID0);
  wait_flag(PIPE_MTE1, PIPE_S, EVENT_ID1);
  wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
  wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);

  return;
}

static __aicore__ void main_incore_0_aiv_BI_TOPDOWN(__gm__ float* v1, __gm__ float* v2, __gm__ float* v3, int32_t v4) {
  unsigned v5 = 0;
  __gm__ void * v6 = nullptr;
  const float v7 = 1.0f;
  const int32_t v8 = 8;
  const int32_t v9 = 16;
  const int32_t v10 = 64;
  const int32_t v11 = 1;
  const int32_t v12 = 512;
  const int64_t v13 = 45056;
  const int64_t v14 = 40960;
  const int64_t v15 = 36864;
  const int64_t v16 = 32768;
  const int32_t v17 = 0;
  using T = float;

  set_mask_norm();
  set_vector_mask(-1, -1);
  int64_t v18 = get_subblockid();
  auto v19 = TPipe<0, Direction::DIR_BOTH, 8192, 4, 2, false>(v6, v17, v17);
  Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v20 = Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v9, v10);
  TASSIGN(v20, v16);
  Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v21 = Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v9, v10);
  __ubuf__ float* v22 = v20.data();
  uint64_t v23 = reinterpret_cast<uint64_t>(v22);
  TASSIGN(v21, v23);
  int32_t v24 = (int32_t) ((uint32_t) ((int32_t) (int64_t) v18) * (uint32_t) v9);
  pto::Shape<1, 1, 1, 16, 64> v25 = pto::Shape<1, 1, 1, 16, 64>();
  pto::Stride<1024, 1024, 1024, 64, 1> v26 = pto::Stride<1024, 1024, 1024, 64, 1>();
  GlobalTensor<float, pto::Shape<1, 1, 1, 16, 64>, pto::Stride<1024, 1024, 1024, 64, 1>, pto::Layout::ND> v27 = GlobalTensor<float, pto::Shape<1, 1, 1, 16, 64>, pto::Stride<1024, 1024, 1024, 64, 1>, pto::Layout::ND>(v2 + (v5 + (unsigned) v24 * (unsigned) v10 + v5 * (unsigned) v11), v25, v26);
  set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
  set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
  set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
  set_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
  set_flag(PIPE_V, PIPE_S, EVENT_ID0);
  set_flag(PIPE_V, PIPE_S, EVENT_ID1);
  TLOAD(v21, v27);
  set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
  wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
  for (size_t v28 = (size_t) v17; v28 < ((size_t) v8); v28 += (size_t) v11) {
    Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v29 = Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v9, v10);
    TASSIGN(v29, v15);
    Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v30 = Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v9, v10);
    __ubuf__ float* v31 = v29.data();
    uint64_t v32 = reinterpret_cast<uint64_t>(v31);
    TASSIGN(v30, v32);
    pto::Shape<1, 1, 1, 16, 64> v33 = pto::Shape<1, 1, 1, 16, 64>();
    pto::Stride<8192, 8192, 8192, 512, 1> v34 = pto::Stride<8192, 8192, 8192, 512, 1>();
    GlobalTensor<float, pto::Shape<1, 1, 1, 16, 64>, pto::Stride<8192, 8192, 8192, 512, 1>, pto::Layout::ND> v35 = GlobalTensor<float, pto::Shape<1, 1, 1, 16, 64>, pto::Stride<8192, 8192, 8192, 512, 1>, pto::Layout::ND>(v1 + (v5 + (unsigned) v24 * (unsigned) v12 + (unsigned) ((int32_t) (uint32_t) ((int32_t) (uint32_t) ((int32_t) (uint32_t) v4 * (uint32_t) v8) + (uint32_t) ((int32_t) v28)) * (uint32_t) v10) * (unsigned) v11), v33, v34);
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    TLOAD(v30, v35);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
    Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v36 = Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v9, v10);
    TASSIGN(v36, v14);
    Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v37 = Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v9, v10);
    __ubuf__ float* v38 = v36.data();
    uint64_t v39 = reinterpret_cast<uint64_t>(v38);
    TASSIGN(v37, v39);
    TADDS(v37, v21, v7);
    Tile<TileType::Vec, float, 16, 64, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v40 = Tile<TileType::Vec, float, 16, 64, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null>(v9, v10);
    TASSIGN(v40, v13);
    Tile<TileType::Vec, float, 16, 64, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v41 = Tile<TileType::Vec, float, 16, 64, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null>(v9, v10);
    __ubuf__ float* v42 = v40.data();
    uint64_t v43 = reinterpret_cast<uint64_t>(v42);
    TASSIGN(v41, v43);
    wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
    TMOV(v41, v37);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    TPUSH<TPipe<0, Direction::DIR_BOTH, 8192, 4, 2, false>, Tile<TileType::Vec, float, 16, 64, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null>, TileSplitAxis::TILE_UP_DOWN>(v19, v41);
    set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
    Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v44 = Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v9, v10);
    wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
    TPOP<TPipe<0, Direction::DIR_BOTH, 8192, 4, 2, false>, Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>, TileSplitAxis::TILE_UP_DOWN>(v19, v44);
    set_flag(PIPE_S, PIPE_V, EVENT_ID0);
    Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v45 = Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v9, v10);
    TASSIGN(v45, v15);
    Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v46 = Tile<TileType::Vec, float, 16, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v9, v10);
    __ubuf__ float* v47 = v45.data();
    uint64_t v48 = reinterpret_cast<uint64_t>(v47);
    TASSIGN(v46, v48);
    wait_flag(PIPE_S, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
    TADD(v46, v30, v44);
    set_flag(PIPE_V, PIPE_S, EVENT_ID0);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
    TFREE<TPipe<0, Direction::DIR_BOTH, 8192, 4, 2, false>, TileSplitAxis::TILE_UP_DOWN>(v19);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
    pipe_barrier(PIPE_MTE3);
    TSTORE(v35, v46);
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
  }
  wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
  wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
  wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
  wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
  wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
  wait_flag(PIPE_V, PIPE_S, EVENT_ID1);

  return;
}

static __aicore__ void main_incore_0_aic_C2V_NOSPLIT(__gm__ float* v1, __gm__ float* v2, __gm__ float* v3, int32_t v4) {
  unsigned v5 = 0;
  __gm__ void * v6 = nullptr;
  const int32_t v7 = 8;
  const int32_t v8 = 64;
  const int32_t v9 = 1;
  const int32_t v10 = 512;
  const int32_t v11 = 32;
  const int64_t v12 = 16384;
  const int64_t v13 = 0;
  const int32_t v14 = 0;
  using T = float;

  auto v15 = TPipe<0, Direction::DIR_C2V, 8192, 8, 2, true>(v6, v14, v14);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
  set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
  set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
  for (size_t v16 = (size_t) v14; v16 < ((size_t) v7); v16 += (size_t) v9) {
    Tile<TileType::Mat, float, 64, 64, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v17 = Tile<TileType::Mat, float, 64, 64, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null>(v8, v8);
    TASSIGN(v17, v13);
    Tile<TileType::Mat, float, 64, 64, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v18 = Tile<TileType::Mat, float, 64, 64, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null>(v8, v8);
    __cbuf__ float* v19 = v17.data();
    uint64_t v20 = reinterpret_cast<uint64_t>(v19);
    TASSIGN(v18, v20);
    pto::Shape<1, 1, 1, 64, 64> v21 = pto::Shape<1, 1, 1, 64, 64>();
    pto::Stride<32768, 32768, 32768, 512, 1> v22 = pto::Stride<32768, 32768, 32768, 512, 1>();
    GlobalTensor<float, pto::Shape<1, 1, 1, 64, 64>, pto::Stride<32768, 32768, 32768, 512, 1>, pto::Layout::ND> v23 = GlobalTensor<float, pto::Shape<1, 1, 1, 64, 64>, pto::Stride<32768, 32768, 32768, 512, 1>, pto::Layout::ND>(v2 + (v5 + v5 * (unsigned) v10 + (unsigned) ((int32_t) (uint32_t) ((int32_t) (uint32_t) ((int32_t) (uint32_t) v4 * (uint32_t) v7) + (uint32_t) ((int32_t) v16)) * (uint32_t) v8) * (unsigned) v9), v21, v22);
    wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
    TLOAD(v18, v23);
    Tile<TileType::Mat, float, 32, 64, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v24 = Tile<TileType::Mat, float, 32, 64, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null>(v11, v8);
    TASSIGN(v24, v12);
    Tile<TileType::Mat, float, 32, 64, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v25 = Tile<TileType::Mat, float, 32, 64, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null>(v11, v8);
    __cbuf__ float* v26 = v24.data();
    uint64_t v27 = reinterpret_cast<uint64_t>(v26);
    TASSIGN(v25, v27);
    pto::Shape<1, 1, 1, 32, 64> v28 = pto::Shape<1, 1, 1, 32, 64>();
    pto::Stride<2048, 2048, 2048, 64, 1> v29 = pto::Stride<2048, 2048, 2048, 64, 1>();
    GlobalTensor<float, pto::Shape<1, 1, 1, 32, 64>, pto::Stride<2048, 2048, 2048, 64, 1>, pto::Layout::ND> v30 = GlobalTensor<float, pto::Shape<1, 1, 1, 32, 64>, pto::Stride<2048, 2048, 2048, 64, 1>, pto::Layout::ND>(v3 + (v5 + v5 * (unsigned) v8 + v5 * (unsigned) v9), v28, v29);
    TLOAD(v25, v30);
    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    Tile<TileType::Left, float, 32, 64, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v31 = Tile<TileType::Left, float, 32, 64, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null>(v11, v8);
    TASSIGN(v31, v13);
    Tile<TileType::Left, float, 32, 64, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v32 = Tile<TileType::Left, float, 32, 64, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null>(v11, v8);
    __ca__ float* v33 = v31.data();
    uint64_t v34 = reinterpret_cast<uint64_t>(v33);
    TASSIGN(v32, v34);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
    TMOV(v32, v25);
    Tile<TileType::Right, float, 64, 64, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null> v35 = Tile<TileType::Right, float, 64, 64, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null>(v8, v8);
    TASSIGN(v35, v13);
    Tile<TileType::Right, float, 64, 64, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null> v36 = Tile<TileType::Right, float, 64, 64, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null>(v8, v8);
    __cb__ float* v37 = v35.data();
    uint64_t v38 = reinterpret_cast<uint64_t>(v37);
    TASSIGN(v36, v38);
    TMOV(v36, v18);
    set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
    set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
    Tile<TileType::Acc, float, 32, 64, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null, CompactMode::Null> v39 = Tile<TileType::Acc, float, 32, 64, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null, CompactMode::Null>(v11, v8);
    TASSIGN(v39, v13);
    Tile<TileType::Acc, float, 32, 64, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null, CompactMode::Null> v40 = Tile<TileType::Acc, float, 32, 64, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null, CompactMode::Null>(v11, v8);
    __cc__ float* v41 = v39.data();
    uint64_t v42 = reinterpret_cast<uint64_t>(v41);
    TASSIGN(v40, v42);
    wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
    wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
    TMATMUL(v40, v32, v36);
    set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
    set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    TPUSH<TPipe<0, Direction::DIR_C2V, 8192, 8, 2, true>, Tile<TileType::Acc, float, 32, 64, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null, CompactMode::Null>, TileSplitAxis::TILE_NO_SPLIT>(v15, v40);
    set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
  }
  wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
  wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
  wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
  wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);

  return;
}

static __aicore__ void main_incore_0_aiv_C2V_NOSPLIT(__gm__ float* v1, __gm__ float* v2, __gm__ float* v3, int32_t v4) {
  unsigned v5 = 0;
  __gm__ void * v6 = nullptr;
  const int32_t v7 = 8;
  const int32_t v8 = 64;
  const int32_t v9 = 1;
  const int32_t v10 = 512;
  const int32_t v11 = 32;
  const int64_t v12 = 65536;
  const int32_t v13 = 0;
  using T = float;

  set_mask_norm();
  set_vector_mask(-1, -1);
  if (get_subblockid() == 0) {
  auto v14 = TPipe<0, Direction::DIR_C2V, 8192, 8, 2, true>(v6, v13, v13);
  set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
  set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
  set_flag(PIPE_V, PIPE_S, EVENT_ID0);
  set_flag(PIPE_V, PIPE_S, EVENT_ID1);
  for (size_t v15 = (size_t) v13; v15 < ((size_t) v7); v15 += (size_t) v9) {
    Tile<TileType::Vec, float, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v16 = Tile<TileType::Vec, float, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v11, v8);
    TASSIGN(v16, v12);
    Tile<TileType::Vec, float, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v17 = Tile<TileType::Vec, float, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v11, v8);
    __ubuf__ float* v18 = v16.data();
    uint64_t v19 = reinterpret_cast<uint64_t>(v18);
    TASSIGN(v17, v19);
    pto::Shape<1, 1, 1, 32, 64> v20 = pto::Shape<1, 1, 1, 32, 64>();
    pto::Stride<16384, 16384, 16384, 512, 1> v21 = pto::Stride<16384, 16384, 16384, 512, 1>();
    GlobalTensor<float, pto::Shape<1, 1, 1, 32, 64>, pto::Stride<16384, 16384, 16384, 512, 1>, pto::Layout::ND> v22 = GlobalTensor<float, pto::Shape<1, 1, 1, 32, 64>, pto::Stride<16384, 16384, 16384, 512, 1>, pto::Layout::ND>(v1 + (v5 + v5 * (unsigned) v10 + (unsigned) ((int32_t) (uint32_t) ((int32_t) (uint32_t) ((int32_t) (uint32_t) v4 * (uint32_t) v7) + (uint32_t) ((int32_t) v15)) * (uint32_t) v8) * (unsigned) v9), v20, v21);
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    TLOAD(v17, v22);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    Tile<TileType::Vec, float, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v23 = Tile<TileType::Vec, float, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v11, v8);
    wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
    TPOP<TPipe<0, Direction::DIR_C2V, 8192, 8, 2, true>, Tile<TileType::Vec, float, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>, TileSplitAxis::TILE_NO_SPLIT>(v14, v23);
    set_flag(PIPE_S, PIPE_V, EVENT_ID0);
    Tile<TileType::Vec, float, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v24 = Tile<TileType::Vec, float, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v11, v8);
    TASSIGN(v24, v12);
    Tile<TileType::Vec, float, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v25 = Tile<TileType::Vec, float, 32, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v11, v8);
    __ubuf__ float* v26 = v24.data();
    uint64_t v27 = reinterpret_cast<uint64_t>(v26);
    TASSIGN(v25, v27);
    wait_flag(PIPE_S, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    TADD(v25, v17, v23);
    set_flag(PIPE_V, PIPE_S, EVENT_ID0);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    TFREE<TPipe<0, Direction::DIR_C2V, 8192, 8, 2, true>, TileSplitAxis::TILE_NO_SPLIT>(v14);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    pipe_barrier(PIPE_MTE3);
    TSTORE(v22, v25);
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
  }
  wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
  wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
  wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
  wait_flag(PIPE_V, PIPE_S, EVENT_ID1);
  }

  return;
}


void* g_shared_storage_ptr = nullptr;
// A simple function matching GetPipeSharedStateInjectedHookFn signature
extern "C" void* GlobalPipeHook(uint64_t key, size_t size) {
    // We'll use a global pointer to store the allocated memory
    return g_shared_storage_ptr;
}



inline void LaunchTPut_BI_LEFTRIGHT(T *out, T *A, T *B, T *C) {
    using MainPipe_1 = TPipe<0, Direction::DIR_BOTH, 8192, 4, 2, true>;
    size_t v5 = 0;
    std::cout<<"Start"<<std::endl;
    // 1. Allocate and zero the shared synchronization state
    size_t required_size = sizeof(MainPipe_1::SharedState);
    void* raw_mem = malloc(required_size);
    // g_shared_storage_ptr = calloc(1, required_size); 
    g_shared_storage_ptr = new (raw_mem) MainPipe_1::SharedState();
    
    std::barrier sync_point(3); 

    // 2. Register the hook for shared storage ONLY
    // We pass nullptr for subblock_id hook so it falls back to the thread_local context
    pto::cpu_sim::register_hooks(nullptr, (void*)GlobalPipeHook);

    auto aiv_func = [&](int32_t id) {
        // This sets the thread_local ExecutionContext
        pto::cpu_sim::ScopedExecutionContext ctx(0, id, 2);

        sync_point.arrive_and_wait(); 
        main_incore_0_aiv_BI_LEFTRIGHT(C, A, B, v5);
    };

    auto aic_func = [&]() {
        // Cube Core: Block 0, Subblock 0, Dim 1
        pto::cpu_sim::ScopedExecutionContext ctx(0, 0, 1);

        sync_point.arrive_and_wait(); 
        main_incore_0_aic_BI_LEFTRIGHT(C, A, B, v5);
    };

    std::thread v0(aiv_func, 0);
    std::thread v1(aiv_func, 1);
    std::thread c0(aic_func);

    v0.join();
    v1.join();
    c0.join();

    // 3. Cleanup
    // pto::cpu_sim::register_hooks(nullptr, nullptr);
    // aclrtFree(pipe_mem);
    static_cast<MainPipe_1::SharedState*>(g_shared_storage_ptr)->~SharedState();
    // free(g_shared_storage_ptr);
}

inline void LaunchTPut_BI_NOSPLIT(T *out, T *A, T *B, T *C) {
    using MainPipe_2 = TPipe<0, Direction::DIR_BOTH, 8192, 4, 2, true>;
    size_t v5 = 0;
    std::cout<<"Start"<<std::endl;
    // 1. Allocate and zero the shared synchronization state
    size_t required_size = sizeof(MainPipe_2::SharedState);
    void* raw_mem = malloc(required_size);
    // g_shared_storage_ptr = calloc(1, required_size); 
    g_shared_storage_ptr = new (raw_mem) MainPipe_2::SharedState();
    
    std::barrier sync_point(2); 

    // 2. Register the hook for shared storage ONLY
    // We pass nullptr for subblock_id hook so it falls back to the thread_local context
    pto::cpu_sim::register_hooks(nullptr, (void*)GlobalPipeHook);

    auto aiv_func = [&](int32_t id) {
        // This sets the thread_local ExecutionContext
        pto::cpu_sim::ScopedExecutionContext ctx(0, id, 1);

        sync_point.arrive_and_wait(); 
        main_incore_0_aiv_BI_NOSPLIT(C, A, B, v5);
    };

    auto aic_func = [&]() {
        // Cube Core: Block 0, Subblock 0, Dim 1
        pto::cpu_sim::ScopedExecutionContext ctx(0, 0, 1);

        sync_point.arrive_and_wait(); 
        main_incore_0_aic_BI_NOSPLIT(C, A, B, v5);
    };

    std::thread v0(aiv_func, 0);
    // std::thread v1(aiv_func, 1);
    std::thread c0(aic_func);

    v0.join();
    // v1.join();
    c0.join();

    // 3. Cleanup
    // pto::cpu_sim::register_hooks(nullptr, nullptr);
    // aclrtFree(pipe_mem);
    static_cast<MainPipe_2::SharedState*>(g_shared_storage_ptr)->~SharedState();
    // free(g_shared_storage_ptr);
}

inline void LaunchTPut_BI_TOPDOWN(T *out, T *A, T *B, T *C) {
    using MainPipe_3 = TPipe<0, Direction::DIR_BOTH, 8192, 4, 2, false>;
    size_t v5 = 0;
    std::cout<<"Start"<<std::endl;
    // 1. Allocate and zero the shared synchronization state
    size_t required_size = sizeof(MainPipe_3::SharedState);
    void* raw_mem = malloc(required_size);
    // g_shared_storage_ptr = calloc(1, required_size); 
    g_shared_storage_ptr = new (raw_mem) MainPipe_3::SharedState();
    
    std::barrier sync_point(3); 

    // 2. Register the hook for shared storage ONLY
    // We pass nullptr for subblock_id hook so it falls back to the thread_local context
    pto::cpu_sim::register_hooks(nullptr, (void*)GlobalPipeHook);

    auto aiv_func = [&](int32_t id) {
        // This sets the thread_local ExecutionContext
        pto::cpu_sim::ScopedExecutionContext ctx(0, id, 2);

        sync_point.arrive_and_wait(); 
        main_incore_0_aiv_BI_TOPDOWN(C, A, B, v5);
    };

    auto aic_func = [&]() {
        // Cube Core: Block 0, Subblock 0, Dim 1
        pto::cpu_sim::ScopedExecutionContext ctx(0, 0, 1);

        sync_point.arrive_and_wait(); 
        main_incore_0_aic_BI_TOPDOWN(C, A, B, v5);
    };

    std::thread v0(aiv_func, 0);
    std::thread v1(aiv_func, 1);
    std::thread c0(aic_func);

    v0.join();
    v1.join();
    c0.join();

    // 3. Cleanup
    // pto::cpu_sim::register_hooks(nullptr, nullptr);
    // aclrtFree(pipe_mem);
    // static_cast<MainPipe::SharedState*>(g_shared_storage_ptr)->~SharedState();
    // free(g_shared_storage_ptr);
}

inline void LaunchTPut_C2V_NOSPLIT(T *out, T *A, T *B, T *C) {
    using MainPipe = TPipe<0, Direction::DIR_C2V, 8192, 8, 2, true>;
    size_t v5 = 0;
    std::cout<<"Start"<<std::endl;
    // 1. Allocate and zero the shared synchronization state
    size_t required_size = sizeof(MainPipe::SharedState);
    void* raw_mem = malloc(required_size);
    // g_shared_storage_ptr = calloc(1, required_size); 
    g_shared_storage_ptr = new (raw_mem) MainPipe::SharedState();
    
    std::barrier sync_point(2); 

    // 2. Register the hook for shared storage ONLY
    // We pass nullptr for subblock_id hook so it falls back to the thread_local context
    pto::cpu_sim::register_hooks(nullptr, (void*)GlobalPipeHook);

    auto aiv_func = [&](int32_t id) {
        // This sets the thread_local ExecutionContext
        pto::cpu_sim::ScopedExecutionContext ctx(0, id, 1);

        sync_point.arrive_and_wait(); 
        main_incore_0_aiv_C2V_NOSPLIT(C, B, A, v5);
    };

    auto aic_func = [&]() {
        // Cube Core: Block 0, Subblock 0, Dim 1
        pto::cpu_sim::ScopedExecutionContext ctx(0, 0, 1);

        sync_point.arrive_and_wait(); 
        main_incore_0_aic_C2V_NOSPLIT(C, B, A, v5);
    };

    std::thread v0(aiv_func, 0);
    // std::thread v1(aiv_func, 1);
    std::thread c0(aic_func);

    c0.join();
    v0.join();
    // v1.join();
    

    // 3. Cleanup
    // pto::cpu_sim::register_hooks(nullptr, nullptr);
    // aclrtFree(pipe_mem);
    // static_cast<MainPipe::SharedState*>(g_shared_storage_ptr)->~SharedState();
    // free(g_shared_storage_ptr);
}


template <int key>
void test_tpush()
{   
     size_t ARow = 32, ACol = 64, BRow = 64, BCol = 512, CRow = 32, CCol = 512;
     size_t ASize = ARow * ACol * sizeof(T);
     size_t BSize = BRow * BCol * sizeof(T);
     size_t CSize = CRow * CCol * sizeof(T);

    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    T *dstHost, *srcAHost, *srcBHost, *srcCHost;
    T *dstDevice, *srcADevice, *srcBDevice, *srcCDevice;

    aclrtMallocHost((void **)(&dstHost), CSize);
    aclrtMallocHost((void **)(&srcAHost), ASize);
    aclrtMallocHost((void **)(&srcBHost), BSize);
    aclrtMallocHost((void **)(&srcCHost), CSize);

    aclrtMalloc((void **)&dstDevice, CSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&srcADevice, ASize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&srcBDevice, BSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&srcCDevice, CSize, ACL_MEM_MALLOC_HUGE_FIRST);

    CHECK_RESULT_GTEST(ReadFile(GetGoldenDir() + "/a.bin", ASize, srcAHost, ASize));
    CHECK_RESULT_GTEST(ReadFile(GetGoldenDir() + "/b.bin", BSize, srcBHost, BSize));
    CHECK_RESULT_GTEST(ReadFile(GetGoldenDir() + "/c.bin", CSize, srcCHost, CSize));

    aclrtMemcpy(srcADevice, ASize, srcAHost, ASize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(srcBDevice, BSize, srcBHost, BSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(srcCDevice, CSize, srcCHost, CSize, ACL_MEMCPY_HOST_TO_DEVICE);

    if(key == 1) {
        LaunchTPut_BI_LEFTRIGHT(dstDevice, srcADevice, srcBDevice, srcCDevice);
    } else if (key == 2) {
        LaunchTPut_BI_NOSPLIT(dstDevice, srcADevice, srcBDevice, srcCDevice);
    } else if (key == 3) {
        LaunchTPut_BI_TOPDOWN(dstDevice, srcADevice, srcBDevice, srcCDevice);
    } else if (key == 4) {
        LaunchTPut_C2V_NOSPLIT(dstDevice, srcADevice, srcBDevice, srcCDevice);
    }

    aclrtSynchronizeStream(stream);
    aclrtMemcpy(dstHost, CSize, dstDevice, CSize, ACL_MEMCPY_DEVICE_TO_HOST);

    WriteFile(GetGoldenDir() + "/output.bin", srcCDevice, CSize);

    aclrtFree(dstDevice);
    aclrtFree(srcADevice);
    aclrtFree(srcBDevice);
    aclrtFree(srcCDevice);

    aclrtFreeHost(dstHost);
    aclrtFreeHost(srcAHost);
    aclrtFreeHost(srcBHost);
    aclrtFreeHost(srcCHost);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    size_t elem_count = CSize / sizeof(T);

    std::vector<T> golden(elem_count);
    std::vector<T> devFinal(elem_count);
    CHECK_RESULT_GTEST(ReadFile(GetGoldenDir() + "/golden.bin", CSize, golden.data(), CSize));
    CHECK_RESULT_GTEST(ReadFile(GetGoldenDir() + "/output.bin", CSize, devFinal.data(), CSize));

    bool ret = ResultCmp<T>(golden, devFinal, 0.001f);

    EXPECT_TRUE(ret);
}

// TEST_F(TPUSH_A5Test, case_1)
// {
//     test_tpush<1>();
// }

// TEST_F(TPUSH_A5Test, case_2)
// {
//     test_tpush<2>();
// }

// TEST_F(TPUSH_A5Test, case_1)
// {
//     test_tpush<3>();
// }

TEST_F(TPUSH_A5Test, case_4)
{
    test_tpush<4>();
}

}
