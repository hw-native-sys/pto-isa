#include "pto/pto-inst.hpp"
using namespace pto;

enum class PTOAutoSyncTailMode : int {
  kBarrierAll = 0,
  kSetWaitMte3ToSEvent0 = 1,
};

static AICORE inline void ptoas_auto_sync_tail(
    PTOAutoSyncTailMode mode = PTOAutoSyncTailMode::kBarrierAll) {
  switch (mode) {
  case PTOAutoSyncTailMode::kSetWaitMte3ToSEvent0:
    set_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
    wait_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
    break;
  case PTOAutoSyncTailMode::kBarrierAll:
  default:
    pipe_barrier(PIPE_ALL);
    break;
  }
}

__global__ AICORE void tilekernels_mhc_head_compute_mix_fwd_m4(__gm__ float* v1, __gm__ float* v2, __gm__ float* v3, __gm__ float* v4, int32_t v5) {
  unsigned v6 = 0;
  const int32_t v7 = 0;
  const int32_t v8 = 1;
  const int32_t v9 = 4;
  const float v10 = -1.0f;
  const float v11 = 1.0f;
  const float v12 = 9.99999997E-7f;
  const int64_t v13 = 96;
  const int64_t v14 = 0;
  const int64_t v15 = 32;
  const int64_t v16 = 64;
  const int64_t v17 = 128;
  const int64_t v18 = 160;
  const int64_t v19 = 192;
  const int64_t v20 = 224;
  const int64_t v21 = 256;
  const int64_t v22 = 288;
  using T = float;

  #if defined(__DAV_VEC__)
  set_mask_norm();
  set_vector_mask(-1, -1);
  int64_t v23 = get_block_idx();
  int64_t v24 = get_block_num();
  int32_t v25 = (int32_t) ((int64_t) v24);
  int32_t v26 = v5 / v25;
  int32_t v27 = v5 % v25 != v7 && v5 < v7 == v25 < v7 ? v26 + v8 : v26;
  int32_t v28 = (int32_t) ((uint32_t) ((int32_t) (int64_t) v23) * (uint32_t) v27);
  int32_t v29 = (int32_t) ((uint32_t) v28 + (uint32_t) v27);
  Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, 1, 8, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v30;
  TASSIGN(v30, v13);
  Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, 1, 4, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v31;
  __ubuf__ float* v32 = v30.data();
  uint64_t v33 = reinterpret_cast<uint64_t>(v32);
  TASSIGN(v31, v33);
  Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, 1, 8, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v34;
  TASSIGN(v34, v14);
  Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, 1, 4, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v35;
  __ubuf__ float* v36 = v34.data();
  uint64_t v37 = reinterpret_cast<uint64_t>(v36);
  TASSIGN(v35, v37);
  Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, 1, 8, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v38;
  TASSIGN(v38, v15);
  Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, 1, 1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v39;
  __ubuf__ float* v40 = v38.data();
  uint64_t v41 = reinterpret_cast<uint64_t>(v40);
  TASSIGN(v39, v41);
  Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, 1, 8, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v42;
  TASSIGN(v42, v16);
  Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, 1, 4, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v43;
  __ubuf__ float* v44 = v42.data();
  uint64_t v45 = reinterpret_cast<uint64_t>(v44);
  TASSIGN(v43, v45);
  Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, 1, 8, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v46;
  TASSIGN(v46, v17);
  Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, 1, 4, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v47;
  __ubuf__ float* v48 = v46.data();
  uint64_t v49 = reinterpret_cast<uint64_t>(v48);
  TASSIGN(v47, v49);
  Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, 1, 8, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v50;
  TASSIGN(v50, v18);
  Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, 1, 4, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v51;
  __ubuf__ float* v52 = v50.data();
  uint64_t v53 = reinterpret_cast<uint64_t>(v52);
  TASSIGN(v51, v53);
  Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, 1, 8, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v54;
  TASSIGN(v54, v19);
  Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, 1, 4, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v55;
  __ubuf__ float* v56 = v54.data();
  uint64_t v57 = reinterpret_cast<uint64_t>(v56);
  TASSIGN(v55, v57);
  Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, 1, 8, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v58;
  TASSIGN(v58, v20);
  Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, 1, 4, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v59;
  __ubuf__ float* v60 = v58.data();
  uint64_t v61 = reinterpret_cast<uint64_t>(v60);
  TASSIGN(v59, v61);
  Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, 1, 8, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v62;
  TASSIGN(v62, v21);
  Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, 1, 4, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v63;
  __ubuf__ float* v64 = v62.data();
  uint64_t v65 = reinterpret_cast<uint64_t>(v64);
  TASSIGN(v63, v65);
  Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, 1, 8, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v66;
  TASSIGN(v66, v22);
  Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, 1, 4, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v67;
  __ubuf__ float* v68 = v66.data();
  uint64_t v69 = reinterpret_cast<uint64_t>(v68);
  TASSIGN(v67, v69);
  pto::Shape<1, 1, 1, 1, 4> v70 = pto::Shape<1, 1, 1, 1, 4>();
  pto::Stride<4, 4, 4, 4, 1> v71 = pto::Stride<4, 4, 4, 4, 1>();
  GlobalTensor<float, pto::Shape<1, 1, 1, 1, 4>, pto::Stride<4, 4, 4, 4, 1>, pto::Layout::ND> v72 = GlobalTensor<float, pto::Shape<1, 1, 1, 1, 4>, pto::Stride<4, 4, 4, 4, 1>, pto::Layout::ND>(v3 + (v6 + v6 * (unsigned) v9 + v6 * (unsigned) v8), v70, v71);
  set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
  set_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
  set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
  set_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
  TLOAD(v35, v72);
  pto::Shape<1, 1, 1, 1, 1> v73 = pto::Shape<1, 1, 1, 1, 1>();
  pto::Stride<1, 1, 1, 1, 1> v74 = pto::Stride<1, 1, 1, 1, 1>();
  GlobalTensor<float, pto::Shape<1, 1, 1, 1, 1>, pto::Stride<1, 1, 1, 1, 1>, pto::Layout::ND> v75 = GlobalTensor<float, pto::Shape<1, 1, 1, 1, 1>, pto::Stride<1, 1, 1, 1, 1>, pto::Layout::ND>(v2 + (v6 + v6 * (unsigned) v8 + v6 * (unsigned) v8), v73, v74);
  TLOAD(v39, v75);
  set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
  wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
  TROWEXPAND(v43, v39);
  for (size_t v76 = (size_t) v28; v76 < ((size_t) ((uint32_t) v29 < (uint32_t) v5 ? v29 : v5)); v76 += (size_t) v8) {
    int32_t v77 = (int32_t) v76;
    pto::Shape<1, 1, 1, 1, 4> v78 = pto::Shape<1, 1, 1, 1, 4>();
    pto::Stride<4, 4, 4, 4, 1> v79 = pto::Stride<4, 4, 4, 4, 1>();
    GlobalTensor<float, pto::Shape<1, 1, 1, 1, 4>, pto::Stride<4, 4, 4, 4, 1>, pto::Layout::ND> v80 = GlobalTensor<float, pto::Shape<1, 1, 1, 1, 4>, pto::Stride<4, 4, 4, 4, 1>, pto::Layout::ND>(v1 + (v6 + (unsigned) v77 * (unsigned) v9 + v6 * (unsigned) v8), v78, v79);
    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
    TLOAD(v31, v80);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
    pipe_barrier(PIPE_ALL);
    TMUL(v47, v31, v43);
    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
    pipe_barrier(PIPE_ALL);
    TADD(v47, v47, v35);
    pipe_barrier(PIPE_ALL);
    TMULS(v51, v47, v10);
    pipe_barrier(PIPE_ALL);
    TEXP(v55, v51);
    pipe_barrier(PIPE_ALL);
    TADDS(v59, v55, v11);
    pipe_barrier(PIPE_ALL);
    TRECIP(v63, v59);
    pipe_barrier(PIPE_ALL);
    wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
    TADDS(v67, v63, v12);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    pto::Shape<1, 1, 1, 1, 4> v81 = pto::Shape<1, 1, 1, 1, 4>();
    pto::Stride<4, 4, 4, 4, 1> v82 = pto::Stride<4, 4, 4, 4, 1>();
    GlobalTensor<float, pto::Shape<1, 1, 1, 1, 4>, pto::Stride<4, 4, 4, 4, 1>, pto::Layout::ND> v83 = GlobalTensor<float, pto::Shape<1, 1, 1, 1, 4>, pto::Stride<4, 4, 4, 4, 1>, pto::Layout::ND>(v4 + (v6 + (unsigned) v77 * (unsigned) v9 + v6 * (unsigned) v8), v81, v82);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    pipe_barrier(PIPE_MTE3);
    TSTORE(v83, v67);
    set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
  }
  wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
  wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
  wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
  wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
  #endif // __DAV_VEC__

  ptoas_auto_sync_tail(PTOAutoSyncTailMode::kBarrierAll);
  return;
}
