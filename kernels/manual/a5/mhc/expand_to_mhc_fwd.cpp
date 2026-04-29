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

__global__ AICORE void tilekernels_mhc_expand_to_mhc_fwd_m4(__gm__ bfloat16_t* v1, __gm__ bfloat16_t* v2, int32_t v3, int32_t v4) {
  unsigned v5 = 0;
  const int32_t v6 = 4;
  const int32_t v7 = 1024;
  const int32_t v8 = 1;
  const int32_t v9 = 0;
  const int32_t v10 = 2;
  const int32_t v11 = 3;
  const int64_t v12 = 0;
  using T = float;
  size_t v13 = (size_t) v8;
  int32_t v14 = (int32_t) ((uint32_t) v3 * (uint32_t) v6);

  #if defined(__DAV_VEC__)
  set_mask_norm();
  set_vector_mask(-1, -1);
  int64_t v15 = get_block_idx();
  int64_t v16 = get_block_num();
  int32_t v17 = (int32_t) ((int64_t) v16);
  int32_t v18 = v3 / v17;
  int32_t v19 = v3 % v17 != v9 && v3 < v9 == v17 < v9 ? v18 + v8 : v18;
  int32_t v20 = (int32_t) ((uint32_t) ((int32_t) (int64_t) v15) * (uint32_t) v19);
  int32_t v21 = (int32_t) ((uint32_t) v20 + (uint32_t) v19);
  int32_t v22 = v4 / v7;
  set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
  set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
  for (size_t v23 = (size_t) v20; v23 < ((size_t) ((uint32_t) v21 < (uint32_t) v3 ? v21 : v3)); v23 += v13) {
    int32_t v24 = (int32_t) v23;
    for (size_t v25 = (size_t) v9; v25 < ((size_t) (v4 % v7 != v9 && v4 < v9 == v7 < v9 ? v22 + v8 : v22)); v25 += v13) {
      int32_t v26 = (int32_t) ((uint32_t) ((int32_t) v25) * (uint32_t) v7);
      int32_t v27 = (int32_t) ((uint32_t) v4 - (uint32_t) v26);
      int32_t v28 = (uint32_t) v27 < (uint32_t) v7 ? v27 : v7;
      Tile<TileType::Vec, bfloat16_t, 8, 1024, BLayout::RowMajor, 8, 1024, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v29;
      TASSIGN(v29, v12);
      Tile<TileType::Vec, bfloat16_t, 8, 1024, BLayout::RowMajor, 1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v30 = Tile<TileType::Vec, bfloat16_t, 8, 1024, BLayout::RowMajor, 1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v28);
      __ubuf__ bfloat16_t* v31 = v29.data();
      uint64_t v32 = reinterpret_cast<uint64_t>(v31);
      TASSIGN(v30, v32);
      unsigned v33 = (unsigned) v28;
      unsigned v34 = (unsigned) v4;
      pto::Shape<1, 1, 1, 1, -1> v35 = pto::Shape<1, 1, 1, 1, -1>(v28);
      pto::Stride<-1, -1, -1, -1, 1> v36 = pto::Stride<-1, -1, -1, -1, 1>(v34, v34, v34, v34);
      GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, 1>, pto::Layout::ND> v37 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, 1>, pto::Layout::ND>(v1 + (v5 + (unsigned) v24 * (unsigned) v4 + (unsigned) v26 * (unsigned) v8), v35, v36);
      wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
      TLOAD(v30, v37);
      set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
      int32_t v38 = (int32_t) ((uint32_t) v24 * (uint32_t) v6);
      unsigned v39 = (unsigned) v28;
      unsigned v40 = (unsigned) v4;
      pto::Shape<1, 1, 1, 1, -1> v41 = pto::Shape<1, 1, 1, 1, -1>(v28);
      pto::Stride<-1, -1, -1, -1, 1> v42 = pto::Stride<-1, -1, -1, -1, 1>(v40, v40, v40, v40);
      GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, 1>, pto::Layout::ND> v43 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, 1>, pto::Layout::ND>(v2 + (v5 + (unsigned) v38 * (unsigned) v4 + (unsigned) v26 * (unsigned) v8), v41, v42);
      wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
      pipe_barrier(PIPE_MTE3);
      TSTORE(v43, v30);
      unsigned v44 = (unsigned) v28;
      unsigned v45 = (unsigned) v4;
      pto::Shape<1, 1, 1, 1, -1> v46 = pto::Shape<1, 1, 1, 1, -1>(v28);
      pto::Stride<-1, -1, -1, -1, 1> v47 = pto::Stride<-1, -1, -1, -1, 1>(v45, v45, v45, v45);
      GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, 1>, pto::Layout::ND> v48 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, 1>, pto::Layout::ND>(v2 + (v5 + (unsigned) ((int32_t) (uint32_t) v38 + (uint32_t) v8) * (unsigned) v4 + (unsigned) v26 * (unsigned) v8), v46, v47);
      pipe_barrier(PIPE_MTE3);
      TSTORE(v48, v30);
      unsigned v49 = (unsigned) v28;
      unsigned v50 = (unsigned) v4;
      pto::Shape<1, 1, 1, 1, -1> v51 = pto::Shape<1, 1, 1, 1, -1>(v28);
      pto::Stride<-1, -1, -1, -1, 1> v52 = pto::Stride<-1, -1, -1, -1, 1>(v50, v50, v50, v50);
      GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, 1>, pto::Layout::ND> v53 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, 1>, pto::Layout::ND>(v2 + (v5 + (unsigned) ((int32_t) (uint32_t) v38 + (uint32_t) v10) * (unsigned) v4 + (unsigned) v26 * (unsigned) v8), v51, v52);
      pipe_barrier(PIPE_MTE3);
      TSTORE(v53, v30);
      unsigned v54 = (unsigned) v28;
      unsigned v55 = (unsigned) v4;
      pto::Shape<1, 1, 1, 1, -1> v56 = pto::Shape<1, 1, 1, 1, -1>(v28);
      pto::Stride<-1, -1, -1, -1, 1> v57 = pto::Stride<-1, -1, -1, -1, 1>(v55, v55, v55, v55);
      GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, 1>, pto::Layout::ND> v58 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, 1>, pto::Layout::ND>(v2 + (v5 + (unsigned) ((int32_t) (uint32_t) v38 + (uint32_t) v11) * (unsigned) v4 + (unsigned) v26 * (unsigned) v8), v56, v57);
      pipe_barrier(PIPE_MTE3);
      TSTORE(v58, v30);
      set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    };
  }
  wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
  wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
  #endif // __DAV_VEC__

  ptoas_auto_sync_tail(PTOAutoSyncTailMode::kBarrierAll);
  return;
}
