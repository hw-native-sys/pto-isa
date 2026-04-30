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

__global__ AICORE void tilekernels_mhc_expand_to_mhc_bwd_m4(__gm__ bfloat16_t* v1, __gm__ bfloat16_t* v2, int32_t v3, int32_t v4) {
  RoundMode v5 = RoundMode::CAST_RINT;
  unsigned v6 = 0;
  const int32_t v7 = 4;
  const int32_t v8 = 1024;
  const int32_t v9 = 1;
  const int32_t v10 = 0;
  const int32_t v11 = 2;
  const int32_t v12 = 3;
  const int64_t v13 = 0;
  const int64_t v14 = 16384;
  const int64_t v15 = 49152;
  const int64_t v16 = 81920;
  using T = float;
  size_t v17 = (size_t) v9;
  int32_t v18 = (int32_t) ((uint32_t) v3 * (uint32_t) v7);

  #if defined(__DAV_VEC__)
  set_mask_norm();
  set_vector_mask(-1, -1);
  int64_t v19 = get_block_idx();
  int64_t v20 = get_block_num();
  int32_t v21 = (int32_t) ((int64_t) v20);
  int32_t v22 = v3 / v21;
  int32_t v23 = v3 % v21 != v10 && v3 < v10 == v21 < v10 ? v22 + v9 : v22;
  int32_t v24 = (int32_t) ((uint32_t) ((int32_t) (int64_t) v19) * (uint32_t) v23);
  int32_t v25 = (int32_t) ((uint32_t) v24 + (uint32_t) v23);
  int32_t v26 = v4 / v8;
  set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
  set_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
  set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
  set_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
  for (size_t v27 = (size_t) v24; v27 < ((size_t) ((uint32_t) v25 < (uint32_t) v3 ? v25 : v3)); v27 += v17) {
    int32_t v28 = (int32_t) v27;
    for (size_t v29 = (size_t) v10; v29 < ((size_t) (v4 % v8 != v10 && v4 < v10 == v8 < v10 ? v26 + v9 : v26)); v29 += v17) {
      int32_t v30 = (int32_t) ((uint32_t) ((int32_t) v29) * (uint32_t) v8);
      int32_t v31 = (int32_t) ((uint32_t) v4 - (uint32_t) v30);
      int32_t v32 = (uint32_t) v31 < (uint32_t) v8 ? v31 : v8;
      Tile<TileType::Vec, bfloat16_t, 8, 1024, BLayout::RowMajor, 8, 1024, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v33;
      TASSIGN(v33, v13);
      Tile<TileType::Vec, bfloat16_t, 8, 1024, BLayout::RowMajor, 1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v34 = Tile<TileType::Vec, bfloat16_t, 8, 1024, BLayout::RowMajor, 1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v32);
      __ubuf__ bfloat16_t* v35 = v33.data();
      uint64_t v36 = reinterpret_cast<uint64_t>(v35);
      TASSIGN(v34, v36);
      Tile<TileType::Vec, float, 8, 1024, BLayout::RowMajor, 8, 1024, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v37;
      TASSIGN(v37, v14);
      Tile<TileType::Vec, float, 8, 1024, BLayout::RowMajor, 1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v38 = Tile<TileType::Vec, float, 8, 1024, BLayout::RowMajor, 1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v32);
      __ubuf__ float* v39 = v37.data();
      uint64_t v40 = reinterpret_cast<uint64_t>(v39);
      TASSIGN(v38, v40);
      Tile<TileType::Vec, float, 8, 1024, BLayout::RowMajor, 8, 1024, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v41;
      TASSIGN(v41, v15);
      Tile<TileType::Vec, float, 8, 1024, BLayout::RowMajor, 1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v42 = Tile<TileType::Vec, float, 8, 1024, BLayout::RowMajor, 1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v32);
      __ubuf__ float* v43 = v41.data();
      uint64_t v44 = reinterpret_cast<uint64_t>(v43);
      TASSIGN(v42, v44);
      Tile<TileType::Vec, bfloat16_t, 8, 1024, BLayout::RowMajor, 8, 1024, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v45;
      TASSIGN(v45, v16);
      Tile<TileType::Vec, bfloat16_t, 8, 1024, BLayout::RowMajor, 1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v46 = Tile<TileType::Vec, bfloat16_t, 8, 1024, BLayout::RowMajor, 1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v32);
      __ubuf__ bfloat16_t* v47 = v45.data();
      uint64_t v48 = reinterpret_cast<uint64_t>(v47);
      TASSIGN(v46, v48);
      int32_t v49 = (int32_t) ((uint32_t) v28 * (uint32_t) v7);
      unsigned v50 = (unsigned) v32;
      unsigned v51 = (unsigned) v4;
      pto::Shape<1, 1, 1, 1, -1> v52 = pto::Shape<1, 1, 1, 1, -1>(v32);
      pto::Stride<-1, -1, -1, -1, 1> v53 = pto::Stride<-1, -1, -1, -1, 1>(v51, v51, v51, v51);
      GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, 1>, pto::Layout::ND> v54 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, 1>, pto::Layout::ND>(v1 + (v6 + (unsigned) v49 * (unsigned) v4 + (unsigned) v30 * (unsigned) v9), v52, v53);
      wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
      TLOAD(v34, v54);
      set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
      wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
      TCVT(v38, v34, v5);
      set_flag(PIPE_V, PIPE_MTE2, EVENT_ID2);
      pipe_barrier(PIPE_ALL);
      TMOV(v42, v38);
      unsigned v55 = (unsigned) v32;
      unsigned v56 = (unsigned) v4;
      pto::Shape<1, 1, 1, 1, -1> v57 = pto::Shape<1, 1, 1, 1, -1>(v32);
      pto::Stride<-1, -1, -1, -1, 1> v58 = pto::Stride<-1, -1, -1, -1, 1>(v56, v56, v56, v56);
      GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, 1>, pto::Layout::ND> v59 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, 1>, pto::Layout::ND>(v1 + (v6 + (unsigned) ((int32_t) (uint32_t) v49 + (uint32_t) v9) * (unsigned) v4 + (unsigned) v30 * (unsigned) v9), v57, v58);
      wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID2);
      TLOAD(v34, v59);
      set_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
      wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
      pipe_barrier(PIPE_ALL);
      TCVT(v38, v34, v5);
      set_flag(PIPE_V, PIPE_MTE2, EVENT_ID3);
      pipe_barrier(PIPE_ALL);
      TADD(v42, v42, v38);
      unsigned v60 = (unsigned) v32;
      unsigned v61 = (unsigned) v4;
      pto::Shape<1, 1, 1, 1, -1> v62 = pto::Shape<1, 1, 1, 1, -1>(v32);
      pto::Stride<-1, -1, -1, -1, 1> v63 = pto::Stride<-1, -1, -1, -1, 1>(v61, v61, v61, v61);
      GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, 1>, pto::Layout::ND> v64 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, 1>, pto::Layout::ND>(v1 + (v6 + (unsigned) ((int32_t) (uint32_t) v49 + (uint32_t) v11) * (unsigned) v4 + (unsigned) v30 * (unsigned) v9), v62, v63);
      wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID3);
      TLOAD(v34, v64);
      set_flag(PIPE_MTE2, PIPE_V, EVENT_ID2);
      wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID2);
      pipe_barrier(PIPE_ALL);
      TCVT(v38, v34, v5);
      set_flag(PIPE_V, PIPE_MTE2, EVENT_ID4);
      pipe_barrier(PIPE_ALL);
      TADD(v42, v42, v38);
      unsigned v65 = (unsigned) v32;
      unsigned v66 = (unsigned) v4;
      pto::Shape<1, 1, 1, 1, -1> v67 = pto::Shape<1, 1, 1, 1, -1>(v32);
      pto::Stride<-1, -1, -1, -1, 1> v68 = pto::Stride<-1, -1, -1, -1, 1>(v66, v66, v66, v66);
      GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, 1>, pto::Layout::ND> v69 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, 1>, pto::Layout::ND>(v1 + (v6 + (unsigned) ((int32_t) (uint32_t) v49 + (uint32_t) v12) * (unsigned) v4 + (unsigned) v30 * (unsigned) v9), v67, v68);
      wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID4);
      TLOAD(v34, v69);
      set_flag(PIPE_MTE2, PIPE_V, EVENT_ID3);
      wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID3);
      pipe_barrier(PIPE_ALL);
      TCVT(v38, v34, v5);
      set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
      pipe_barrier(PIPE_ALL);
      TADD(v42, v42, v38);
      pipe_barrier(PIPE_ALL);
      wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
      TCVT(v46, v42, v5);
      set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
      unsigned v70 = (unsigned) v32;
      unsigned v71 = (unsigned) v4;
      pto::Shape<1, 1, 1, 1, -1> v72 = pto::Shape<1, 1, 1, 1, -1>(v32);
      pto::Stride<-1, -1, -1, -1, 1> v73 = pto::Stride<-1, -1, -1, -1, 1>(v71, v71, v71, v71);
      GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, 1>, pto::Layout::ND> v74 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, 1>, pto::Layout::ND>(v2 + (v6 + (unsigned) v28 * (unsigned) v4 + (unsigned) v30 * (unsigned) v9), v72, v73);
      wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
      pipe_barrier(PIPE_MTE3);
      TSTORE(v74, v46);
      set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
    };
  }
  wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
  wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
  wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
  wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
  #endif // __DAV_VEC__

  ptoas_auto_sync_tail(PTOAutoSyncTailMode::kBarrierAll);
  return;
}
