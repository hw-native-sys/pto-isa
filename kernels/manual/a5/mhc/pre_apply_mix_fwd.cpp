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

__global__ AICORE void tilekernels_mhc_pre_apply_mix_fwd_m4(__gm__ bfloat16_t* v1, __gm__ float* v2, __gm__ bfloat16_t* v3, int32_t v4, int32_t v5) {
  unsigned v6 = 3;
  unsigned v7 = 2;
  unsigned v8 = 1;
  RoundMode v9 = RoundMode::CAST_RINT;
  unsigned v10 = 0;
  const int32_t v11 = 4;
  const int32_t v12 = 1024;
  const int32_t v13 = 1;
  const int32_t v14 = 0;
  const int32_t v15 = 2;
  const int32_t v16 = 3;
  const int64_t v17 = 0;
  const int64_t v18 = 16384;
  const int64_t v19 = 49152;
  const int64_t v20 = 49408;
  const int64_t v21 = 82176;
  const int64_t v22 = 114944;
  const int64_t v23 = 147712;
  using T = float;
  size_t v24 = (size_t) v13;
  int32_t v25 = (int32_t) ((uint32_t) v4 * (uint32_t) v11);

  #if defined(__DAV_VEC__)
  set_mask_norm();
  set_vector_mask(-1, -1);
  int64_t v26 = get_block_idx();
  int64_t v27 = get_block_num();
  int32_t v28 = (int32_t) ((int64_t) v27);
  int32_t v29 = v4 / v28;
  int32_t v30 = v4 % v28 != v14 && v4 < v14 == v28 < v14 ? v29 + v13 : v29;
  int32_t v31 = (int32_t) ((uint32_t) ((int32_t) (int64_t) v26) * (uint32_t) v30);
  int32_t v32 = (int32_t) ((uint32_t) v31 + (uint32_t) v30);
  int32_t v33 = v5 / v12;
  set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
  set_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
  set_flag(PIPE_V, PIPE_MTE2, EVENT_ID2);
  set_flag(PIPE_V, PIPE_MTE2, EVENT_ID3);
  set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
  set_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
  for (size_t v34 = (size_t) v31; v34 < ((size_t) ((uint32_t) v32 < (uint32_t) v4 ? v32 : v4)); v34 += v24) {
    int32_t v35 = (int32_t) v34;
    for (size_t v36 = (size_t) v14; v36 < ((size_t) (v5 % v12 != v14 && v5 < v14 == v12 < v14 ? v33 + v13 : v33)); v36 += v24) {
      int32_t v37 = (int32_t) ((uint32_t) ((int32_t) v36) * (uint32_t) v12);
      int32_t v38 = (int32_t) ((uint32_t) v5 - (uint32_t) v37);
      int32_t v39 = (uint32_t) v38 < (uint32_t) v12 ? v38 : v12;
      Tile<TileType::Vec, bfloat16_t, 8, 1024, BLayout::RowMajor, 8, 1024, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v40;
      TASSIGN(v40, v17);
      Tile<TileType::Vec, bfloat16_t, 8, 1024, BLayout::RowMajor, 1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v41 = Tile<TileType::Vec, bfloat16_t, 8, 1024, BLayout::RowMajor, 1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v39);
      __ubuf__ bfloat16_t* v42 = v40.data();
      uint64_t v43 = reinterpret_cast<uint64_t>(v42);
      TASSIGN(v41, v43);
      Tile<TileType::Vec, float, 8, 1024, BLayout::RowMajor, 8, 1024, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v44;
      TASSIGN(v44, v18);
      Tile<TileType::Vec, float, 8, 1024, BLayout::RowMajor, 1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v45 = Tile<TileType::Vec, float, 8, 1024, BLayout::RowMajor, 1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v39);
      __ubuf__ float* v46 = v44.data();
      uint64_t v47 = reinterpret_cast<uint64_t>(v46);
      TASSIGN(v45, v47);
      Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, 8, 8, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v48;
      TASSIGN(v48, v19);
      Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, 1, 1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v49;
      __ubuf__ float* v50 = v48.data();
      uint64_t v51 = reinterpret_cast<uint64_t>(v50);
      TASSIGN(v49, v51);
      Tile<TileType::Vec, float, 8, 1024, BLayout::RowMajor, 8, 1024, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v52;
      TASSIGN(v52, v20);
      Tile<TileType::Vec, float, 8, 1024, BLayout::RowMajor, 1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v53 = Tile<TileType::Vec, float, 8, 1024, BLayout::RowMajor, 1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v39);
      __ubuf__ float* v54 = v52.data();
      uint64_t v55 = reinterpret_cast<uint64_t>(v54);
      TASSIGN(v53, v55);
      Tile<TileType::Vec, float, 8, 1024, BLayout::RowMajor, 8, 1024, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v56;
      TASSIGN(v56, v21);
      Tile<TileType::Vec, float, 8, 1024, BLayout::RowMajor, 1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v57 = Tile<TileType::Vec, float, 8, 1024, BLayout::RowMajor, 1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v39);
      __ubuf__ float* v58 = v56.data();
      uint64_t v59 = reinterpret_cast<uint64_t>(v58);
      TASSIGN(v57, v59);
      Tile<TileType::Vec, float, 8, 1024, BLayout::RowMajor, 8, 1024, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v60;
      TASSIGN(v60, v22);
      Tile<TileType::Vec, float, 8, 1024, BLayout::RowMajor, 1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v61 = Tile<TileType::Vec, float, 8, 1024, BLayout::RowMajor, 1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v39);
      __ubuf__ float* v62 = v60.data();
      uint64_t v63 = reinterpret_cast<uint64_t>(v62);
      TASSIGN(v61, v63);
      Tile<TileType::Vec, bfloat16_t, 8, 1024, BLayout::RowMajor, 8, 1024, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v64;
      TASSIGN(v64, v23);
      Tile<TileType::Vec, bfloat16_t, 8, 1024, BLayout::RowMajor, 1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v65 = Tile<TileType::Vec, bfloat16_t, 8, 1024, BLayout::RowMajor, 1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v39);
      __ubuf__ bfloat16_t* v66 = v64.data();
      uint64_t v67 = reinterpret_cast<uint64_t>(v66);
      TASSIGN(v65, v67);
      int32_t v68 = (int32_t) ((uint32_t) v35 * (uint32_t) v11);
      unsigned v69 = (unsigned) v39;
      unsigned v70 = (unsigned) v5;
      pto::Shape<1, 1, 1, 1, -1> v71 = pto::Shape<1, 1, 1, 1, -1>(v39);
      pto::Stride<-1, -1, -1, -1, 1> v72 = pto::Stride<-1, -1, -1, -1, 1>(v70, v70, v70, v70);
      GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, 1>, pto::Layout::ND> v73 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, 1>, pto::Layout::ND>(v1 + (v10 + (unsigned) v68 * (unsigned) v5 + (unsigned) v37 * (unsigned) v13), v71, v72);
      wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
      TLOAD(v41, v73);
      set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
      wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
      TCVT(v45, v41, v9);
      set_flag(PIPE_V, PIPE_MTE2, EVENT_ID4);
      pto::Shape<1, 1, 1, 1, 1> v74 = pto::Shape<1, 1, 1, 1, 1>();
      pto::Stride<4, 4, 4, 4, 1> v75 = pto::Stride<4, 4, 4, 4, 1>();
      GlobalTensor<float, pto::Shape<1, 1, 1, 1, 1>, pto::Stride<4, 4, 4, 4, 1>, pto::Layout::ND> v76 = GlobalTensor<float, pto::Shape<1, 1, 1, 1, 1>, pto::Stride<4, 4, 4, 4, 1>, pto::Layout::ND>(v2 + (v10 + (unsigned) v35 * (unsigned) v11 + v10 * (unsigned) v13), v74, v75);
      wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID2);
      TLOAD(v49, v76);
      set_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
      wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
      TROWEXPAND(v53, v49);
      set_flag(PIPE_V, PIPE_MTE2, EVENT_ID5);
      pipe_barrier(PIPE_ALL);
      TMUL(v57, v45, v53);
      pipe_barrier(PIPE_ALL);
      TMOV(v61, v57);
      unsigned v77 = (unsigned) v39;
      unsigned v78 = (unsigned) v5;
      pto::Shape<1, 1, 1, 1, -1> v79 = pto::Shape<1, 1, 1, 1, -1>(v39);
      pto::Stride<-1, -1, -1, -1, 1> v80 = pto::Stride<-1, -1, -1, -1, 1>(v78, v78, v78, v78);
      GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, 1>, pto::Layout::ND> v81 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, 1>, pto::Layout::ND>(v1 + (v10 + (unsigned) ((int32_t) (uint32_t) v68 + (uint32_t) v13) * (unsigned) v5 + (unsigned) v37 * (unsigned) v13), v79, v80);
      wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID4);
      TLOAD(v41, v81);
      set_flag(PIPE_MTE2, PIPE_V, EVENT_ID2);
      wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID2);
      TCVT(v45, v41, v9);
      set_flag(PIPE_V, PIPE_MTE2, EVENT_ID6);
      pto::Shape<1, 1, 1, 1, 1> v82 = pto::Shape<1, 1, 1, 1, 1>();
      pto::Stride<4, 4, 4, 4, 1> v83 = pto::Stride<4, 4, 4, 4, 1>();
      GlobalTensor<float, pto::Shape<1, 1, 1, 1, 1>, pto::Stride<4, 4, 4, 4, 1>, pto::Layout::ND> v84 = GlobalTensor<float, pto::Shape<1, 1, 1, 1, 1>, pto::Stride<4, 4, 4, 4, 1>, pto::Layout::ND>(v2 + (v10 + (unsigned) v35 * (unsigned) v11 + v8 * (unsigned) v13), v82, v83);
      wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID5);
      TLOAD(v49, v84);
      set_flag(PIPE_MTE2, PIPE_V, EVENT_ID3);
      wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID3);
      TROWEXPAND(v53, v49);
      set_flag(PIPE_V, PIPE_MTE2, EVENT_ID7);
      pipe_barrier(PIPE_ALL);
      TMUL(v57, v45, v53);
      pipe_barrier(PIPE_ALL);
      TADD(v61, v61, v57);
      unsigned v85 = (unsigned) v39;
      unsigned v86 = (unsigned) v5;
      pto::Shape<1, 1, 1, 1, -1> v87 = pto::Shape<1, 1, 1, 1, -1>(v39);
      pto::Stride<-1, -1, -1, -1, 1> v88 = pto::Stride<-1, -1, -1, -1, 1>(v86, v86, v86, v86);
      GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, 1>, pto::Layout::ND> v89 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, 1>, pto::Layout::ND>(v1 + (v10 + (unsigned) ((int32_t) (uint32_t) v68 + (uint32_t) v15) * (unsigned) v5 + (unsigned) v37 * (unsigned) v13), v87, v88);
      wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID6);
      TLOAD(v41, v89);
      set_flag(PIPE_MTE2, PIPE_V, EVENT_ID4);
      wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID4);
      TCVT(v45, v41, v9);
      set_flag(PIPE_V, PIPE_MTE2, EVENT_ID4);
      pto::Shape<1, 1, 1, 1, 1> v90 = pto::Shape<1, 1, 1, 1, 1>();
      pto::Stride<4, 4, 4, 4, 1> v91 = pto::Stride<4, 4, 4, 4, 1>();
      GlobalTensor<float, pto::Shape<1, 1, 1, 1, 1>, pto::Stride<4, 4, 4, 4, 1>, pto::Layout::ND> v92 = GlobalTensor<float, pto::Shape<1, 1, 1, 1, 1>, pto::Stride<4, 4, 4, 4, 1>, pto::Layout::ND>(v2 + (v10 + (unsigned) v35 * (unsigned) v11 + v7 * (unsigned) v13), v90, v91);
      wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID7);
      TLOAD(v49, v92);
      set_flag(PIPE_MTE2, PIPE_V, EVENT_ID5);
      wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID5);
      TROWEXPAND(v53, v49);
      set_flag(PIPE_V, PIPE_MTE2, EVENT_ID5);
      pipe_barrier(PIPE_ALL);
      TMUL(v57, v45, v53);
      pipe_barrier(PIPE_ALL);
      TADD(v61, v61, v57);
      unsigned v93 = (unsigned) v39;
      unsigned v94 = (unsigned) v5;
      pto::Shape<1, 1, 1, 1, -1> v95 = pto::Shape<1, 1, 1, 1, -1>(v39);
      pto::Stride<-1, -1, -1, -1, 1> v96 = pto::Stride<-1, -1, -1, -1, 1>(v94, v94, v94, v94);
      GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, 1>, pto::Layout::ND> v97 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, 1>, pto::Layout::ND>(v1 + (v10 + (unsigned) ((int32_t) (uint32_t) v68 + (uint32_t) v16) * (unsigned) v5 + (unsigned) v37 * (unsigned) v13), v95, v96);
      wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID4);
      TLOAD(v41, v97);
      set_flag(PIPE_MTE2, PIPE_V, EVENT_ID6);
      wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID6);
      TCVT(v45, v41, v9);
      set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
      pto::Shape<1, 1, 1, 1, 1> v98 = pto::Shape<1, 1, 1, 1, 1>();
      pto::Stride<4, 4, 4, 4, 1> v99 = pto::Stride<4, 4, 4, 4, 1>();
      GlobalTensor<float, pto::Shape<1, 1, 1, 1, 1>, pto::Stride<4, 4, 4, 4, 1>, pto::Layout::ND> v100 = GlobalTensor<float, pto::Shape<1, 1, 1, 1, 1>, pto::Stride<4, 4, 4, 4, 1>, pto::Layout::ND>(v2 + (v10 + (unsigned) v35 * (unsigned) v11 + v6 * (unsigned) v13), v98, v99);
      wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID5);
      TLOAD(v49, v100);
      set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
      wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
      TROWEXPAND(v53, v49);
      set_flag(PIPE_V, PIPE_MTE2, EVENT_ID2);
      pipe_barrier(PIPE_ALL);
      TMUL(v57, v45, v53);
      pipe_barrier(PIPE_ALL);
      TADD(v61, v61, v57);
      pipe_barrier(PIPE_ALL);
      wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
      TCVT(v65, v61, v9);
      set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
      unsigned v101 = (unsigned) v39;
      unsigned v102 = (unsigned) v5;
      pto::Shape<1, 1, 1, 1, -1> v103 = pto::Shape<1, 1, 1, 1, -1>(v39);
      pto::Stride<-1, -1, -1, -1, 1> v104 = pto::Stride<-1, -1, -1, -1, 1>(v102, v102, v102, v102);
      GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, 1>, pto::Layout::ND> v105 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, 1>, pto::Layout::ND>(v3 + (v10 + (unsigned) v35 * (unsigned) v5 + (unsigned) v37 * (unsigned) v13), v103, v104);
      wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
      pipe_barrier(PIPE_MTE3);
      TSTORE(v105, v65);
      set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
    };
  }
  wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
  wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
  wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID2);
  wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID3);
  wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
  wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
  #endif // __DAV_VEC__

  ptoas_auto_sync_tail(PTOAutoSyncTailMode::kBarrierAll);
  return;
}
