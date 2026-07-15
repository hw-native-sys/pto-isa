/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#include <acl/acl.h>
#include <pto/pto-inst.hpp>

#include "pto_macro_matmul.hpp"
#include "pto_macro_fa_softmax.hpp"
#include "pto_macro_fa_gu.hpp"

using namespace std;
using namespace pto;

#ifndef FFTS_BUFFER_FLAG_ENUM
#define FFTS_BUFFER_FLAG_ENUM
// Buffer flag values for FFTS pipeline coordination
enum FftsBufferFlag : uint32_t {
    BUF0_QK_READY,    // Buffer 0: QK data ready
    BUF0_SM_CONSUMED, // Buffer 0: Softmax consumed
    BUF1_SM_READY,    // Buffer 1: Softmax output ready
    BUF1_SV_CONSUMED, // Buffer 1: SV consumed
    UPDATE_READY,     // Update stage ready
    UPDATE_CONSUMED   // Update stage consumed
};
#endif

enum CoreEvtID : uint32_t {
    QK_EVENT_ID0,
    QK_EVENT_ID1,
    PV_EVENT_ID0,
    PV_EVENT_ID1,
};

#define VEC_CORES 2

// Inline macro used for small, performance-sensitive functions
#ifndef PTO_INLINE
#define PTO_INLINE __attribute__((always_inline)) inline
#endif

// Detect build-time macros and expose as constexpr flags for clearer conditionals
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

constexpr std::size_t MAX_TILE_L1_BYTES = 512U * 1024U;
constexpr std::size_t MAX_VEC_UB_BYTES = 192U * 1024U;

template <typename TileType>
constexpr AICORE std::size_t tile_storage_bytes()
{
    using ElementType = typename TileType::DType;
    return static_cast<std::size_t>(TileType::Rows * TileType::Cols) * sizeof(ElementType);
}

template <typename TileType, std::size_t NumBuffers>
constexpr AICORE std::size_t tile_buffer_total_bytes()
{
    return tile_storage_bytes<TileType>() * NumBuffers;
}

template <typename TileType, std::size_t NumBuffers>
PTO_INLINE AICORE uint32_t assign_tile_buffers(TileType (&tiles)[NumBuffers], uint32_t base_offset)
{
    if constexpr (NumBuffers == 0) {
        return base_offset;
    }

    constexpr std::size_t total_storage_bytes = tile_buffer_total_bytes<TileType, NumBuffers>();
    static_assert(total_storage_bytes <= MAX_TILE_L1_BYTES, "Tile buffer L1 allocation exceeds 512KB");

    for (std::size_t idx = 0; idx < NumBuffers; ++idx) {
        const uint32_t tile_offset = base_offset + static_cast<uint32_t>(idx * tile_storage_bytes<TileType>());
        TASSIGN(tiles[idx], tile_offset);
    }

    return base_offset + static_cast<uint32_t>(total_storage_bytes);
}

template <
    typename TileQType, std::size_t NumQ, typename TileKType, std::size_t NumK, typename TilePType, std::size_t NumP,
    typename TileVType, std::size_t NumV>
PTO_INLINE AICORE void allocate_cube_tile_buffers(
    TileQType (&qTiles)[NumQ], TileKType (&kTiles)[NumK], TilePType (&pTiles)[NumP], TileVType (&vTiles)[NumV])
{
    constexpr std::size_t total_bytes =
        tile_buffer_total_bytes<TileQType, NumQ>() + tile_buffer_total_bytes<TileKType, NumK>() +
        tile_buffer_total_bytes<TilePType, NumP>() + tile_buffer_total_bytes<TileVType, NumV>();
    static_assert(total_bytes <= MAX_TILE_L1_BYTES, "Total cube L1 allocation exceeds 512KB");

    uint32_t l1_offset = 0;
    l1_offset = assign_tile_buffers(qTiles, l1_offset);
    l1_offset = assign_tile_buffers(kTiles, l1_offset);
    l1_offset = assign_tile_buffers(pTiles, l1_offset);
    l1_offset = assign_tile_buffers(vTiles, l1_offset);
    (void)l1_offset;
}

template <
    typename TileDataF_T, typename ReduceTileF_T, typename TileDataH_T, typename TileOutT, std::size_t SrcBuffers,
    std::size_t XexpBuffers, std::size_t pvVecBuffers, std::size_t ExpMaxBuffers>
PTO_INLINE AICORE void allocate_vec_tile_buffers(
    TileDataF_T (&srcTiles)[SrcBuffers], ReduceTileF_T& m1_local_max, TileDataF_T& input_reduce_tmp,
    ReduceTileF_T& l1_local_sum, ReduceTileF_T& m2_global_max, ReduceTileF_T& l2_global_sum,
    ReduceTileF_T (&l1_exp_max)[ExpMaxBuffers], TileDataH_T (&x_expT)[XexpBuffers], TileOutT (&pvTile)[pvVecBuffers],
    TileOutT& runningOTile)
{
    constexpr std::size_t src_bytes = tile_buffer_total_bytes<TileDataF_T, SrcBuffers>();
    constexpr std::size_t float_tile_bytes = tile_storage_bytes<TileDataF_T>();
    constexpr std::size_t reduce_tile_bytes = tile_storage_bytes<ReduceTileF_T>();
    constexpr std::size_t xexp_bytes = tile_buffer_total_bytes<TileDataH_T, XexpBuffers>();
    constexpr std::size_t out_tile_bytes = tile_storage_bytes<TileOutT>();
    constexpr std::size_t total_bytes =
        src_bytes + xexp_bytes + (reduce_tile_bytes * (3U + ExpMaxBuffers)) + (float_tile_bytes * 1U) + out_tile_bytes;
    static_assert(total_bytes <= MAX_VEC_UB_BYTES, "Vec tile UB allocation exceeds 192KB");

    uint32_t offset = 0;
    offset = assign_tile_buffers(srcTiles, offset);

    TASSIGN(m1_local_max, offset);
    offset += static_cast<uint32_t>(reduce_tile_bytes);

    TASSIGN(m2_global_max, offset);
    offset += static_cast<uint32_t>(reduce_tile_bytes);

    uint32_t tmp_float_offset = offset;
    TASSIGN(input_reduce_tmp, tmp_float_offset);
    offset += static_cast<uint32_t>(float_tile_bytes);

    TASSIGN(l1_local_sum, offset);
    offset += static_cast<uint32_t>(reduce_tile_bytes);

    TASSIGN(l2_global_sum, offset);
    offset += static_cast<uint32_t>(reduce_tile_bytes);

    offset = assign_tile_buffers(l1_exp_max, offset);

    uint32_t tail_offset = assign_tile_buffers(x_expT, offset);

    TASSIGN(runningOTile, tail_offset);

    assign_tile_buffers(pvTile, 0);

    tail_offset += static_cast<uint32_t>(out_tile_bytes);
    (void)tail_offset;
}

// Helper to assign an accumulator tile to one of two ping-pong UB addresses (0x0 / 0x10000).
// Keeps a per-type static running index that toggles on every call. Caller may pass
// `initial_id` (0 or 1) to set the starting buffer index on the first call for that tile type.
template <typename AccTileT>
PTO_INLINE AICORE int assign_running_acc_tile(AccTileT& accTile, int initial_id = -1)
{
    static int running_tile_buffer_idx = 0; // per-instantiation running buffer index: 0 -> base0, 1 -> base1
    if (initial_id == 0 || initial_id == 1) {
        running_tile_buffer_idx = initial_id;
    }
    const int id = running_tile_buffer_idx;
    const uint32_t base_addr = (id == 0) ? 0x0u : 0x10000u;
    TASSIGN(accTile, base_addr);
    running_tile_buffer_idx ^= 1; // toggle for next call
    return id;
}

template <
    int S0, int HEAD_SIZE, int S1, int CUBE_S1, int QKV_CV_FIFO, bool INTERMEDIATE_CHECK, typename TileMatQData,
    typename TileMatKData, typename TileQKData>
PTO_INLINE AICORE void compute_qk(
    int tile_idx, __gm__ half* q, __gm__ half* k, __gm__ float* qk_out, TileMatQData& qMatTile, TileMatKData& kMatTile,
    TileQKData& qkAccTile, uint64_t qkMatTileEventId)
{
    if constexpr (DAV_CUBE) {
        constexpr uint32_t Cube_S0 = S0;
        constexpr uint32_t Cube_S1 = CUBE_S1;
        constexpr uint32_t Cube_HEAD = HEAD_SIZE;
        static_assert(QKV_CV_FIFO >= 1, "QKV_CV_FIFO must be >= 1");

        const int s1_index = tile_idx * static_cast<int>(Cube_S1);

        using GlobalDataQ =
            GlobalTensor<half, pto::Shape<1, 1, 1, Cube_S0, HEAD_SIZE>, pto::Stride<1, 1, 1, HEAD_SIZE, 1>>;
        using GlobalDataK = GlobalTensor<
            half, pto::Shape<1, 1, 1, HEAD_SIZE, Cube_S1>, pto::Stride<1, 1, 1, 1, HEAD_SIZE>,
            Layout::DN>; // BNSD - (N, K) layout

        GlobalDataQ qGlobal(q);
        GlobalDataK kGlobal(k + s1_index * HEAD_SIZE);

        wait_flag(PIPE_MTE1, PIPE_MTE2, qkMatTileEventId);

        if (tile_idx == 0) {
            TLOAD(qMatTile, qGlobal);
        }

        TLOAD(kMatTile, kGlobal);

        set_flag(PIPE_MTE1, PIPE_MTE2, qkMatTileEventId);

        set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);

        pto_macro_matmul<Cube_S0, Cube_HEAD, Cube_S1>(qMatTile, kMatTile, qkAccTile);

        set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
        wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);

        wait_flag_dev(BUF0_SM_CONSUMED); // wait for SM consume data

        if constexpr (INTERMEDIATE_CHECK) {
            using GlobalDataQK =
                GlobalTensor<float, pto::Shape<1, 1, 1, Cube_S0, Cube_S1>, pto::Stride<1, 1, 1, S1, 1>>;
            GlobalDataQK qkGlobalTile(qk_out + s1_index);
            TSTORE(qkGlobalTile, qkAccTile);
        } else {
            using GlobalDataQK =
                GlobalTensor<float, pto::Shape<1, 1, 1, Cube_S0, Cube_S1>, pto::Stride<1, 1, 1, Cube_S1, 1>>;
            const uint32_t buf_idx = static_cast<uint32_t>(tile_idx % QKV_CV_FIFO);
            const size_t base_elems =
                static_cast<size_t>(buf_idx) * static_cast<size_t>(Cube_S0) * static_cast<size_t>(Cube_S1);
            GlobalDataQK qkGlobalTile(qk_out + base_elems);
            TSTORE(qkGlobalTile, qkAccTile);
        }

        ffts_cross_core_sync(PIPE_FIX, getFFTSMsg(0x2, BUF0_QK_READY)); // notify for QK produce data
    }
}

template <
    int S0, int HEAD_SIZE, int S1, int CUBE_S1, int QKV_CV_FIFO, int PV_CV_FIFO, bool INTERMEDIATE_CHECK,
    typename TileMatPData, typename TileMatVData, typename TilePVData>
PTO_INLINE AICORE void compute_pv(
    int tile_idx, __gm__ half* p_out, __gm__ half* v, __gm__ float* pv_out, TileMatPData& pMatTile,
    TileMatVData& vMatTile, TilePVData& pvAccTile, uint64_t svMatTileEventId)
{
    constexpr uint32_t Cube_S0 = S0;
    constexpr uint32_t Cube_S1 = CUBE_S1;
    constexpr uint32_t Cube_HEAD = HEAD_SIZE;
    static_assert(QKV_CV_FIFO >= 1, "QKV_CV_FIFO must be >= 1");

    const int s1_index = tile_idx * static_cast<int>(Cube_S1);

    if constexpr (DAV_CUBE) {
        using GlobalVT =
            GlobalTensor<half, pto::Shape<1, 1, 1, Cube_S1, HEAD_SIZE>, pto::Stride<1, 1, 1, HEAD_SIZE, 1>>;

        wait_flag(PIPE_MTE1, PIPE_MTE2, svMatTileEventId);

        GlobalVT vLoad((__gm__ half*)(v + s1_index * HEAD_SIZE));
        TLOAD(vMatTile, vLoad);

        wait_flag_dev(BUF1_SM_READY); // wait for softmax produce data

        if constexpr (INTERMEDIATE_CHECK) {
            using GlobalXexpTileT =
                GlobalTensor<half, pto::Shape<1, 1, 1, Cube_S0, Cube_S1>, pto::Stride<1, 1, 1, S1, 1>>;
            GlobalXexpTileT xexpLoad(p_out + s1_index);
            TLOAD(pMatTile, xexpLoad);
        } else {
            using GlobalXexpTileT =
                GlobalTensor<half, pto::Shape<1, 1, 1, Cube_S0, Cube_S1>, pto::Stride<1, 1, 1, Cube_S1, 1>>;
            const uint32_t buf_idx = static_cast<uint32_t>(tile_idx % QKV_CV_FIFO);
            const size_t base_elems =
                static_cast<size_t>(buf_idx) * static_cast<size_t>(Cube_S0) * static_cast<size_t>(Cube_S1);
            GlobalXexpTileT xexpLoad(p_out + base_elems);
            TLOAD(pMatTile, xexpLoad);
        }
        ffts_cross_core_sync(PIPE_MTE2, getFFTSMsg(0x2, BUF1_SV_CONSUMED)); // notify SV consume data

        set_flag(PIPE_MTE1, PIPE_MTE2, svMatTileEventId);

        set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);

        pto_macro_matmul<Cube_S0, Cube_S1, Cube_HEAD>(pMatTile, vMatTile, pvAccTile);

        set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
        wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);

        wait_flag_dev(UPDATE_CONSUMED); // wait for update consume data

        if constexpr (INTERMEDIATE_CHECK) {
            size_t partialOffsetElems =
                static_cast<size_t>(tile_idx) * static_cast<size_t>(S0) * static_cast<size_t>(HEAD_SIZE);
            using GlobalDataPV =
                GlobalTensor<float, pto::Shape<1, 1, 1, Cube_S0, HEAD_SIZE>, pto::Stride<1, 1, 1, HEAD_SIZE, 1>>;
            GlobalDataPV pvGlobalTile((__gm__ float*)(pv_out + partialOffsetElems));
            TSTORE(pvGlobalTile, pvAccTile);
        } else {
            using GlobalDataPV =
                GlobalTensor<float, pto::Shape<1, 1, 1, Cube_S0, HEAD_SIZE>, pto::Stride<1, 1, 1, HEAD_SIZE, 1>>;

            // GlobalDataPV pvGlobalTile((__gm__ float *)(pv_out));
            const uint32_t buf_idx = static_cast<uint32_t>(tile_idx % PV_CV_FIFO);
            const size_t base_elems =
                static_cast<size_t>(buf_idx) * static_cast<size_t>(Cube_S0) * static_cast<size_t>(HEAD_SIZE);
            GlobalDataPV pvGlobalTile((__gm__ float*)(pv_out + base_elems));
            TSTORE(pvGlobalTile, pvAccTile);
        }
        ffts_cross_core_sync(PIPE_FIX, getFFTSMsg(0x2, UPDATE_READY)); // notify update produce data
    }
}

template <
    int S0, int HEAD_SIZE, int S1, int CUBE_S1, int QKV_CV_FIFO, bool INTERMEDIATE_CHECK, typename TileDataF_T,
    typename TileDataH_T, typename ReduceTileF_T>
PTO_INLINE AICORE void compute_p(
    int tile_idx, bool initFlag, __gm__ float* qk_out, __gm__ half* p_out, __gm__ float* p_out_fp32,
    __gm__ float* global_sum_out, __gm__ float* exp_max_out, TileDataF_T& qkVecTile, TileDataH_T& x_expT,
    TileDataF_T& input_reduce_tmp, ReduceTileF_T& m1_local_max, ReduceTileF_T& l1_local_sum,
    ReduceTileF_T& m2_global_max, ReduceTileF_T& l2_global_sum, ReduceTileF_T& l1_exp_max, uint64_t pTileEventId)
{
    constexpr uint32_t Vec_S0 = S0 / VEC_CORES;
    constexpr uint32_t Cube_S0 = S0;
    constexpr uint32_t Cube_S1 = CUBE_S1;
    static_assert(QKV_CV_FIFO >= 1, "QKV_CV_FIFO must be >= 1");
    if constexpr (DAV_VEC) {
        const int s1_index = tile_idx * static_cast<int>(Cube_S1);

        wait_flag(PIPE_V, PIPE_MTE2, pTileEventId);
        wait_flag_dev(BUF0_QK_READY); // wait for QK produce data

        if constexpr (INTERMEDIATE_CHECK) {
            __gm__ float* qk_ptr = qk_out + Vec_S0 * S1 * get_subblockid();
            qk_ptr += s1_index;
            using GlobalDataQK_VEC =
                GlobalTensor<float, pto::Shape<1, 1, 1, Vec_S0, Cube_S1>, pto::Stride<1, 1, 1, S1, 1>>;
            GlobalDataQK_VEC qkGlobalTile(qk_ptr);
            TLOAD(qkVecTile, qkGlobalTile);
        } else {
            const uint32_t buf_idx = static_cast<uint32_t>(tile_idx % QKV_CV_FIFO);
            const size_t base_elems =
                static_cast<size_t>(buf_idx) * static_cast<size_t>(Cube_S0) * static_cast<size_t>(Cube_S1);
            __gm__ float* qk_ptr = qk_out + base_elems + Vec_S0 * Cube_S1 * get_subblockid();
            using GlobalDataQK_VEC =
                GlobalTensor<float, pto::Shape<1, 1, 1, Vec_S0, Cube_S1>, pto::Stride<1, 1, 1, Cube_S1, 1>>;
            GlobalDataQK_VEC qkGlobalTile(qk_ptr);
            TLOAD(qkVecTile, qkGlobalTile);
        }

        ffts_cross_core_sync(PIPE_MTE2, getFFTSMsg(0x2, BUF0_SM_CONSUMED)); // notify for SM consume data
        set_flag(PIPE_V, PIPE_MTE2, pTileEventId);

        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

        wait_flag(PIPE_MTE3, PIPE_V, pTileEventId);
        if (initFlag) {
            pto_macro_fa_softmax<true, HEAD_SIZE>(
                x_expT, qkVecTile, m1_local_max, l1_local_sum, m2_global_max, l2_global_sum, l1_exp_max,
                input_reduce_tmp, qkVecTile);
        } else {
            pto_macro_fa_softmax<false, HEAD_SIZE>(
                x_expT, qkVecTile, m1_local_max, l1_local_sum, m2_global_max, l2_global_sum, l1_exp_max,
                input_reduce_tmp, qkVecTile);
        }

        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);

        wait_flag_dev(BUF1_SV_CONSUMED); // wait for SV consume data

        if constexpr (INTERMEDIATE_CHECK) {
            __gm__ half* p_ptr = p_out + Vec_S0 * S1 * get_subblockid();
            p_ptr += s1_index;
            using GlobalPTileHalf =
                GlobalTensor<half, pto::Shape<1, 1, 1, Vec_S0, Cube_S1>, pto::Stride<1, 1, 1, S1, 1>>;
            GlobalPTileHalf pTileHalf((__gm__ half*)(p_ptr));
            TSTORE(pTileHalf, x_expT);
        } else {
            __gm__ half* p_ptr = p_out + Vec_S0 * Cube_S1 * get_subblockid();
            using GlobalPTileHalf =
                GlobalTensor<half, pto::Shape<1, 1, 1, Vec_S0, Cube_S1>, pto::Stride<1, 1, 1, Cube_S1, 1>>;
            const uint32_t buf_idx = static_cast<uint32_t>(tile_idx % QKV_CV_FIFO);
            const size_t base_elems =
                static_cast<size_t>(buf_idx) * static_cast<size_t>(Cube_S0) * static_cast<size_t>(Cube_S1);
            p_ptr += base_elems;
            GlobalPTileHalf pTileHalf((__gm__ half*)(p_ptr));
            TSTORE(pTileHalf, x_expT);
        }

        ffts_cross_core_sync(PIPE_MTE3, getFFTSMsg(0x2, BUF1_SM_READY)); // notify softmax produce data

        if constexpr (INTERMEDIATE_CHECK) {
            __gm__ float* p_fp32_ptr = p_out_fp32 + Vec_S0 * Cube_S1 * get_subblockid() + s1_index;
            using GlobalPTileFP32 =
                GlobalTensor<float, pto::Shape<1, 1, 1, Vec_S0, Cube_S1>, pto::Stride<1, 1, 1, S1, 1>>;
            GlobalPTileFP32 pTileFp32((__gm__ float*)(p_fp32_ptr));
            TSTORE(pTileFp32, qkVecTile); // p_tile_fp32 reuse qk_vec_tile

            size_t gsumOffsetElems = static_cast<size_t>(tile_idx) * static_cast<size_t>(S0);
            using GlobalSumT = GlobalTensor<float, pto::Shape<1, 1, 1, Vec_S0, 1>, pto::Stride<1, 1, 1, 1, 1>>;
            GlobalSumT gsum((__gm__ float*)(global_sum_out + gsumOffsetElems + Vec_S0 * get_subblockid()));
            TSTORE(gsum, l2_global_sum);

            size_t expOffsetElems = static_cast<size_t>(tile_idx) * static_cast<size_t>(S0);
            using GlobalExpT = GlobalTensor<float, pto::Shape<1, 1, 1, Vec_S0, 1>, pto::Stride<1, 1, 1, 1, 1>>;
            GlobalExpT gexp((__gm__ float*)(exp_max_out + expOffsetElems + Vec_S0 * get_subblockid()));
            TSTORE(gexp, l1_exp_max);
        }

        set_flag(PIPE_MTE3, PIPE_V, pTileEventId);
        // pipe_barrier(PIPE_ALL);
    }
}

template <
    int S0, int HEAD_SIZE, int S1, int PV_CV_FIFO, bool INTERMEDIATE_CHECK, typename TileOutT, typename ReduceTileF_T>
PTO_INLINE AICORE void compute_gu(
    int tile_idx, int num_tiles_s1, __gm__ float* pv_out, __gm__ float* o_out, __gm__ float* o_parts_out,
    TileOutT& runningOTile, TileOutT& pvVecTile, ReduceTileF_T& l1_exp_max, ReduceTileF_T& l2_global_sum,
    uint64_t guEventId)
{
    constexpr uint32_t Vec_S0 = S0 / VEC_CORES;
    constexpr uint32_t Cube_S0 = S0;

    using GlobalDataPV_VEC =
        GlobalTensor<float, pto::Shape<1, 1, 1, Vec_S0, HEAD_SIZE>, pto::Stride<1, 1, 1, HEAD_SIZE, 1>>;

    if constexpr (DAV_VEC) {
        const uint32_t buf_idx = static_cast<uint32_t>(tile_idx % PV_CV_FIFO);
        const size_t base_elems =
            static_cast<size_t>(buf_idx) * static_cast<size_t>(Cube_S0) * static_cast<size_t>(HEAD_SIZE);

        __gm__ float* pv_out_ptr = pv_out + base_elems + Vec_S0 * HEAD_SIZE * get_subblockid();
        if constexpr (INTERMEDIATE_CHECK) {
            size_t partialOffsetElemsVec =
                static_cast<size_t>(tile_idx) * static_cast<size_t>(S0) * static_cast<size_t>(HEAD_SIZE);
            pv_out_ptr += partialOffsetElemsVec;
        }

        GlobalDataPV_VEC pvGlobalVec(pv_out_ptr);

        wait_flag_dev(UPDATE_READY); // wait for update consume data

        // softamx output and gu input buffer reuse

        wait_flag(PIPE_V, PIPE_MTE2, guEventId);

        if (tile_idx == 0) {
            TLOAD(runningOTile, pvGlobalVec);
            set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
            wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        } else {
            TLOAD(pvVecTile, pvGlobalVec);
            set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
            wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

            if (tile_idx < num_tiles_s1 - 1) {
                pto_macro_fa_gu<ReduceTileF_T, TileOutT>(runningOTile, pvVecTile, l1_exp_max);
            } else {
                pto_macro_fa_gu_last<ReduceTileF_T, TileOutT>(runningOTile, pvVecTile, l1_exp_max, l2_global_sum);
            }
        }

        set_flag(PIPE_V, PIPE_MTE2, guEventId);
        ffts_cross_core_sync(PIPE_MTE2, getFFTSMsg(0x2, UPDATE_CONSUMED)); // notify update consume data

        if (tile_idx == num_tiles_s1 - 1) {
            set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
            wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
            using GlobalOutT =
                GlobalTensor<float, pto::Shape<1, 1, 1, Vec_S0, HEAD_SIZE>, pto::Stride<1, 1, 1, HEAD_SIZE, 1>>;
            GlobalOutT outGlobal((__gm__ float*)(o_out + Vec_S0 * HEAD_SIZE * get_subblockid()));
            TSTORE(outGlobal, runningOTile);
        }

        if constexpr (INTERMEDIATE_CHECK) {
            set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
            wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
            size_t oPartOffsetElems =
                static_cast<size_t>(tile_idx) * static_cast<size_t>(S0) * static_cast<size_t>(HEAD_SIZE);
            using GlobalOutPartT =
                GlobalTensor<float, pto::Shape<1, 1, 1, Vec_S0, HEAD_SIZE>, pto::Stride<1, 1, 1, HEAD_SIZE, 1>>;
            GlobalOutPartT outPartGM(
                (__gm__ float*)(o_parts_out + oPartOffsetElems + Vec_S0 * HEAD_SIZE * get_subblockid()));
            TSTORE(outPartGM, runningOTile);
        }
    }
}

template <int S0, int HEAD_SIZE, int S1, int CUBE_S1, bool INTERMEDIATE_CHECK = false>
__global__ AICORE void runTFA(
    __gm__ uint64_t* ffts_addr, __gm__ half* q, __gm__ half* k, __gm__ half* v, __gm__ half* p_out,
    __gm__ float* p_out_fp32, __gm__ float* global_sum_out, __gm__ float* exp_max_out, __gm__ float* o_out,
    __gm__ float* o_parts_out, __gm__ float* qk_out, __gm__ float* pv_out)
{
    uint64_t tStart = get_sys_cnt();

    set_ffts_base_addr((uint64_t)ffts_addr);
    if constexpr (DAV_CUBE) {
        set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
        set_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
    }

    // Rename dimensions for clarity: S0 (rows), S1 (cols), HEAD_SIZE (inner)
    constexpr uint32_t Cube_S0 = S0;
    constexpr uint32_t Cube_S1 = CUBE_S1; // per-tile S1 chunk
    constexpr uint32_t Cube_HEAD = HEAD_SIZE;
    constexpr uint32_t Vec_S0 = S0 / VEC_CORES;
    // Buffer counts for optional double-buffering (default 1)
    constexpr uint32_t qkPreloadNum = 5;
    constexpr uint32_t srcVecTNBuffers = 2;
    constexpr uint32_t xexpVecTNBuffers = 2;
    constexpr uint32_t outOTileNBuffers = 2;
    constexpr uint32_t qMatTNBuffers = 1;
    constexpr uint32_t kMatTNBuffers = 2;
    constexpr uint32_t pMatTNBuffers = 2;
    constexpr uint32_t vMatTNBuffers = 2;
    constexpr uint32_t qkGlobalTensorNBuffers = 1 + qkPreloadNum;
    constexpr uint32_t pvGlobalTensorNBuffers = 1 + qkPreloadNum;
    constexpr uint32_t guGlobalTensorNBuffers = 1 + qkPreloadNum;
    constexpr uint32_t preload_pv_offset = qkPreloadNum;
    constexpr uint32_t pvPreloadNum = 1;

    // Define tile types for first QK matmul
    using TileMatQData =
        Tile<TileType::Mat, half, S0, HEAD_SIZE, BLayout::ColMajor, S0, HEAD_SIZE, SLayout::RowMajor, 512>;
    using TileMatKData =
        Tile<TileType::Mat, half, HEAD_SIZE, Cube_S1, BLayout::RowMajor, HEAD_SIZE, Cube_S1, SLayout::ColMajor, 512>;
    using TileQKData = TileAcc<float, S0, Cube_S1, Cube_S0, Cube_S1>;

    TileMatQData qMatTile[qMatTNBuffers];
    TileMatKData kMatTile[kMatTNBuffers];
    TileQKData qkAccTile;

    // Define tile types for second PV matmul
    using TileMatPData = Tile<TileType::Mat, half, S0, Cube_S1, BLayout::ColMajor, S0, Cube_S1, SLayout::RowMajor, 512>;
    using TileMatVData =
        Tile<TileType::Mat, half, Cube_S1, HEAD_SIZE, BLayout::ColMajor, Cube_S1, HEAD_SIZE, SLayout::RowMajor, 512>;
    using TilePVData = TileAcc<float, S0, HEAD_SIZE, Cube_S0, HEAD_SIZE>;

    TileMatPData pMatTile[pMatTNBuffers];
    TileMatVData vMatTile[vMatTNBuffers];
    TilePVData pvAccTile;

    allocate_cube_tile_buffers(qMatTile, kMatTile, pMatTile, vMatTile);

    // Assign accumulator tiles using ping-pong helper. qk starts at 0, pv starts at 1.
    assign_running_acc_tile(qkAccTile, 0);
    assign_running_acc_tile(pvAccTile, 1);

    // Define tile types for FA softmax P computation
    // UB offsets for softmax tiles
    // Define per-tile vector tiles sized to Cube_S1
    using TileDataF_T = Tile<TileType::Vec, float, Vec_S0, Cube_S1, BLayout::RowMajor, Vec_S0, Cube_S1>;
    using TileDataH_T = Tile<TileType::Vec, half, Vec_S0, Cube_S1, BLayout::RowMajor, Vec_S0, Cube_S1>;
    using ReduceTileF_T = Tile<TileType::Vec, float, Vec_S0, 1, BLayout::ColMajor, Vec_S0, 1>;

    TileDataF_T qkVecTile[srcVecTNBuffers];
    ReduceTileF_T m1_local_max;
    TileDataF_T input_reduce_tmp;
    ReduceTileF_T l1_local_sum;
    ReduceTileF_T m2_global_max;
    ReduceTileF_T l2_global_sum;
    ReduceTileF_T l1_exp_max[qkGlobalTensorNBuffers];
    TileDataH_T x_expT[xexpVecTNBuffers];

    using TileOutT = Tile<TileType::Vec, float, Vec_S0, HEAD_SIZE, BLayout::RowMajor, Vec_S0, HEAD_SIZE>;
    TileOutT pvVecTile[outOTileNBuffers];
    TileOutT runningOTile;
    allocate_vec_tile_buffers<
        TileDataF_T, ReduceTileF_T, TileDataH_T, TileOutT, srcVecTNBuffers, xexpVecTNBuffers, outOTileNBuffers>(
        qkVecTile, m1_local_max, input_reduce_tmp, l1_local_sum, m2_global_max, l2_global_sum, l1_exp_max, x_expT,
        pvVecTile, runningOTile);

    int num_tiles_s1 = S1 / Cube_S1;
    if constexpr (DAV_CUBE) {
        for (int i = 0; i < qkGlobalTensorNBuffers; i++) {
            st_dev(getFFTSMsg(0x2, BUF1_SV_CONSUMED), ffts_addr, 0);
        }
        set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
        set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
        set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);
        set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID3);
        set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
        set_flag(PIPE_FIX, PIPE_M, EVENT_ID1);
    }
    if constexpr (DAV_VEC) {
        st_dev(getFFTSMsg(0x2, UPDATE_CONSUMED), ffts_addr, 0);
        for (int i = 0; i < qkGlobalTensorNBuffers; i++) {
            ffts_cross_core_sync(PIPE_MTE2, getFFTSMsg(0x2, BUF0_SM_CONSUMED));
        }
        set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
        set_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
        set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
        set_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
    }

    int preload_pv_tile_idx = 0;

    for (int preload_qk_idx = 0; preload_qk_idx < qkPreloadNum && preload_qk_idx < num_tiles_s1; preload_qk_idx++) {
        if constexpr (DAV_CUBE) {
            int accId = assign_running_acc_tile(qkAccTile);
            wait_flag(PIPE_FIX, PIPE_M, accId);
            compute_qk<S0, HEAD_SIZE, S1, CUBE_S1, qkGlobalTensorNBuffers, INTERMEDIATE_CHECK>(
                preload_qk_idx, q, k, qk_out, qMatTile[0], kMatTile[preload_qk_idx % kMatTNBuffers], qkAccTile,
                preload_qk_idx % kMatTNBuffers);
            set_flag(PIPE_FIX, PIPE_M, accId);
            if (preload_qk_idx >= preload_pv_offset && preload_pv_tile_idx < pvPreloadNum) {
                int accId = assign_running_acc_tile(pvAccTile);
                wait_flag(PIPE_FIX, PIPE_M, accId);
                compute_pv<
                    S0, HEAD_SIZE, S1, CUBE_S1, pvGlobalTensorNBuffers, guGlobalTensorNBuffers, INTERMEDIATE_CHECK>(
                    preload_pv_tile_idx, p_out, v, pv_out, pMatTile[preload_pv_tile_idx % pMatTNBuffers],
                    vMatTile[preload_pv_tile_idx % vMatTNBuffers], pvAccTile,
                    preload_pv_tile_idx % vMatTNBuffers + PV_EVENT_ID0);
                set_flag(PIPE_FIX, PIPE_M, accId);
                preload_pv_tile_idx++;
            }
        }

        if constexpr (DAV_VEC) {
            int tile_idx = preload_qk_idx;
            bool initFlag = (preload_qk_idx == 0);
            compute_p<S0, HEAD_SIZE, S1, CUBE_S1, qkGlobalTensorNBuffers, INTERMEDIATE_CHECK>(
                tile_idx, initFlag, qk_out, p_out, p_out_fp32, global_sum_out, exp_max_out,
                qkVecTile[tile_idx % srcVecTNBuffers], x_expT[tile_idx % xexpVecTNBuffers], input_reduce_tmp,
                m1_local_max, l1_local_sum, m2_global_max, l2_global_sum,
                l1_exp_max[preload_qk_idx % qkGlobalTensorNBuffers], tile_idx % xexpVecTNBuffers);
        }
    }

    for (int S1_tile_idx = 0; S1_tile_idx < num_tiles_s1; S1_tile_idx++) {
        int next_qk_tile_idx = (S1_tile_idx + qkPreloadNum) >= num_tiles_s1 ? -1 : (S1_tile_idx + qkPreloadNum); // 1

        if constexpr (DAV_CUBE) {
            if (next_qk_tile_idx != -1) {
                int accId = assign_running_acc_tile(qkAccTile);
                wait_flag(PIPE_FIX, PIPE_M, accId);
                compute_qk<S0, HEAD_SIZE, S1, CUBE_S1, qkGlobalTensorNBuffers, INTERMEDIATE_CHECK>(
                    next_qk_tile_idx, q, k, qk_out, qMatTile[0], kMatTile[next_qk_tile_idx % kMatTNBuffers], qkAccTile,
                    next_qk_tile_idx % kMatTNBuffers);
                set_flag(PIPE_FIX, PIPE_M, accId);
            }
        }

        if constexpr (DAV_VEC) {
            if (next_qk_tile_idx != -1) {
                bool initFlag = ((S1_tile_idx + qkPreloadNum) == 0);
                compute_p<S0, HEAD_SIZE, S1, CUBE_S1, qkGlobalTensorNBuffers, INTERMEDIATE_CHECK>(
                    next_qk_tile_idx, initFlag, qk_out, p_out, p_out_fp32, global_sum_out, exp_max_out,
                    qkVecTile[next_qk_tile_idx % srcVecTNBuffers], x_expT[next_qk_tile_idx % xexpVecTNBuffers],
                    input_reduce_tmp, m1_local_max, l1_local_sum, m2_global_max, l2_global_sum,
                    l1_exp_max[next_qk_tile_idx % qkGlobalTensorNBuffers], next_qk_tile_idx % xexpVecTNBuffers);
            }
        }

        if constexpr (DAV_CUBE) {
            int S1_pv_tile_idx = preload_pv_tile_idx >= num_tiles_s1 ? -1 : preload_pv_tile_idx;
            if (S1_pv_tile_idx != -1) {
                int accId = assign_running_acc_tile(pvAccTile);
                wait_flag(PIPE_FIX, PIPE_M, accId);
                compute_pv<
                    S0, HEAD_SIZE, S1, CUBE_S1, pvGlobalTensorNBuffers, guGlobalTensorNBuffers, INTERMEDIATE_CHECK>(
                    S1_pv_tile_idx, p_out, v, pv_out, pMatTile[S1_pv_tile_idx % pMatTNBuffers],
                    vMatTile[S1_pv_tile_idx % vMatTNBuffers], pvAccTile, S1_pv_tile_idx % vMatTNBuffers + PV_EVENT_ID0);
                set_flag(PIPE_FIX, PIPE_M, accId);
                preload_pv_tile_idx++;
            }
        }

        if constexpr (DAV_VEC) {
            compute_gu<S0, HEAD_SIZE, S1, guGlobalTensorNBuffers, INTERMEDIATE_CHECK>(
                S1_tile_idx, num_tiles_s1, pv_out, o_out, o_parts_out, runningOTile,
                pvVecTile[S1_tile_idx % outOTileNBuffers], l1_exp_max[S1_tile_idx % qkGlobalTensorNBuffers],
                l2_global_sum, S1_tile_idx % outOTileNBuffers);
        }
    }

    if constexpr (DAV_CUBE) {
        wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
        wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
        wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
        wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
        wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);
        wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID3);
        for (int i = 0; i < qkGlobalTensorNBuffers; i++) {
            wait_flag_dev(BUF0_SM_CONSUMED);
        }
        wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
        wait_flag(PIPE_FIX, PIPE_M, EVENT_ID1);
    }

    if constexpr (DAV_VEC) {
        wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
        wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
    }

    uint64_t tEnd = get_sys_cnt();
#ifdef DEBUGLOG
    cce::printf("TQKSV total cycles: %llu\n", tEnd - tStart);
#endif
}

// Host wrapper
template <int S0, int HEAD_SIZE, int S1, int CUBE_S1, bool INTERMEDIATE_CHECK = false>
void LaunchTFA(
    uint16_t* ffts, aclFloat16* q, aclFloat16* k, aclFloat16* v, aclFloat16* p_out, float* p_out_fp32,
    float* global_sum_out, float* exp_max_out, float* o_out, float* o_parts_out, float* qk_out, float* pv_out,
    void* stream)
{
    runTFA<S0, HEAD_SIZE, S1, CUBE_S1, INTERMEDIATE_CHECK><<<1, nullptr, stream>>>(
        (__gm__ uint64_t*)ffts, (half*)q, (half*)k, (half*)v, (half*)p_out, p_out_fp32, global_sum_out, exp_max_out,
        o_out, o_parts_out, qk_out, pv_out);
}

// Explicit instantiations for currently used cases
template void LaunchTFA<64, 128, 256, 128>(
    uint16_t* ffts, aclFloat16* q, aclFloat16* k, aclFloat16* v, aclFloat16* p_out, float* p_out_fp32,
    float* global_sum_out, float* exp_max_out, float* o_out, float* o_parts_out, float* qk_out, float* pv_out,
    void* stream);
template void LaunchTFA<64, 128, 128, 128>(
    uint16_t* ffts, aclFloat16* q, aclFloat16* k, aclFloat16* v, aclFloat16* p_out, float* p_out_fp32,
    float* global_sum_out, float* exp_max_out, float* o_out, float* o_parts_out, float* qk_out, float* pv_out,
    void* stream);
template void LaunchTFA<64, 128, 512, 128>(
    uint16_t* ffts, aclFloat16* q, aclFloat16* k, aclFloat16* v, aclFloat16* p_out, float* p_out_fp32,
    float* global_sum_out, float* exp_max_out, float* o_out, float* o_parts_out, float* qk_out, float* pv_out,
    void* stream);
// instantiate debug variant with intermediate checks enabled
template void LaunchTFA<64, 128, 128, 128, true>(
    uint16_t* ffts, aclFloat16* q, aclFloat16* k, aclFloat16* v, aclFloat16* p_out, float* p_out_fp32,
    float* global_sum_out, float* exp_max_out, float* o_out, float* o_parts_out, float* qk_out, float* pv_out,
    void* stream);
template void LaunchTFA<64, 128, 256, 128, true>(
    uint16_t* ffts, aclFloat16* q, aclFloat16* k, aclFloat16* v, aclFloat16* p_out, float* p_out_fp32,
    float* global_sum_out, float* exp_max_out, float* o_out, float* o_parts_out, float* qk_out, float* pv_out,
    void* stream);
template void LaunchTFA<64, 128, 512, 128, true>(
    uint16_t* ffts, aclFloat16* q, aclFloat16* k, aclFloat16* v, aclFloat16* p_out, float* p_out_fp32,
    float* global_sum_out, float* exp_max_out, float* o_out, float* o_parts_out, float* qk_out, float* pv_out,
    void* stream);

// Explicit instantiations for S0=128, HEAD=128
template void LaunchTFA<128, 128, 128, 128>(
    uint16_t* ffts, aclFloat16* q, aclFloat16* k, aclFloat16* v, aclFloat16* p_out, float* p_out_fp32,
    float* global_sum_out, float* exp_max_out, float* o_out, float* o_parts_out, float* qk_out, float* pv_out,
    void* stream);
template void LaunchTFA<128, 128, 256, 128>(
    uint16_t* ffts, aclFloat16* q, aclFloat16* k, aclFloat16* v, aclFloat16* p_out, float* p_out_fp32,
    float* global_sum_out, float* exp_max_out, float* o_out, float* o_parts_out, float* qk_out, float* pv_out,
    void* stream);
template void LaunchTFA<128, 128, 512, 128>(
    uint16_t* ffts, aclFloat16* q, aclFloat16* k, aclFloat16* v, aclFloat16* p_out, float* p_out_fp32,
    float* global_sum_out, float* exp_max_out, float* o_out, float* o_parts_out, float* qk_out, float* pv_out,
    void* stream);

template void LaunchTFA<128, 128, 2048, 128>(
    uint16_t* ffts, aclFloat16* q, aclFloat16* k, aclFloat16* v, aclFloat16* p_out, float* p_out_fp32,
    float* global_sum_out, float* exp_max_out, float* o_out, float* o_parts_out, float* qk_out, float* pv_out,
    void* stream);

template void LaunchTFA<128, 128, 8192, 128>(
    uint16_t* ffts, aclFloat16* q, aclFloat16* k, aclFloat16* v, aclFloat16* p_out, float* p_out_fp32,
    float* global_sum_out, float* exp_max_out, float* o_out, float* o_parts_out, float* qk_out, float* pv_out,
    void* stream);

// Debug instantiations for S0=128, HEAD=128
template void LaunchTFA<128, 128, 128, 128, true>(
    uint16_t* ffts, aclFloat16* q, aclFloat16* k, aclFloat16* v, aclFloat16* p_out, float* p_out_fp32,
    float* global_sum_out, float* exp_max_out, float* o_out, float* o_parts_out, float* qk_out, float* pv_out,
    void* stream);
template void LaunchTFA<128, 128, 256, 128, true>(
    uint16_t* ffts, aclFloat16* q, aclFloat16* k, aclFloat16* v, aclFloat16* p_out, float* p_out_fp32,
    float* global_sum_out, float* exp_max_out, float* o_out, float* o_parts_out, float* qk_out, float* pv_out,
    void* stream);
template void LaunchTFA<128, 128, 512, 128, true>(
    uint16_t* ffts, aclFloat16* q, aclFloat16* k, aclFloat16* v, aclFloat16* p_out, float* p_out_fp32,
    float* global_sum_out, float* exp_max_out, float* o_out, float* o_parts_out, float* qk_out, float* pv_out,
    void* stream);

template void LaunchTFA<128, 128, 2048, 128, true>(
    uint16_t* ffts, aclFloat16* q, aclFloat16* k, aclFloat16* v, aclFloat16* p_out, float* p_out_fp32,
    float* global_sum_out, float* exp_max_out, float* o_out, float* o_parts_out, float* qk_out, float* pv_out,
    void* stream);

template void LaunchTFA<128, 128, 8192, 128, true>(
    uint16_t* ffts, aclFloat16* q, aclFloat16* k, aclFloat16* v, aclFloat16* p_out, float* p_out_fp32,
    float* global_sum_out, float* exp_max_out, float* o_out, float* o_parts_out, float* qk_out, float* pv_out,
    void* stream);