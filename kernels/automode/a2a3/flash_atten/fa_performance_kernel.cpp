/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#include <acl/acl.h>
#include <pto/pto-inst.hpp>

#include "fa_performance_kernel.h"
#include <pto/npu/kernels/Pto_prefetch.hpp>
#if defined(__DAV_C220_CUBE__) || defined(__DAV_C220_VEC__)
#include <pto/npu/a2a3/custom/TSyncCVID.hpp>
#include <pto/npu/a2a3/custom/TSync_Custom.hpp>
#define UF_ENABLE 1
#elif defined(__DAV_C310_CUBE__) || defined(__DAV_C310_VEC__)
#include <pto/npu/a5/custom/TSyncCVID.hpp>
#include <pto/npu/a5/custom/TSync_Custom.hpp>
#define UF_ENABLE 1
#endif
#include "pto_macro_matmul.hpp"
#include "pto_macro_fa_softmax.hpp"
#include "pto_macro_fa_gu.hpp"
#include "multiBuffer.hpp"
using namespace std;
using namespace pto;
using namespace pto_auto;

#ifndef FFTS_BUFFER_FLAG_ENUM
#define FFTS_BUFFER_FLAG_ENUM
// Buffer flag values for FFTS pipeline coordination
enum FftsBufferFlag : uint32_t {
    BUF0_QK_READY,    // Buffer 0: QK data ready
    BUF0_SM_CONSUMED, // Buffer 0: Softmax consumed
    BUF1_SM_READY,    // Buffer 1: Softmax output ready
    BUF1_SV_CONSUMED, // Buffer 1: SV consumed
    UPDATE_READY,     // Update stage ready
    UPDATE_CONSUMED,  // Update stage consumed
    CV_BLOCK_END = 7, // CV comm slot block end (CV_COMM_CTRL reserved in TSyncCVID)
};
#endif

#define VEC_CORES 2
// -----------------------------------------------------------------------------
// Performance tuning knobs (high-level)
//
// The kernel is a cross-core pipeline (Cube + Vec) with explicit FIFOs:
//   QK (Cube):  compute_qk   -> qk_tile_fifo (fp32)
//   P  (Vec):   compute_p    -> p_tile_fifo  (fp16 x_exp) + l1_exp_max_ififo
//   PV (Cube):  compute_pv   -> pv_tile_fifo (fp32)
//   GU (Vec):   compute_gu   -> o_out (fp32) with running rescale/update
//
// Key knobs that impact throughput (see runTFA<> below):
// - CUBE_S0 / CUBE_S1: tile sizes for QK/PV cube matmuls (compute intensity vs. buffer pressure)
// - qkPreloadNum: pipeline warmup depth (more overlap vs. more L1 FIFO footprint)
// - *_TNBuffers: ping/pong depth for Mat tiles (overlap) and Vec tiles (latency hiding)
// - QKV_CV_FIFO / PV_CV_FIFO: FIFO depth between stages (avoid backpressure)
// -----------------------------------------------------------------------------

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

// Decide whether to block or signal consumption flags for a given tile index.
// Reverse dependency: notify one step before the corresponding wait within each sync period.
template <int FifoSize, int SyncPeriod>
AICORE inline bool should_wait_consumption(int sync_iter)
{
    static_assert(FifoSize >= 1, "CV FIFO size must be >= 1");
    constexpr int period = (SyncPeriod > 0) ? SyncPeriod : 1;
    static_assert(period >= 1, "CV FIFO consume sync period must be >= 1");
    if (sync_iter < static_cast<int>(FifoSize))
        return false;
    return (sync_iter % period) == 0;
}

template <int FifoSize, int SyncPeriod>
AICORE inline bool should_notify_consumption(int sync_iter)
{
    static_assert(FifoSize >= 1, "CV FIFO size must be >= 1");
    constexpr int period = (SyncPeriod > 0) ? SyncPeriod : 1;
    static_assert(period >= 1, "CV FIFO consume sync period must be >= 1");
    return ((sync_iter + 1) % period) == 0; // notify one tile earlier than the wait check
}

// Compute how many consumption notifications have not been waited on yet so we can drain them at kernel tail.
AICORE inline int pending_consumption_events(int tiles_processed, int fifo_size, int sync_period)
{
    if (tiles_processed <= 0 || sync_period <= 0 || fifo_size <= 0)
        return 0;

    const int notify_count = tiles_processed / sync_period; // notifications fire every sync_period tiles

    int wait_count = 0;
    if (tiles_processed > fifo_size) {
        const int last_iter = tiles_processed - 1;
        wait_count = (last_iter / sync_period) - ((fifo_size - 1) / sync_period); // waits start after FIFO is filled
        if (wait_count < 0)
            wait_count = 0;
    }

    int pending = notify_count - wait_count;
    if (pending < 0)
        pending = 0;

    const int max_pending = (fifo_size + sync_period - 1) / sync_period; // ceil(fifo_size / sync_period)
    return (pending > max_pending) ? max_pending : pending;
}

template <
    typename QKPipe, int S0, int HEAD_SIZE, int S1, int CUBE_S0, int CUBE_S1, int TILE_S1, bool INTERMEDIATE_CHECK,
    bool CAUSAL_MASK, typename TileMatQData, typename TileMatKData, typename TileQKData>
AICORE inline void compute_qk(
    QKPipe& qkPipe, int tile_id, int sub_tile_id, __gm__ half* q, __gm__ half* k, TileMatQData& qMatTile,
    TileMatKData& kMatTile, TileQKData& qkAccTile, int blk_idx)
{
    if constexpr (DAV_CUBE) {
        constexpr uint32_t Cube_S0 = CUBE_S0;
        constexpr uint32_t Cube_S1 = CUBE_S1;
        constexpr uint32_t Tile_S1 = TILE_S1;
        constexpr uint32_t kTileFactor = Tile_S1 / Cube_S1;
        constexpr uint32_t Cube_HEAD = HEAD_SIZE;

        constexpr int QKP_CV_FIFO = QKPipe::DataFiFo::fifoDepth;
        constexpr int CV_FIFO_CONS_SYNC_PERIOD = QKPipe::DataFiFo::fifoPeriod;
        static_assert(QKP_CV_FIFO >= 1, "QKP_CV_FIFO must be >= 1");
        static_assert(Tile_S1 % Cube_S1 == 0, "TILE_S1 must be divisible by CUBE_S1");

        const int s0_index = blk_idx * CUBE_S0;
        const int s1_index = tile_id * static_cast<int>(Tile_S1) + sub_tile_id * static_cast<int>(Cube_S1);
        const int sync_iter = tile_id;
        const bool should_wait_consume = should_wait_consumption<QKP_CV_FIFO, CV_FIFO_CONS_SYNC_PERIOD>(sync_iter);
        if constexpr (CAUSAL_MASK) {
            if (s1_index > s0_index) {
                if (sub_tile_id == 0 && should_wait_consume) {
                    qkPipe.prod.allocate(); // wait for SM consume data
                }
                if (sub_tile_id == static_cast<int>(kTileFactor) - 1) {
                    qkPipe.prod.record(); // notify for QK produce data
                }
                return;
            }
        }
        using GlobalDataK = GlobalTensor<
            half, pto::Shape<1, 1, 1, HEAD_SIZE, Cube_S1>, pto::Stride<1, 1, 1, 1, HEAD_SIZE>,
            Layout::DN>; // BNSD - (N, K) layout

        GlobalDataK kGlobal(k + s1_index * HEAD_SIZE);

        TLOAD(kMatTile, kGlobal);

#if UF_ENABLE
        pto_macro_matmul<Cube_S0, Cube_HEAD, Cube_S1>(qMatTile, kMatTile, qkAccTile, AccMode::InitFinalSum);
#else
        pto_macro_matmul<Cube_S0, Cube_HEAD, Cube_S1>(qMatTile, kMatTile, qkAccTile, AccMode::Init);
#endif

        bool isAllocate = (sub_tile_id == 0 && should_wait_consume);
        bool isRecord = (sub_tile_id == (static_cast<int>(kTileFactor) - 1));
        qkPipe.prod.setAllocateStatus(isAllocate);
        qkPipe.prod.setRecordStatus(isRecord);
        qkPipe.prod.setEntryOffset(sub_tile_id * Cube_S0 * Cube_S1 * sizeof(float));
        TPUSH(qkAccTile, qkPipe);
    }
}

template <
    typename PPipe, typename PVPipe, int S0, int HEAD_SIZE, int S1, int CUBE_S0, int CUBE_S1, int TILE_S1,
    bool INTERMEDIATE_CHECK, bool CAUSAL_MASK, typename TileMatPData, typename TileMatVData, typename TilePVData>
AICORE inline void compute_pv(
    PPipe& pPipe, PVPipe& pvPipe, int tile_id, int sub_tile_id, __gm__ half* v, TileMatPData& pMatTile,
    TileMatVData& vMatTile, TilePVData& pvAccTile, int blk_idx)
{
    constexpr uint32_t Cube_S0 = CUBE_S0;
    constexpr uint32_t Cube_S1 = CUBE_S1;
    constexpr uint32_t Tile_S1 = TILE_S1;
    constexpr uint32_t kTileFactor = Tile_S1 / Cube_S1;
    constexpr uint32_t Cube_HEAD = HEAD_SIZE;
    constexpr uint32_t TileElems = Cube_S0 * Tile_S1;
    constexpr int QKP_CV_FIFO = PVPipe::DataFiFo::fifoDepth;
    constexpr int CV_FIFO_CONS_SYNC_PERIOD = PVPipe::DataFiFo::fifoPeriod;
    static_assert(QKP_CV_FIFO >= 1, "QKP_CV_FIFO must be >= 1");
    static_assert(Tile_S1 % Cube_S1 == 0, "TILE_S1 must be divisible by CUBE_S1");

    const int s0_index = blk_idx * Cube_S0;
    const int s1_index = tile_id * static_cast<int>(Tile_S1) + sub_tile_id * static_cast<int>(Cube_S1);
    const int sync_iter = tile_id;
    const bool should_wait_consume = should_wait_consumption<QKP_CV_FIFO, CV_FIFO_CONS_SYNC_PERIOD>(sync_iter);
    const bool should_notify_consume = should_notify_consumption<QKP_CV_FIFO, CV_FIFO_CONS_SYNC_PERIOD>(sync_iter);
    const bool is_last_subtile = (sub_tile_id + 1 == static_cast<int>(kTileFactor));
    const bool next_will_be_skipped = (s1_index + static_cast<int>(Cube_S1)) > s0_index && CAUSAL_MASK;

    if constexpr (DAV_CUBE) {
        if constexpr (CAUSAL_MASK) {
            if (s1_index > s0_index) {
                if (sub_tile_id == 0) {
                    pPipe.cons.wait(); // wait for softmax produce data
                }
                if (sub_tile_id == static_cast<int>(kTileFactor) - 1 && should_notify_consume) {
                    pPipe.cons.free(); // notify SV consume data
                }
                return;
            }
        }

        using GlobalVT =
            GlobalTensor<half, pto::Shape<1, 1, 1, Cube_S1, HEAD_SIZE>, pto::Stride<1, 1, 1, HEAD_SIZE, 1>>;

        GlobalVT vLoad((__gm__ half*)(v + s1_index * HEAD_SIZE));
        TLOAD(vMatTile, vLoad);

        bool isWait = (sub_tile_id == 0);
        bool isFree = (sub_tile_id == static_cast<int>(kTileFactor) - 1 && should_notify_consume);
        pPipe.cons.setWaitStatus(isWait);
        pPipe.cons.setFreeStatus(isFree);
        pPipe.cons.setEntryOffset(sub_tile_id * Cube_S0 * Cube_S1 * sizeof(half));
        TPOP(pMatTile, pPipe);

#if UF_ENABLE
        const AccMode accMode =
            (sub_tile_id == 0) ?
                (is_last_subtile || next_will_be_skipped ? AccMode::InitFinalSum : AccMode::InitPartialSum) :
                (is_last_subtile || next_will_be_skipped ? AccMode::AccFinalSum : AccMode::AccPartialSum);
        pto_macro_matmul<Cube_S0, Cube_S1, Cube_HEAD>(pMatTile, vMatTile, pvAccTile, accMode);
#else
        const AccMode accMode = (sub_tile_id == 0) ? AccMode::Init : AccMode::Acc;
        pto_macro_matmul<Cube_S0, Cube_S1, Cube_HEAD>(pMatTile, vMatTile, pvAccTile, accMode);
#endif

        if (sub_tile_id == static_cast<int>(kTileFactor) - 1 || next_will_be_skipped) {
            pvPipe.prod.setAllocateStatus(should_wait_consume);
            pvPipe.prod.setRecordStatus(true);
            pvPipe.prod.setEntryOffset(0);
            TPUSH(pvAccTile, pvPipe);
        } // end loop
    }
}

template <
    typename QKPipe, typename PPipe, int S0, int HEAD_SIZE, int S1, int CUBE_S0, int CUBE_S1, int TILE_S1,
    bool INTERMEDIATE_CHECK, bool CAUSAL_MASK, typename TileDataF_T, typename TileDataH_T, typename ReduceTileF_T>
AICORE inline void compute_p(
    QKPipe& qkPipe, PPipe& pPipe, int tile_id, int row_slice, __gm__ float* exp_max_ififo, __gm__ float* global_sum_out,
    __gm__ float* exp_max_out, TileDataF_T& qkVecTile, TileDataH_T& x_expT, TileDataF_T& input_reduce_tmp,
    ReduceTileF_T& m1_local_max, ReduceTileF_T& l1_local_sum, ReduceTileF_T& m2_global_max,
    ReduceTileF_T& l2_global_sum, ReduceTileF_T& l1_exp_max_ififo, TileDataF_T& triu, int blk_idx)
{
    constexpr uint32_t Cube_S0 = CUBE_S0;
    constexpr uint32_t Cube_S1 = CUBE_S1;
    constexpr uint32_t Tile_S1 = TILE_S1;
    constexpr uint32_t kTileFactor = Tile_S1 / Cube_S1;
    constexpr uint32_t Vec_S0 = Cube_S0 / VEC_CORES / kTileFactor;
    constexpr int QKP_CV_FIFO = QKPipe::DataFiFo::fifoDepth;
    constexpr int CV_FIFO_CONS_SYNC_PERIOD = QKPipe::DataFiFo::fifoPeriod;
    const bool initFlag = (tile_id == 0);
    static_assert(QKP_CV_FIFO >= 1, "QKP_CV_FIFO must be >= 1");
    static_assert(Tile_S1 % Cube_S1 == 0, "TILE_S1 must be divisible by CUBE_S1");
    static_assert(Cube_S0 % (VEC_CORES * kTileFactor) == 0, "Vec rows must divide evenly across tile slices");
    if constexpr (DAV_VEC) {
        const size_t subblock_base_rows =
            static_cast<size_t>(Cube_S0 / VEC_CORES) * static_cast<size_t>(get_subblockid());
        const size_t row_offset = subblock_base_rows + static_cast<size_t>(row_slice * Vec_S0);
        const int s0_index = blk_idx * Cube_S0 + row_offset;
        const int s1_index = tile_id * static_cast<int>(Tile_S1);
        const int sync_iter = tile_id;
        const bool should_notify_consume = should_notify_consumption<QKP_CV_FIFO, CV_FIFO_CONS_SYNC_PERIOD>(sync_iter);

        bool isWait = (row_slice == 0);
        bool isFree = (row_slice == static_cast<int>(kTileFactor) - 1 && should_notify_consume);
        qkPipe.cons.setWaitStatus(isWait);
        qkPipe.cons.setFreeStatus(isFree);
        qkPipe.cons.setEntryOffset(row_offset * Cube_S1 * sizeof(float));
        TPOP(qkVecTile, qkPipe);

        // Extract per-slice views into the per-core reduce tiles so each slice writes into its row range
        using ReduceSliceTile = Tile<TileType::Vec, float, ReduceTileF_T::Rows, 1, BLayout::ColMajor, Vec_S0, 1>;

        ReduceSliceTile m1_local_max_slice;
        ReduceSliceTile l1_local_sum_slice;
        ReduceSliceTile m2_global_max_slice;
        ReduceSliceTile l2_global_sum_slice;
        ReduceSliceTile l1_exp_max_slice;

        TSUBVIEW(m1_local_max_slice, m1_local_max, row_slice * Vec_S0, 0);
        TSUBVIEW(l1_local_sum_slice, l1_local_sum, row_slice * Vec_S0, 0);
        TSUBVIEW(m2_global_max_slice, m2_global_max, row_slice * Vec_S0, 0);
        TSUBVIEW(l2_global_sum_slice, l2_global_sum, row_slice * Vec_S0, 0);
        TSUBVIEW(l1_exp_max_slice, l1_exp_max_ififo, row_slice * Vec_S0, 0);

        // Extract current slice state from full-length reduce tiles
        if (initFlag) {
            pto_macro_fa_softmax<true, HEAD_SIZE, CAUSAL_MASK>(
                x_expT, qkVecTile, m1_local_max_slice, l1_local_sum_slice, m2_global_max_slice, l2_global_sum_slice,
                l1_exp_max_slice, input_reduce_tmp, qkVecTile, triu, s0_index, s1_index);
        } else {
            pto_macro_fa_softmax<false, HEAD_SIZE, CAUSAL_MASK>(
                x_expT, qkVecTile, m1_local_max_slice, l1_local_sum_slice, m2_global_max_slice, l2_global_sum_slice,
                l1_exp_max_slice, input_reduce_tmp, qkVecTile, triu, s0_index, s1_index);
        }

        const bool should_wait_consume = should_wait_consumption<QKP_CV_FIFO, CV_FIFO_CONS_SYNC_PERIOD>(sync_iter);
        bool isAllocate = (row_slice == 0 && should_wait_consume);
        const bool isRecord = (row_slice == (static_cast<int>(kTileFactor) - 1));
        pPipe.prod.setAllocateStatus(isAllocate);
        pPipe.prod.setRecordStatus(isRecord);
        pPipe.prod.setEntryOffset(row_offset * Cube_S1 * sizeof(half));
        TPUSH(x_expT, pPipe);

        if constexpr (INTERMEDIATE_CHECK) {
            // On the final row_slice, emit the exp_max for this subblock only (Cube_S0 / VEC_CORES rows)
            if (row_slice == static_cast<int>(kTileFactor) - 1) {
                constexpr uint32_t SubblockRows = Cube_S0 / VEC_CORES;
                using GlobalPMaxFloatSub =
                    GlobalTensor<float, pto::Shape<1, 1, 1, 1, SubblockRows>, pto::Stride<1, 1, 1, Cube_S0, 1>>;
                using ExpMaxSub = Tile<TileType::Vec, float, 1, SubblockRows, BLayout::RowMajor, 1, SubblockRows>;
                const size_t base_elems_pmax =
                    static_cast<size_t>(tile_id % QKP_CV_FIFO) * static_cast<size_t>(Cube_S0) + subblock_base_rows;
                __gm__ float* p_ptr_fp32 = exp_max_ififo + base_elems_pmax;
                GlobalPMaxFloatSub pMaxGlobal(p_ptr_fp32);
                ExpMaxSub l1_exp_max_rowmajor;
                TRESHAPE(l1_exp_max_rowmajor, l1_exp_max_ififo);
                TSTORE(pMaxGlobal, l1_exp_max_rowmajor);
            }
        }
    }
}

template <
    typename PVPipe, int S0, int HEAD_SIZE, int S1, int CUBE_S0, int TILE_S1, int PV_CV_FIFO,
    int CV_FIFO_CONS_SYNC_PERIOD, bool INTERMEDIATE_CHECK, bool CAUSAL_MASK, Phase GU_Phase, typename TileOutT,
    typename ReduceTileF_T>
AICORE inline void compute_gu(
    PVPipe& pvPipe, int tile_id, int num_tiles, __gm__ float* o_out, __gm__ float* o_parts_out, TileOutT& runningOTile,
    TileOutT& pvVecTile, ReduceTileF_T& l1_exp_max_ififo, ReduceTileF_T& l2_global_sum)
{
    constexpr uint32_t Cube_S0 = CUBE_S0;
    constexpr uint32_t Vec_S0 = Cube_S0 / VEC_CORES;

    using GlobalDataPV_VEC =
        GlobalTensor<float, pto::Shape<1, 1, 1, Vec_S0, HEAD_SIZE>, pto::Stride<1, 1, 1, HEAD_SIZE, 1>>;
    if constexpr (DAV_VEC) {
        const size_t subblock_base_rows =
            static_cast<size_t>(Cube_S0 / VEC_CORES) * static_cast<size_t>(get_subblockid());
        // softamx output and gu input buffer reuse
        const bool should_notify_consume = should_notify_consumption<PV_CV_FIFO, CV_FIFO_CONS_SYNC_PERIOD>(tile_id);
        pvPipe.cons.setWaitStatus(true);
        pvPipe.cons.setFreeStatus(should_notify_consume);
        pvPipe.cons.setEntryOffset(subblock_base_rows * HEAD_SIZE * sizeof(float));

        if (tile_id == 0) {
            TPOP(runningOTile, pvPipe);

            if constexpr (CAUSAL_MASK) {
                if (tile_id == num_tiles - 1) {
                    pto_macro_fa_gu_single_and_last_tile(runningOTile, l2_global_sum);
                }
            }
        } else {
            TPOP(pvVecTile, pvPipe);

            if (tile_id < num_tiles - 1) {
                pto_macro_fa_gu<ReduceTileF_T, TileOutT>(runningOTile, pvVecTile, l1_exp_max_ififo);
            } else {
                pto_macro_fa_gu_last<ReduceTileF_T, TileOutT>(runningOTile, pvVecTile, l1_exp_max_ififo, l2_global_sum);
            }
        }
        if constexpr (GU_Phase == Phase::Epilogue) {
            using GlobalOutT = GlobalTensor<
                float, pto::Shape<1, 1, 1, CUBE_S0 / VEC_CORES, HEAD_SIZE>, pto::Stride<1, 1, 1, HEAD_SIZE, 1>>;
            GlobalOutT outGlobal((__gm__ float*)(o_out + subblock_base_rows * HEAD_SIZE));
            TSTORE(outGlobal, runningOTile);
        }
    }
}

template <
    int S0, int HEAD_SIZE, int S1, int CUBE_S0, int CUBE_S1, int TILE_S1, int QK_PRELOAD, int CV_FIFO_SIZE,
    bool INTERMEDIATE_CHECK, bool CAUSAL_MASK, int CV_FIFO_CONS_SYNC_PERIOD>
__global__ AICORE void runTFA(
    __gm__ uint64_t* ffts_addr, __gm__ half* q, __gm__ half* k, __gm__ half* v, __gm__ half* p_tile_fifo,
    __gm__ float* exp_max_ififo, __gm__ float* global_sum_out, __gm__ float* exp_max_out, __gm__ float* o_out,
    __gm__ float* o_parts_out, __gm__ float* qk_tile_fifo, __gm__ float* pv_tile_fifo, __gm__ uint8_t* cv_comm_buf,
    __gm__ uint8_t* profile_buf)
{
    uint64_t tStart = get_sys_cnt();

    set_ffts_base_addr((uint64_t)ffts_addr);

    // Rename dimensions for clarity: S0 (rows total), Cube_S0 (per-block rows), S1 (cols), HEAD_SIZE (inner)
    constexpr uint32_t Cube_S0 = CUBE_S0;
    constexpr uint32_t block_rows = S0 / CUBE_S0;
    constexpr uint32_t Cube_S1 = CUBE_S1; // per-tile S1 chunk
    constexpr uint32_t Tile_S1 = TILE_S1; // logical tile along S1
    static_assert(Tile_S1 % Cube_S1 == 0, "TILE_S1 must be divisible by CUBE_S1");
    constexpr uint32_t kTileFactor = Tile_S1 / Cube_S1; // sub-tiles per TILE_S1
    constexpr uint32_t Cube_HEAD = HEAD_SIZE;
    constexpr uint32_t Vec_S0 = Cube_S0 / VEC_CORES / kTileFactor;
    constexpr uint32_t VecGuRows = Cube_S0 / VEC_CORES;
    static_assert(Cube_S0 % (VEC_CORES * kTileFactor) == 0, "Vec rows must divide evenly across tile slices");

    // --------------------------
    // Tuning knobs (pipeline)
    //
    // qkPreloadNum controls how many (QK -> P) tiles we warm up before entering the steady-state loop.
    // - Larger preload improves overlap (Cube/VEC concurrency) for long S1.
    // - Larger preload increases FIFO footprint (qkGlobalTensorNBuffers / pvGlobalTensorNBuffers /
    // guGlobalTensorNBuffers).
    constexpr uint32_t qkPreloadNum = QK_PRELOAD;

    // Buffer counts for optional double-buffering (default 1)
    // - srcVecTNBuffers/xexpVecTNBuffers: Vec ping-pong for QK load and x_exp output
    // - *MatTNBuffers: L1 ping-pong for Cube stage (K/P/V)
    // Keep these small (1-2) unless you have measured stall bubbles that require deeper buffering.
    constexpr uint32_t srcVecTNBuffers = 2;
    constexpr uint32_t xexpVecTNBuffers = 2;
    constexpr uint32_t outOTileNBuffers = 2;
    constexpr uint32_t qkPvOverlapNBuffers = 1;
    constexpr uint32_t qMatTNBuffers = 1;
    constexpr uint32_t kMatTNBuffers = 2;
    constexpr uint32_t pMatTNBuffers = 2;
    constexpr uint32_t vMatTNBuffers = 2;
    constexpr uint32_t qkp_tile_fifo_size = CV_FIFO_SIZE;
    constexpr uint32_t pv_tile_fifo_size = CV_FIFO_SIZE;
    static_assert(qkPreloadNum >= 1, "qkPreloadNum must be >= 1");
    static_assert(CV_FIFO_CONS_SYNC_PERIOD >= 1, "CV_FIFO_CONS_SYNC_PERIOD must be >= 1");
    static_assert((qkPreloadNum > 1) || (kTileFactor == 1), "qkPreloadNum must be > 1 unless kTileFactor == 1");

    // Define tile types for first QK matmul
    using TileMatQData =
        Tile<TileType::Mat, half, Cube_S0, HEAD_SIZE, BLayout::ColMajor, Cube_S0, HEAD_SIZE, SLayout::RowMajor, 512>;
    using TileMatKData =
        Tile<TileType::Mat, half, HEAD_SIZE, Cube_S1, BLayout::RowMajor, HEAD_SIZE, Cube_S1, SLayout::ColMajor, 512>;
    // Accumulator rows must match Cube_S0 (per-block rows), not logical S0
    using TileQKData = TileAcc<float, Cube_S0, Cube_S1, Cube_S0, Cube_S1>;

    TileMatQData qMatTile;
    TileQKData qkAccTile;

    // Define tile types for second PV matmul
    using TileMatPData =
        Tile<TileType::Mat, half, Cube_S0, Cube_S1, BLayout::ColMajor, Cube_S0, Cube_S1, SLayout::RowMajor, 512>;
    using TileMatVData =
        Tile<TileType::Mat, half, Cube_S1, HEAD_SIZE, BLayout::ColMajor, Cube_S1, HEAD_SIZE, SLayout::RowMajor, 512>;
    using TilePVData = TileAcc<float, Cube_S0, HEAD_SIZE, Cube_S0, HEAD_SIZE>;

    TilePVData pvAccTile;

    // Define tile types for FA softmax P computation
    // UB offsets for softmax tiles
    // Define per-tile vector tiles sized to Cube_S1
    using TileDataF_T = Tile<TileType::Vec, float, Vec_S0, Tile_S1, BLayout::RowMajor, Vec_S0, Tile_S1>;
    using TileDataH_T = Tile<TileType::Vec, half, Vec_S0, Tile_S1, BLayout::RowMajor, Vec_S0, Tile_S1>;
    constexpr uint32_t SubblockRows = Cube_S0 / VEC_CORES;
    // Reduce tiles cover one vector core's rows (Cube_S0 / VEC_CORES); slices are extracted per row_slice
    using ReduceTileF_T = Tile<TileType::Vec, float, SubblockRows, 1, BLayout::ColMajor, SubblockRows, 1>;

    ReduceTileF_T m1_local_max;
    TileDataF_T input_reduce_tmp;
    TileDataF_T triu;
    ReduceTileF_T l1_local_sum;
    ReduceTileF_T m2_global_max;
    ReduceTileF_T l2_global_sum;
    ReduceTileF_T l1_exp_max_ififo[qkp_tile_fifo_size];

    using TileOutGuT = Tile<TileType::Vec, float, VecGuRows, HEAD_SIZE, BLayout::RowMajor, VecGuRows, HEAD_SIZE>;
    TileOutGuT pvVecTile;
    TileOutGuT runningOTile;

    // block offset for logical S0
#if defined(__DAV_C220_CUBE__) || defined(__DAV_C220_VEC__) // A5 defined macro, don't need to reassign
    const int block_idx = get_block_idx();
#endif
    const int block_offset_rows = block_idx * static_cast<int>(Cube_S0);

    constexpr bool use_cv_comm = (!INTERMEDIATE_CHECK) && (block_rows >= static_cast<uint32_t>(pto::kCvMaxCores));
    int comm_slot = block_idx;

    if constexpr (use_cv_comm) {
        comm_slot = pto::TSYNC_CVID(block_idx, cv_comm_buf);
    }
    __gm__ uint64_t* profile_entry = nullptr;
    if (profile_buf != nullptr) {
        std::size_t profile_block_base = static_cast<std::size_t>(block_idx) * kFaProfileBytesPerBlock;
        std::size_t profile_offset = profile_block_base;
        if constexpr (DAV_VEC) {
            profile_offset +=
                (static_cast<std::size_t>(get_subblockid()) + 1U) * 1024U; // vec subblock 0/1 use 2nd/3rd KB
        }
        profile_entry = reinterpret_cast<__gm__ uint64_t*>(profile_buf + profile_offset);
        profile_entry[0] = tStart;
    }
    const size_t p_fifo_block_stride =
        static_cast<size_t>(qkp_tile_fifo_size) * static_cast<size_t>(Cube_S0) * static_cast<size_t>(Tile_S1);
    const size_t p_max_fifo_block_stride = static_cast<size_t>(qkp_tile_fifo_size) * static_cast<size_t>(Cube_S0);
    const size_t qk_fifo_block_stride = p_fifo_block_stride;
    const size_t pv_fifo_block_stride =
        static_cast<size_t>(pv_tile_fifo_size) * static_cast<size_t>(Cube_S0) * static_cast<size_t>(HEAD_SIZE);

    __gm__ half* q_block = q + block_offset_rows * HEAD_SIZE;
    __gm__ half* p_tile_fifo_block = p_tile_fifo + static_cast<size_t>(comm_slot) * p_fifo_block_stride;
    __gm__ float* exp_max_ififo_block = exp_max_ififo + static_cast<size_t>(comm_slot) * p_max_fifo_block_stride;
    __gm__ float* global_sum_block = global_sum_out + block_offset_rows;
    __gm__ float* exp_max_block = exp_max_out + block_offset_rows;
    __gm__ float* o_out_block = o_out + static_cast<size_t>(block_offset_rows) * static_cast<size_t>(HEAD_SIZE);
    __gm__ float* o_parts_block = o_parts_out + static_cast<size_t>(block_offset_rows) * static_cast<size_t>(HEAD_SIZE);
    __gm__ float* qk_tile_fifo_block = qk_tile_fifo + static_cast<size_t>(comm_slot) * qk_fifo_block_stride;
    __gm__ float* pv_tile_fifo_block = pv_tile_fifo + static_cast<size_t>(comm_slot) * pv_fifo_block_stride;

    constexpr int num_tiles_s1 = S1 / Tile_S1;

    constexpr uint8_t FiFoDepth = CV_FIFO_SIZE;
    constexpr uint8_t FiFoSyncT = CV_FIFO_CONS_SYNC_PERIOD;
    using QKPipe = TMPipe<
        BUF0_QK_READY, FIFOType::GM_FIFO, FiFoDepth, FiFoSyncT, TileQKData, TileDataF_T, UF_ENABLE ? true : false, 0>;
    QKPipe qkPipe(qk_tile_fifo_block);

    // pFiFo, pProd, pCons
    using PPipe = TMPipe<BUF1_SM_READY, FIFOType::GM_FIFO, FiFoDepth, FiFoSyncT, TileDataH_T, TileMatPData, false, 0>;
    PPipe pPipe(p_tile_fifo_block);

    // pvFiFo, pvProd, pvCons
    using PVPipe = TMPipe<
        UPDATE_READY, FIFOType::GM_FIFO, FiFoDepth, FiFoSyncT, TilePVData, TileOutGuT, UF_ENABLE ? true : false, 0>;
    PVPipe pvPipe(pv_tile_fifo_block);

    // ---------------------------------- CUBE COMPUTATIONS BEGIN -----------------------------------------------------
    // 1. pre computation: qk only, 2. main loop: qk and pv, 3. post computation: pv only
    if constexpr (DAV_CUBE) {
        using GlobalDataQ =
            GlobalTensor<half, pto::Shape<1, 1, 1, Cube_S0, HEAD_SIZE>, pto::Stride<1, 1, 1, HEAD_SIZE, 1>>;
        GlobalDataQ qGlobal(q_block);
        TLOAD(qMatTile, qGlobal);

        // QK pre-computation (tile_id based)
        // nested double-buffered loop:
        // if the inner loops iteration > 1, the inner loop gets unrolled
        // otherwise we unroll the outerloop
        MultiBuffered<kMatTNBuffers> mb;
        using InnerMultiBuffered = MultiBuffered<kMatTNBuffers>::NestedLoopInvoker<Range<kTileFactor>>;
        mb.loop<Range<qkPreloadNum, kTileFactor>>([&](auto ctxOuter, InnerMultiBuffered inner) {
            int tile_id = ctxOuter.iter;
            inner.loop([&](auto ctxInner) {
                int sub_tile = ctxInner.iter;
                TileMatKData kMatTile;
                qkPipe.prod.setTileId(tile_id, sub_tile);
                compute_qk<QKPipe, S0, HEAD_SIZE, S1, CUBE_S0, CUBE_S1, Tile_S1, INTERMEDIATE_CHECK, CAUSAL_MASK>(
                    qkPipe, tile_id, sub_tile, q_block, k, qMatTile, kMatTile, qkAccTile, block_idx);
            });
        });

        // we don't do double buffering for the main loop because it hurts the performance
        TileMatKData kMatTile;
        TileMatPData pMatTile;
        TileMatVData vMatTile;
        constexpr int NumStages = 2;
        MultiStaged<NumStages> qk_pv_stages;
        for (int tile_id = 0; tile_id < num_tiles_s1 - qkPreloadNum; ++tile_id) {
            for (int sub_tile = 0; sub_tile < kTileFactor; ++sub_tile) {
                qk_pv_stages.run(
                    [&]() {
                        qkPipe.prod.setTileId(tile_id + qkPreloadNum, sub_tile);
                        compute_qk<
                            QKPipe, S0, HEAD_SIZE, S1, CUBE_S0, CUBE_S1, Tile_S1, INTERMEDIATE_CHECK, CAUSAL_MASK>(
                            qkPipe, tile_id + qkPreloadNum, sub_tile, q_block, k, qMatTile, kMatTile, qkAccTile,
                            block_idx);
                    },
                    [&]() {
                        pPipe.cons.setTileId(tile_id, sub_tile);
                        pvPipe.prod.setTileId(tile_id, sub_tile);
                        compute_pv<
                            PPipe, PVPipe, S0, HEAD_SIZE, S1, CUBE_S0, CUBE_S1, Tile_S1, INTERMEDIATE_CHECK,
                            CAUSAL_MASK>(pPipe, pvPipe, tile_id, sub_tile, v, pMatTile, vMatTile, pvAccTile, block_idx);
                    });
            }
        }

        mb.loop<Range<qkPreloadNum, kTileFactor>>([&](auto ctxOuter, auto inner) {
            int tile_id = ctxOuter.iter;
            inner.loop([&](auto ctxInner) {
                int sub_tile = ctxInner.iter;
                TileMatVData vMatTile;
                TileMatPData pMatTile;
                pPipe.cons.setTileId(tile_id + num_tiles_s1 - qkPreloadNum, sub_tile);
                pvPipe.prod.setTileId(tile_id + num_tiles_s1 - qkPreloadNum, sub_tile);
                compute_pv<
                    PPipe, PVPipe, S0, HEAD_SIZE, S1, CUBE_S0, CUBE_S1, Tile_S1, INTERMEDIATE_CHECK, CAUSAL_MASK>(
                    pPipe, pvPipe, tile_id + num_tiles_s1 - qkPreloadNum, sub_tile, v, pMatTile, vMatTile, pvAccTile,
                    block_idx);
            });
        });
    }
    // ---------------------------------- CUBE COMPUTATIONS END -----------------------------------------------------

    // ---------------------------------- VEC COMPUTATIONS BEGIN -----------------------------------------------------
    if constexpr (DAV_VEC) {
        MultiBuffered<kMatTNBuffers> mb;
        mb.loop<Range<qkPreloadNum, kTileFactor>>([&](auto ctxOuter, auto inner) {
            int tile_id = ctxOuter.iter;
            inner.loop([&](auto ctxInner) {
                int row_slice = ctxInner.iter;
                // Init only on the very first S1 tile; row_slice partitions rows within that tile
                TileDataF_T qkVecTile;
                TileDataH_T x_expT;
                pPipe.prod.setTileId(tile_id, row_slice);
                qkPipe.cons.setTileId(tile_id, row_slice);
                compute_p<QKPipe, PPipe, S0, HEAD_SIZE, S1, CUBE_S0, CUBE_S1, Tile_S1, INTERMEDIATE_CHECK, CAUSAL_MASK>(
                    qkPipe, pPipe, tile_id, row_slice, exp_max_ififo_block, global_sum_block, exp_max_block, qkVecTile,
                    x_expT, input_reduce_tmp, m1_local_max, l1_local_sum, m2_global_max, l2_global_sum,
                    l1_exp_max_ififo[tile_id % qkp_tile_fifo_size], triu, block_idx);
            });
        });

        TileOutGuT pvVecTile;

        for (int tile_id = 0; tile_id < num_tiles_s1 - qkPreloadNum; ++tile_id) {
            mb.loop<Range<kTileFactor>>([&](auto ctx) {
                TileDataF_T qkVecTile;
                TileDataH_T x_expT;
                pPipe.prod.setTileId(tile_id + qkPreloadNum, ctx.iter);
                qkPipe.cons.setTileId(tile_id + qkPreloadNum, ctx.iter);
                compute_p<QKPipe, PPipe, S0, HEAD_SIZE, S1, CUBE_S0, CUBE_S1, Tile_S1, INTERMEDIATE_CHECK, CAUSAL_MASK>(
                    qkPipe, pPipe, tile_id + qkPreloadNum, ctx.iter, exp_max_ififo_block, global_sum_block,
                    exp_max_block, qkVecTile, x_expT, input_reduce_tmp, m1_local_max, l1_local_sum, m2_global_max,
                    l2_global_sum, l1_exp_max_ififo[(tile_id + qkPreloadNum) % qkp_tile_fifo_size], triu, block_idx);
            });

            pvPipe.cons.setTileId(tile_id, -1);
            compute_gu<
                PVPipe, S0, HEAD_SIZE, S1, CUBE_S0, Tile_S1, pv_tile_fifo_size, CV_FIFO_CONS_SYNC_PERIOD,
                INTERMEDIATE_CHECK, CAUSAL_MASK, Phase::Main>(
                pvPipe, tile_id, num_tiles_s1, o_out_block, o_parts_block, runningOTile, pvVecTile,
                l1_exp_max_ififo[tile_id % qkp_tile_fifo_size], l2_global_sum);
        }
        mb.loop<Range<qkPreloadNum>, 0, 2>([&](auto ctx) {
            TileOutGuT pvVecTile;
            pvPipe.cons.setTileId(ctx.iter + num_tiles_s1 - qkPreloadNum, -1);
            compute_gu<
                PVPipe, S0, HEAD_SIZE, S1, CUBE_S0, Tile_S1, pv_tile_fifo_size, CV_FIFO_CONS_SYNC_PERIOD,
                INTERMEDIATE_CHECK, CAUSAL_MASK, ctx.phase>(
                pvPipe, ctx.iter + num_tiles_s1 - qkPreloadNum, num_tiles_s1, o_out_block, o_parts_block, runningOTile,
                pvVecTile, l1_exp_max_ififo[(ctx.iter + num_tiles_s1 - qkPreloadNum) % qkp_tile_fifo_size],
                l2_global_sum);
        });
    }
    // ---------------------------------- VEC COMPUTATIONS END -----------------------------------------------------

    const int pending_qk_sm_consumed =
        pending_consumption_events(num_tiles_s1, static_cast<int>(qkp_tile_fifo_size), CV_FIFO_CONS_SYNC_PERIOD);
    const int pending_sv_consumed = pending_qk_sm_consumed; // same schedule and FIFO settings
    const int pending_update_consumed =
        pending_consumption_events(num_tiles_s1, static_cast<int>(qkp_tile_fifo_size), CV_FIFO_CONS_SYNC_PERIOD);

    if constexpr (DAV_CUBE) {
        for (int i = 0; i < pending_qk_sm_consumed; ++i)
            qkPipe.prod.allocate();
        for (int i = 0; i < pending_update_consumed; ++i)
            pvPipe.prod.allocate();
#ifdef __DAV_C220_CUBE__
        wait_flag_dev(CV_BLOCK_END); // wait for vector done all reading
#endif
    }

    if constexpr (DAV_VEC) {
        for (int i = 0; i < pending_sv_consumed; ++i)
            pPipe.prod.allocate();
#ifdef __DAV_C220_VEC__
        ffts_cross_core_sync(PIPE_MTE2, _getFFTSMsg(CV_CORE_SYNC, CV_BLOCK_END)); // cube can exit CV comm now
#endif
    }

    uint64_t tEnd = get_sys_cnt();
    if (profile_entry != nullptr) {
        profile_entry[1] = tEnd;
    }
#ifdef _DEBUG
    if constexpr (DAV_CUBE) {
        cce::printf(
            "Core %d Cube Block %d, Start @%d End @%d (%d us)\n", get_coreid(), block_idx, int(tStart), int(tEnd),
            int(tEnd - tStart) * 20 / 1000);
    } else {
        cce::printf(
            "Core %d Vec Block %d, SubBlock %d, Start @%d End @%d (%d us)\n", get_coreid(), block_idx,
            int(get_subblockid()), int(tStart), int(tEnd), int(tEnd - tStart) * 20 / 1000);
    }
#endif
}

// Empty kernel to warm up cores
__global__ AICORE __attribute__((aic)) void warmup_kernel() {}

// Host wrapper
template <
    int S0, int HEAD_SIZE, int S1, int CUBE_S0, int CUBE_S1, int TILE_S1, int QK_PRELOAD, int CV_FIFO_SIZE,
    bool INTERMEDIATE_CHECK, bool CAUSAL_MASK, int CV_FIFO_CONS_SYNC_PERIOD>
void LaunchTFA(
    uint16_t* ffts, aclFloat16* q, aclFloat16* k, aclFloat16* v, aclFloat16* p_tile_fifo, float* exp_max_ififo,
    float* global_sum_out, float* exp_max_out, float* o_out, float* o_parts_out, float* qk_tile_fifo,
    float* pv_tile_fifo, uint8_t* profile_data, aclrtStream stream, uint8_t* cv_comm_buf)
{
    static_assert(S0 % CUBE_S0 == 0, "S0 must be divisible by CUBE_S0");
    constexpr uint32_t block_rows = S0 / CUBE_S0;

#if defined(__DAV_C220_CUBE__) || defined(__DAV_C220_VEC__)
    // Warm up all cores first, then prefetch q/k/v into L2
    warmup_kernel<<<24, nullptr, stream>>>();

    const uint64_t tensor_elems = static_cast<uint64_t>(S0) * static_cast<uint64_t>(HEAD_SIZE);
    const uint64_t tensor_bytes = tensor_elems * sizeof(half);
    constexpr bool kPrefetchUseSdma = true; // simulation cannot use sdma
    constexpr int kPrefetchAivCores = 40;   // only used when kPrefetchUseSdma is false

    if constexpr (kPrefetchUseSdma) {
        PTO_PREFETCH((__gm__ void*)q, tensor_bytes, stream);
        PTO_PREFETCH((__gm__ void*)k, tensor_bytes, stream);
        PTO_PREFETCH((__gm__ void*)v, tensor_bytes, stream);
    } else {
        PTO_PREFETCH<false, kPrefetchAivCores>((__gm__ void*)q, tensor_bytes, stream);
        PTO_PREFETCH<false, kPrefetchAivCores>((__gm__ void*)k, tensor_bytes, stream);
        PTO_PREFETCH<false, kPrefetchAivCores>((__gm__ void*)v, tensor_bytes, stream);
    }
#endif

    runTFA<
        S0, HEAD_SIZE, S1, CUBE_S0, CUBE_S1, TILE_S1, QK_PRELOAD, CV_FIFO_SIZE, INTERMEDIATE_CHECK, CAUSAL_MASK,
        CV_FIFO_CONS_SYNC_PERIOD><<<block_rows, nullptr, stream>>>(
        (__gm__ uint64_t*)ffts, (half*)q, (half*)k, (half*)v, (half*)p_tile_fifo, exp_max_ififo, global_sum_out,
        exp_max_out, o_out, o_parts_out, qk_tile_fifo, pv_tile_fifo, cv_comm_buf, profile_data);
}

// Backward-compatible overload without profiling buffer
template <
    int S0, int HEAD_SIZE, int S1, int CUBE_S0, int CUBE_S1, int TILE_S1, int QK_PRELOAD, int CV_FIFO_SIZE,
    bool INTERMEDIATE_CHECK, bool CAUSAL_MASK, int CV_FIFO_CONS_SYNC_PERIOD>
void LaunchTFA(
    uint16_t* ffts, aclFloat16* q, aclFloat16* k, aclFloat16* v, aclFloat16* p_tile_fifo, float* exp_max_ififo,
    float* global_sum_out, float* exp_max_out, float* o_out, float* o_parts_out, float* qk_tile_fifo,
    float* pv_tile_fifo, aclrtStream stream, uint8_t* cv_comm_buf)
{
    LaunchTFA<
        S0, HEAD_SIZE, S1, CUBE_S0, CUBE_S1, TILE_S1, QK_PRELOAD, CV_FIFO_SIZE, INTERMEDIATE_CHECK, CAUSAL_MASK,
        CV_FIFO_CONS_SYNC_PERIOD>(
        ffts, q, k, v, p_tile_fifo, exp_max_ififo, global_sum_out, exp_max_out, o_out, o_parts_out, qk_tile_fifo,
        pv_tile_fifo, nullptr, stream, cv_comm_buf);
}

#include "generated_cases.h"

#define INSTANTIATE_TFA(S0, HEAD, S1, CUBE_S0, CUBE_S1, TILE_S1, QK_PRELOAD, CAUSAL_MASK)                              \
    template void LaunchTFA<                                                                                           \
        S0, HEAD, S1, CUBE_S0, CUBE_S1, TILE_S1, QK_PRELOAD, kFaCvFifoSize, false, CAUSAL_MASK,                        \
        kFaCvFifoConsSyncPeriod>(                                                                                      \
        ate void LaunchTFA<                                                                                            \
            S0, HEAD, S1, CUBE_S0, CUBE_S1, TILE_S1, QK_PRELOAD, kFaCvFifoSize, true, CAUSAL_MASK,                     \
            kFaCvFifoConsSyncPeriod>(                                                                                  \
            uint16_t * ffts, aclFloat16 * q, aclFloat16 * k, aclFloat16 * v, aclFloat16 * p_out, float* p_out_fp32,    \
            uint16_t* ffts, aclFloat16* q, aclFloat16* k, aclFloat16* v, aclFloat16* p_out, float* p_out_fp32,         \
            float* global_sum_out, float* exp_max_out, float* o_out, float* o_parts_out, float* qk_out, float* pv_out, \
            uint8_t* profile_data, aclrtStream stream, uint8_t* cv_comm_buf);                                          \
        template void LaunchTFA<                                                                                       \
            S0, HEAD, S1, CUBE_S0, CUBE_S1, TILE_S1, QK_PRELOAD, kFaCvFifoSize, false, CAUSAL_MASK,                    \
            kFaCvFifoConsSyncPeriod>(                                                                                  \
            uint16_t * ffts, aclFloat16 * q, aclFloat16 * k, aclFloat16 * v, aclFloat16 * p_out, float* p_out_fp32,    \
            float* global_sum_out, float* exp_max_out, float* o_out, float* o_parts_out, float* qk_out, float* pv_out, \
            aclrtStream stream, uint8_t* cv_comm_buf);                                                                 \
        templ float* global_sum_out, float* exp_max_out, float* o_out, float* o_parts_out, float* qk_out,              \
        float* pv_out, uint8_t* profile_data, aclrtStream stream, uint8_t* cv_comm_buf);                               \
    template void LaunchTFA<                                                                                           \
        S0, HEAD, S1, CUBE_S0, CUBE_S1, TILE_S1, QK_PRELOAD, kFaCvFifoSize, true, CAUSAL_MASK,                         \
        kFaCvFifoConsSyncPeriod>(                                                                                      \
        uint16_t * ffts, aclFloat16 * q, aclFloat16 * k, aclFloat16 * v, aclFloat16 * p_out, float* p_out_fp32,        \
        float* global_sum_out, float* exp_max_out, float* o_out, float* o_parts_out, float* qk_out, float* pv_out,     \
        aclrtStream stream, uint8_t* cv_comm_buf);

TFA_FOR_EACH_CASE(INSTANTIATE_TFA)

#undef INSTANTIATE_TFA