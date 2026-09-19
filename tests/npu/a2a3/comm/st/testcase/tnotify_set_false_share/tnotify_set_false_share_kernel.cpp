/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

// =============================================================================
// MFE: NotifyOp::Set dcci write-backs the whole 64-byte GM cache line
//
// ISA under test: include/pto/comm/a2a3/TNotify.hpp NotifyOp::Set.
// Old path: dcci + scalar store + dcci — write-backs the whole 64 B line.
// Patched Set: ld_dev + st_atomic SUM(value - old). Packed stride=1 must pass.
//
// Protocol: every rank Sets one slot in *rank 0's* HCCL window, then rank 0 reads.
//   slots[rank * stride] := 1000 + rank
//
// stride=1 (packed) — P=2 diagram, one 64 B line:
//
//   byte  0        4        8                               64
//        [ slot0  | slot1  |  unused padding …              ]
//          rank0     rank1
//          1000      1001     both words share the line
//
// Race (P=2; P=4 is the same with four writers on that line):
//   1. Rank 0 dcci+store+dcci publishes [1000, 0, …] to GM.
//   2. Rank 1 still holds a stale cached line [0, 0, …].
//   3. Rank 1 stores slot1 → cached line is [0, 1001, …].
//   4. Rank 1 dcci write-backs the WHOLE line → GM is [0, 1001, …].
//   5. Rank 0 reads slot0 → 0  (expected 1000).
//
// stride=16 — each rank owns a full line (16×int32). Same Set; no neighbor wipe.
//
// Device code is small. Host is the tnotify HCCL+MPI harness (window prefix,
// barriers, device copy-out). WindowMemInit/Read exist because aclrtMemcpy on
// the HCCL window is not reliable (same as tnotify_kernel.cpp).
//
// Reproduced in:
//   https://github.com/georgebisbas/pypto-docker/blob/main/Dockerfile.hw-native-sys.cann9.0
// =============================================================================

#include <cstdint>
#include <iostream>
#include <vector>

#ifndef AICORE
#define AICORE [aicore]
#endif

#include "../common.hpp"

// HCCL keeps a sync prefix at the start of the window (tnotify uses the same).
static constexpr size_t HCCL_WIN_SYNC_PREFIX = 64 * sizeof(int32_t);
static constexpr int32_t kSlotBase = 1000;
static constexpr int kIters = 50; // bursty race: one iter is not a verdict

// Must match TNOTIFY_IMPL NotifyOp::Set in include/pto/comm/a2a3/TNotify.hpp.
// Word-safe Set: atomic add of wrapping (value - current). a2a3 has no
// atomic-store op (set_st_atomic_cfg op range is [0,0] = SUM only). Inlined
// so stock CANN bisheng can compile this without --cce-pto-enable.
AICORE inline void TNotifySet(__gm__ int32_t* ptr, int32_t value)
{
    const uint32_t old = ld_dev(reinterpret_cast<__gm__ uint32_t*>(ptr), 0);
    const int32_t delta = static_cast<int32_t>(static_cast<uint32_t>(value) - old);
    set_st_atomic_cfg(ATOMIC_S32, ATOMIC_SUM);
    __asm__ __volatile__("");
    dcci(ptr, cache_line_t::SINGLE_CACHE_LINE);
    __asm__ __volatile__("");
    st_atomic<int32_t>(delta, ptr);
    __asm__ __volatile__("");
    dcci(ptr, cache_line_t::SINGLE_CACHE_LINE);
    __asm__ __volatile__("");
    dsb(DSB_DDR);
    pipe_barrier(PIPE_ALL);
}

__global__ AICORE void WindowMemInit(__gm__ int32_t* ptr, int32_t value, int count)
{
    for (int i = 0; i < count; ++i) {
        ptr[i] = value;
    }
    pipe_barrier(PIPE_ALL);
}

__global__ AICORE void WindowMemRead(__gm__ int32_t* dst, __gm__ int32_t* src, int count)
{
    for (int i = 0; i < count; ++i) {
        dst[i] = src[i];
    }
    pipe_barrier(PIPE_ALL);
}

// Same packing as TNotifyScoreboardKernel (tnotify_kernel.cpp): each rank Sets
// rank0_base[my_rank] (here with an explicit stride so the control can be 16).
// Rank r → rank0.slots[r * slot_stride] = 1000 + r  (remote Set for r != 0).
__global__ AICORE void TNotifySetFalseShareKernel(
    __gm__ int32_t* slots, __gm__ CommDeviceContext* hcclCtx, int nranks, int slot_stride, int32_t slot_base)
{
    (void)nranks;
    int my_rank = static_cast<int>(hcclCtx->rankId);
    __gm__ int32_t* rank0_slots = CommRemotePtr(hcclCtx, slots, /*pe=*/0);
    TNotifySet(rank0_slots + my_rank * slot_stride, slot_base + my_rank);
}

static bool RunSetFalseShareKernel(
    int rank_id, int n_ranks, int n_devices, int first_device_id, const HcclRootInfo* rootInfo, int slot_stride)
{
    TestContext ctx;
    if (!ctx.Init(rank_id, n_ranks, n_devices, first_device_id, rootInfo)) {
        return false;
    }

    const int n_elems = n_ranks * slot_stride;
    size_t off = 0;
    uint64_t win = ctx.hostCtx.windowsIn[rank_id];
    // Skip HCCL's reserved prefix (same constant as tnotify/twait). The race
    // is *inside* `slots`, not between this prefix and the array.
    WindowAlloc(win, off, HCCL_WIN_SYNC_PREFIX);
    int32_t* slots = static_cast<int32_t*>(WindowAlloc(win, off, static_cast<size_t>(n_elems) * sizeof(int32_t)));

    int32_t* scratch = nullptr;
    std::vector<int32_t> copy(static_cast<size_t>(n_elems));
    if (rank_id == 0) {
        aclrtMalloc(reinterpret_cast<void**>(&scratch), copy.size() * sizeof(int32_t), ACL_MEM_MALLOC_HUGE_FIRST);
    }

    int fail_count = 0;
    for (int iter = 0; iter < kIters; ++iter) {
        // Fresh zeros so a leftover 1000 cannot masquerade as a successful Set.
        WindowMemInit<<<1, nullptr, ctx.stream>>>(slots, 0, n_elems);
        aclrtSynchronizeStream(ctx.stream);
        HcclHostBarrier(ctx.comm, ctx.stream);

        TNotifySetFalseShareKernel<<<1, nullptr, ctx.stream>>>(slots, ctx.deviceCtx, n_ranks, slot_stride, kSlotBase);
        ctx.aclStatus = aclrtSynchronizeStream(ctx.stream);
        HcclHostBarrier(ctx.comm, ctx.stream);
        // Overwrite with a smaller value. If Set were SUM we'd see 1042+rank, not 42+rank.
        TNotifySetFalseShareKernel<<<1, nullptr, ctx.stream>>>(
            slots, ctx.deviceCtx, n_ranks, slot_stride, /*slot_base=*/42);
        ctx.aclStatus = aclrtSynchronizeStream(ctx.stream);
        HcclHostBarrier(ctx.comm, ctx.stream);

        if (rank_id != 0) {
            // Non-zero ranks return success from the check (fail_count stays 0).
            // Rank 0 is the only observer; ForkAndRun ANDs all ranks, so if we
            // counted fails here as 0 on every rank, GTest would go green.
            continue;
        }
        WindowMemRead<<<1, nullptr, ctx.stream>>>(scratch, slots, n_elems);
        aclrtSynchronizeStream(ctx.stream);
        aclrtMemcpy(
            copy.data(), copy.size() * sizeof(int32_t), scratch, copy.size() * sizeof(int32_t),
            ACL_MEMCPY_DEVICE_TO_HOST);

        bool ok = true;
        for (int r = 0; r < n_ranks; ++r) {
            int32_t got = copy[static_cast<size_t>(r * slot_stride)];
            int32_t exp = 42 + r;
            if (got != exp) {
                std::cerr << "iter " << iter << " slot " << r << " expected " << exp << " got " << got
                          << " (slot_stride=" << slot_stride << ")\n";
                ok = false;
            }
        }
        fail_count += ok ? 0 : 1;
    }

    if (rank_id == 0) {
        aclrtFree(scratch);
        std::cout << "TNOTIFY Set false-share: slot_stride=" << slot_stride << " failed " << fail_count << "/" << kIters
                  << " (" << (fail_count * 100 / kIters) << "%)\n";
    }
    return ctx.Finalize() && (fail_count == 0);
}

bool RunSetFalseShare(int n_ranks, int n_devices, int first_rank_id, int first_device_id, int slot_stride)
{
    return ForkAndRunWithHcclRootInfo(
        n_ranks, first_rank_id, first_device_id, [&](int rankId, const HcclRootInfo* rootInfo) {
            return RunSetFalseShareKernel(rankId, n_ranks, n_devices, first_device_id, rootInfo, slot_stride);
        });
}
