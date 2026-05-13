/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

// Mock layer that stands in for LPU WSE SPR + WFE primitives.
//
// Design doc section 5 defines the LPU WSE GridPipe lowering as:
//   - mtspr SPR_RDY_<DIR>    -> cross-core SPR write, wakes neighbor WFE
//   - mtspr SPR_FREE_<DIR>   -> cross-core SPR write, wakes producer WFE
//   - mfspr SPR_RDY_<DIR>    -> read local mirror of ready flag
//   - wfe   SPR_RDY_<DIR>, N -> block until ready >= N
//   - wfe   SPR_FREE_<DIR>,N -> block until free  >= N
//
// On A2/A3 silicon there is no native cross-core mesh SPR + event line.  The
// mocks below emulate the contract via GM atomic flags + spin-wait, so the
// GridPipe semantics can be exercised on real A2/A3 boards while staying
// trivially swappable for a real LPU WSE SPR backend later.
//
// Each MOCK_* function carries a comment naming the corresponding LPU WSE
// pseudo-asm line from design doc section 5.3, so `grep -n "LPU WSE:"` returns
// the substitution sites.

#ifndef PTO_GRID_PIPE_MOCK_SPR_HPP
#define PTO_GRID_PIPE_MOCK_SPR_HPP

#include <cstdint>

#include <pto/common/arch_macro.hpp>
#include <pto/common/grid_pipe.hpp>

namespace pto {
namespace grid_mock {

#ifndef PTO_GRID_MOCK_WFE_MAX_SPINS
#define PTO_GRID_MOCK_WFE_MAX_SPINS 100000000U
#endif

inline constexpr uint32_t kDefaultWfeMaxSpins = PTO_GRID_MOCK_WFE_MAX_SPINS;
inline constexpr uint32_t kFaultFlagWordOffset = 8;

// MOCK: design doc 5.3 producer step (4), consumer step (4).
//
// LPU WSE: mtspr SPR_RDY_<DIR>, newValue      (cross-core, wakes neighbor WFE)
// LPU WSE: mtspr SPR_FREE_<DIR>, newValue     (cross-core, wakes producer WFE)
//
// A2/A3 mock: producer / consumer holds a pointer into the *neighbor's* GM
// window (resolved by HcclRemotePtr at runtime); we write the new monotonic
// counter into that remote location.  Pairing read happens via MockWfe* below.
//
// volatile cast prevents the compiler from caching the write.  Cross-rank
// visibility on A2/A3 requires an explicit dsb(DSB_DDR) + dcci pair around the
// store: AICORE caches are not coherent between cores, so without the dcci the
// pairing MockWfe spin on the remote rank may never observe the write.  This
// matches the SetLocalSummaryReady pattern in ready_queue.hpp (allgather_gemm).
inline AICORE void MockMtsprReady(__gm__ uint32_t *remoteFlag, uint32_t newValue)
{
    if (remoteFlag != nullptr) {
        volatile __gm__ uint32_t *ptr = reinterpret_cast<volatile __gm__ uint32_t *>(remoteFlag);
        // Match the canonical TNotify Set pattern: pre-invalidate, store,
        // post-invalidate, dsb(DSB_DDR).  Compiler barriers prevent reordering.
        __asm__ __volatile__("" ::: "memory");
        dcci(reinterpret_cast<__gm__ void *>(const_cast<__gm__ uint32_t *>(ptr)), SINGLE_CACHE_LINE);
        __asm__ __volatile__("" ::: "memory");
        *ptr = newValue;
        __asm__ __volatile__("" ::: "memory");
        dcci(reinterpret_cast<__gm__ void *>(const_cast<__gm__ uint32_t *>(ptr)), SINGLE_CACHE_LINE);
        __asm__ __volatile__("" ::: "memory");
        dsb(DSB_DDR);
    }
}

inline AICORE void MockMtsprFree(__gm__ uint32_t *remoteFlag, uint32_t newValue)
{
    if (remoteFlag != nullptr) {
        volatile __gm__ uint32_t *ptr = reinterpret_cast<volatile __gm__ uint32_t *>(remoteFlag);
        __asm__ __volatile__("" ::: "memory");
        dcci(reinterpret_cast<__gm__ void *>(const_cast<__gm__ uint32_t *>(ptr)), SINGLE_CACHE_LINE);
        __asm__ __volatile__("" ::: "memory");
        *ptr = newValue;
        __asm__ __volatile__("" ::: "memory");
        dcci(reinterpret_cast<__gm__ void *>(const_cast<__gm__ uint32_t *>(ptr)), SINGLE_CACHE_LINE);
        __asm__ __volatile__("" ::: "memory");
        dsb(DSB_DDR);
    }
}

// MOCK: design doc 5.3 producer step (1), consumer step (1).
//
// LPU WSE: mfspr r_ready, SPR_RDY_<DIR>
// LPU WSE: mfspr r_free,  SPR_FREE_<DIR>
//
// A2/A3 mock: volatile read of the local mirror of the flag.
inline AICORE uint32_t MockMfspr(__gm__ uint32_t *localFlag)
{
    if (localFlag == nullptr) {
        return 0;
    }
    return *reinterpret_cast<volatile __gm__ uint32_t *>(localFlag);
}

// MOCK: design doc 5.3 producer step (1) wait, consumer step (1) wait.
//
// LPU WSE: wfe SPR_RDY_<DIR>,  threshold      (block until SPR >= threshold)
// LPU WSE: wfe SPR_FREE_<DIR>, threshold
//
// A2/A3 mock: spin-poll the GM flag with `dcci` each iteration to invalidate
// the local cache line so the next load fetches from DDR.  Without the dcci,
// the AICORE may cache the original 0 indefinitely and never observe the
// producer's write (cross-core caches are not auto-coherent on A2/A3).
inline AICORE bool MockTryWfeReady(__gm__ uint32_t *localFlag, uint32_t threshold,
                                   uint32_t maxSpins = kDefaultWfeMaxSpins)
{
    if (localFlag == nullptr) {
        return true;
    }
    volatile __gm__ uint32_t *p = reinterpret_cast<volatile __gm__ uint32_t *>(localFlag);
    uint32_t spin = 0;
    constexpr uint32_t kFenceInterval = 64;
    while (true) {
        __asm__ __volatile__("" ::: "memory");
        dcci(reinterpret_cast<__gm__ void *>(const_cast<__gm__ uint32_t *>(p)), SINGLE_CACHE_LINE);
        __asm__ __volatile__("" ::: "memory");
        if (*p >= threshold) {
            return true;
        }
        if (maxSpins != 0 && spin >= maxSpins) {
            return false;
        }
        if ((++spin % kFenceInterval) == 0) {
            pipe_barrier(PIPE_ALL);
        }
    }
}

inline AICORE bool MockTryWfeFree(__gm__ uint32_t *localFlag, uint32_t threshold,
                                  uint32_t maxSpins = kDefaultWfeMaxSpins)
{
    if (localFlag == nullptr) {
        return true;
    }
    volatile __gm__ uint32_t *p = reinterpret_cast<volatile __gm__ uint32_t *>(localFlag);
    uint32_t spin = 0;
    constexpr uint32_t kFenceInterval = 64;
    while (true) {
        __asm__ __volatile__("" ::: "memory");
        dcci(reinterpret_cast<__gm__ void *>(const_cast<__gm__ uint32_t *>(p)), SINGLE_CACHE_LINE);
        __asm__ __volatile__("" ::: "memory");
        if (*p >= threshold) {
            return true;
        }
        if (maxSpins != 0 && spin >= maxSpins) {
            return false;
        }
        if ((++spin % kFenceInterval) == 0) {
            pipe_barrier(PIPE_ALL);
        }
    }
}

inline AICORE void MockWfeReady(__gm__ uint32_t *localFlag, uint32_t threshold)
{
    (void)MockTryWfeReady(localFlag, threshold, 0);
}

inline AICORE void MockWfeFree(__gm__ uint32_t *localFlag, uint32_t threshold)
{
    (void)MockTryWfeFree(localFlag, threshold, 0);
}

inline AICORE void MockSetFault(__gm__ uint32_t *faultFlag, uint32_t faultCode)
{
    if (faultFlag != nullptr) {
        volatile __gm__ uint32_t *ptr = reinterpret_cast<volatile __gm__ uint32_t *>(faultFlag);
        __asm__ __volatile__("" ::: "memory");
        *ptr = faultCode;
        __asm__ __volatile__("" ::: "memory");
        dcci(reinterpret_cast<__gm__ void *>(const_cast<__gm__ uint32_t *>(ptr)), SINGLE_CACHE_LINE);
        dsb(DSB_DDR);
    }
}

// MOCK: design doc 5.4 SPR_BOUNDARY_MASK fault.
//
// LPU WSE: writing SPR_RDY_<DIR> when SPR_BOUNDARY_MASK has that direction
// disabled triggers a hardware fault / squash.
//
// A2/A3 mock: explicit early-exit + sentinel write so the host can detect the
// out-of-bound attempt.  Real boards will raise a fault; here we trap softly
// by writing a sentinel and aborting the kernel branch.  The host launcher
// inspects a "fault sentinel" GM word after each kernel and fails the run.
inline AICORE void MockBoundaryFault(__gm__ uint32_t *faultSentinel, uint32_t faultCode)
{
    if (faultSentinel != nullptr) {
        *reinterpret_cast<volatile __gm__ uint32_t *>(faultSentinel) = faultCode;
    }
    // Best-effort halt of the current kernel branch.  Real silicon will fault
    // here; on A2/A3 we just stop emitting further GridPipe ops in this branch.
}

// Fault codes mirror SPR_BOUNDARY_MASK fields (design doc section 5.2).
inline constexpr uint32_t kFaultPushNorth  = 0x101;
inline constexpr uint32_t kFaultPushEast   = 0x102;
inline constexpr uint32_t kFaultPushWest   = 0x103;
inline constexpr uint32_t kFaultPushSource = 0x104;  // Always illegal.
inline constexpr uint32_t kFaultPopNorth   = 0x201;
inline constexpr uint32_t kFaultPopEast    = 0x202;
inline constexpr uint32_t kFaultPopWest    = 0x203;
inline constexpr uint32_t kFaultWaitReadyTimeout = 0x301;
inline constexpr uint32_t kFaultWaitFreeTimeout  = 0x302;

}  // namespace grid_mock
}  // namespace pto

#endif  // PTO_GRID_PIPE_MOCK_SPR_HPP
