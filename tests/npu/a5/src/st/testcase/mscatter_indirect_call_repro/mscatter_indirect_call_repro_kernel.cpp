/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

// ============================================================================
// Standalone regression / repro for the runtime-dispatch limitation documented
// in tests/npu/a5/src/st/testcase/mgather/MGATHER.md
// §"Runtime Dispatch Requirement".
//
// THIS TEST IS EXPECTED TO HANG OR PRODUCE INCORRECT RESULTS until the
// toolchain / firmware gains support for SIMT-aware lowering of indirect
// calls into SIMT-bearing functions. When that lands, this test should
// start passing without any test-side change — it serves as a regression
// detector for the fix.
//
// What it demonstrates
// --------------------
// Standard rtKernelLaunch launch path (host `<<<1, nullptr, stream>>>`,
// identical to every other ST in this suite) — kernel entry is
// `runMSCATTER_indirect_call_elem2d_float_8x32_random_256size`. Inside,
// the SIMT body (one MSCATTER) is reached through ONE volatile fn-pointer
// indirect call instead of a direct call:
//
//     volatile FnT fn = &simt_body;   // bisheng forced to emit HiIPUISD::LongCALL
//     fn(out, src, indices);          // BLR — SIMT body runs after this
//
// Direct call (without the `volatile`) → MSCATTER PASSes, identical to the
// MSCATTERTest case_elem2d_float_8x32_random_256size in
// tests/npu/a5/src/st/testcase/mscatter/.
//
// Indirect call (the variant in this file) → chip hang. Confirmed on
// dav-c310-vec (a5) with CANN 9.1.T500 / bisheng 15.0.5 / driver
// 25.6.rc1.b108 — task-submit times out at 200s, no errcode raised.
//
// Why this is a meaningful regression test
// ----------------------------------------
// The standard CANN launch path installs TID / warp / vec-pipe scheduling
// state correctly. The first `cce::async_invoke` inside MSCATTER is then
// dispatched correctly. With an intervening BLR (HiIPUISD::LongCALL),
// bisheng skips the SIMT-aware pre-call setup it emits for short BL —
// the SIMT scheduler is not staged for the post-BLR target, and the
// first `async_invoke` finds no warp scheduler to dispatch into.
//
// All custom AICore dispatchers that dispatch user kernels via runtime
// function pointer (e.g. ringbuffer-based SU dispatchers like
// hw-native-sys/simpler#970) hit the same issue.
// ============================================================================

#include <pto/pto-inst.hpp>
#include <pto/common/constants.hpp>
#include <pto/npu/a5/MScatter.hpp>
#include "acl/acl.h"
#include "mscatter_indirect_call_repro_common.h"

using namespace std;
using namespace pto;

__global__ AICORE __attribute__((aiv)) void mscatter_indirect_warmup_kernel()
{}

AICORE PTO_INLINE void FlushScatterOutput()
{
    dcci(static_cast<__gm__ void *>(0), ENTIRE_DATA_CACHE);
    dsb(DSB_DDR);
}

// ----------------------------------------------------------------------------
// SIMT body — same as runElem2D in tests/.../mscatter/mscatter_kernel.cpp,
// instantiated for the 8x32 → 256-slot float case.
// noinline so the call boundary survives -O3.
// ----------------------------------------------------------------------------
AICORE __attribute__((noinline)) void simt_body_elem2d_float_8x32_random_256size(
    __gm__ float __out__ *out, __gm__ float __in__ *src, __gm__ int32_t __in__ *indices)
{
    constexpr int kSrcRows = 8;
    constexpr int kSrcCols = 32;
    constexpr int kTableSize = 256;
    using SrcShape  = pto::Shape <1, 1, 1, kSrcRows, kSrcCols>;
    using SrcStride = pto::Stride<1, 1, 1, kSrcCols, 1>;
    using IdxShape  = pto::Shape <1, 1, 1, kSrcRows, kSrcCols>;
    using IdxStride = pto::Stride<1, 1, 1, kSrcCols, 1>;
    using OutShape  = pto::Shape <1, 1, 1, 1, kTableSize>;
    using OutStride = pto::Stride<1, 1, 1, kTableSize, 1>;

    GlobalTensor<float,   SrcShape, SrcStride> srcGlobal(src);
    GlobalTensor<int32_t, IdxShape, IdxStride> idxGlobal(indices);
    GlobalTensor<float,   OutShape, OutStride> outGlobal(out);

    using SrcTile = Tile<TileType::Vec, float,   kSrcRows, kSrcCols,
                         BLayout::RowMajor, kSrcRows, kSrcCols>;
    using IdxTile = Tile<TileType::Vec, int32_t, kSrcRows, kSrcCols,
                         BLayout::RowMajor, kSrcRows, kSrcCols>;

    SrcTile srcTile;
    IdxTile idxTile;

    constexpr int idxBytes = ((kSrcRows * kSrcCols * (int)sizeof(int32_t) + 31) / 32) * 32;
    TASSIGN(idxTile, 0x0);
    TASSIGN(srcTile, idxBytes);

    TLOAD(idxTile, idxGlobal);
    TLOAD(srcTile, srcGlobal);

    set_flag (PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

    // ↓ The SIMT instruction — fails when reached via BLR.
    MSCATTER<pto::Coalesce::Elem,
             pto::ScatterAtomicOp::None,
             pto::ScatterOOB::Undefined,
             pto::ScatterConflict::Last>(outGlobal, srcTile, idxTile);

    pipe_barrier(PIPE_ALL);
    set_flag (PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    FlushScatterOutput();
}

// ----------------------------------------------------------------------------
// Kernel entry — launched by host via <<<1, nullptr, stream>>>.
// `volatile FnT fn = &simt_body;` forces bisheng to lower the call as
// HiIPUISD::LongCALL (4-word indirect call, target via linker reloc).
// Same SIMT body code, same data, same launch path — only the call form
// changes vs the corresponding case in tests/.../mscatter/mscatter_kernel.cpp.
// ----------------------------------------------------------------------------
extern "C" __global__ AICORE void runMSCATTER_indirect_call_elem2d_float_8x32_random_256size(
    __gm__ float *out, __gm__ float *src, __gm__ int32_t *indices)
{
    using FnT = void (*)(__gm__ float *, __gm__ float *, __gm__ int32_t *);
    volatile FnT fn = &simt_body_elem2d_float_8x32_random_256size;
    fn(out, src, indices);   // <-- BLR; SIMT scheduler unprepared on this path
}

void Launch_indirect_call_elem2d_float_8x32_random_256size(
    float *out, float *src, int32_t *indices, void *stream)
{
    mscatter_indirect_warmup_kernel<<<64, nullptr, stream>>>();
    runMSCATTER_indirect_call_elem2d_float_8x32_random_256size<<<1, nullptr, stream>>>(
        reinterpret_cast<float *>(out), reinterpret_cast<float *>(src), indices);
}
