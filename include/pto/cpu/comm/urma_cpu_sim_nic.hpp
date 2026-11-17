/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_CPU_COMM_URMA_CPU_SIM_NIC_HPP
#define PTO_CPU_COMM_URMA_CPU_SIM_NIC_HPP

// CPU-SIM stand-in for the URMA NIC.
//
// On real hardware a URMA op (put/get/notify) is fire-and-forget: the core
// fills WQEs into a WQ ring, rings the doorbell, and the NIC asynchronously
// reads the WQEs, performs the DMA, and later writes CQEs into the CQ ring that
// the shared completion logic (WaitForUrmaCqe / UrmaPollCqAt) polls. None of
// that NIC behaviour exists on CPU-SIM (no device, no DMA engine), so the
// completion logic would spin forever waiting for CQEs that never come.
//
// This file supplies the missing NIC as a CPU-side simulation. The drain core
// (CpuSimUrmaDrainNIC) lives in urma_async_intrin.hpp under __CPU_SIM so the
// shared engine can call it from its ring-backpressure points: when a post
// would fill the WQ/CQ ring, the engine lets the simulated NIC consume the
// WQEs committed so far instead of waiting for an asynchronous completion that
// can never arrive. The CPU-side TPut/TGet wrappers additionally call
// CpuSimUrmaDrainPost after __urma_put_async / __urma_get_async /
// __urma_put_async_notify returns, so any WQEs still queued are drained before
// Wait/Test can run.
//
// The drain walks exactly the WQEs the core committed, performs the copy the
// NIC would have done (WRITE local->remote, READ remote->local, FAA
// signal-add), and writes one CQE per SQE with the owner bit flipped to match
// what the consumer expects.
//
// The simulated NIC is a faithful reader of the shared ring format: it derives
// the SQE/SGE layout, remote address, BB/CQE accounting and owner-bit rule from
// the same urma_types.hpp structures the shared engine uses, so it stays in
// lockstep with the real wire format without touching the shared headers. This
// file is CPU-only (#ifdef __CPU_SIM) and is NOT compiled on NPU.

// Ring mechanics used below (producer/consumer convention):
//   WQ (send work queue): the CORE is the producer and the NIC is the
//       consumer. WQEs live in a ring of wq->depth slots; each slot is a BB
//       (basic block) of 2^wq->wqeShiftSize bytes and holds one SQE with its
//       SGE(s) immediately after it. The NIC's consumption frontier is
//       wq->tailAddr (next BB to consume); the core's production frontier is
//       wq->headAddr. The region of newly posted WQEs is therefore
//       [tail, head), and 'targetBb' passed in from the post wrapper is the
//       core's head, i.e. the number of BBs the post committed. Note this is a
//       BB counter, not a WQE counter: an FAA WQE spans 2 BBs but counts as one
//       submitted WQE (tracked separately in wq->submittedWqeCount for the
//       completion side). Draining advances tail to head.
//   CQ (completion queue): the NIC is the producer and the CORE is the
//       consumer. CQEs live in a ring of cq->depth slots of
//       2^cq->cqeShiftSize bytes. cq->tailAddr is the next CQE index to
//       produce; the consumer polls headAddr and checks each CQE's owner bit.
//       The owner bit is 0 for the first cq->depth CQEs, 1 for the next
//       cq->depth, and so on, flipping every wrap of the ring; the producer
//       must write the bit the consumer expects for that sequence number.

#ifdef __CPU_SIM

#include "pto/comm/async_common/async_types.hpp"
#include "pto/comm/async/urma/urma_types.hpp"
#ifdef PTO_URMA_SUPPORTED
#include "pto/comm/async/urma/urma_async_intrin.hpp"
#endif

namespace pto {
namespace comm {
namespace urma {
namespace detail {

// Drain one jetty's slice of a just-posted operation. Resolves the jetty's WQ
// and CQ from the shared workspace and performs the copy + CQE publication.
// Called by the CPU-side TPut/TGet wrappers once per jetty right after the
// shared engine commits the WQEs; targetBb is the number of BBs the post
// committed for that jetty (0 means nothing to drain).
AICORE inline bool CpuSimUrmaDrainPost(const AsyncSession& session, uint32_t peer, uint32_t jetty, uint32_t targetBb)
{
    if (targetBb == 0U) {
        return true;
    }
    __gm__ UrmaWQCtx* wq = GetWqContextAt(session, peer, jetty);
    __gm__ UrmaCqCtx* cq = GetCqContextAt(session, peer, jetty);
    return CpuSimUrmaDrainNIC(session, peer, jetty, wq, cq, targetBb);
}

} // namespace detail
} // namespace urma
} // namespace comm
} // namespace pto

#endif // __CPU_SIM
#endif // PTO_CPU_COMM_URMA_CPU_SIM_NIC_HPP
