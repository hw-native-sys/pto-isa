/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

// HNS_1825 (Hi1825) RoCE RDMA device-side backend for A5.
//
// Posts RDMA WRITE / READ from AIV using the RdmaInfo table supplied via RdmaExecContext.
// WQE/CQE staging uses each AIV's own UB scratch. Local scalar atomics coordinate
// shared queues; remote RDMA atomic operations are not supported.

#ifndef PTO_COMM_ASYNC_RDMA_BACKENDS_HNS_1825_BACKEND_HPP
#define PTO_COMM_ASYNC_RDMA_BACKENDS_HNS_1825_BACKEND_HPP

#if defined(__NPU_ARCH__) && __NPU_ARCH__ != 0 && __NPU_ARCH__ != 3510
#error "The HNS1825 shared-queue backend requires Ascend950 (__NPU_ARCH__ == 3510)."
#endif

#include "pto/pto-inst.hpp"
#include "pto/comm/async_common/async_types.hpp"
#include "pto/comm/async/rdma/rdma_device_common.hpp"
#include "pto/comm/async/rdma/backends/hns_1825/hns_1825_types.hpp"
// AscendC::GetSystemCycle only — do NOT include full kernel_operator.h here.
// That pulls adv_api/hccl (`using namespace HcclApi`) and collides with ST
// common.hpp symbols (MAX_CC_TILING_NUM / GROUP_NAME_SIZE / ALG_CONFIG_SIZE).
#if !defined(__CPU_SIM) && !defined(__COSTMODEL)
#include "kernel_operator_sys_var_intf.h"
#endif

namespace pto {
namespace comm {
namespace rdma {
namespace hns_1825 {
namespace detail {

constexpr uint32_t kCqeReadSize = sizeof(Hns1825Cqe);

struct PostSendResult {
    uint32_t curHead;
    uint32_t status;
};

// NIC consumes big-endian; AscendC is little-endian. SQ/CQ head-tail mirrors stay host-order;
// only NIC-visible payloads (WQE dwords / software doorbell) are byte-swapped.
AICORE inline uint16_t Htobe16(uint16_t v) { return (uint16_t)(((v & 0x00ffU) << 8) | ((v & 0xff00U) >> 8)); }

AICORE inline uint32_t Htobe32(uint32_t v)
{
    return ((v & 0x000000ffU) << 24) | ((v & 0x0000ff00U) << 8) | ((v & 0x00ff0000U) >> 8) | ((v & 0xff000000U) >> 24);
}

AICORE inline uint64_t Htobe64(uint64_t v)
{
    return ((v & 0x00000000000000ffULL) << 56) | ((v & 0x000000000000ff00ULL) << 40) |
           ((v & 0x0000000000ff0000ULL) << 24) | ((v & 0x00000000ff000000ULL) << 8) |
           ((v & 0x000000ff00000000ULL) >> 8) | ((v & 0x0000ff0000000000ULL) >> 24) |
           ((v & 0x00ff000000000000ULL) >> 40) | ((v & 0xff00000000000000ULL) >> 56);
}

// Host-order u32 mirror access (SQ/CQ head-tail): bypass the scalar data cache.
AICORE inline uint32_t ReadU32Gm(uint64_t addr) { return ld_dev((__gm__ uint32_t*)addr, 0); }

AICORE inline void WriteU32Gm(uint64_t addr, uint32_t value) { st_dev(value, (__gm__ uint32_t*)addr, 0); }

// The HNS1825 backend runs on Ascend950 (3510). Scalar atomics serialize local
// GM queue state; DSB orders the scalar accesses and the completed MTE copies
// before ownership crosses AIVs or passes to the NIC.
AICORE inline void QueueFence()
{
    __asm__ __volatile__("" ::: "memory");
    dsb(DSB_DDR);
    __asm__ __volatile__("" ::: "memory");
}

AICORE inline uint64_t AtomicLoad(__gm__ uint64_t* address) { return atomicAdd(address, uint64_t{0}); }

AICORE inline __gm__ QueueState* GetQueueState(__gm__ RoceSqCtx* sq)
{
    return reinterpret_cast<__gm__ QueueState*>(sq->stateAddr);
}

AICORE inline bool TryQueueLock(__gm__ uint64_t* lock)
{
    if (atomicCAS(lock, uint64_t{0}, uint64_t{1}) != 0) {
        return false;
    }
    QueueFence();
    return true;
}

AICORE inline void UnlockQueue(__gm__ uint64_t* lock)
{
    QueueFence();
    (void)atomicCAS(lock, uint64_t{1}, uint64_t{0});
}

AICORE inline uint32_t QueueError(__gm__ QueueState* state) { return static_cast<uint32_t>(AtomicLoad(&state->error)); }

AICORE inline uint32_t FailQueue(__gm__ QueueState* state, uint32_t status)
{
    QueueFence();
    const uint64_t previous = atomicCAS(&state->error, uint64_t{0}, static_cast<uint64_t>(status));
    return previous == 0 ? status : static_cast<uint32_t>(previous);
}

AICORE inline bool QueueTimedOut(uint64_t startCycles)
{
    return static_cast<uint64_t>(AscendC::GetSystemCycle()) - startCycles >= kHns1825PollCqTimeoutCycles;
}

AICORE inline bool TryPostSend(__gm__ RoceSqCtx* sq, uint64_t end, __ubuf__ uint8_t* ub, uint32_t syncId);

// Completion and capacity waits can flush a ready prefix whose producer returned
// before ringing the doorbell. Never wait for the doorbell lock while progressing.
AICORE inline void SubmitReady(__gm__ RoceSqCtx* sq, __ubuf__ uint8_t* ub, uint32_t syncId)
{
    __gm__ QueueState* state = GetQueueState(sq);
    const uint64_t ready = AtomicLoad(&state->readyHead);
    if (ready > AtomicLoad(&state->postedHead)) {
        (void)TryPostSend(sq, ready, ub, syncId);
    }
}

// Order scalar UB writes before MTE3 copies a NIC-visible record (WQE / software doorbell) to GM, then order the
// MTE3 store before the subsequent scalar dcci / st_dev. Post-copy sync MUST be MTE3->Scalar (not MTE3->MTE2):
// RoCE consumers are scalar NIC doorbell paths.
AICORE inline void WriteUbToGmWithSync(uint64_t gmAddr, __ubuf__ uint8_t* ub, uint32_t size, uint32_t syncId)
{
    set_flag(PIPE_S, PIPE_MTE3, syncId);
    wait_flag(PIPE_S, PIPE_MTE3, syncId);
    copy_ubuf_to_gm_align_v2(
        reinterpret_cast<__gm__ uint32_t*>(gmAddr), reinterpret_cast<__ubuf__ uint32_t*>(ub), 0, 1, size, 0, 0, 0);
    set_flag(PIPE_MTE3, PIPE_S, syncId);
    wait_flag(PIPE_MTE3, PIPE_S, syncId);
}

// Copy `size` bytes from GM to the UB scratch, then make them scalar-readable.
AICORE inline void ReadGmToUb(__ubuf__ uint8_t* ub, uint64_t gmAddr, uint32_t size, uint32_t syncId)
{
    copy_gm_to_ubuf_align_v2(
        reinterpret_cast<__ubuf__ uint32_t*>(ub), reinterpret_cast<__gm__ uint32_t*>(gmAddr), 0, 1, size, 0, 0, 0, 0, 0,
        0);
    set_flag(PIPE_MTE2, PIPE_S, syncId);
    wait_flag(PIPE_MTE2, PIPE_S, syncId);
}

// SQ ring slot (one 64B WQEBB).
AICORE inline __gm__ uint8_t* GetSendWqe(__gm__ RoceSqCtx* sqCtx, uint32_t idx)
{
    return (__gm__ uint8_t*)sqCtx->bufAddr + (uint64_t)idx * kHns1825WqebbSize;
}

// A CQE is ready when its owner bit matches the lap parity from the current consumer index.
AICORE inline bool CheckCqeOwner(__ubuf__ Hns1825Cqe* cqe, uint32_t curTail, uint32_t cqRing)
{
    constexpr uint32_t kOwnerShift = 31; // owner bit at dw1[31]
    uint32_t curOwner = ((cqe->owner_id_qpn & (1U << kOwnerShift)) != 0);
    uint32_t expectOwner = (uint32_t)(((curTail & cqRing) == 0));
    return (expectOwner ^ curOwner) != 0;
}

// CQ doorbell: keep host-order CI mirrors in CQ/SQ tail records, publish big-endian CI to CQ software doorbell.
AICORE inline void RingCqDoorbell(
    __gm__ RdmaInfo* info, uint32_t pe, uint32_t qpIdx, uint32_t curTail, __ubuf__ uint8_t* ub, uint32_t syncId)
{
    constexpr uint32_t kCqUpdateCiMask = 0xffffff; // CQ consumer index is 24-bit
    uint32_t qpNum = info->qpNum;
    __gm__ RoceCqCtx* cqCtx = (__gm__ RoceCqCtx*)(info->scqPtr + ((uint64_t)pe * qpNum + qpIdx) * sizeof(RoceCqCtx));
    __gm__ RoceSqCtx* sqCtx = (__gm__ RoceSqCtx*)(info->sqPtr + ((uint64_t)pe * qpNum + qpIdx) * sizeof(RoceSqCtx));

    WriteU32Gm(cqCtx->tailAddr, curTail);
    WriteU32Gm(sqCtx->tailAddr, curTail);

    *reinterpret_cast<__ubuf__ uint32_t*>(ub) = Htobe32(curTail & kCqUpdateCiMask);
    WriteUbToGmWithSync(cqCtx->dbSwAddr, ub, sizeof(uint32_t), syncId);
    dcci((__gm__ void*)cqCtx->dbSwAddr, SINGLE_CACHE_LINE);
}

// All CQ consumers, including capacity waits and Test, share this short lock.
// Inspect a bounded batch of ready CQEs and release it before waiting for NIC
// progress. Physical CI includes an error CQE; completedHead only counts success.
AICORE inline void ProgressCq(__gm__ RdmaInfo* info, uint32_t pe, uint32_t qpIdx, __ubuf__ uint8_t* ub, uint32_t syncId)
{
    constexpr uint32_t kCqeOpcodeShift = 27;
    constexpr uint32_t kCqeOpcodeMask = 0x1f;
    constexpr uint32_t kCqeOptypeError = 0x1e;
    constexpr uint32_t kCqeOptypeInvalid = 0x1f;

    const uint64_t index = static_cast<uint64_t>(pe) * info->qpNum + qpIdx;
    __gm__ RoceSqCtx* sq = reinterpret_cast<__gm__ RoceSqCtx*>(info->sqPtr) + index;
    __gm__ RoceCqCtx* cq = reinterpret_cast<__gm__ RoceCqCtx*>(info->scqPtr) + index;
    __gm__ QueueState* state = GetQueueState(sq);
    if (!TryQueueLock(&state->cqLock)) {
        return;
    }
    if (QueueError(state) != 0) {
        UnlockQueue(&state->cqLock);
        return;
    }
    const uint32_t cqeSize = cq->cqeSize == 0 ? kHns1825DefaultCqeSize : cq->cqeSize;
    const uint64_t startTail = AtomicLoad(&state->completedHead);
    const uint64_t posted = AtomicLoad(&state->postedHead);
    QueueFence();
    uint64_t curTail = startTail;
    uint64_t completed = startTail;
    uint32_t status = 0;
    for (uint32_t count = 0; count < kCqProgressBatch && curTail < posted; ++count) {
        const uint64_t cqeAddr = cq->bufAddr + (curTail & (cq->depth - 1U)) * cqeSize;
        dcci(reinterpret_cast<__gm__ void*>(cqeAddr), SINGLE_CACHE_LINE);
        ReadGmToUb(ub, cqeAddr, kCqeReadSize, syncId);
        __ubuf__ Hns1825Cqe* cqe = reinterpret_cast<__ubuf__ Hns1825Cqe*>(ub);
        const uint32_t cqeType = (cqe->op_sr_wqebb >> kCqeOpcodeShift) & kCqeOpcodeMask;
        if (cqeType == kCqeOptypeInvalid || !CheckCqeOwner(cqe, static_cast<uint32_t>(curTail), cq->depth)) {
            break;
        }
        ++curTail;
        if (cqeType == kCqeOptypeError) {
            status = cqe->syndrome == 0 ? kHns1825CqeError : cqe->syndrome;
            break;
        }
        completed = curTail;
    }
    if (curTail != startTail) {
        RingCqDoorbell(info, pe, qpIdx, static_cast<uint32_t>(curTail), ub, syncId);
        QueueFence();
        (void)atomicCAS(&state->completedHead, startTail, completed);
    }
    if (status != 0) {
        (void)FailQueue(state, status);
    }
    UnlockQueue(&state->cqLock);
}

// Poll outside both queue locks. A timeout poisons the queue without reclaiming
// uncompleted slots, rolling back reservations, or forcing another AIV's lock.
AICORE inline uint32_t PollCq(
    __gm__ RdmaInfo* info, uint32_t pe, uint32_t qpIdx, uint32_t targetIdx, __ubuf__ uint8_t* ub, uint32_t syncId)
{
    __gm__ RoceSqCtx* sq =
        reinterpret_cast<__gm__ RoceSqCtx*>(info->sqPtr) + static_cast<uint64_t>(pe) * info->qpNum + qpIdx;
    __gm__ QueueState* state = GetQueueState(sq);
    const uint64_t startCycles = static_cast<uint64_t>(AscendC::GetSystemCycle());
    while (true) {
        if (AtomicLoad(&state->completedHead) >= targetIdx) {
            QueueFence();
            return 0;
        }
        const uint32_t status = QueueError(state);
        if (status != 0) {
            // A CQ consumer publishes its successful prefix before the error.
            // Re-read it in case our first load raced that publication.
            QueueFence();
            return AtomicLoad(&state->completedHead) >= targetIdx ? 0U : status;
        }
        if (QueueTimedOut(startCycles)) {
            return FailQueue(state, kHns1825PollCqTimeoutError);
        }
        SubmitReady(sq, ub, syncId);
        ProgressCq(info, pe, qpIdx, ub, syncId);
    }
}

// Fill the 16B WQE control segment in the UB staging buffer; returns pointer just after it.
AICORE inline __ubuf__ uint8_t* FillWqeCtrlSeg(__ubuf__ uint8_t* ubBase, uint32_t curHead, uint32_t depth)
{
    constexpr uint32_t kCtrlValue = 0x40;   // owner_sl fixed part
    constexpr uint32_t kVaValue = 0x20;     // df_tsl VA bit
    constexpr uint32_t kCqeSignalShift = 7; // df_tsl CR bit
    constexpr uint32_t kOwnerShift = 7;     // owner_sl owner bit
    constexpr uint32_t kCmpTaskLenShift = 28;
    constexpr uint32_t kMsnShift = 12;
    constexpr uint32_t kMsnMask = 0x3;
    constexpr uint32_t kSegLenUnit = sizeof(uint64_t); // section length unit = 8B
    constexpr uint32_t kDataSegBdsl = sizeof(Hns1825WqeDataSeg) / kSegLenUnit;
    uint16_t wf_bdsl = (uint16_t)(kDataSegBdsl | ((curHead & kMsnMask) << kMsnShift));

    __ubuf__ Hns1825WqeCtrlSeg* ctrl = (__ubuf__ Hns1825WqeCtrlSeg*)(__ubuf__ void*)ubBase;
    ctrl->owner_sl = (((curHead & depth) == 0) ? (1U << kOwnerShift) : 0) | kCtrlValue;
    ctrl->df_tsl = (uint8_t)((1U << kCqeSignalShift) | kVaValue | (sizeof(Hns1825WqeRdmaTaskSeg) / kSegLenUnit));
    ctrl->wf_bdsl = Htobe16(wf_bdsl);
    ctrl->cl_pi = Htobe32(1U << kCmpTaskLenShift);
    ctrl->db = 0;
    return ubBase + sizeof(Hns1825WqeCtrlSeg);
}

// Inline WRITE uses the bytes after the task segment instead of a data segment.
AICORE inline __ubuf__ uint8_t* FillInlineWqeCtrlSeg(__ubuf__ uint8_t* ubBase, uint32_t curHead, uint32_t depth)
{
    constexpr uint32_t kCtrlValue = 0x40;
    constexpr uint32_t kDataInlineShift = 6;
    constexpr uint32_t kVaValue = 0x20;
    constexpr uint32_t kCqeSignalShift = 7;
    constexpr uint32_t kOwnerShift = 7;
    constexpr uint32_t kCmpTaskLenShift = 28;
    constexpr uint32_t kMsnShift = 12;
    constexpr uint32_t kMsnMask = 0x3;
    constexpr uint32_t kSegLenUnit = sizeof(uint64_t);
    constexpr uint32_t kInlineBytes = sizeof(int32_t);
    constexpr uint32_t kInlineBdsl = (kInlineBytes + kSegLenUnit - 1U) / kSegLenUnit;
    const uint16_t wfBdsl = static_cast<uint16_t>(kInlineBdsl | ((curHead & kMsnMask) << kMsnShift));

    __ubuf__ Hns1825WqeCtrlSeg* ctrl = reinterpret_cast<__ubuf__ Hns1825WqeCtrlSeg*>(ubBase);
    ctrl->owner_sl = (((curHead & depth) == 0U) ? (1U << kOwnerShift) : 0U) | kCtrlValue;
    ctrl->df_tsl = static_cast<uint8_t>(
        (1U << kCqeSignalShift) | (1U << kDataInlineShift) | kVaValue | (sizeof(Hns1825WqeRdmaTaskSeg) / kSegLenUnit));
    ctrl->wf_bdsl = Htobe16(wfBdsl);
    ctrl->cl_pi = Htobe32(1U << kCmpTaskLenShift);
    ctrl->db = 0;
    return ubBase + sizeof(Hns1825WqeCtrlSeg);
}

// Fill the 32B RDMA task segment (opcode / len / remote VA / rkey / ulp).
AICORE inline __ubuf__ uint8_t* FillWqeTaskSeg(
    __ubuf__ uint8_t* addr, const RdmaSendWr& wr, RdmaOpcode opcode, bool fence = false)
{
    constexpr uint32_t kMsgWrite = 0x04;
    constexpr uint32_t kMsgRead = 0x08;
    constexpr uint32_t kReadLastExtLen = 4;
    uint32_t hwOpcode = (opcode == RdmaOpcode::OP_RDMA_READ) ? kMsgRead : kMsgWrite;

    __ubuf__ Hns1825WqeRdmaTaskSeg* task = (__ubuf__ Hns1825WqeRdmaTaskSeg*)(__ubuf__ void*)addr;
    task->com_tsk.value = 0;
    task->com_tsk.bs.signal = 1;
    task->com_tsk.bs.fence = fence ? 1U : 0U;
    task->com_tsk.bs.opcode = hwOpcode;
    task->com_tsk.value = Htobe32(task->com_tsk.value);
    task->data_len = Htobe32((uint32_t)wr.message_len);
    task->imm_data = 0;
    task->dw3.value = 0;
    if (opcode == RdmaOpcode::OP_RDMA_READ) {
        task->dw3.bs.last_ext_len = kReadLastExtLen;
        task->dw3.value = Htobe32(task->dw3.value);
    }
    task->va = Htobe64((uint64_t)wr.remote_addr);
    task->rkey = Htobe32(wr.rkey);
    task->ulp = Htobe32(wr.lkey & 0xffffU);
    return addr + sizeof(Hns1825WqeRdmaTaskSeg);
}

// Fill the 16B data segment (single SGE).
AICORE inline __ubuf__ uint8_t* FillWqeDataSeg(__ubuf__ uint8_t* addr, const RdmaSendWr& wr)
{
    constexpr uint32_t kNextSgeInvalid = 1U << 31; // le_key L bit
    constexpr uint32_t kLkeyMask = 0x3fffffffU;    // le_key key[29:0]

    __ubuf__ Hns1825WqeDataSeg* data = (__ubuf__ Hns1825WqeDataSeg*)(__ubuf__ void*)addr;
    data->buf_addr = Htobe64((uint64_t)wr.local_addr);
    data->r_len = Htobe32((uint32_t)wr.message_len);
    data->le_key = Htobe32((wr.lkey & kLkeyMask) | kNextSgeInvalid);
    return addr + sizeof(Hns1825WqeDataSeg);
}

// Copy only this producer's reserved slot, with its owner still invalid. The
// untouched next slot is invalid from initialization or the preceding ring lap.
AICORE inline uint32_t FillWqeWriteRead(
    const RdmaSendWr& wr, __gm__ RoceSqCtx* sqCtx, __gm__ uint8_t* wqeAddr, uint32_t curHead, RdmaOpcode opcode,
    __ubuf__ uint8_t* ub, uint32_t syncId)
{
    __ubuf__ uint8_t* dataUb = FillWqeTaskSeg(FillWqeCtrlSeg(ub, curHead, sqCtx->depth), wr, opcode);
    (void)FillWqeDataSeg(dataUb, wr);
    WriteUbToGmWithSync((uint64_t)wqeAddr, ub, kHns1825WriteReadWqeSize, syncId);
    return kHns1825WriteReadWqeSize;
}

// Assemble one fenced 4-byte inline RDMA WRITE. The inline value is payload
// data and therefore stays in host byte order; only WQE control fields are
// converted to the NIC's big-endian format.
AICORE inline void FillWqeInlineSet(
    const RdmaSendWr& wr, int32_t signalValue, __gm__ RoceSqCtx* sqCtx, __gm__ uint8_t* wqeAddr, uint32_t curHead,
    __ubuf__ uint8_t* ub, uint32_t syncId)
{
    __ubuf__ uint8_t* inlineData =
        FillWqeTaskSeg(FillInlineWqeCtrlSeg(ub, curHead, sqCtx->depth), wr, RdmaOpcode::OP_RDMA_WRITE, true);
    *reinterpret_cast<__ubuf__ uint64_t*>(inlineData) = static_cast<uint32_t>(signalValue);
    *reinterpret_cast<__ubuf__ uint64_t*>(inlineData + sizeof(uint64_t)) = 0U;
    WriteUbToGmWithSync(reinterpret_cast<uint64_t>(wqeAddr), ub, kHns1825WriteReadWqeSize, syncId);
}

// Ring the SQ doorbell: update PI mirror, publish software doorbell, write hardware doorbell via st_dev.
AICORE inline void RingSqDoorbell(__gm__ RoceSqCtx* sqCtx, uint32_t curHead, __ubuf__ uint8_t* ub, uint32_t syncId)
{
    constexpr uint32_t kPiHighShift = 8;
    constexpr uint32_t kPiFieldShift = 32;
    constexpr uint32_t kDbTypeSq = 21;
    constexpr uint32_t kSgidIdx = 1;
    constexpr uint32_t kCntxSize = 1;
    WriteU32Gm(sqCtx->headAddr, curHead);

    // Software shadow doorbell (big-endian PI).
    *reinterpret_cast<__ubuf__ uint32_t*>(ub) = Htobe32(curHead);
    WriteUbToGmWithSync(sqCtx->dbSwAddr, ub, sizeof(uint32_t), syncId);
    dcci((__gm__ void*)sqCtx->dbSwAddr, SINGLE_CACHE_LINE);
    QueueFence();

    // Hardware doorbell register: single 64-bit st_dev write.
    Hns1825SqDb db;
    db.value = 0;
    db.bs.c = 0;
    db.bs.rsvd0 = 0;
    db.bs.cntx_size = kCntxSize;
    db.bs.qpn = sqCtx->wqn;
    db.bs.sub_type = 0;
    db.bs.rsvd1 = 0;
    db.bs.pi = 0;
    db.bs.sgid_index = kSgidIdx;
    db.bs.type = kDbTypeSq;
    db.bs.mtu_shift = sqCtx->mtuShift;
    db.bs.cos = sqCtx->dbCos;
    uint64_t dbValue = db.value | ((((uint64_t)curHead >> kPiHighShift) & 0xffULL) << kPiFieldShift);

    st_dev(dbValue, (__gm__ uint64_t*)sqCtx->dbAddr, 0);
    pipe_barrier(PIPE_ALL);
}

// Only elected submitters and completion helpers contend for this lock. An older
// target can arrive after a newer one; it must not repeat owners or regress PI.
AICORE inline bool TryPostSend(__gm__ RoceSqCtx* sq, uint64_t end, __ubuf__ uint8_t* ub, uint32_t syncId)
{
    __gm__ QueueState* state = GetQueueState(sq);
    if (AtomicLoad(&state->postedHead) >= end) {
        return true;
    }
    if (!TryQueueLock(&state->postLock)) {
        return false;
    }
    const uint64_t posted = AtomicLoad(&state->postedHead);
    if (QueueError(state) == 0 && end > posted) {
        // The ready handoff follows each producer's completed MTE copies. HNS1825
        // owners are published here, in the same critical section as the DB.
        for (uint64_t index = posted; index < end; ++index) {
            const uint64_t address = reinterpret_cast<uint64_t>(GetSendWqe(sq, index & (sq->depth - 1U)));
            WriteU32Gm(address, ReadU32Gm(address) ^ 0x80U);
        }
        QueueFence();
        RingSqDoorbell(sq, static_cast<uint32_t>(end), ub, syncId);
        QueueFence();
        (void)atomicCAS(&state->postedHead, posted, end);
    }
    UnlockQueue(&state->postLock);
    return true;
}

AICORE inline uint32_t PostSend(__gm__ RoceSqCtx* sq, uint64_t end, __ubuf__ uint8_t* ub, uint32_t syncId)
{
    __gm__ QueueState* state = GetQueueState(sq);
    const uint64_t startCycles = static_cast<uint64_t>(AscendC::GetSystemCycle());
    while (true) {
        const uint32_t status = QueueError(state);
        if (status != 0) {
            return status;
        }
        if (TryPostSend(sq, end, ub, syncId)) {
            return QueueError(state);
        }
        if (QueueTimedOut(startCycles)) {
            return FailQueue(state, kHns1825PollCqTimeoutError);
        }
    }
}

AICORE inline uint32_t QueueCapacity(__gm__ RoceSqCtx* sq, __gm__ RoceCqCtx* cq)
{
    const uint32_t sqCapacity = sq->depth - kHns1825PollCqThreshold;
    const uint32_t cqCapacity = cq->depth - 1U;
    return sqCapacity < cqCapacity ? sqCapacity : cqCapacity;
}

// Reserve adjacent WQEBBs before preparing them. Both SQ and CQ capacity matter:
// each WQE is signaled, and CQ entries must not be overwritten before CI reaches
// the NIC. Reservations beyond capacity wait without touching their SQ slots.
AICORE inline uint32_t ReserveWqes(const RdmaExecContext& ctx, uint32_t count, uint64_t& start)
{
    __gm__ RdmaInfo* info = reinterpret_cast<__gm__ RdmaInfo*>(ctx.contextGm);
    const uint64_t index = static_cast<uint64_t>(ctx.destRankId) * info->qpNum + ctx.qpIdx;
    __gm__ RoceSqCtx* sq = reinterpret_cast<__gm__ RoceSqCtx*>(info->sqPtr) + index;
    __gm__ RoceCqCtx* cq = reinterpret_cast<__gm__ RoceCqCtx*>(info->scqPtr) + index;
    __gm__ QueueState* state = GetQueueState(sq);
    const uint32_t capacity = QueueCapacity(sq, cq);
    if (count > capacity) {
        return kHns1825InvalidContextError;
    }
    uint32_t status = QueueError(state);
    if (status != 0) {
        return status;
    }
    start = atomicAdd(&state->reserveHead, static_cast<uint64_t>(count));
    const uint64_t end = start + count;
    if (end > kMaxQueueIndex) {
        return FailQueue(state, kQueueIndexExhaustedError);
    }
    if (end > capacity) {
        status =
            PollCq(info, ctx.destRankId, ctx.qpIdx, static_cast<uint32_t>(end - capacity), ctx.tmpBuf.addr, ctx.syncId);
        if (status != 0) {
            return status;
        }
    }
    QueueFence();
    return QueueError(state);
}

// Hand off readiness after the MTE copies complete, in reservation order.
// As in IBGDA, only the reservation tail or a batch boundary must post-send;
// other producers can return early.
AICORE inline uint32_t SubmitReadyWqes(
    __gm__ RoceSqCtx* sq, __gm__ RoceCqCtx* cq, uint64_t start, uint64_t end, __ubuf__ uint8_t* ub, uint32_t syncId)
{
    __gm__ QueueState* state = GetQueueState(sq);
    const uint64_t startCycles = static_cast<uint64_t>(AscendC::GetSystemCycle());
    QueueFence();
    while (true) {
        const uint32_t status = QueueError(state);
        if (status != 0) {
            return status;
        }
        if (atomicCAS(&state->readyHead, start, end) == start) {
            break;
        }
        if (QueueTimedOut(startCycles)) {
            return FailQueue(state, kHns1825PollCqTimeoutError);
        }
    }
    QueueFence();
    uint32_t batch = kPostSendBatch;
    const uint32_t capacity = QueueCapacity(sq, cq);
    // A batch must fit even when the CQ is shallower than the SQ.
    while (batch > capacity) {
        batch >>= 1U;
    }
    const uint64_t mask = ~(static_cast<uint64_t>(batch) - 1U);
    const bool batchLimitReached = (start & mask) != (end & mask);
    const bool noConcurrentSubmissions = end == AtomicLoad(&state->reserveHead);
    return batchLimitReached || noConcurrentSubmissions ? PostSend(sq, end, ub, syncId) : QueueError(state);
}

// Post one WRITE/READ WQE. The event's index is this reservation's end, even
// when a different AIV publishes a larger prefix on its behalf.
template <RdmaOpcode OP>
AICORE inline PostSendResult PostSendReadWrite(const RdmaExecContext& ctx, RdmaSendWr& wr)
{
    __gm__ RdmaInfo* info = (__gm__ RdmaInfo*)ctx.contextGm;
    uint32_t pe = ctx.destRankId;
    uint32_t qpIdx = ctx.qpIdx;
    uint32_t qpNum = info->qpNum;
    __ubuf__ uint8_t* ub = ctx.tmpBuf.addr;
    uint32_t syncId = ctx.syncId;

    __gm__ RoceSqCtx* sqCtx = (__gm__ RoceSqCtx*)(info->sqPtr + ((uint64_t)pe * qpNum + qpIdx) * sizeof(RoceSqCtx));
    __gm__ RoceCqCtx* cqCtx = (__gm__ RoceCqCtx*)(info->scqPtr + ((uint64_t)pe * qpNum + qpIdx) * sizeof(RoceCqCtx));
    uint64_t start = 0;
    const uint32_t status = ReserveWqes(ctx, 1, start);
    if (status != 0) {
        return {0, status};
    }
    const uint32_t curHead = static_cast<uint32_t>(start);

    __gm__ RdmaMemInfo* remoteMem = (__gm__ RdmaMemInfo*)(info->memPtr + sizeof(RdmaMemInfo) * pe);
    __gm__ RdmaMemInfo* localMem = (__gm__ RdmaMemInfo*)(info->memPtr + sizeof(RdmaMemInfo) * ctx.myPe);
    wr.rkey = remoteMem->rkey;
    wr.lkey = localMem->lkey;

    __gm__ uint8_t* wqeAddr = GetSendWqe(sqCtx, curHead & (sqCtx->depth - 1));
    (void)FillWqeWriteRead(wr, sqCtx, wqeAddr, curHead, OP, ub, syncId);
    return {curHead + 1U, SubmitReadyWqes(sqCtx, cqCtx, start, start + 1U, ub, syncId)};
}

// Post payload WRITE followed by a fenced inline signal WRITE. Both WQEs are
// signaled and consume one WQEBB/CQE each, so the existing head/tail mapping
// remains valid. The SQ is published once after both WQEs are complete.
AICORE inline PostSendResult PostSendWriteNotify(
    const RdmaExecContext& ctx, RdmaSendWr& payloadWr, RdmaSendWr& signalWr, int32_t signalValue)
{
    __gm__ RdmaInfo* info = reinterpret_cast<__gm__ RdmaInfo*>(ctx.contextGm);
    const uint32_t pe = ctx.destRankId;
    const uint32_t qpIdx = ctx.qpIdx;
    const uint32_t qpNum = info->qpNum;
    __ubuf__ uint8_t* ub = ctx.tmpBuf.addr;
    const uint32_t syncId = ctx.syncId;

    __gm__ RoceSqCtx* sqCtx = reinterpret_cast<__gm__ RoceSqCtx*>(
        info->sqPtr + (static_cast<uint64_t>(pe) * qpNum + qpIdx) * sizeof(RoceSqCtx));
    __gm__ RoceCqCtx* cqCtx = reinterpret_cast<__gm__ RoceCqCtx*>(
        info->scqPtr + (static_cast<uint64_t>(pe) * qpNum + qpIdx) * sizeof(RoceCqCtx));
    const uint32_t depth = sqCtx->depth;
    uint64_t start = 0;
    const uint32_t status = ReserveWqes(ctx, 2, start);
    if (status != 0) {
        return {0, status};
    }
    uint32_t curHead = static_cast<uint32_t>(start);

    __gm__ RdmaMemInfo* remoteMem = reinterpret_cast<__gm__ RdmaMemInfo*>(info->memPtr + sizeof(RdmaMemInfo) * pe);
    __gm__ RdmaMemInfo* localMem = reinterpret_cast<__gm__ RdmaMemInfo*>(info->memPtr + sizeof(RdmaMemInfo) * ctx.myPe);
    payloadWr.rkey = remoteMem->rkey;
    payloadWr.lkey = localMem->lkey;
    signalWr.rkey = remoteMem->rkey;
    signalWr.lkey = localMem->lkey;

    __gm__ uint8_t* payloadWqe = GetSendWqe(sqCtx, curHead & (depth - 1U));
    (void)FillWqeWriteRead(payloadWr, sqCtx, payloadWqe, curHead, RdmaOpcode::OP_RDMA_WRITE, ub, syncId);

    ++curHead;
    __gm__ uint8_t* signalWqe = GetSendWqe(sqCtx, curHead & (depth - 1U));
    FillWqeInlineSet(signalWr, signalValue, sqCtx, signalWqe, curHead, ub, syncId);

    ++curHead;
    return {curHead, SubmitReadyWqes(sqCtx, cqCtx, start, start + 2U, ub, syncId)};
}

AICORE inline bool IsRangeInsideMr(uint64_t address, uint64_t length, __gm__ const RdmaMemInfo* mem)
{
    if (mem == nullptr || address < mem->addr || length == 0 || length > mem->size) {
        return false;
    }
    return address - mem->addr <= mem->size - length;
}

AICORE inline bool IsQueueValid(
    __gm__ RdmaInfo* info, uint32_t peer, uint32_t qpIdx, const RdmaTmpBuffer& tmpBuf, uint32_t syncId)
{
    if (!IsWorkspaceHeaderValid(info) || info->backend != RdmaBackend::HNS_1825 || peer >= info->rankCount ||
        qpIdx >= info->qpNum || info->sqPtr == 0 || info->scqPtr == 0 || tmpBuf.addr == nullptr ||
        tmpBuf.size < kHns1825WriteReadWqeSize || syncId > kHns1825MaxSyncId) {
        return false;
    }
    const uint64_t index = static_cast<uint64_t>(peer) * info->qpNum + qpIdx;
    __gm__ RoceSqCtx* sq = reinterpret_cast<__gm__ RoceSqCtx*>(info->sqPtr) + index;
    __gm__ RoceCqCtx* cq = reinterpret_cast<__gm__ RoceCqCtx*>(info->scqPtr) + index;
    const uint32_t cqeSize = cq->cqeSize == 0 ? kHns1825DefaultCqeSize : cq->cqeSize;
    return sq->stateAddr != 0 && (sq->stateAddr % alignof(QueueState)) == 0 && sq->depth > kHns1825PollCqThreshold &&
           (sq->depth & (sq->depth - 1U)) == 0 && cq->depth >= 4 && (cq->depth & (cq->depth - 1U)) == 0 &&
           (cqeSize == sizeof(Hns1825Cqe) || cqeSize == kHns1825DefaultCqeSize);
}

AICORE inline uint32_t ValidateTransfer(
    const RdmaExecContext& ctx, uint64_t localAddr, uint64_t remoteAddr, uint64_t len)
{
    __gm__ RdmaInfo* info = reinterpret_cast<__gm__ RdmaInfo*>(ctx.contextGm);
    if (!IsQueueValid(info, ctx.destRankId, ctx.qpIdx, ctx.tmpBuf, ctx.syncId) ||
        ctx.backend != RdmaBackend::HNS_1825 || ctx.destRankId >= info->rankCount || ctx.myPe >= info->rankCount ||
        info->memPtr == 0) {
        return kHns1825InvalidContextError;
    }
    if (len > kHns1825MaxTransferBytes) {
        return kHns1825InvalidArgumentError;
    }
    __gm__ RdmaMemInfo* remoteMem =
        reinterpret_cast<__gm__ RdmaMemInfo*>(info->memPtr + sizeof(RdmaMemInfo) * ctx.destRankId);
    __gm__ RdmaMemInfo* localMem = reinterpret_cast<__gm__ RdmaMemInfo*>(info->memPtr + sizeof(RdmaMemInfo) * ctx.myPe);
    return IsRangeInsideMr(localAddr, len, localMem) && IsRangeInsideMr(remoteAddr, len, remoteMem) ?
               0 :
               kHns1825InvalidArgumentError;
}

AICORE inline uint32_t ValidateNotifySignal(const RdmaExecContext& ctx, uint64_t remoteSignalAddr)
{
    if ((remoteSignalAddr & (alignof(int32_t) - 1U)) != 0U) {
        return kHns1825InvalidArgumentError;
    }
    __gm__ RdmaInfo* info = reinterpret_cast<__gm__ RdmaInfo*>(ctx.contextGm);
    __gm__ RdmaMemInfo* remoteMem =
        reinterpret_cast<__gm__ RdmaMemInfo*>(info->memPtr + sizeof(RdmaMemInfo) * ctx.destRankId);
    return IsRangeInsideMr(remoteSignalAddr, sizeof(int32_t), remoteMem) ? 0U : kHns1825InvalidArgumentError;
}

} // namespace detail

// ============================================================================
// HNS1825 entry points consumed by the RDMA device dispatcher.
// ============================================================================

AICORE inline bool BuildSession(
    __gm__ uint8_t* workspace, uint32_t myPe, const RdmaTmpBuffer& tmpBuf, uint32_t syncId, RdmaSession& session)
{
    session = {};
    __gm__ RdmaInfo* info = reinterpret_cast<__gm__ RdmaInfo*>(workspace);
    bool valid = IsWorkspaceHeaderValid(info) && info->backend == RdmaBackend::HNS_1825 && tmpBuf.addr != nullptr &&
                 tmpBuf.size >= kHns1825WriteReadWqeSize && syncId <= kHns1825MaxSyncId && myPe < info->rankCount &&
                 info->sqPtr != 0 && info->scqPtr != 0 && info->memPtr != 0;
    if (!valid) {
        return false;
    }

    session.execCtx.contextGm = workspace;
    session.execCtx.backend = RdmaBackend::HNS_1825;
    session.execCtx.destRankId = 0;
    session.execCtx.qpIdx = 0;
    session.execCtx.myPe = myPe;
    session.execCtx.tmpBuf = tmpBuf;
    session.execCtx.syncId = syncId;
    session.eventCtx.contextGm = workspace;
    session.eventCtx.backend = RdmaBackend::HNS_1825;
    session.eventCtx.tmpBuf = tmpBuf;
    session.eventCtx.syncId = syncId;
    session.valid = true;
    return true;
}

AICORE inline bool BuildSession(
    __gm__ uint8_t* workspace, uint32_t destRankId, uint32_t myPe, const RdmaTmpBuffer& tmpBuf, uint32_t syncId,
    RdmaSession& session)
{
    if (!BuildSession(workspace, myPe, tmpBuf, syncId, session)) {
        return false;
    }

    __gm__ RdmaInfo* info = reinterpret_cast<__gm__ RdmaInfo*>(workspace);
    bool valid = destRankId < info->rankCount;
    if (valid) {
        __gm__ RoceCqCtx* cqCtx = reinterpret_cast<__gm__ RoceCqCtx*>(
            info->scqPtr + static_cast<uint64_t>(destRankId) * info->qpNum * sizeof(RoceCqCtx));
        const uint32_t cqeSize = cqCtx->cqeSize == 0 ? kHns1825DefaultCqeSize : cqCtx->cqeSize;
        valid = cqeSize >= sizeof(Hns1825Cqe) && cqeSize <= tmpBuf.size;
    }
    if (!valid) {
        session = {};
        return false;
    }

    session.execCtx.destRankId = destRankId;
    return true;
}

// RDMA WRITE: dst is remote, src is local.
AICORE inline uint64_t Write(const RdmaExecContext& ctx, __gm__ uint8_t* dst, __gm__ uint8_t* src, uint64_t len)
{
    const uint32_t validateStatus =
        detail::ValidateTransfer(ctx, reinterpret_cast<uint64_t>(src), reinterpret_cast<uint64_t>(dst), len);
    if (validateStatus != 0) {
        return EncodeErrorHandle(validateStatus);
    }
    RdmaSendWr wr = {};
    wr.remote_addr = dst;
    wr.local_addr = src;
    wr.message_len = len;
    detail::PostSendResult result = detail::PostSendReadWrite<RdmaOpcode::OP_RDMA_WRITE>(ctx, wr);
    return result.status == 0 ? EncodeHandle(ctx.destRankId, result.curHead) : EncodeErrorHandle(result.status);
}

// RDMA WRITE followed by a fenced 4-byte inline RDMA WRITE to remoteSignal.
AICORE inline uint64_t WriteNotify(
    const RdmaExecContext& ctx, __gm__ uint8_t* dst, __gm__ uint8_t* src, uint64_t len, __gm__ int32_t* remoteSignal,
    int32_t signalValue)
{
    uint32_t status =
        detail::ValidateTransfer(ctx, reinterpret_cast<uint64_t>(src), reinterpret_cast<uint64_t>(dst), len);
    if (status == 0U) {
        status = detail::ValidateNotifySignal(ctx, reinterpret_cast<uint64_t>(remoteSignal));
    }
    if (status != 0U) {
        return EncodeErrorHandle(status);
    }

    RdmaSendWr payloadWr{};
    payloadWr.remote_addr = dst;
    payloadWr.local_addr = src;
    payloadWr.message_len = len;
    RdmaSendWr signalWr{};
    signalWr.remote_addr = reinterpret_cast<__gm__ uint8_t*>(remoteSignal);
    signalWr.message_len = sizeof(int32_t);

    const detail::PostSendResult result = detail::PostSendWriteNotify(ctx, payloadWr, signalWr, signalValue);
    return result.status == 0U ? EncodeHandle(ctx.destRankId, result.curHead) : EncodeErrorHandle(result.status);
}

// RDMA READ: dst is local, src is remote (remote goes into wr.remote_addr).
AICORE inline uint64_t Read(const RdmaExecContext& ctx, __gm__ uint8_t* dst, __gm__ uint8_t* src, uint64_t len)
{
    const uint32_t validateStatus =
        detail::ValidateTransfer(ctx, reinterpret_cast<uint64_t>(dst), reinterpret_cast<uint64_t>(src), len);
    if (validateStatus != 0) {
        return EncodeErrorHandle(validateStatus);
    }
    RdmaSendWr wr = {};
    wr.remote_addr = src;
    wr.local_addr = dst;
    wr.message_len = len;
    detail::PostSendResult result = detail::PostSendReadWrite<RdmaOpcode::OP_RDMA_READ>(ctx, wr);
    return result.status == 0 ? EncodeHandle(ctx.destRankId, result.curHead) : EncodeErrorHandle(result.status);
}

// Symmetric MR base VA of a peer (device VA), read from the RdmaInfo mem table.
// The kernel uses it to compute the remote target address: peerBase + offset.
AICORE inline uint64_t PeerMrBaseAddr(__gm__ uint8_t* rdmaWorkspace, uint32_t peer)
{
    __gm__ RdmaInfo* info = (__gm__ RdmaInfo*)rdmaWorkspace;
    if (!IsWorkspaceHeaderValid(info) || info->backend != RdmaBackend::HNS_1825 || peer >= info->rankCount ||
        info->memPtr == 0) {
        return 0;
    }
    __gm__ RdmaMemInfo* mem = (__gm__ RdmaMemInfo*)(info->memPtr + sizeof(RdmaMemInfo) * peer);
    return mem->addr;
}

// Blocking completion: poll CQ until WQEs up to curHead complete.
// Returns 0 on success, kHns1825PollCqTimeoutError on timeout, or CQE syndrome on error.
AICORE inline uint32_t WaitEventStatus(uint64_t handle, const RdmaEventContext& ctx)
{
    if (handle == 0) {
        return 0;
    }
    if (IsErrorHandle(handle)) {
        return static_cast<uint32_t>(handle);
    }
    uint32_t destRankId = 0;
    uint32_t curHead = 0;
    DecodeHandle(handle, destRankId, curHead);
    __gm__ RdmaInfo* info = (__gm__ RdmaInfo*)ctx.contextGm;
    if (!detail::IsQueueValid(info, destRankId, 0, ctx.tmpBuf, ctx.syncId) || ctx.backend != RdmaBackend::HNS_1825) {
        return kHns1825InvalidContextError;
    }
    return detail::PollCq(info, destRankId, /*qpIdx*/ 0, curHead, ctx.tmpBuf.addr, ctx.syncId);
}

AICORE inline bool WaitEvent(uint64_t handle, const RdmaEventContext& ctx) { return WaitEventStatus(handle, ctx) == 0; }

// Non-blocking completion check. Flush ready submissions and inspect a bounded
// CQ batch; neither attempt waits on a busy lock.
// True denotes a terminal event, including errors reported by WaitEventStatus.
AICORE inline bool TestEvent(uint64_t handle, const RdmaEventContext& ctx)
{
    if (handle == 0) {
        return true;
    }
    if (IsErrorHandle(handle)) {
        return true;
    }
    uint32_t destRankId = 0;
    uint32_t curHead = 0;
    DecodeHandle(handle, destRankId, curHead);

    __gm__ RdmaInfo* info = (__gm__ RdmaInfo*)ctx.contextGm;
    if (!detail::IsQueueValid(info, destRankId, 0, ctx.tmpBuf, ctx.syncId) || ctx.backend != RdmaBackend::HNS_1825) {
        return true;
    }
    __gm__ RoceSqCtx* sq =
        reinterpret_cast<__gm__ RoceSqCtx*>(info->sqPtr) + static_cast<uint64_t>(destRankId) * info->qpNum;
    __gm__ QueueState* state = detail::GetQueueState(sq);
    if (detail::AtomicLoad(&state->completedHead) < curHead && detail::QueueError(state) == 0) {
        detail::SubmitReady(sq, ctx.tmpBuf.addr, ctx.syncId);
        detail::ProgressCq(info, destRankId, 0, ctx.tmpBuf.addr, ctx.syncId);
    }
    const bool terminal = detail::AtomicLoad(&state->completedHead) >= curHead || detail::QueueError(state) != 0;
    if (terminal) {
        detail::QueueFence();
    }
    return terminal;
}

// Publish the last expected completion's CQE as a record a non-AICore consumer can poll with one
// read: the address it must watch and the owner value that marks the slot done. `expected` is the
// value a finished slot carries, which is the lap parity inverted -- CheckCqeOwner reports ready
// when the slot owner DIFFERS from that parity. Publishing the parity itself would invert the
// predicate for every consumer.
AICORE inline bool GetRdmaCompletionRecord(uint64_t handle, const RdmaEventContext& ctx, AsyncCompletionRecord& record)
{
    if (handle == 0 || IsErrorHandle(handle)) {
        return false;
    }
    uint32_t destRankId = 0;
    uint32_t curHead = 0;
    DecodeHandle(handle, destRankId, curHead);
    if (curHead == 0U) {
        return false;
    }

    __gm__ RdmaInfo* info = (__gm__ RdmaInfo*)ctx.contextGm;
    if (!IsWorkspaceHeaderValid(info)) {
        return false;
    }
    uint32_t qpNum = info->qpNum;
    __gm__ RoceCqCtx* cqCtx =
        (__gm__ RoceCqCtx*)(info->scqPtr + ((uint64_t)destRankId * qpNum + 0) * sizeof(RoceCqCtx));

    uint32_t cqeSize = cqCtx->cqeSize == 0 ? kHns1825DefaultCqeSize : cqCtx->cqeSize;
    uint32_t cqRing = cqCtx->depth;
    if (cqCtx->bufAddr == 0U || cqRing == 0U) {
        return false;
    }

    const uint32_t lastIdx = curHead - 1U;
    record.addr = reinterpret_cast<__gm__ uint8_t*>(cqCtx->bufAddr + (uint64_t)(lastIdx & (cqRing - 1U)) * cqeSize);
    record.expected = static_cast<uint64_t>(((lastIdx & cqRing) == 0U) ? 0U : 1U);
    record.kind = CompletionKind::RDMA_HNS1825_CQE;
    return true;
}

} // namespace hns_1825
} // namespace rdma
} // namespace comm
} // namespace pto

#endif // PTO_COMM_ASYNC_RDMA_BACKENDS_HNS_1825_BACKEND_HPP
