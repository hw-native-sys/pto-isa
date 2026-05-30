/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_NPU_COMM_ASYNC_URMA_INTRIN_HPP
#define PTO_NPU_COMM_ASYNC_URMA_INTRIN_HPP

#ifdef PTO_URMA_SUPPORTED

#include "pto/common/debug.h"
#include "pto/comm/async_common/async_types.hpp"
#include "pto/npu/comm/async/urma/urma_types.hpp"

namespace pto {
namespace comm {
namespace urma {

// ============================================================================
// DcciCachelines — flush multiple cache lines covering [addr, addr+length)
// ============================================================================
AICORE inline void DcciCachelines(__gm__ uint8_t *addr, uint64_t length)
{
    __gm__ uint8_t *start = (__gm__ uint8_t *)((uint64_t)addr / kCacheLineSize * kCacheLineSize);
    __gm__ uint8_t *end = (__gm__ uint8_t *)(((uint64_t)addr + length) / kCacheLineSize * kCacheLineSize);
    for (uint64_t i = 0; i <= static_cast<uint64_t>(end - start); i += kCacheLineSize) {
        __asm__ __volatile__("");
        dcci((__gm__ void *)(start + i), SINGLE_CACHE_LINE);
        __asm__ __volatile__("");
    }
}

namespace detail {

// ============================================================================
// UrmaPollCqUpdateInfo — update CQ/WQ tail and ring CQ doorbell after polling (URMA)
// ============================================================================
AICORE inline void UrmaPollCqUpdateInfo(uint32_t curTail, __gm__ UrmaCqCtx *cqCtxEntry, __gm__ UrmaWQCtx *wqCtxEntry)
{
    __gm__ uint32_t *dbAddr = (__gm__ uint32_t *)cqCtxEntry->dbAddr;
    st_dev(static_cast<uint32_t>(curTail & 0xFFFFFF), dbAddr, 0);

    __gm__ uint32_t *wqTailAddr = (__gm__ uint32_t *)wqCtxEntry->tailAddr;
    st_dev(curTail, wqTailAddr, 0);
}

// ============================================================================
// UrmaPollCq — poll completion queue until idx entries are consumed
// Returns 0 on success, non-zero on error/timeout.
// ============================================================================
AICORE inline uint32_t UrmaPollCq(__gm__ uint8_t *contextGm, uint32_t destRankId, uint32_t qpIdx, uint32_t idx)
{
    if (idx == 0) {
        return 0;
    }

    __gm__ UrmaInfo *urmaInfo = (__gm__ UrmaInfo *)contextGm;
    uint32_t qpNum = urmaInfo->qpNum;

    __gm__ UrmaCqCtx *cqCtxEntry =
        (__gm__ UrmaCqCtx *)(urmaInfo->scqPtr + (destRankId * qpNum + qpIdx) * sizeof(UrmaCqCtx));
    uint64_t cqBaseAddr = cqCtxEntry->bufAddr;
    uint32_t cqeSize = 1U << cqCtxEntry->cqeShiftSize;
    uint32_t depth = cqCtxEntry->depth;

    uint32_t curTail = ld_dev((__gm__ uint32_t *)cqCtxEntry->tailAddr, 0);

    while (curTail != idx) {
        __gm__ UrmaJfcCqeCtx *cqeAddr = (__gm__ UrmaJfcCqeCtx *)(cqBaseAddr + cqeSize * (curTail & (depth - 1)));
        bool validOwner = (curTail / depth) & 1;

        uint32_t times = 0;
        while ((validOwner ^ cqeAddr->owner) == 0 && times < kUrmaMaxPollTimes) {
            DcciCachelines((__gm__ uint8_t *)cqeAddr, sizeof(UrmaJfcCqeCtx));
            times++;
        }
        if (times >= kUrmaMaxPollTimes) {
            trap();
            return 0xFF;
        }

        curTail++;

        uint8_t status = cqeAddr->status & 0xFF;
        uint8_t subStatus = cqeAddr->substatus & 0xFF;
        constexpr uint8_t kStatusShift = 8;
        if (status != 0 || subStatus != 0) {
            return (status << kStatusShift) | subStatus;
        }
    }

    st_dev(curTail, (__gm__ uint32_t *)cqCtxEntry->tailAddr, 0);

    __gm__ UrmaWQCtx *wqCtxEntry =
        (__gm__ UrmaWQCtx *)(urmaInfo->sqPtr + (destRankId * qpNum + qpIdx) * sizeof(UrmaWQCtx));
    UrmaPollCqUpdateInfo(curTail, cqCtxEntry, wqCtxEntry);

    return 0;
}

// ============================================================================
// UrmaPostSendUpdateInfo — ring SQ doorbell and update head
// ============================================================================
AICORE inline void UrmaPostSendUpdateInfo(uint32_t curHead, __gm__ UrmaWQCtx *qpCtxEntry)
{
    __gm__ uint32_t *doorBellAddr = (__gm__ uint32_t *)qpCtxEntry->dbAddr;
    st_dev(curHead, doorBellAddr, 0);
    st_dev(curHead, (__gm__ uint32_t *)qpCtxEntry->headAddr, 0);
}

// ============================================================================
// FillSqeCtx — populate SQE fields from remote memory info and opcode
// ============================================================================
AICORE inline void FillSqeCtx(__gm__ UrmaSqeCtx *sqeCtx, __gm__ UrmaMemInfo *remoteMemInfo, __gm__ uint8_t *remoteAddr,
                              UrmaOpcode opcode, uint32_t curHead, uint32_t depth)
{
    sqeCtx->sqeBbIdx = static_cast<uint16_t>(curHead % depth);
    sqeCtx->opcode = static_cast<uint32_t>(opcode);
    sqeCtx->flag = 0b00100010;
    sqeCtx->rsv0 = 0;
    sqeCtx->nf = 0;
    sqeCtx->tokenEn = remoteMemInfo->tokenValueValid;
    sqeCtx->rmtJettyType = remoteMemInfo->rmtJettyType;
    sqeCtx->owner = (curHead & (depth << kMaxSgeNumShift)) == 0 ? 1 : 0;
    sqeCtx->targetHint = remoteMemInfo->targetHint;
    sqeCtx->rsv1 = 0;
    sqeCtx->inlineMsgLen = 0;
    sqeCtx->tpId = remoteMemInfo->tpn;
    sqeCtx->sgeNum = 1;
    sqeCtx->rmtJettyOrSegId = remoteMemInfo->tid;
    sqeCtx->rsv2 = 0;
    sqeCtx->rmtTokenValue = remoteMemInfo->rmtTokenValue;
    sqeCtx->udfType = 0;
    sqeCtx->reduceDataType = 0;
    sqeCtx->reduceOpcode = 0;
    sqeCtx->rsv3 = 0;

    uint64_t remoteAddrValue = reinterpret_cast<uint64_t>(remoteAddr);
    sqeCtx->rmtAddrLOrTokenId = static_cast<uint32_t>(remoteAddrValue & 0xFFFFFFFF);
    sqeCtx->rmtAddrHOrTokenValue = static_cast<uint32_t>((remoteAddrValue >> 32) & 0xFFFFFFFF);

    __gm__ uint64_t *rmtEid = (__gm__ uint64_t *)(remoteMemInfo->eidAddr);
    sqeCtx->rmtEidL = rmtEid[0];
    sqeCtx->rmtEidH = rmtEid[1];
}

// ============================================================================
// UrmaPostSend — prepare WQE+SGE, flush cache, ring doorbell
// Returns curHead after post (used to encode AsyncEvent handle).
// ============================================================================
AICORE inline uint32_t UrmaPostSend(__gm__ uint8_t *contextGm, __gm__ uint8_t *remoteAddr, __gm__ uint8_t *localAddr,
                                    uint32_t destRankId, uint32_t qpIdx, UrmaOpcode opcode, uint64_t messageLen)
{
    __gm__ UrmaInfo *urmaInfo = (__gm__ UrmaInfo *)contextGm;
    PTO_ASSERT(destRankId < urmaInfo->rankCount, "UrmaPostSend: destRankId out of range");
    uint32_t qpNum = urmaInfo->qpNum;

    __gm__ UrmaWQCtx *qpCtxEntry =
        (__gm__ UrmaWQCtx *)(urmaInfo->sqPtr + (destRankId * qpNum + qpIdx) * sizeof(UrmaWQCtx));
    uint32_t wqeSize = 1U << qpCtxEntry->wqeShiftSize;
    uint32_t depth = qpCtxEntry->depth;

    uint32_t curHead = ld_dev((__gm__ uint32_t *)qpCtxEntry->headAddr, 0);
    uint32_t curTail = ld_dev((__gm__ uint32_t *)qpCtxEntry->tailAddr, 0);

    if ((curHead + kUrmaPollCqThreshold) % depth == curTail % depth) {
        (void)UrmaPollCq(contextGm, destRankId, qpIdx, curTail + kNumCqePerPollCq);
    }

    __gm__ UrmaMemInfo *remoteMemInfo = (__gm__ UrmaMemInfo *)(urmaInfo->memPtr + sizeof(UrmaMemInfo) * destRankId);

    __gm__ uint8_t *wqeAddr = (__gm__ uint8_t *)(qpCtxEntry->bufAddr + wqeSize * (curHead % depth));
    FillSqeCtx((__gm__ UrmaSqeCtx *)wqeAddr, remoteMemInfo, remoteAddr, opcode, curHead, depth);

    __gm__ UrmaSgeCtx *sgeCtx = (__gm__ UrmaSgeCtx *)(wqeAddr + sizeof(UrmaSqeCtx));
    sgeCtx->len = static_cast<uint32_t>(messageLen);
    sgeCtx->tokenId = urmaInfo->localTokenId;
    sgeCtx->va = reinterpret_cast<uint64_t>(localAddr);

    DcciCachelines(wqeAddr, sizeof(UrmaSqeCtx) + sizeof(UrmaSgeCtx));
    curHead++;
    UrmaPostSendUpdateInfo(curHead, qpCtxEntry);

    return curHead;
}

// ============================================================================
// Handle encoding/decoding for AsyncEvent
// ============================================================================
constexpr uint32_t kHandleRankIdShift = 32;

AICORE inline uint64_t EncodeHandle(uint32_t destRankId, uint32_t curHead)
{
    return (static_cast<uint64_t>(destRankId) << kHandleRankIdShift) | static_cast<uint64_t>(curHead);
}

AICORE inline void DecodeHandle(uint64_t handle, uint32_t &destRankId, uint32_t &curHead)
{
    destRankId = static_cast<uint32_t>(handle >> kHandleRankIdShift);
    curHead = static_cast<uint32_t>(handle & 0xFFFFFFFF);
}

// ============================================================================
// UrmaWaitEvent — blocking wait for URMA completion (polls CQ)
// ============================================================================
AICORE inline bool UrmaWaitEvent(uint64_t eventHandle, const UrmaEventContext &eventCtx)
{
    uint32_t destRankId = 0;
    uint32_t curHead = 0;
    DecodeHandle(eventHandle, destRankId, curHead);
    uint32_t ret = UrmaPollCq(eventCtx.contextGm, destRankId, 0, curHead);
    return ret == 0;
}

// ============================================================================
// UrmaTestEvent — non-blocking completion check
// ============================================================================
AICORE inline bool UrmaTestEvent(uint64_t eventHandle, const UrmaEventContext &eventCtx)
{
    uint32_t destRankId = 0;
    uint32_t curHead = 0;
    DecodeHandle(eventHandle, destRankId, curHead);

    __gm__ UrmaInfo *urmaInfo = (__gm__ UrmaInfo *)eventCtx.contextGm;
    uint32_t qpNum = urmaInfo->qpNum;
    __gm__ UrmaCqCtx *cqCtxEntry =
        (__gm__ UrmaCqCtx *)(urmaInfo->scqPtr + (destRankId * qpNum + 0) * sizeof(UrmaCqCtx));
    uint32_t curTail = ld_dev((__gm__ uint32_t *)cqCtxEntry->tailAddr, 0);
    if (static_cast<int32_t>(curTail - curHead) >= 0) {
        return true;
    }

    uint64_t cqBaseAddr = cqCtxEntry->bufAddr;
    uint32_t cqeSize = 1U << cqCtxEntry->cqeShiftSize;
    uint32_t depth = cqCtxEntry->depth;

    uint32_t lastIdx = curHead - 1;
    __gm__ UrmaJfcCqeCtx *lastCqe = (__gm__ UrmaJfcCqeCtx *)(cqBaseAddr + cqeSize * (lastIdx & (depth - 1)));
    bool validOwner = (lastIdx / depth) & 1;

    DcciCachelines((__gm__ uint8_t *)lastCqe, sizeof(UrmaJfcCqeCtx));

    return (validOwner ^ lastCqe->owner) != 0;
}

} // namespace detail

// ============================================================================
// Public API: __urma_put_async / __urma_get_async
// ============================================================================

AICORE inline uint64_t __urma_put_async(__gm__ uint8_t *dst, __gm__ uint8_t *src, uint64_t transferSize,
                                        const UrmaExecContext &execCtx)
{
    uint32_t curHead = detail::UrmaPostSend(execCtx.contextGm, dst, src, execCtx.destRankId, execCtx.qpIdx,
                                            UrmaOpcode::WRITE, transferSize);
    return detail::EncodeHandle(execCtx.destRankId, curHead);
}

AICORE inline uint64_t __urma_get_async(__gm__ uint8_t *dst, __gm__ uint8_t *src, uint64_t transferSize,
                                        const UrmaExecContext &execCtx)
{
    // RDMA READ: remote addr = src (SQE remote field), local addr = dst (SGE.va)
    uint32_t curHead = detail::UrmaPostSend(execCtx.contextGm, src, dst, execCtx.destRankId, execCtx.qpIdx,
                                            UrmaOpcode::READ, transferSize);
    return detail::EncodeHandle(execCtx.destRankId, curHead);
}

// ============================================================================
// BuildUrmaSession — fill UrmaSession from workspace and destRankId
// ============================================================================

AICORE inline bool BuildUrmaSession(__gm__ uint8_t *contextGm, uint32_t destRankId, UrmaSession &session)
{
    session.execCtx.contextGm = contextGm;
    session.execCtx.destRankId = destRankId;
    session.execCtx.qpIdx = 0;
    session.eventCtx.contextGm = contextGm;
    session.valid = (contextGm != nullptr);
    return session.valid;
}

// ============================================================================
// UrmaPeerMrBaseAddr — symmetric MR base (device VA) for peer index peerRank
//
// Indexes into the per-peer UrmaMemInfo array at memPtr + sizeof(UrmaMemInfo) * peerRank.
// peerRank uses UrmaWorkspaceManager allgather order (MPI rank order, 0 .. rankCount-1).
// ============================================================================
AICORE inline uint64_t UrmaPeerMrBaseAddr(__gm__ uint8_t *urmaWorkspace, uint32_t peerRank)
{
    __gm__ UrmaInfo *info = (__gm__ UrmaInfo *)urmaWorkspace;
    PTO_ASSERT(peerRank < info->rankCount, "UrmaPeerMrBaseAddr: peerRank out of range");
    __gm__ UrmaMemInfo *memRow = reinterpret_cast<__gm__ UrmaMemInfo *>(info->memPtr) + peerRank;
    return memRow->addr;
}

} // namespace urma
} // namespace comm
} // namespace pto

#endif // PTO_URMA_SUPPORTED
#endif // PTO_NPU_COMM_ASYNC_URMA_INTRIN_HPP
