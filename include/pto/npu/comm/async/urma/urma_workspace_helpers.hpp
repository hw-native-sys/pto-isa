/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_NPU_COMM_ASYNC_URMA_WORKSPACE_HELPERS_HPP
#define PTO_NPU_COMM_ASYNC_URMA_WORKSPACE_HELPERS_HPP

#include <cstdint>
#include <iostream>

#include "acl/acl.h"

#include "pto/npu/comm/async/urma/urma_types.hpp"
#include "pto/npu/comm/async/urma/urma_hccp_loader.hpp"

namespace pto {
namespace comm {
namespace urma {

// ============================================================================
// UrmaBootstrapHandle: generic cross-rank information exchange abstraction.
// ============================================================================
struct UrmaBootstrapHandle {
    int (*allgather)(const void *sendbuf, void *recvbuf, int size, void *ctx);
    int (*barrier)(void *ctx);
    void *ctx;
};

inline uint32_t Log2U32(uint32_t n)
{
    return (n <= 1) ? 0 : __builtin_ctz(n);
}

// Byte-swap and cross-swap the two 64-bit halves of an EID (network -> host byte order).
// Uses __builtin_memcpy to avoid strict-aliasing UB when type-punning uint8_t[] <-> uint64_t.
inline void SwapEidByteOrder(hccp::HccpEid &eid)
{
    static_assert(sizeof(hccp::HccpEid) == 16, "EID must be 16 bytes");
    uint64_t lo, hi;
    __builtin_memcpy(&lo, eid.raw, sizeof(uint64_t));
    __builtin_memcpy(&hi, eid.raw + sizeof(uint64_t), sizeof(uint64_t));
    lo = __builtin_bswap64(lo);
    hi = __builtin_bswap64(hi);
    __builtin_memcpy(eid.raw, &hi, sizeof(uint64_t));
    __builtin_memcpy(eid.raw + sizeof(uint64_t), &lo, sizeof(uint64_t));
}

// Compute UrmaInfo pointer layout from a base address.
// Sets sqPtr/rqPtr/scqPtr/rcqPtr/memPtr in copyInfo relative to baseAddr.
inline void ComputeInfoLayout(UrmaInfo *copyInfo, uint8_t *baseAddr, uint32_t rankCount)
{
    constexpr uint32_t qpNum = 1;
    uint8_t *cursor = baseAddr + sizeof(UrmaInfo);
    copyInfo->sqPtr = reinterpret_cast<uint64_t>(cursor);
    cursor += sizeof(UrmaWQCtx) * rankCount * qpNum;
    copyInfo->rqPtr = reinterpret_cast<uint64_t>(cursor);
    cursor += sizeof(UrmaWQCtx) * rankCount * qpNum;
    copyInfo->scqPtr = reinterpret_cast<uint64_t>(cursor);
    cursor += sizeof(UrmaCqCtx) * rankCount * qpNum;
    copyInfo->rcqPtr = reinterpret_cast<uint64_t>(cursor);
    cursor += sizeof(UrmaCqCtx) * rankCount * qpNum;
    copyInfo->memPtr = reinterpret_cast<uint64_t>(cursor);
}

inline bool AllocAndCopyEidTable(void *&hccpEidDevice, const std::vector<hccp::HccpEid> &hccpEidList,
                                 uint32_t rankCount)
{
    size_t eidTableSize = rankCount * sizeof(hccp::HccpEid);
    aclError err = aclrtMalloc(&hccpEidDevice, eidTableSize, ACL_MEM_MALLOC_HUGE_FIRST);
    if (err != ACL_SUCCESS) {
        std::cerr << "[URMA] aclrtMalloc for hccpEid failed: " << err << std::endl;
        return false;
    }
    err = aclrtMemcpy(hccpEidDevice, eidTableSize, hccpEidList.data(), eidTableSize, ACL_MEMCPY_HOST_TO_DEVICE);
    if (err != ACL_SUCCESS) {
        std::cerr << "[URMA] aclrtMemcpy for hccpEid failed: " << err << std::endl;
        return false;
    }
    return true;
}

inline void FillPerRankData(UrmaInfo *copyInfo, const std::vector<UrmaWQCtx> &wqInfoList,
                            const std::vector<UrmaCqCtx> &cqInfoList, std::vector<UrmaMemInfo> &ubMemInfoList,
                            const std::vector<uint32_t> &tpnList, void *hccpEidDevice, uint32_t rankId,
                            uint32_t rankCount)
{
    auto &localWq = wqInfoList[rankId];
    auto &localCq = cqInfoList[rankId];

    for (uint32_t rank = 0; rank < rankCount; ++rank) {
        ubMemInfoList[rank].tpn = tpnList[rank];

        reinterpret_cast<UrmaWQCtx *>(copyInfo->sqPtr)[rank] = localWq;
        reinterpret_cast<UrmaWQCtx *>(copyInfo->rqPtr)[rank] = localWq;
        reinterpret_cast<UrmaCqCtx *>(copyInfo->scqPtr)[rank] = localCq;
        reinterpret_cast<UrmaCqCtx *>(copyInfo->rcqPtr)[rank] = localCq;
        reinterpret_cast<UrmaMemInfo *>(copyInfo->memPtr)[rank] = ubMemInfoList[rank];
        reinterpret_cast<UrmaMemInfo *>(copyInfo->memPtr)[rank].eidAddr =
            reinterpret_cast<uint64_t>(static_cast<hccp::HccpEid *>(hccpEidDevice) + rank);
    }
}

// Allocate a device-side uint32_t counter, zero-initialized.
inline bool AllocDeviceCounter(void *&addr, const char *name)
{
    if (aclrtMalloc(&addr, sizeof(uint32_t), ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS || !addr) {
        std::cerr << "[URMA] aclrtMalloc for " << name << " failed" << std::endl;
        return false;
    }
    aclrtMemset(addr, sizeof(uint32_t), 0, sizeof(uint32_t));
    return true;
}

inline void FreeDeviceAddr(void *&addr)
{
    if (addr) {
        aclrtFree(addr);
        addr = nullptr;
    }
}

// Step 1: Open TSD (process-level singleton)
inline bool OpenTsdIfNeeded(uint32_t deviceId, bool &tsdOpened)
{
    if (tsdOpened)
        return true;
    auto &api = HccpV2Loader::Instance();
    hccp::ProcOpenArgs args{};
    args.procType = hccp::TSD_SUB_PROC_HCCP;
    char paramStr[] = "--hdcType=18";
    hccp::ProcExtParam extParam{paramStr, sizeof("--hdcType=18")};
    args.extParamList = &extParam;
    args.extParamCnt = 1;
    int subPid = 0;
    args.subPid = &subPid;

    int ret = api.tsdProcessOpen(deviceId, &args);
    if (ret != 0) {
        std::cerr << "[URMA] TsdProcessOpen failed: " << ret << std::endl;
        return false;
    }
    tsdOpened = true;
    return true;
}

// Step 2: Initialize RA (process-level singleton)
inline bool RaInitIfNeeded(uint32_t deviceId, bool &raInitialized)
{
    if (raInitialized)
        return true;
    auto &api = HccpV2Loader::Instance();
    hccp::RaInitConfig config{};
    config.phyId = deviceId;
    config.nicPosition = hccp::NETWORK_OFFLINE;
    config.hdcType = hccp::HDC_SERVICE_TYPE_RDMA_V2;
    config.enableHdcAsync = true;

    int ret = api.raInit(&config);
    if (ret != 0) {
        std::cerr << "[URMA] RaInit failed: " << ret << std::endl;
        return false;
    }
    raInitialized = true;
    return true;
}

// Initialize default QP creation attributes.
inline void InitDefaultQpAttr(hccp::QpCreateAttr &qpAttr, void *cqHandle, void *tokenIdHandle,
                              hccp::TransportModeT transportMode)
{
    qpAttr.scqHandle = cqHandle;
    qpAttr.rcqHandle = cqHandle;
    qpAttr.srqHandle = cqHandle;
    qpAttr.sqDepth = hccp::kSqDepthDefault;
    qpAttr.rqDepth = hccp::kRqDepthDefault;
    qpAttr.transportMode = transportMode;
    qpAttr.ub.mode = hccp::JETTY_MODE_USER_CTL_NORMAL;
    qpAttr.ub.jettyId = 0;
    qpAttr.ub.flag.value = 1;
    constexpr uint32_t kJfsFlagValueDefault = 2;
    qpAttr.ub.jfsFlag.value = kJfsFlagValueDefault;
    qpAttr.ub.tokenValue = hccp::kTokenValue;
    qpAttr.ub.priority = 0;
    qpAttr.ub.rnrRetry = hccp::kRnrRetryCountDefault;
    qpAttr.ub.errTimeout = 0;
    qpAttr.ub.extMode.piType = false;
    qpAttr.ub.extMode.cstmFlag.bs.sqCstm = 0;
    qpAttr.ub.extMode.sqebbNum = hccp::kSqDepthDefault;
    qpAttr.ub.tokenIdHandle = tokenIdHandle;
}

// Query the first available EID for a given device.
inline bool QueryFirstEid(uint32_t deviceId, hccp::HccpEid &eid, unsigned int &eidIndex)
{
    auto &api = HccpV2Loader::Instance();
    hccp::RaInfo info{hccp::NETWORK_OFFLINE, deviceId};

    unsigned int eidNum = 0;
    int ret = api.raGetDevEidInfoNum(info, &eidNum);
    if (ret != 0 || eidNum == 0) {
        std::cerr << "[URMA] RaGetDevEidInfoNum failed: ret=" << ret << " eidNum=" << eidNum << std::endl;
        return false;
    }

    std::vector<hccp::DevEidInfo> eidInfoList(eidNum);
    unsigned int infoListNum = eidNum;
    ret = api.raGetDevEidInfoList(info, eidInfoList.data(), &infoListNum);
    if (ret != 0 || infoListNum != eidNum) {
        std::cerr << "[URMA] RaGetDevEidInfoList failed: ret=" << ret << std::endl;
        return false;
    }

    eid = eidInfoList[0].eid;
    eidIndex = eidInfoList[0].eidIndex;
    return true;
}

inline void PopulateMrResult(hccp::RegMemResultInfo &localMR, const hccp::MrRegInfoT &mrInfo, void *lmemHandle,
                             void *tokenIdHandle)
{
    localMR.address = mrInfo.in.mem.addr;
    localMR.size = mrInfo.in.mem.size;
    localMR.lmemHandle = lmemHandle;
    localMR.key = mrInfo.out.key;
    localMR.tokenId = mrInfo.out.ub.tokenId;
    localMR.tokenValue = hccp::kTokenValue;
    localMR.targetSegHandle = mrInfo.out.ub.targetSegHandle;
    localMR.tokenIdHandle = tokenIdHandle;
    localMR.cacheable = 0;
    localMR.access = hccp::MEM_SEG_ACCESS_DEFAULT;
}

inline void PopulateLocalMemInfo(UrmaMemInfo &memInfo, uint32_t tokenId, uint64_t symmetricSize, void *symmetricAddr)
{
    memInfo.tokenValueValid = true;
    memInfo.rmtJettyType = 1;
    memInfo.targetHint = 0;
    memInfo.tpn = 0;
    constexpr uint32_t kTokenIdTidShift = 8;
    memInfo.tid = tokenId >> kTokenIdTidShift;
    memInfo.rmtTokenValue = hccp::kTokenValue;
    memInfo.len = static_cast<uint32_t>(symmetricSize);
    memInfo.addr = reinterpret_cast<uint64_t>(symmetricAddr);
}

} // namespace urma
} // namespace comm
} // namespace pto

#endif // PTO_NPU_COMM_ASYNC_URMA_WORKSPACE_HELPERS_HPP
