/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_COMM_ASYNC_URMA_CHANNEL_HELPER_HPP
#define PTO_COMM_ASYNC_URMA_CHANNEL_HELPER_HPP

#if defined(__CCE_KT_TEST__)
#error "urma_channel_helper.hpp is a host-only header and cannot be included in device code."
#endif

#include <cstdint>
#include <cstring>
#include <iostream>

#include "securec.h"

#include "acl/acl.h"
#include "hccl/hccl.h"
#include "hccl/hccl_res.h"

#include "pto/comm/async/urma/urma_hccl_defs.hpp"

namespace pto {
namespace comm {
namespace urma {

// ============================================================================
// UrmaChannelHelper: reads ChannelEntity and sub-structures from device memory.
// ============================================================================
class UrmaChannelHelper {
public:
    static bool IsLikelyDevicePtr(const void *ptr)
    {
        return reinterpret_cast<uintptr_t>(ptr) >= kDeviceVaThreshold;
    }

    static bool IsValidChannelEntityHeader(const ChannelEntity &entity)
    {
        const uint32_t magic = entity.abiHeader.magicWord;
        if (magic != kHcclChannelEntityMagic && magic != kHcommChannelEntityMagic) {
            return false;
        }
        return entity.engine == COMM_ENGINE_AIV;
    }

    static bool CopyChannelSubStruct(const void *srcPtr, void *dst, size_t size, uint32_t peer, const char *name)
    {
        if (srcPtr == nullptr) {
            return false;
        }
        if (IsLikelyDevicePtr(srcPtr)) {
            aclError err = aclrtMemcpy(dst, size, srcPtr, size, ACL_MEMCPY_DEVICE_TO_HOST);
            if (err != ACL_SUCCESS) {
                std::cerr << "[URMA] aclrtMemcpy(" << name << ") peer=" << peer << " err=" << err << std::endl;
                return false;
            }
            return true;
        }
        errno_t rc = memcpy_s(dst, size, srcPtr, size);
        return rc == EOK;
    }

    static bool FillChannelSubStructs(uint32_t peer, const ChannelEntity &hostEntity, SqContext &sq, CqContext &cq,
                                      RegedBufferEntity &remoteBuf, RegedBufferEntity &localBuf)
    {
        if (hostEntity.sqContextAddr != nullptr && hostEntity.sqNum > 0) {
            if (!CopyChannelSubStruct(hostEntity.sqContextAddr, &sq, sizeof(SqContext), peer, "SqContext")) {
                return false;
            }
        }

        if (hostEntity.cqContextAddr != nullptr && hostEntity.cqNum > 0) {
            if (!CopyChannelSubStruct(hostEntity.cqContextAddr, &cq, sizeof(CqContext), peer, "CqContext")) {
                return false;
            }
        }

        if (hostEntity.remoteBufferAddr != nullptr && hostEntity.remoteBufferNum > 0) {
            if (!CopyChannelSubStruct(hostEntity.remoteBufferAddr, &remoteBuf, sizeof(RegedBufferEntity), peer,
                                      "RemoteBuffer")) {
                return false;
            }
        }

        if (hostEntity.localBufferAddr != nullptr && hostEntity.localBufferNum > 0) {
            if (!CopyChannelSubStruct(hostEntity.localBufferAddr, &localBuf, sizeof(RegedBufferEntity), peer,
                                      "LocalBuffer")) {
                return false;
            }
        }
        return true;
    }

    static bool TryReadChannelEntity(ChannelHandle handle, uint32_t peer, ChannelEntity &hostEntity, SqContext &sq,
                                     CqContext &cq, RegedBufferEntity &remoteBuf, RegedBufferEntity &localBuf)
    {
        void *devEntityPtr = reinterpret_cast<void *>(static_cast<uintptr_t>(handle));
        aclError err = aclrtMemcpy(&hostEntity, sizeof(ChannelEntity), devEntityPtr, sizeof(ChannelEntity),
                                   ACL_MEMCPY_DEVICE_TO_HOST);
        if (err != ACL_SUCCESS) {
            std::cerr << "[URMA] aclrtMemcpy(ChannelEntity) peer=" << peer << " err=" << err << std::endl;
            return false;
        }

        if (!IsValidChannelEntityHeader(hostEntity)) {
            std::cerr << "[URMA] invalid ChannelEntity header peer=" << peer << " magic=0x" << std::hex
                      << hostEntity.abiHeader.magicWord << " engine=" << std::dec << static_cast<int>(hostEntity.engine)
                      << std::endl;
            return false;
        }

        return FillChannelSubStructs(peer, hostEntity, sq, cq, remoteBuf, localBuf);
    }

    static bool ReadRegedBufferEntityAt(RegedBufferEntity *array, uint32_t count, uint32_t index, uint32_t peer,
                                        RegedBufferEntity &out)
    {
        if (array == nullptr || index >= count) {
            return false;
        }
        const void *ptr = reinterpret_cast<const void *>(reinterpret_cast<uintptr_t>(array) +
                                                         static_cast<uintptr_t>(index) * sizeof(RegedBufferEntity));
        return CopyChannelSubStruct(ptr, &out, sizeof(RegedBufferEntity), peer, "RegedBufferEntity");
    }

    static bool SelectSymmetricRemoteBuffer(HcclComm comm, const char *symMemTag, uint64_t symmetricSize,
                                            ChannelHandle handle, uint32_t peer, const ChannelEntity &entity,
                                            RegedBufferEntity &selected, uint64_t &rmaAddr, uint32_t &rmaSize)
    {
        void *symAddr = nullptr;
        uint64_t symSize = 0;
        if (!GetRemoteMemByTag(comm, symMemTag, handle, peer, &symAddr, &symSize)) {
            return false;
        }
        rmaAddr = reinterpret_cast<uint64_t>(symAddr);
        rmaSize = static_cast<uint32_t>(symSize);

        if (entity.remoteBufferAddr == nullptr || entity.remoteBufferNum == 0) {
            return false;
        }

        bool found = false;
        for (uint32_t i = 0; i < entity.remoteBufferNum; ++i) {
            RegedBufferEntity buf{};
            if (!ReadRegedBufferEntityAt(entity.remoteBufferAddr, entity.remoteBufferNum, i, peer, buf)) {
                continue;
            }
            if (buf.type != REGED_BUFFER_RMA) {
                continue;
            }
            if (buf.bufferInfo.rma.addr != rmaAddr || buf.bufferInfo.rma.size != symmetricSize) {
                continue;
            }
            selected = buf;
            found = true;
            break;
        }
        if (!found) {
            std::cerr << "[URMA] peer=" << peer << " no RegedBufferEntity matches " << symMemTag << std::endl;
            return false;
        }
        return true;
    }

    static bool SelectSymmetricLocalBuffer(uint64_t symmetricSize, const ChannelEntity &entity, uint32_t peer,
                                           RegedBufferEntity &selected)
    {
        if (entity.localBufferAddr == nullptr || entity.localBufferNum == 0) {
            return false;
        }
        for (uint32_t i = 0; i < entity.localBufferNum; ++i) {
            RegedBufferEntity buf{};
            if (!ReadRegedBufferEntityAt(entity.localBufferAddr, entity.localBufferNum, i, peer, buf)) {
                continue;
            }
            if (buf.type == REGED_BUFFER_RMA && buf.bufferInfo.rma.size == symmetricSize) {
                selected = buf;
                return true;
            }
        }
        for (uint32_t i = 0; i < entity.localBufferNum; ++i) {
            RegedBufferEntity buf{};
            if (!ReadRegedBufferEntityAt(entity.localBufferAddr, entity.localBufferNum, i, peer, buf)) {
                continue;
            }
            if (buf.type == REGED_BUFFER_RMA) {
                selected = buf;
                return true;
            }
        }
        return false;
    }

private:
    static bool GetRemoteMemByTag(HcclComm comm, const char *symMemTag, ChannelHandle handle, uint32_t peer,
                                  void **outAddr, uint64_t *outSize)
    {
        uint32_t memNum = 0;
        CommMem *remoteMems = nullptr;
        char **memTags = nullptr;
        HcclResult rc = HcclChannelGetRemoteMems(comm, handle, &memNum, &remoteMems, &memTags);
        if (rc != HCCL_SUCCESS) {
            std::cerr << "[URMA] HcclChannelGetRemoteMems peer=" << peer << " ret=" << static_cast<int>(rc)
                      << std::endl;
            return false;
        }
        for (uint32_t i = 0; i < memNum; ++i) {
            const char *tag = memTags[i] ? memTags[i] : "";
            if (strcmp(tag, symMemTag) == 0) {
                *outAddr = remoteMems[i].addr;
                *outSize = remoteMems[i].size;
                return true;
            }
        }
        std::cerr << "[URMA] peer=" << peer << " tag " << symMemTag << " not found" << std::endl;
        return false;
    }

    static constexpr uint32_t kHcclChannelEntityMagic = 0x0f0f0f0fU;
    static constexpr uint32_t kHcommChannelEntityMagic = 0x0fcf0f0fU;
    static constexpr uint64_t kDeviceVaThreshold = 0x100000000000ULL;
};

} // namespace urma
} // namespace comm
} // namespace pto

#endif // PTO_COMM_ASYNC_URMA_CHANNEL_HELPER_HPP
