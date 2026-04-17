/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/
#ifndef PTO_MOCKER_COMMON_ACLRT_STUB_HPP
#define PTO_MOCKER_COMMON_ACLRT_STUB_HPP

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdlib>

#include <pto/costmodel/common/qualifiers.hpp>

inline int aclInit(...)
{
    return 0;
}
inline int aclFinalize(...)
{
    return 0;
}
inline int aclrtSetDevice(...)
{
    return 0;
}
inline int aclrtResetDevice(...)
{
    return 0;
}
inline int aclrtGetDevice(int *deviceId)
{
    if (deviceId != nullptr) {
        *deviceId = 0;
    }
    return 0;
}
inline int aclrtGetCurrentContext(aclrtContext *ctx)
{
    if (ctx != nullptr) {
        *ctx = nullptr;
    }
    return 0;
}
inline int aclrtCreateStream(aclrtStream *stream)
{
    if (stream != nullptr) {
        *stream = nullptr;
    }
    return 0;
}
inline int aclrtCreateStreamWithConfig(aclrtStream *stream, uint32_t, uint32_t)
{
    if (stream != nullptr) {
        *stream = nullptr;
    }
    return 0;
}
inline int aclrtDestroyStream(aclrtStream)
{
    return 0;
}
inline int aclrtSynchronizeStream(aclrtStream)
{
    return 0;
}
inline int aclrtStreamGetId(aclrtStream, uint32_t *streamId)
{
    if (streamId != nullptr) {
        *streamId = 0;
    }
    return 0;
}
inline int aclrtSetStreamAttribute(aclrtStream, int, const void *)
{
    return 0;
}
inline int aclrtMallocHost(void **ptr, size_t size)
{
    *ptr = std::malloc(size);
    return (*ptr == nullptr) ? 1 : 0;
}
inline int aclrtMalloc(void **ptr, size_t size, uint32_t)
{
    return aclrtMallocHost(ptr, size);
}
inline int aclrtMemcpy(void *dst, size_t dstSize, const void *src, size_t srcSize, int)
{
    const size_t bytes = (srcSize < dstSize) ? srcSize : dstSize;
    if (bytes == 0) {
        return 0;
    }
    std::copy_n(reinterpret_cast<const unsigned char *>(src), bytes, reinterpret_cast<unsigned char *>(dst));
    return 0;
}
inline int aclrtMemset(void *dst, size_t dstSize, int value, size_t count)
{
    const size_t bytes = (count < dstSize) ? count : dstSize;
    if (bytes == 0) {
        return 0;
    }
    std::fill_n(reinterpret_cast<unsigned char *>(dst), bytes, static_cast<unsigned char>(value));
    return 0;
}
inline int aclrtFree(void *ptr)
{
    std::free(ptr);
    return 0;
}
inline int aclrtFreeHost(void *ptr)
{
    std::free(ptr);
    return 0;
}
inline int aclrtCmoAsync(void *, size_t, int, aclrtStream)
{
    return 0;
}

#endif
