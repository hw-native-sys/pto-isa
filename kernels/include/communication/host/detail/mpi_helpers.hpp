/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_COMM_DOMAIN_HOST_DETAIL_MPI_HELPERS_HPP
#define PTO_COMM_DOMAIN_HOST_DETAIL_MPI_HELPERS_HPP

#if defined(__CCE_KT_TEST__)
#error "communication/host/detail/mpi_helpers.hpp is host-only"
#endif

#include <cstdlib>
#include <dlfcn.h>

#include "pto/comm/logger/host_log.hpp"

namespace pto {
namespace comm {
namespace domain {
namespace detail {

// MPICH ABI only — MPI_COMM_WORLD and MPI_CHAR constants are hard-coded to
// MPICH values. OpenMPI uses a different ABI (pointer-based handles) and is
// NOT supported by this dlopen-based wrapper.
using MpiComm = int;
inline constexpr MpiComm kMpiCommWorld = static_cast<MpiComm>(0x44000000);
using MpiDatatype = int;
inline constexpr MpiDatatype kMpiChar = static_cast<MpiDatatype>(0x4c000101);

using MpiCommRankFn = int (*)(MpiComm, int*);
using MpiCommSizeFn = int (*)(MpiComm, int*);
using MpiBcastFn = int (*)(void*, int, MpiDatatype, int, MpiComm);
using MpiBarrierFn = int (*)(MpiComm);

inline void*& MpiHandle()
{
    static void* handle = nullptr;
    return handle;
}

inline void* LoadMpiLibrary()
{
    void*& h = MpiHandle();
    if (h != nullptr) {
        return h;
    }
    const char* envPath = std::getenv("MPI_LIB_PATH");
    if (envPath != nullptr) {
        h = dlopen(envPath, RTLD_NOW);
        if (h != nullptr) {
            return h;
        }
    }
    static const char* kCandidates[] = {
        "/usr/local/mpich/lib/libmpi.so",
        "/lib/aarch64-linux-gnu/libmpich.so",
        "/lib/x86_64-linux-gnu/libmpich.so",
        "/usr/lib/libmpi.so",
        "libmpi.so",
        "libmpich.so",
        nullptr};
    for (int i = 0; kCandidates[i] != nullptr; ++i) {
        h = dlopen(kCandidates[i], RTLD_NOW);
        if (h != nullptr) {
            return h;
        }
    }
    PTO_HOST_LOG_ERROR("DOMAIN", "[PTO-DOMAIN] Cannot find MPI library; set MPI_LIB_PATH");
    return nullptr;
}

template <typename T>
inline T GetMpiFunc(const char* name)
{
    void* h = LoadMpiLibrary();
    if (h == nullptr) {
        return nullptr;
    }
    return reinterpret_cast<T>(dlsym(h, name));
}

inline int DomainMpiRank()
{
    int rank = 0;
    auto fn = GetMpiFunc<MpiCommRankFn>("MPI_Comm_rank");
    if (fn != nullptr) {
        fn(kMpiCommWorld, &rank);
    }
    return rank;
}

inline int DomainMpiSize()
{
    int size = 1;
    auto fn = GetMpiFunc<MpiCommSizeFn>("MPI_Comm_size");
    if (fn != nullptr) {
        fn(kMpiCommWorld, &size);
    }
    return size;
}

inline void DomainMpiBcast(void* buf, int count, MpiDatatype dt, int root)
{
    auto fn = GetMpiFunc<MpiBcastFn>("MPI_Bcast");
    if (fn != nullptr) {
        fn(buf, count, dt, root, kMpiCommWorld);
    }
}

inline void DomainMpiBarrier()
{
    auto fn = GetMpiFunc<MpiBarrierFn>("MPI_Barrier");
    if (fn != nullptr) {
        fn(kMpiCommWorld);
    }
}

} // namespace detail
} // namespace domain
} // namespace comm
} // namespace pto

#endif // PTO_COMM_DOMAIN_HOST_DETAIL_MPI_HELPERS_HPP
