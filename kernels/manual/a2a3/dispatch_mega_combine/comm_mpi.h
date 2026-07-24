/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#pragma once

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <iostream>

using MPI_Comm = int;
#define COMM_MPI_COMM_WORLD ((MPI_Comm)0x44000000)
using MPI_Datatype = int;
#define COMM_MPI_CHAR ((MPI_Datatype)0x4c000101)

using MpiInitFunc = int (*)(int*, char***);
using MpiCommSizeFunc = int (*)(MPI_Comm, int*);
using MpiCommRankFunc = int (*)(MPI_Comm, int*);
using MpiBcastFunc = int (*)(void*, int, MPI_Datatype, int, MPI_Comm);
using MpiGatherFunc = int (*)(const void*, int, MPI_Datatype, void*, int, MPI_Datatype, int, MPI_Comm);
using MpiBarrierFunc = int (*)(MPI_Comm);
using MpiFinalizeFunc = int (*)();

namespace comm_mpi {

inline void*& MpiHandle()
{
    static void* handle = nullptr;
    return handle;
}

inline void* LoadMpiLibrary()
{
    void*& h = MpiHandle();
    if (h) {
        return h;
    }

    const char* envPath = std::getenv("MPI_LIB_PATH");
    if (envPath != nullptr) {
        h = dlopen(envPath, RTLD_NOW);
        if (h != nullptr) {
            return h;
        }
    }

    static const char* candidates[] = {
        "/usr/local/mpich/lib/libmpi.so",
        "/lib/aarch64-linux-gnu/libmpich.so",
        "/lib/x86_64-linux-gnu/libmpich.so",
        "/usr/lib/libmpi.so",
        "/usr/lib/libmpich.so",
        "libmpi.so",
        "libmpich.so",
        nullptr,
    };
    for (int i = 0; candidates[i] != nullptr; ++i) {
        h = dlopen(candidates[i], RTLD_NOW);
        if (h != nullptr) {
            return h;
        }
    }

    std::cerr << "[ERROR] Cannot find MPI library. Set MPI_LIB_PATH to libmpi.so path." << std::endl;
    return nullptr;
}

template <typename T>
inline T GetFunc(const char* name)
{
    void* h = LoadMpiLibrary();
    if (h == nullptr) {
        return nullptr;
    }
    return reinterpret_cast<T>(dlsym(h, name));
}

} // namespace comm_mpi

inline bool CommMpiInit(int* argc, char*** argv)
{
    auto fn = comm_mpi::GetFunc<MpiInitFunc>("MPI_Init");
    if (fn == nullptr) {
        return false;
    }
    return fn(argc, argv) == 0;
}

inline void CommMpiFinalize()
{
    auto fn = comm_mpi::GetFunc<MpiFinalizeFunc>("MPI_Finalize");
    if (fn != nullptr) {
        fn();
    }
}

inline int CommMpiRank()
{
    int rank = 0;
    auto fn = comm_mpi::GetFunc<MpiCommRankFunc>("MPI_Comm_rank");
    if (fn != nullptr) {
        fn(COMM_MPI_COMM_WORLD, &rank);
    }
    return rank;
}

inline int CommMpiSize()
{
    int size = 1;
    auto fn = comm_mpi::GetFunc<MpiCommSizeFunc>("MPI_Comm_size");
    if (fn != nullptr) {
        fn(COMM_MPI_COMM_WORLD, &size);
    }
    return size;
}

inline void CommMpiBcast(void* buf, int count, MPI_Datatype dt, int root)
{
    auto fn = comm_mpi::GetFunc<MpiBcastFunc>("MPI_Bcast");
    if (fn != nullptr) {
        fn(buf, count, dt, root, COMM_MPI_COMM_WORLD);
    }
}

inline void CommMpiGather(
    const void* sendbuf, int sendcount, MPI_Datatype sendtype, void* recvbuf, int recvcount, MPI_Datatype recvtype,
    int root)
{
    auto fn = comm_mpi::GetFunc<MpiGatherFunc>("MPI_Gather");
    if (fn != nullptr) {
        fn(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, root, COMM_MPI_COMM_WORLD);
    }
}

inline void CommMpiBarrier()
{
    auto fn = comm_mpi::GetFunc<MpiBarrierFunc>("MPI_Barrier");
    if (fn != nullptr) {
        fn(COMM_MPI_COMM_WORLD);
    }
}
