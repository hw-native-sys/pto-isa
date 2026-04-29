/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
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
#include <iostream>
#include <dlfcn.h>
#include <stdlib.h>

// ============================================================================
// MPI dynamic loading infrastructure (following PyPTO's approach).
// MPI is loaded at runtime via dlopen to avoid a hard link-time dependency.
// Set MPI_LIB_PATH env to override the library search path.
// ============================================================================
using MPI_Comm = int;
#define COMM_MPI_COMM_WORLD ((MPI_Comm)0x44000000)
using MPI_Datatype = int;
#define COMM_MPI_CHAR ((MPI_Datatype)0x4c000101)
#define COMM_MPI_INT ((MPI_Datatype)0x4c000405)

using MpiInitFunc = int (*)(int *, char ***);
using MpiCommSizeFunc = int (*)(MPI_Comm, int *);
using MpiCommRankFunc = int (*)(MPI_Comm, int *);
using MpiBcastFunc = int (*)(void *, int, MPI_Datatype, int, MPI_Comm);
using MpiBarrierFunc = int (*)(MPI_Comm);
using MpiFinalizeFunc = int (*)();

namespace comm_mpi {

inline void *&MpiHandle()
{
    static void *handle = nullptr;
    return handle;
}

// Defensive sanity check on a user-supplied MPI library path.
// This is NOT a security boundary (the program runs with the invoking user's
// privileges and many other env vars such as LD_LIBRARY_PATH/LD_PRELOAD have
// equivalent effect). It only catches obvious misconfiguration and makes the
// loaded path explicit in logs.
inline bool LooksLikeMpiLibName(const char *path)
{
    const char *slash = std::strrchr(path, '/');
    const char *base = slash ? slash + 1 : path;
    if (std::strncmp(base, "libmpi", 6) != 0 && std::strncmp(base, "libmpich", 8) != 0)
        return false;
    return std::strstr(base, ".so") != nullptr;
}

inline void *LoadMpiLibrary()
{
    void *&h = MpiHandle();
    if (h)
        return h;

    const char *envPath = std::getenv("MPI_LIB_PATH");
    if (envPath && envPath[0] != '\0') {
        if (envPath[0] != '/') {
            std::cerr << "[MPI] Ignoring MPI_LIB_PATH (must be an absolute path): " << envPath << std::endl;
        } else if (!LooksLikeMpiLibName(envPath)) {
            std::cerr << "[MPI] Ignoring MPI_LIB_PATH (filename does not look like an MPI library, "
                         "expected libmpi*.so* or libmpich*.so*): "
                      << envPath << std::endl;
        } else {
            h = dlopen(envPath, RTLD_NOW);
            if (h) {
                std::cerr << "[MPI] Loaded from MPI_LIB_PATH: " << envPath << std::endl;
                return h;
            }
            std::cerr << "[MPI] dlopen failed for MPI_LIB_PATH=" << envPath << ": " << dlerror() << std::endl;
        }
    }

    static const char *candidates[] = {"libmpi.so",
                                       "libmpich.so",
                                       "/usr/local/mpich/lib/libmpi.so",
                                       "/lib/aarch64-linux-gnu/libmpich.so",
                                       "/lib/x86_64-linux-gnu/libmpich.so",
                                       "/usr/lib/libmpi.so",
                                       "/usr/lib/libmpich.so",
                                       nullptr};
    for (int i = 0; candidates[i]; ++i) {
        h = dlopen(candidates[i], RTLD_NOW);
        if (h) {
            std::cerr << "[MPI] Loaded: " << candidates[i] << std::endl;
            return h;
        }
    }
    std::cerr << "[ERROR] Cannot find MPI library. Set MPI_LIB_PATH to the path of libmpi.so" << std::endl;
    return nullptr;
}

template <typename T>
inline T GetFunc(const char *name)
{
    void *h = LoadMpiLibrary();
    if (!h)
        return nullptr;
    return reinterpret_cast<T>(dlsym(h, name));
}

} // namespace comm_mpi

inline bool CommMpiInit(int *argc, char ***argv)
{
    auto fn = comm_mpi::GetFunc<MpiInitFunc>("MPI_Init");
    if (!fn)
        return false;
    int ret = fn(argc, argv);
    if (ret != 0) {
        std::cerr << "[ERROR] MPI_Init failed: " << ret << std::endl;
        return false;
    }
    return true;
}

inline void CommMpiFinalize()
{
    auto fn = comm_mpi::GetFunc<MpiFinalizeFunc>("MPI_Finalize");
    if (fn)
        fn();
}

inline int CommMpiRank()
{
    int rank = 0;
    auto fn = comm_mpi::GetFunc<MpiCommRankFunc>("MPI_Comm_rank");
    if (fn)
        fn(COMM_MPI_COMM_WORLD, &rank);
    return rank;
}

inline int CommMpiSize()
{
    int size = 1;
    auto fn = comm_mpi::GetFunc<MpiCommSizeFunc>("MPI_Comm_size");
    if (fn)
        fn(COMM_MPI_COMM_WORLD, &size);
    return size;
}

inline void CommMpiBcast(void *buf, int count, MPI_Datatype dt, int root)
{
    auto fn = comm_mpi::GetFunc<MpiBcastFunc>("MPI_Bcast");
    if (fn)
        fn(buf, count, dt, root, COMM_MPI_COMM_WORLD);
}

inline void CommMpiBarrier()
{
    auto fn = comm_mpi::GetFunc<MpiBarrierFunc>("MPI_Barrier");
    if (fn)
        fn(COMM_MPI_COMM_WORLD);
}
