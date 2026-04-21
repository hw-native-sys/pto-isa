/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

// Allgather Async Demo — Host Entry Point
//
// A2/A3 build (default):  Demos 1-3, SDMA-based (TPUT_ASYNC / TGET_ASYNC via HCCL)
// A5 build (ALLGATHER_DEMO_SOC_A5): Demos 4-6, URMA-based (TPUT_ASYNC / TGET_ASYNC via URMA)
//
// Usage: mpirun -n <N> ./allgather_demo

#include <cstdlib>
#include <iostream>
#include "comm_mpi.h"

#ifndef ALLGATHER_DEMO_SOC_A5
#define ALLGATHER_DEMO_SOC_A5 0
#endif

#if ALLGATHER_DEMO_SOC_A5
#include "../kernel/allgather_urma_kernel.h"
#else
#include "../kernel/allgather_kernel.h"
#endif

int main(int argc, char **argv)
{
    if (!CommMpiInit(&argc, &argv)) {
        std::cerr << "[FATAL] MPI init failed. Launch with: mpirun -n <N> ./allgather_demo" << std::endl;
        return 1;
    }

    int rank = CommMpiRank();
    int size = CommMpiSize();

    if (size < 2) {
        if (rank == 0) {
            std::cerr << "[ERROR] Allgather requires at least 2 MPI ranks." << std::endl;
            std::cerr << "        Launch with: mpirun -n <N> ./allgather_demo" << std::endl;
        }
        CommMpiFinalize();
        return 1;
    }

    if (rank == 0) {
        std::cout << "========================================" << std::endl;
        std::cout << " PTO Allgather Async Demo" << std::endl;
        std::cout << " Ranks: " << size << std::endl;
        std::cout << "========================================" << std::endl;
    }

    int failures = 0;

#if ALLGATHER_DEMO_SOC_A5
    // A5: URMA demos only
    if (rank == 0)
        std::cout << "\n--- Demo 4: URMA Multi-core TPUT_ASYNC ---" << std::endl;
    if (!RunAllgatherUrmaPutMC(size, size, 0, 0)) {
        if (rank == 0)
            std::cerr << "[URMA_TPUT_MC FAIL]" << std::endl;
        ++failures;
    }

    CommMpiBarrier();

    if (rank == 0)
        std::cout << "\n--- Demo 5: URMA Multi-core TGET_ASYNC ---" << std::endl;
    if (!RunAllgatherUrmaGetMC(size, size, 0, 0)) {
        if (rank == 0)
            std::cerr << "[URMA_TGET_MC FAIL]" << std::endl;
        ++failures;
    }

    CommMpiBarrier();

    if (rank == 0)
        std::cout << "\n--- Demo 6: URMA Ring TPUT_ASYNC ---" << std::endl;
    if (!RunAllgatherUrmaRing(size, size, 0, 0)) {
        if (rank == 0)
            std::cerr << "[URMA_RING_TPUT FAIL]" << std::endl;
        ++failures;
    }
    CommMpiBarrier();
#else
    // A2/A3: SDMA demos only
    if (rank == 0)
        std::cout << "\n--- Demo 1: Multi-core TPUT_ASYNC ---" << std::endl;
    if (!RunAllgatherPutAsyncMC(size, 0, 0)) {
        if (rank == 0)
            std::cerr << "[TPUT_ASYNC_MC FAIL]" << std::endl;
        ++failures;
    }

    CommMpiBarrier();

    if (rank == 0)
        std::cout << "\n--- Demo 2: Multi-core TGET_ASYNC ---" << std::endl;
    if (!RunAllgatherGetAsyncMC(size, 0, 0)) {
        if (rank == 0)
            std::cerr << "[TGET_ASYNC_MC FAIL]" << std::endl;
        ++failures;
    }

    CommMpiBarrier();

    if (rank == 0)
        std::cout << "\n--- Demo 3: Ring TPUT_ASYNC ---" << std::endl;
    if (!RunAllgatherRing(size, 0, 0)) {
        if (rank == 0)
            std::cerr << "[RING_TPUT_ASYNC FAIL]" << std::endl;
        ++failures;
    }
#endif

    if (rank == 0) {
        std::cout << "\n========================================" << std::endl;
        if (failures == 0)
            std::cout << " All demos PASSED" << std::endl;
        else
            std::cout << " " << failures << " demo(s) FAILED" << std::endl;
        std::cout << "========================================" << std::endl;
    }

    CommMpiFinalize();
    return (failures == 0) ? 0 : 1;
}
