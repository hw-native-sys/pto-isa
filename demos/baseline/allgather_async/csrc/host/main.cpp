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
// Demonstrates the allgather collective using PTO's SDMA-based instructions:
//   1. Multi-core TPUT_ASYNC
//   2. Multi-core TGET_ASYNC
//
// Usage: mpirun -n <N> ./allgather_demo

#include <cstdlib>
#include <iostream>
#include "comm_mpi.h"
#include "../kernel/allgather_kernel.h"

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
