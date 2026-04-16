# --------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# --------------------------------------------------------------------------------

cmake_minimum_required(VERSION 3.16)

# CMake must see the Ascend toolchain before project() on a fresh configure.
set(CMAKE_COMPILER bisheng)
set(CMAKE_C_COMPILER ${CMAKE_COMPILER})
set(CMAKE_CXX_COMPILER ${CMAKE_COMPILER})

project(allgather_gemm_demo)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

if(NOT DEFINED ENV{ASCEND_HOME_PATH})
    message(FATAL_ERROR "Cannot find ASCEND_HOME_PATH, please run set_env.sh.")
else()
    set(ASCEND_HOME_PATH $ENV{ASCEND_HOME_PATH})
endif()

# HCCL backend (replaces SHMEM)
message(STATUS "Using HCCL backend for communication")

# =============================================================================
# GEMM Size Configuration
# =============================================================================
if(NOT DEFINED G_M)
    set(G_M 2048)
endif()
if(NOT DEFINED G_K)
    set(G_K 2048)
endif()
if(NOT DEFINED G_N)
    set(G_N 1024)
endif()
if(NOT DEFINED G_BASE_M)
    set(G_BASE_M 128)
endif()
if(NOT DEFINED G_BASE_N)
    set(G_BASE_N 256)
endif()
if(DEFINED COMPUTE_BLOCKS AND NOT DEFINED COMPUTE_BLOCK_NUM)
    set(COMPUTE_BLOCK_NUM ${COMPUTE_BLOCKS})
endif()
if(DEFINED COMM_BLOCKS AND NOT DEFINED COMM_BLOCK_NUM)
    set(COMM_BLOCK_NUM ${COMM_BLOCKS})
endif()
if(NOT DEFINED COMPUTE_BLOCK_NUM)
    if(DEFINED SOC_VERSION AND (SOC_VERSION MATCHES "Ascend910B[34]" OR SOC_VERSION MATCHES "Ascend910_93[67]"))
        set(COMPUTE_BLOCK_NUM 20)
    else()
        set(COMPUTE_BLOCK_NUM 24)
    endif()
endif()
if(NOT DEFINED COMM_BLOCK_NUM)
    if(DEFINED SOC_VERSION AND (SOC_VERSION MATCHES "Ascend910B[34]" OR SOC_VERSION MATCHES "Ascend910_93[67]"))
        set(COMM_BLOCK_NUM 40)
    else()
        set(COMM_BLOCK_NUM 48)
    endif()
endif()
# Original (unpadded) dimensions for golden verification
if(NOT DEFINED ORIG_M)
    set(ORIG_M ${G_M})
endif()
if(NOT DEFINED ORIG_K)
    set(ORIG_K ${G_K})
endif()
if(NOT DEFINED ORIG_N)
    set(ORIG_N ${G_N})
endif()
message(STATUS "GEMM Size: M=${G_M}, K=${G_K}, N=${G_N} (orig: ${ORIG_M}x${ORIG_K}x${ORIG_N})")
message(STATUS "GEMM Tile: BASE_M=${G_BASE_M}, BASE_N=${G_BASE_N} (BASE_K derived as BASE_N/4 in gemm_config.hpp)")
message(STATUS "Block Configuration: COMPUTE_BLOCK_NUM=${COMPUTE_BLOCK_NUM}, COMM_BLOCK_NUM=${COMM_BLOCK_NUM}")

if(NOT DEFINED ENV{ASCEND_DRIVER_PATH})
    set(ASCEND_DRIVER_PATH /usr/local/Ascend/driver)
else()
    set(ASCEND_DRIVER_PATH $ENV{ASCEND_DRIVER_PATH})
endif()

add_compile_options(
    -D_FORTIFY_SOURCE=2
    -O2 -std=c++17
    -Wno-macro-redefined -Wno-ignored-attributes
    -fstack-protector-strong
)
add_compile_definitions(_GLIBCXX_USE_CXX11_ABI=0)
add_compile_definitions(PTO_NPU_ARCH_A5)

add_link_options(
    -s
    -Wl,-z,relro
    -Wl,-z,now
)

set(CMAKE_CCE_COMPILE_OPTIONS
    -xcce
    -Xhost-start -Xhost-end
    "SHELL:-mllvm -cce-aicore-stack-size=0x8000"
    "SHELL:-mllvm -cce-aicore-function-stack-size=0x8000"
    "SHELL:-mllvm -cce-aicore-record-overflow=true"
    "SHELL:-mllvm -cce-aicore-addr-transform"
    "SHELL:-mllvm -cce-aicore-dcci-insert-for-scalar=false"
)

if(DEBUG_MODE)
    message(STATUS "Debug Mode Enabled, Add Debug Options")
    add_compile_definitions(COMM_DEBUG)
    set(CMAKE_CCE_COMPILE_OPTIONS "${CMAKE_CCE_COMPILE_OPTIONS} --cce-enable-print")
endif()

set(CMAKE_CPP_COMPILE_OPTIONS
    -xc++
    "SHELL:-include stdint.h"
    "SHELL:-include stddef.h"
)

include_directories(
    ${PROJECT_SOURCE_DIR}/../../../../include
    ${ASCEND_HOME_PATH}/include
    ${ASCEND_HOME_PATH}/include/hccl
    ${ASCEND_DRIVER_PATH}/kernel/inc
)

if(EXISTS ${ASCEND_HOME_PATH}/x86_64-linux/asc)
    set(ASCEND_KERNEL_PLATFORM_ROOT ${ASCEND_HOME_PATH}/x86_64-linux)
elseif(EXISTS ${ASCEND_HOME_PATH}/aarch64-linux/asc)
    set(ASCEND_KERNEL_PLATFORM_ROOT ${ASCEND_HOME_PATH}/aarch64-linux)
else()
    message(FATAL_ERROR "Cannot find Ascend kernel include root under ${ASCEND_HOME_PATH}/x86_64-linux or aarch64-linux")
endif()

set(A5_TESTCASE_DIR ${PROJECT_SOURCE_DIR}/../../../../tests/npu/a5/comm/st/testcase)
set(HOST_EXTRA_INCLUDE_DIRS)

if(EXISTS ${ASCEND_KERNEL_PLATFORM_ROOT}/asc/include/adv_api/hccl/hccl_tiling.h)
    list(APPEND HOST_EXTRA_INCLUDE_DIRS ${ASCEND_KERNEL_PLATFORM_ROOT}/asc/include/adv_api)
elseif(EXISTS ${ASCEND_HOME_PATH}/aarch64-linux/asc/include/adv_api/hccl/hccl_tiling.h)
    list(APPEND HOST_EXTRA_INCLUDE_DIRS ${ASCEND_HOME_PATH}/aarch64-linux/asc/include/adv_api)
endif()

if(EXISTS ${ASCEND_KERNEL_PLATFORM_ROOT}/ascendc/include/highlevel_api/kernel_tiling/kernel_tiling.h)
    list(APPEND HOST_EXTRA_INCLUDE_DIRS ${ASCEND_KERNEL_PLATFORM_ROOT}/ascendc/include/highlevel_api)
elseif(EXISTS ${ASCEND_HOME_PATH}/aarch64-linux/ascendc/include/highlevel_api/kernel_tiling/kernel_tiling.h)
    list(APPEND HOST_EXTRA_INCLUDE_DIRS ${ASCEND_HOME_PATH}/aarch64-linux/ascendc/include/highlevel_api)
endif()

# =============================================================================
# 1. Compute Kernel (Cube arch) - GEMM consumer
# =============================================================================
add_library(allgather_gemm_compute_kernel SHARED allgather_gemm_compute_kernel.cpp)
target_compile_options(allgather_gemm_compute_kernel PRIVATE
    ${CMAKE_CCE_COMPILE_OPTIONS}
    --cce-aicore-arch=dav-c310-cube
    -DMEMORY_BASE -std=c++17
    -DCONFIG_G_M=${G_M} -DCONFIG_G_K=${G_K} -DCONFIG_G_N=${G_N}
    -DCONFIG_G_BASE_M=${G_BASE_M} -DCONFIG_G_BASE_N=${G_BASE_N}
    -DCONFIG_COMPUTE_BLOCK_NUM=${COMPUTE_BLOCK_NUM}
)
target_include_directories(allgather_gemm_compute_kernel PRIVATE
    ${PROJECT_SOURCE_DIR}/../../../../include/
    ${A5_TESTCASE_DIR}
)
target_link_options(allgather_gemm_compute_kernel PRIVATE --cce-fatobj-link)

# =============================================================================
# 2. Communication Kernel (Vec arch + shmem) - AllGather producer
# =============================================================================
add_library(allgather_gemm_comm_kernel SHARED allgather_gemm_comm_kernel.cpp)
target_compile_options(allgather_gemm_comm_kernel PRIVATE
    ${CMAKE_CCE_COMPILE_OPTIONS}
    --cce-aicore-arch=dav-c310-vec
    -DMEMORY_BASE -std=c++17
    -DCONFIG_G_M=${G_M} -DCONFIG_G_K=${G_K} -DCONFIG_G_N=${G_N}
    -DCONFIG_G_BASE_M=${G_BASE_M} -DCONFIG_G_BASE_N=${G_BASE_N}
    -DCONFIG_COMM_BLOCK_NUM=${COMM_BLOCK_NUM}
)
target_include_directories(allgather_gemm_comm_kernel PRIVATE
    ${PROJECT_SOURCE_DIR}/../../../../include/
    ${A5_TESTCASE_DIR}
    ${ASCEND_KERNEL_PLATFORM_ROOT}/asc
    ${ASCEND_KERNEL_PLATFORM_ROOT}/asc/include
    ${ASCEND_KERNEL_PLATFORM_ROOT}/ascendc/include/basic_api
    ${ASCEND_KERNEL_PLATFORM_ROOT}/include/ascendc/basic_api
    ${ASCEND_KERNEL_PLATFORM_ROOT}/asc/impl/basic_api
    ${ASCEND_KERNEL_PLATFORM_ROOT}/ascendc/include/basic_api/impl
    ${ASCEND_KERNEL_PLATFORM_ROOT}/asc/include/interface
    ${ASCEND_KERNEL_PLATFORM_ROOT}/asc/include/utils
)
target_link_directories(allgather_gemm_comm_kernel PRIVATE
    ${ASCEND_HOME_PATH}/lib64
)
target_link_libraries(allgather_gemm_comm_kernel PRIVATE
    hcomm runtime nnopbase
)
target_link_options(allgather_gemm_comm_kernel PRIVATE --cce-fatobj-link)

# =============================================================================
# 3. Executable
# =============================================================================
add_executable(allgather_gemm main.cpp)
target_compile_options(allgather_gemm PRIVATE ${CMAKE_CPP_COMPILE_OPTIONS})
target_include_directories(allgather_gemm PRIVATE
    ${PROJECT_SOURCE_DIR}/../../../../include/
    ${PROJECT_SOURCE_DIR}/../../../../tests/common
    ${A5_TESTCASE_DIR}
    ${HOST_EXTRA_INCLUDE_DIRS}
)
target_compile_definitions(allgather_gemm PRIVATE
    CONFIG_G_M=${G_M} CONFIG_G_K=${G_K} CONFIG_G_N=${G_N}
    CONFIG_G_BASE_M=${G_BASE_M} CONFIG_G_BASE_N=${G_BASE_N}
    CONFIG_ORIG_M=${ORIG_M} CONFIG_ORIG_K=${ORIG_K} CONFIG_ORIG_N=${ORIG_N}
    CONFIG_COMPUTE_BLOCK_NUM=${COMPUTE_BLOCK_NUM}
    CONFIG_COMM_BLOCK_NUM=${COMM_BLOCK_NUM}
)

target_link_directories(allgather_gemm PUBLIC
    ${ASCEND_HOME_PATH}/lib64
    ${ASCEND_HOME_PATH}/simulator/${SOC_VERSION}/lib
    ${ASCEND_HOME_PATH}/tools/simulator/${SOC_VERSION}/lib
)

target_link_libraries(allgather_gemm PRIVATE
    allgather_gemm_compute_kernel
    allgather_gemm_comm_kernel
    $<BUILD_INTERFACE:$<$<STREQUAL:${RUN_MODE},sim>:runtime_camodel>>
    $<BUILD_INTERFACE:$<$<STREQUAL:${RUN_MODE},npu>:runtime>>
    stdc++ ascendcl hcomm m tiling_api register platform c_sec dl nnopbase pthread
)
