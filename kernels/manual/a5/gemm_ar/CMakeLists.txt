# --------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# --------------------------------------------------------------------------------

# ============================================================================
# GEMM AllReduce Dual Stream  A5 Demo — HCCL backend
#
# Targets:
#   gemm_compute_kernel    — Cube-only kernel (dav-c310-cube)
#   comm_kernel            — Vector-only kernel (dav-c310-vec)
#   gemm_allreduce          — Host executable (dual-stream launcher)
# ============================================================================

cmake_minimum_required(VERSION 3.16)

set(CMAKE_COMPILER bisheng)
set(CMAKE_C_COMPILER ${CMAKE_COMPILER})
set(CMAKE_CXX_COMPILER ${CMAKE_COMPILER})

project(pto_gemm_allreduce)

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

if(DEFINED ENV{ASCEND_DRIVER_PATH})
    set(ASCEND_DRIVER_PATH $ENV{ASCEND_DRIVER_PATH})
else()
    set(ASCEND_DRIVER_PATH /usr/local/Ascend/driver)
endif()

add_compile_options(
    -D_FORTIFY_SOURCE=2
    -O2 -std=c++17
    -Wno-macro-redefined -Wno-ignored-attributes
    -fstack-protector-strong
)
add_compile_definitions(_GLIBCXX_USE_CXX11_ABI=0)
add_compile_definitions(PTO_NPU_ARCH_A5)

# Matrix dimensions (configurable via -DCONFIG_G_M=... -DCONFIG_G_K=... -DCONFIG_G_N=...)
if(DEFINED CONFIG_G_M)
    add_compile_definitions(CONFIG_G_M=${CONFIG_G_M})
endif()
if(DEFINED CONFIG_G_K)
    add_compile_definitions(CONFIG_G_K=${CONFIG_G_K})
endif()
if(DEFINED CONFIG_G_N)
    add_compile_definitions(CONFIG_G_N=${CONFIG_G_N})
endif()

# Tile sizes (configurable via -DCONFIG_G_BASE_M=... -DCONFIG_G_BASE_K=... -DCONFIG_G_BASE_N=...)
if(DEFINED CONFIG_G_BASE_M)
    add_compile_definitions(CONFIG_G_BASE_M=${CONFIG_G_BASE_M})
endif()
if(DEFINED CONFIG_G_BASE_K)
    add_compile_definitions(CONFIG_G_BASE_K=${CONFIG_G_BASE_K})
endif()
if(DEFINED CONFIG_G_BASE_N)
    add_compile_definitions(CONFIG_G_BASE_N=${CONFIG_G_BASE_N})
endif()

add_link_options(-s -Wl,-z,relro -Wl,-z,now)

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
    add_compile_definitions(_DEBUG)
    set(CMAKE_CCE_COMPILE_OPTIONS "${CMAKE_CCE_COMPILE_OPTIONS} --cce-enable-print")
endif()

set(CMAKE_CPP_COMPILE_OPTIONS
    -xc++
    "SHELL:-include stdint.h"
    "SHELL:-include stddef.h"
)

include_directories(
    ${PROJECT_SOURCE_DIR}/../../../../include
    ${PROJECT_SOURCE_DIR}
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

set(KERNEL_INCLUDE_DIRS
    ${PROJECT_SOURCE_DIR}/../../../../include
    ${PROJECT_SOURCE_DIR}
    ${ASCEND_KERNEL_PLATFORM_ROOT}/asc
    ${ASCEND_KERNEL_PLATFORM_ROOT}/asc/include
    ${ASCEND_KERNEL_PLATFORM_ROOT}/ascendc/include/basic_api
    ${ASCEND_KERNEL_PLATFORM_ROOT}/include/ascendc/basic_api
    ${ASCEND_KERNEL_PLATFORM_ROOT}/asc/impl/basic_api
    ${ASCEND_KERNEL_PLATFORM_ROOT}/ascendc/include/basic_api/impl
    ${ASCEND_KERNEL_PLATFORM_ROOT}/asc/include/interface
    ${ASCEND_KERNEL_PLATFORM_ROOT}/asc/include/utils
)

set(HOST_INCLUDE_DIRS
    ${PROJECT_SOURCE_DIR}/../../../../include
    ${PROJECT_SOURCE_DIR}
    ${ASCEND_HOME_PATH}/include/hccl
)

if(EXISTS ${ASCEND_HOME_PATH}/aarch64-linux/asc/include/adv_api/hccl/hccl_tiling.h)
    list(APPEND HOST_INCLUDE_DIRS ${ASCEND_HOME_PATH}/aarch64-linux/asc/include/adv_api)
endif()
if(EXISTS ${ASCEND_HOME_PATH}/aarch64-linux/ascendc/include/highlevel_api/kernel_tiling/kernel_tiling.h)
    list(APPEND HOST_INCLUDE_DIRS ${ASCEND_HOME_PATH}/aarch64-linux/ascendc/include/highlevel_api)
endif()
if(EXISTS ${ASCEND_KERNEL_PLATFORM_ROOT}/asc/include/adv_api/hccl/hccl_tiling.h)
    list(APPEND HOST_INCLUDE_DIRS ${ASCEND_KERNEL_PLATFORM_ROOT}/asc/include/adv_api)
endif()
if(EXISTS ${ASCEND_KERNEL_PLATFORM_ROOT}/ascendc/include/highlevel_api/kernel_tiling/kernel_tiling.h)
    list(APPEND HOST_INCLUDE_DIRS ${ASCEND_KERNEL_PLATFORM_ROOT}/ascendc/include/highlevel_api)
endif()

set(HOST_LINK_DIRS
    ${ASCEND_HOME_PATH}/lib64
    ${ASCEND_HOME_PATH}/simulator/${SOC_VERSION}/lib
    ${ASCEND_HOME_PATH}/tools/simulator/${SOC_VERSION}/lib
)

# Block allocation (configurable via -DCOMPUTE_BLOCKS=N -DCOMM_BLOCKS=M)
if(DEFINED COMPUTE_BLOCKS)
    set(BLOCK_DEFS_COMPUTE -DCONFIG_COMPUTE_BLOCK_NUM=${COMPUTE_BLOCKS})
    set(BLOCK_DEFS_ALL -DCONFIG_COMPUTE_BLOCK_NUM=${COMPUTE_BLOCKS})
else()
    set(BLOCK_DEFS_COMPUTE "")
    set(BLOCK_DEFS_ALL "")
endif()
if(DEFINED COMM_BLOCKS)
    list(APPEND BLOCK_DEFS_ALL -DCONFIG_COMM_BLOCK_NUM=${COMM_BLOCKS})
endif()

# ============================================================================
# GEMM Compute Kernel (Cube-only: dav-c220-cube)
# ============================================================================
add_library(gemm_compute_kernel SHARED gemm_compute_kernel.cpp)
target_compile_options(gemm_compute_kernel PRIVATE
    ${CMAKE_CCE_COMPILE_OPTIONS}
    --cce-aicore-arch=dav-c310-cube
    -DMEMORY_BASE -std=c++17
    ${BLOCK_DEFS_COMPUTE}
)
target_include_directories(gemm_compute_kernel PRIVATE ${KERNEL_INCLUDE_DIRS})
target_link_directories(gemm_compute_kernel PRIVATE ${ASCEND_HOME_PATH}/lib64)
target_link_libraries(gemm_compute_kernel PRIVATE runtime)
target_link_options(gemm_compute_kernel PRIVATE --cce-fatobj-link)

# ============================================================================
# Communication Kernel (Vector-only: dav-c220-vec)
# ============================================================================
add_library(comm_kernel SHARED comm_kernel.cpp)
target_compile_options(comm_kernel PRIVATE
    ${CMAKE_CCE_COMPILE_OPTIONS}
    --cce-aicore-arch=dav-c310-vec
    -DMEMORY_BASE -std=c++17
    ${BLOCK_DEFS_ALL}
)
target_include_directories(comm_kernel PRIVATE ${KERNEL_INCLUDE_DIRS})
target_link_directories(comm_kernel PRIVATE ${ASCEND_HOME_PATH}/lib64)
target_link_libraries(comm_kernel PRIVATE runtime)
target_link_options(comm_kernel PRIVATE --cce-fatobj-link)

# ============================================================================
# Host Executable
# ============================================================================
add_executable(gemm_allreduce main.cpp)
target_compile_options(gemm_allreduce PRIVATE ${CMAKE_CPP_COMPILE_OPTIONS} ${BLOCK_DEFS_ALL})
target_include_directories(gemm_allreduce PRIVATE ${HOST_INCLUDE_DIRS})
target_link_directories(gemm_allreduce PUBLIC ${HOST_LINK_DIRS})
target_link_libraries(gemm_allreduce PRIVATE
    gemm_compute_kernel
    comm_kernel
    $<BUILD_INTERFACE:$<$<STREQUAL:${RUN_MODE},sim>:runtime_camodel>>
    $<BUILD_INTERFACE:$<$<STREQUAL:${RUN_MODE},npu>:runtime>>
    stdc++ ascendcl hcomm m tiling_api register platform c_sec dl nnopbase pthread
)
