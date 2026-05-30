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
# MoE Dispatch Communication Operator — PTO-ISA
#
# Standalone Dispatch kernel that pulls quantized tokens from remote ranks
# via TGET, matching MegaMoE's dispatch logic.
#
# Targets:
#   moe_dispatch_kernel — Vector-only kernel (dav-c220-vec)
#   moe_dispatch        — Host executable (MPI + HCCL launcher)
# ============================================================================

cmake_minimum_required(VERSION 3.16)

set(CMAKE_COMPILER bisheng)
set(CMAKE_C_COMPILER ${CMAKE_COMPILER})
set(CMAKE_CXX_COMPILER ${CMAKE_COMPILER})

project(pto_moe_dispatch)

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
add_compile_definitions(PTO_NPU_ARCH_A2A3)

# Shape parameters (configurable via cmake -D)
if(DEFINED CONFIG_EP)
    add_compile_definitions(CONFIG_EP=${CONFIG_EP})
endif()
if(DEFINED CONFIG_EXPERT_PER_RANK)
    add_compile_definitions(CONFIG_EXPERT_PER_RANK=${CONFIG_EXPERT_PER_RANK})
endif()
if(DEFINED CONFIG_HIDDEN_SIZE)
    add_compile_definitions(CONFIG_HIDDEN_SIZE=${CONFIG_HIDDEN_SIZE})
endif()
if(DEFINED CONFIG_MAX_TOKENS_PER_RANK)
    add_compile_definitions(CONFIG_MAX_TOKENS_PER_RANK=${CONFIG_MAX_TOKENS_PER_RANK})
endif()
if(DEFINED CONFIG_MAX_OUTPUT_SIZE)
    add_compile_definitions(CONFIG_MAX_OUTPUT_SIZE=${CONFIG_MAX_OUTPUT_SIZE})
endif()
if(DEFINED CONFIG_FIRST_DEVICE_ID)
    add_compile_definitions(CONFIG_FIRST_DEVICE_ID=${CONFIG_FIRST_DEVICE_ID})
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
    message(STATUS "Debug Mode Enabled")
    add_compile_definitions(_DEBUG)
    add_compile_definitions(COMM_DEBUG)
    set(CMAKE_CCE_COMPILE_OPTIONS "${CMAKE_CCE_COMPILE_OPTIONS} --cce-enable-print")
endif()

set(CMAKE_CPP_COMPILE_OPTIONS
    -xc++
    "SHELL:-include stdint.h"
    "SHELL:-include stddef.h"
)

# PTO include path MUST come first to override CANN's headers
set(PTO_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/../../../../include)
set(COMM_ST_DIR ${PROJECT_SOURCE_DIR}/../../../../tests/npu/a2a3/comm/st/testcase)

include_directories(
    ${PTO_INCLUDE_DIR}
    ${PROJECT_SOURCE_DIR}
    ${COMM_ST_DIR}
    ${ASCEND_HOME_PATH}/include
    ${ASCEND_HOME_PATH}/include/hccl
    ${ASCEND_DRIVER_PATH}/kernel/inc
)

set(KERNEL_INCLUDE_DIRS
    ${PTO_INCLUDE_DIR}
    ${PROJECT_SOURCE_DIR}
    ${COMM_ST_DIR}
    $ENV{ASCEND_HOME_PATH}/aarch64-linux/asc
    $ENV{ASCEND_HOME_PATH}/aarch64-linux/asc/include
    $ENV{ASCEND_HOME_PATH}/aarch64-linux/ascendc/include/basic_api
    $ENV{ASCEND_HOME_PATH}/aarch64-linux/include/ascendc/basic_api
    $ENV{ASCEND_HOME_PATH}/aarch64-linux/asc/impl/basic_api
    $ENV{ASCEND_HOME_PATH}/aarch64-linux/ascendc/include/basic_api/impl
    $ENV{ASCEND_HOME_PATH}/aarch64-linux/asc/include/interface
    $ENV{ASCEND_HOME_PATH}/aarch64-linux/asc/include/utils
)

set(HOST_INCLUDE_DIRS
    ${PTO_INCLUDE_DIR}
    ${PROJECT_SOURCE_DIR}
    ${COMM_ST_DIR}
    ${ASCEND_HOME_PATH}/include/hccl
)

set(HOST_LINK_DIRS
    ${ASCEND_HOME_PATH}/lib64
    ${ASCEND_HOME_PATH}/simulator/${SOC_VERSION}/lib
    ${ASCEND_HOME_PATH}/tools/simulator/${SOC_VERSION}/lib
)

# ============================================================================
# Dispatch Communication Kernel (Vector-only: dav-c220-vec)
# ============================================================================
add_library(moe_dispatch_kernel SHARED moe_dispatch_kernel.cpp)
target_compile_options(moe_dispatch_kernel PRIVATE
    ${CMAKE_CCE_COMPILE_OPTIONS}
    --cce-aicore-arch=dav-c220-vec
    -DMEMORY_BASE -std=c++17
)
target_include_directories(moe_dispatch_kernel PRIVATE ${KERNEL_INCLUDE_DIRS})
target_link_directories(moe_dispatch_kernel PRIVATE ${ASCEND_HOME_PATH}/lib64)
target_link_libraries(moe_dispatch_kernel PRIVATE runtime)
target_link_options(moe_dispatch_kernel PRIVATE --cce-fatobj-link)

# ============================================================================
# Host Executable
# ============================================================================
add_executable(moe_dispatch main.cpp)
target_compile_options(moe_dispatch PRIVATE ${CMAKE_CPP_COMPILE_OPTIONS})
target_include_directories(moe_dispatch PRIVATE ${HOST_INCLUDE_DIRS})
target_link_directories(moe_dispatch PUBLIC ${HOST_LINK_DIRS})
target_link_libraries(moe_dispatch PRIVATE
    moe_dispatch_kernel
    $<BUILD_INTERFACE:$<$<STREQUAL:${RUN_MODE},sim>:runtime_camodel>>
    $<BUILD_INTERFACE:$<$<STREQUAL:${RUN_MODE},npu>:runtime>>
    stdc++ ascendcl hcomm m tiling_api platform c_sec dl nnopbase pthread
)
