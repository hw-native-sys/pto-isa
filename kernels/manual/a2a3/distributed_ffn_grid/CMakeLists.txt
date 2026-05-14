# --------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# --------------------------------------------------------------------------------

cmake_minimum_required(VERSION 3.16)

set(CMAKE_COMPILER bisheng)
set(CMAKE_C_COMPILER ${CMAKE_COMPILER})
set(CMAKE_CXX_COMPILER ${CMAKE_COMPILER})

project(distributed_ffn_grid_demo)

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

# =============================================================================
# Grid + tile dimensions (override via -DGRID_ROWS=... etc).
# =============================================================================
if(NOT DEFINED GRID_ROWS)
    set(GRID_ROWS 2)
endif()
if(NOT DEFINED GRID_COLS)
    set(GRID_COLS 2)
endif()
if(NOT DEFINED TOKEN_TILE)
    set(TOKEN_TILE 16)
endif()
if(NOT DEFINED MODEL_TILE)
    set(MODEL_TILE 64)
endif()
if(NOT DEFINED FFN_TILE)
    set(FFN_TILE 64)
endif()

message(STATUS "Grid: ${GRID_ROWS}x${GRID_COLS}  Tile: ${TOKEN_TILE}x${MODEL_TILE}  FfnTile: ${FFN_TILE}")

add_compile_options(
    -D_FORTIFY_SOURCE=2
    -O2 -std=c++17
    -Wno-macro-redefined -Wno-ignored-attributes
    -fstack-protector-strong
)

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

set(CMAKE_CPP_COMPILE_OPTIONS
    -xc++
    "SHELL:-include stdint.h"
    "SHELL:-include stddef.h"
)

include_directories(
    ${PROJECT_SOURCE_DIR}/../../../../include
    ${ASCEND_HOME_PATH}/include
    ${ASCEND_DRIVER_PATH}/kernel/inc
)

# =============================================================================
# 1. Mixed Cube/Vec Kernel.  Cube and Vec branches run concurrently and use
#    regular A2/A3 TPipe synchronization for intermediate FFN tiles.
# =============================================================================
add_library(distributed_ffn_grid_mixed_kernel SHARED distributed_ffn_grid_compute_kernel.cpp)
target_compile_options(distributed_ffn_grid_mixed_kernel PRIVATE
    ${CMAKE_CCE_COMPILE_OPTIONS}
    --cce-aicore-arch=dav-c220
    -DMEMORY_BASE -std=c++17
    -DCONFIG_GRID_ROWS=${GRID_ROWS} -DCONFIG_GRID_COLS=${GRID_COLS}
    -DCONFIG_TOKEN_TILE=${TOKEN_TILE} -DCONFIG_MODEL_TILE=${MODEL_TILE}
    -DCONFIG_FFN_TILE=${FFN_TILE}
)
target_include_directories(distributed_ffn_grid_mixed_kernel PRIVATE
    ${PROJECT_SOURCE_DIR}/../../../../include/
    ${PROJECT_SOURCE_DIR}/../../../../tests/npu/a2a3/comm/st/testcase/
    $ENV{ASCEND_HOME_PATH}/aarch64-linux/asc
    $ENV{ASCEND_HOME_PATH}/aarch64-linux/asc/include
    $ENV{ASCEND_HOME_PATH}/aarch64-linux/ascendc/include/basic_api
    $ENV{ASCEND_HOME_PATH}/aarch64-linux/include/ascendc/basic_api
    $ENV{ASCEND_HOME_PATH}/aarch64-linux/asc/impl/basic_api
    $ENV{ASCEND_HOME_PATH}/aarch64-linux/ascendc/include/basic_api/impl
    $ENV{ASCEND_HOME_PATH}/aarch64-linux/asc/include/interface
    $ENV{ASCEND_HOME_PATH}/aarch64-linux/asc/include/utils
)
target_link_directories(distributed_ffn_grid_mixed_kernel PRIVATE ${ASCEND_HOME_PATH}/lib64)
target_link_libraries(distributed_ffn_grid_mixed_kernel PRIVATE hcomm runtime nnopbase)
target_link_options(distributed_ffn_grid_mixed_kernel PRIVATE --cce-fatobj-link)

# =============================================================================
# 2. Host driver executable
# =============================================================================
add_executable(distributed_ffn_grid main.cpp)
target_compile_options(distributed_ffn_grid PRIVATE ${CMAKE_CPP_COMPILE_OPTIONS})
target_include_directories(distributed_ffn_grid PRIVATE
    ${PROJECT_SOURCE_DIR}/../../../../include/
    ${PROJECT_SOURCE_DIR}/../../../../tests/common
    ${PROJECT_SOURCE_DIR}/../../../../tests/npu/a2a3/comm/st/testcase/
    ${ASCEND_HOME_PATH}/pkg_inc/runtime
    ${ASCEND_HOME_PATH}/pkg_inc/toolchain
    ${ASCEND_HOME_PATH}/pkg_inc/profiling
    ${ASCEND_HOME_PATH}/aarch64-linux/pkg_inc/runtime
    ${ASCEND_HOME_PATH}/aarch64-linux/pkg_inc
    ${ASCEND_HOME_PATH}/aarch64-linux/pkg_inc/profiling
)
target_compile_definitions(distributed_ffn_grid PRIVATE
    CONFIG_GRID_ROWS=${GRID_ROWS} CONFIG_GRID_COLS=${GRID_COLS}
    CONFIG_TOKEN_TILE=${TOKEN_TILE} CONFIG_MODEL_TILE=${MODEL_TILE}
    CONFIG_FFN_TILE=${FFN_TILE}
)

target_link_directories(distributed_ffn_grid PUBLIC
    ${ASCEND_HOME_PATH}/lib64
    ${ASCEND_HOME_PATH}/simulator/${SOC_VERSION}/lib
    ${ASCEND_HOME_PATH}/tools/simulator/${SOC_VERSION}/lib
)

target_link_libraries(distributed_ffn_grid PRIVATE
    distributed_ffn_grid_mixed_kernel
    $<BUILD_INTERFACE:$<$<STREQUAL:${RUN_MODE},sim>:runtime_camodel>>
    $<BUILD_INTERFACE:$<$<STREQUAL:${RUN_MODE},npu>:runtime>>
    stdc++ ascendcl hcomm m tiling_api platform c_sec dl nnopbase pthread
)
