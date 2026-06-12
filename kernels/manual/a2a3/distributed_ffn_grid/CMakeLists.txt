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

# K-hop unicast smoke kernel config (independent of the FFN grid/tile sizes).
if(NOT DEFINED KHOP_ROWS)
    set(KHOP_ROWS 1)
endif()
if(NOT DEFINED KHOP_COLS)
    set(KHOP_COLS 4)
endif()
if(NOT DEFINED KHOP_DIST)
    set(KHOP_DIST 2)
endif()
if(NOT DEFINED KHOP_T)
    set(KHOP_T 16)
endif()
if(NOT DEFINED KHOP_W)
    set(KHOP_W 64)
endif()

# Single-source broadcast smoke kernel config (independent of the FFN sizes).
if(NOT DEFINED BCAST_ROWS)
    set(BCAST_ROWS 1)
endif()
if(NOT DEFINED BCAST_COLS)
    set(BCAST_COLS 5)
endif()
if(NOT DEFINED BCAST_SRC)
    set(BCAST_SRC 2)
endif()
if(NOT DEFINED BCAST_SPAN_COL)
    set(BCAST_SPAN_COL 0)
endif()
if(NOT DEFINED BCAST_T)
    set(BCAST_T 16)
endif()
if(NOT DEFINED BCAST_W)
    set(BCAST_W 64)
endif()

message(STATUS "Grid: ${GRID_ROWS}x${GRID_COLS}  Tile: ${TOKEN_TILE}x${MODEL_TILE}  FfnTile: ${FFN_TILE}")
message(STATUS "KHop smoke: ${KHOP_ROWS}x${KHOP_COLS}  DIST=${KHOP_DIST}  Tile: ${KHOP_T}x${KHOP_W}")
message(STATUS "Bcast smoke: ${BCAST_ROWS}x${BCAST_COLS}  SRC=${BCAST_SRC}  SPAN_COL=${BCAST_SPAN_COL}  Tile: ${BCAST_T}x${BCAST_W}")

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
# 1. ReduceSum split variant.  Cube and Vec branches run concurrently and use
#    regular A2/A3 TPipe synchronization for intermediate FFN tiles.
# =============================================================================
add_library(distributed_ffn_grid_reducesum_mixed_kernel SHARED distributed_ffn_grid_reducesum_compute_kernel.cpp)
target_compile_options(distributed_ffn_grid_reducesum_mixed_kernel PRIVATE
    ${CMAKE_CCE_COMPILE_OPTIONS}
    --cce-aicore-arch=dav-c220
    -DMEMORY_BASE -std=c++17
    -DCONFIG_GRID_ROWS=${GRID_ROWS} -DCONFIG_GRID_COLS=${GRID_COLS}
    -DCONFIG_TOKEN_TILE=${TOKEN_TILE} -DCONFIG_MODEL_TILE=${MODEL_TILE}
    -DCONFIG_FFN_TILE=${FFN_TILE}
)
target_include_directories(distributed_ffn_grid_reducesum_mixed_kernel PRIVATE
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
target_link_directories(distributed_ffn_grid_reducesum_mixed_kernel PRIVATE ${ASCEND_HOME_PATH}/lib64)
target_link_libraries(distributed_ffn_grid_reducesum_mixed_kernel PRIVATE hcomm runtime nnopbase)
target_link_options(distributed_ffn_grid_reducesum_mixed_kernel PRIVATE --cce-fatobj-link)

# AllGather split variant: hidden shards are gathered through GridPipe, and
# down projection writes one output-H shard per column.
add_library(distributed_ffn_grid_allgather_mixed_kernel SHARED distributed_ffn_grid_allgather_compute_kernel.cpp)
target_compile_options(distributed_ffn_grid_allgather_mixed_kernel PRIVATE
    ${CMAKE_CCE_COMPILE_OPTIONS}
    --cce-aicore-arch=dav-c220
    -DMEMORY_BASE -std=c++17
    -DCONFIG_FFN_GRID_ALLGATHER
    -DCONFIG_GRID_ROWS=${GRID_ROWS} -DCONFIG_GRID_COLS=${GRID_COLS}
    -DCONFIG_TOKEN_TILE=${TOKEN_TILE} -DCONFIG_MODEL_TILE=${MODEL_TILE}
    -DCONFIG_FFN_TILE=${FFN_TILE}
)
target_include_directories(distributed_ffn_grid_allgather_mixed_kernel PRIVATE
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
target_link_directories(distributed_ffn_grid_allgather_mixed_kernel PRIVATE ${ASCEND_HOME_PATH}/lib64)
target_link_libraries(distributed_ffn_grid_allgather_mixed_kernel PRIVATE hcomm runtime nnopbase)
target_link_options(distributed_ffn_grid_allgather_mixed_kernel PRIVATE --cce-fatobj-link)

# =============================================================================
# 2. Host driver executable
# =============================================================================
add_executable(distributed_ffn_grid_reducesum main_reducesum.cpp)
target_compile_options(distributed_ffn_grid_reducesum PRIVATE ${CMAKE_CPP_COMPILE_OPTIONS})
target_include_directories(distributed_ffn_grid_reducesum PRIVATE
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
target_compile_definitions(distributed_ffn_grid_reducesum PRIVATE
    CONFIG_GRID_ROWS=${GRID_ROWS} CONFIG_GRID_COLS=${GRID_COLS}
    CONFIG_TOKEN_TILE=${TOKEN_TILE} CONFIG_MODEL_TILE=${MODEL_TILE}
    CONFIG_FFN_TILE=${FFN_TILE}
)

target_link_directories(distributed_ffn_grid_reducesum PUBLIC
    ${ASCEND_HOME_PATH}/lib64
    ${ASCEND_HOME_PATH}/simulator/${SOC_VERSION}/lib
    ${ASCEND_HOME_PATH}/tools/simulator/${SOC_VERSION}/lib
)

target_link_libraries(distributed_ffn_grid_reducesum PRIVATE
    distributed_ffn_grid_reducesum_mixed_kernel
    $<BUILD_INTERFACE:$<$<STREQUAL:${RUN_MODE},sim>:runtime_camodel>>
    $<BUILD_INTERFACE:$<$<STREQUAL:${RUN_MODE},npu>:runtime>>
    stdc++ ascendcl hcomm m tiling_api platform c_sec dl nnopbase pthread
)

add_executable(distributed_ffn_grid_allgather main_allgather.cpp)
target_compile_options(distributed_ffn_grid_allgather PRIVATE ${CMAKE_CPP_COMPILE_OPTIONS})
target_include_directories(distributed_ffn_grid_allgather PRIVATE
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
target_compile_definitions(distributed_ffn_grid_allgather PRIVATE
    CONFIG_FFN_GRID_ALLGATHER
    CONFIG_GRID_ROWS=${GRID_ROWS} CONFIG_GRID_COLS=${GRID_COLS}
    CONFIG_TOKEN_TILE=${TOKEN_TILE} CONFIG_MODEL_TILE=${MODEL_TILE}
    CONFIG_FFN_TILE=${FFN_TILE}
)

target_link_directories(distributed_ffn_grid_allgather PUBLIC
    ${ASCEND_HOME_PATH}/lib64
    ${ASCEND_HOME_PATH}/simulator/${SOC_VERSION}/lib
    ${ASCEND_HOME_PATH}/tools/simulator/${SOC_VERSION}/lib
)

target_link_libraries(distributed_ffn_grid_allgather PRIVATE
    distributed_ffn_grid_allgather_mixed_kernel
    $<BUILD_INTERFACE:$<$<STREQUAL:${RUN_MODE},sim>:runtime_camodel>>
    $<BUILD_INTERFACE:$<$<STREQUAL:${RUN_MODE},npu>:runtime>>
    stdc++ ascendcl hcomm m tiling_api platform c_sec dl nnopbase pthread
)

# =============================================================================
# 4. GridPipe routed K-hop unicast smoke test (Scheme A).  Vec-only data
#    movement: TPUSH<EAST, KHOP_DIST> / TPOP<EAST, KHOP_DIST> across a 1xCOLS
#    row, exercising the K-hop data path and the routed ready/free doorbell.
# =============================================================================
add_library(khop_smoke_kernel SHARED smoke/khop_smoke_kernel.cpp)
target_compile_options(khop_smoke_kernel PRIVATE
    ${CMAKE_CCE_COMPILE_OPTIONS}
    --cce-aicore-arch=dav-c220
    -DMEMORY_BASE -std=c++17
    -DCONFIG_KHOP_ROWS=${KHOP_ROWS} -DCONFIG_KHOP_COLS=${KHOP_COLS}
    -DCONFIG_KHOP_DIST=${KHOP_DIST} -DCONFIG_KHOP_T=${KHOP_T} -DCONFIG_KHOP_W=${KHOP_W}
)
target_include_directories(khop_smoke_kernel PRIVATE
    ${PROJECT_SOURCE_DIR}
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
target_link_directories(khop_smoke_kernel PRIVATE ${ASCEND_HOME_PATH}/lib64)
target_link_libraries(khop_smoke_kernel PRIVATE hcomm runtime nnopbase)
target_link_options(khop_smoke_kernel PRIVATE --cce-fatobj-link)

add_executable(khop_smoke smoke/main_khop_smoke.cpp)
target_compile_options(khop_smoke PRIVATE ${CMAKE_CPP_COMPILE_OPTIONS})
target_include_directories(khop_smoke PRIVATE
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
target_compile_definitions(khop_smoke PRIVATE
    CONFIG_KHOP_ROWS=${KHOP_ROWS} CONFIG_KHOP_COLS=${KHOP_COLS}
    CONFIG_KHOP_DIST=${KHOP_DIST} CONFIG_KHOP_T=${KHOP_T} CONFIG_KHOP_W=${KHOP_W}
)
target_link_directories(khop_smoke PUBLIC
    ${ASCEND_HOME_PATH}/lib64
    ${ASCEND_HOME_PATH}/simulator/${SOC_VERSION}/lib
    ${ASCEND_HOME_PATH}/tools/simulator/${SOC_VERSION}/lib
)
target_link_libraries(khop_smoke PRIVATE
    khop_smoke_kernel
    $<BUILD_INTERFACE:$<$<STREQUAL:${RUN_MODE},sim>:runtime_camodel>>
    $<BUILD_INTERFACE:$<$<STREQUAL:${RUN_MODE},npu>:runtime>>
    stdc++ ascendcl hcomm m tiling_api platform c_sec dl nnopbase pthread
)

# =============================================================================
# 5. GridPipe single-source broadcast smoke test (Tier 1).  Vec-only data
#    movement: one source TBCAST<ROW|COL>-es a stamped fp32 tile to its whole
#    span in a single multicast (batched writes + one publish fence + batched
#    doorbells, NOT a per-hop TPUSH loop); every other cell TPOP<dir,dist>s it.
# =============================================================================
add_library(bcast_smoke_kernel SHARED smoke/bcast_smoke_kernel.cpp)
target_compile_options(bcast_smoke_kernel PRIVATE
    ${CMAKE_CCE_COMPILE_OPTIONS}
    --cce-aicore-arch=dav-c220
    -DMEMORY_BASE -std=c++17
    -DCONFIG_BCAST_ROWS=${BCAST_ROWS} -DCONFIG_BCAST_COLS=${BCAST_COLS}
    -DCONFIG_BCAST_SRC=${BCAST_SRC} -DCONFIG_BCAST_SPAN_COL=${BCAST_SPAN_COL}
    -DCONFIG_BCAST_T=${BCAST_T} -DCONFIG_BCAST_W=${BCAST_W}
)
target_include_directories(bcast_smoke_kernel PRIVATE
    ${PROJECT_SOURCE_DIR}
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
target_link_directories(bcast_smoke_kernel PRIVATE ${ASCEND_HOME_PATH}/lib64)
target_link_libraries(bcast_smoke_kernel PRIVATE hcomm runtime nnopbase)
target_link_options(bcast_smoke_kernel PRIVATE --cce-fatobj-link)

add_executable(bcast_smoke smoke/main_bcast_smoke.cpp)
target_compile_options(bcast_smoke PRIVATE ${CMAKE_CPP_COMPILE_OPTIONS})
target_include_directories(bcast_smoke PRIVATE
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
target_compile_definitions(bcast_smoke PRIVATE
    CONFIG_BCAST_ROWS=${BCAST_ROWS} CONFIG_BCAST_COLS=${BCAST_COLS}
    CONFIG_BCAST_SRC=${BCAST_SRC} CONFIG_BCAST_SPAN_COL=${BCAST_SPAN_COL}
    CONFIG_BCAST_T=${BCAST_T} CONFIG_BCAST_W=${BCAST_W}
)
target_link_directories(bcast_smoke PUBLIC
    ${ASCEND_HOME_PATH}/lib64
    ${ASCEND_HOME_PATH}/simulator/${SOC_VERSION}/lib
    ${ASCEND_HOME_PATH}/tools/simulator/${SOC_VERSION}/lib
)
target_link_libraries(bcast_smoke PRIVATE
    bcast_smoke_kernel
    $<BUILD_INTERFACE:$<$<STREQUAL:${RUN_MODE},sim>:runtime_camodel>>
    $<BUILD_INTERFACE:$<$<STREQUAL:${RUN_MODE},npu>:runtime>>
    stdc++ ascendcl hcomm m tiling_api platform c_sec dl nnopbase pthread
)
