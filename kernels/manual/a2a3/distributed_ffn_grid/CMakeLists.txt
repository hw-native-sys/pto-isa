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

# Pure 1D N-cut 32-cell AllGather topology (方案①) -- real DeepSeek-v4 Pro shapes.
# Only the AllGather variant consumes these; ReduceSum keeps the legacy tiles above.
if(NOT DEFINED NCUT_GRID_ROWS)
    set(NCUT_GRID_ROWS 4)
endif()
if(NOT DEFINED NCUT_GRID_COLS)
    set(NCUT_GRID_COLS 8)
endif()
if(NOT DEFINED NCUT_T)
    set(NCUT_T 8)
endif()
if(NOT DEFINED NCUT_H)
    set(NCUT_H 7168)
endif()
if(NOT DEFINED NCUT_I)
    set(NCUT_I 3072)
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
if(NOT DEFINED BCAST_SUBRECT)
    set(BCAST_SUBRECT 0)
endif()
if(NOT DEFINED BCAST_RECT_R0)
    set(BCAST_RECT_R0 0)
endif()
if(NOT DEFINED BCAST_RECT_R1)
    set(BCAST_RECT_R1 ${BCAST_ROWS})
endif()
if(NOT DEFINED BCAST_RECT_C0)
    set(BCAST_RECT_C0 0)
endif()
if(NOT DEFINED BCAST_RECT_C1)
    set(BCAST_RECT_C1 ${BCAST_COLS})
endif()
if(NOT DEFINED BCAST_RECT_SRC)
    set(BCAST_RECT_SRC 0)
endif()
if(NOT DEFINED BCAST_ROUNDS)
    set(BCAST_ROUNDS 1)
endif()
if(NOT DEFINED BCAST_MAX_SPINS)
    # Spin bound for the all-sources branch.  0 = the public overloads (block
    # forever, the shipping path); a small n (1000 is plenty) switches to the TRY
    # forms so a stuck handshake reports a fault code instead of hanging.  One
    # spin is a whole service pass, so n is counted in passes, not in loads.
    set(BCAST_MAX_SPINS 0)
endif()
if(NOT DEFINED BCAST_ALL_SOURCES)
    set(BCAST_ALL_SOURCES 0)
endif()
if(NOT DEFINED BCAST_T)
    set(BCAST_T 16)
endif()
# How many publishers one receiver may have in flight (PTO_GRID_BCAST_TICKET_BATCH).
# It is the grant WINDOW width, so it is also the collective's concurrency degree:
# 1 serialises the publishers strictly by basek, SlotCount lets a whole ring's
# worth of them publish at once.  The pipe clamps it to SlotCount, so the default
# below simply means "as concurrent as the ring allows"; set it to 1 to exercise
# the strictly-serialised grant path.
if(NOT DEFINED BCAST_TICKET_BATCH)
    set(BCAST_TICKET_BATCH 32)
endif()
# Broadcast ring depth in slots; 0 = one slot per group member.  Smaller than the
# group width is what forces the kernel's WAVE loop -- the case the caller-supplied
# address exists for.
if(NOT DEFINED BCAST_SLOT_COUNT)
    set(BCAST_SLOT_COUNT 0)
endif()
if(NOT DEFINED BCAST_W)
    set(BCAST_W 64)
endif()

# Unicast HANDOVER smoke: two producers take turns on one consumer channel and the
# second takes it over while the first one's tiles are still undrained.
if(NOT DEFINED UNICAST_TILES)
    set(UNICAST_TILES 2)
endif()
if(NOT DEFINED UNICAST_SLOT_COUNT)
    set(UNICAST_SLOT_COUNT 4)
endif()
if(NOT DEFINED UNICAST_MAX_SPINS)
    set(UNICAST_MAX_SPINS 0)
endif()
if(NOT DEFINED UNICAST_T)
    set(UNICAST_T 16)
endif()
if(NOT DEFINED UNICAST_W)
    set(UNICAST_W 64)
endif()

# Reduce <-> unicast CHANNEL RELAY smoke: one shared-pool channel carries a group
# reduce, then a unicast flow, then a group reduce again.
if(NOT DEFINED RELAY_ROUNDS)
    set(RELAY_ROUNDS 2)
endif()
if(NOT DEFINED RELAY_TILES)
    set(RELAY_TILES 3)
endif()
if(NOT DEFINED RELAY_SIDE_TILES)
    set(RELAY_SIDE_TILES 1)
endif()
if(NOT DEFINED RELAY_SLOT_COUNT)
    set(RELAY_SLOT_COUNT 2)
endif()
if(NOT DEFINED RELAY_MAX_SPINS)
    set(RELAY_MAX_SPINS 0)
endif()
if(NOT DEFINED RELAY_T)
    set(RELAY_T 16)
endif()
if(NOT DEFINED RELAY_W)
    set(RELAY_W 64)
endif()

message(STATUS "Grid: ${GRID_ROWS}x${GRID_COLS}  Tile: ${TOKEN_TILE}x${MODEL_TILE}  FfnTile: ${FFN_TILE}")
message(STATUS "NCut topology: ${NCUT_GRID_ROWS}x${NCUT_GRID_COLS}  T=${NCUT_T} H=${NCUT_H} I=${NCUT_I}")
message(STATUS "Unicast handover smoke: tiles/turn=${UNICAST_TILES} slots=${UNICAST_SLOT_COUNT} maxSpins=${UNICAST_MAX_SPINS} tile=${UNICAST_T}x${UNICAST_W}")
message(STATUS "Channel relay smoke: rounds/reduce=${RELAY_ROUNDS} unicast tiles=${RELAY_TILES} side tiles=${RELAY_SIDE_TILES} slots=${RELAY_SLOT_COUNT} maxSpins=${RELAY_MAX_SPINS} tile=${RELAY_T}x${RELAY_W}")
message(STATUS "Bcast smoke: ${BCAST_ROWS}x${BCAST_COLS}  SRC=${BCAST_SRC}  SPAN_COL=${BCAST_SPAN_COL}  SUBRECT=${BCAST_SUBRECT}  Rect=[r${BCAST_RECT_R0}:${BCAST_RECT_R1},c${BCAST_RECT_C0}:${BCAST_RECT_C1}] RectSRC=${BCAST_RECT_SRC}  Rounds=${BCAST_ROUNDS}  AllSrc=${BCAST_ALL_SOURCES}  Tile: ${BCAST_T}x${BCAST_W}  TicketBatch=${BCAST_TICKET_BATCH}  Slots=${BCAST_SLOT_COUNT}")

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
add_library(distributed_ffn_grid_treduce_reducesum_mixed_kernel SHARED distributed_ffn_grid_treduce_reducesum_compute_kernel.cpp)
target_compile_options(distributed_ffn_grid_treduce_reducesum_mixed_kernel PRIVATE
    ${CMAKE_CCE_COMPILE_OPTIONS}
    --cce-aicore-arch=dav-c220
    -DMEMORY_BASE -std=c++17
    -DCONFIG_GRID_ROWS=${GRID_ROWS} -DCONFIG_GRID_COLS=${GRID_COLS}
    -DCONFIG_TOKEN_TILE=${TOKEN_TILE} -DCONFIG_MODEL_TILE=${MODEL_TILE}
    -DCONFIG_FFN_TILE=${FFN_TILE}
    -DCONFIG_NCUT_GRID_ROWS=${NCUT_GRID_ROWS} -DCONFIG_NCUT_GRID_COLS=${NCUT_GRID_COLS}
    -DCONFIG_NCUT_T=${NCUT_T} -DCONFIG_NCUT_H=${NCUT_H} -DCONFIG_NCUT_I=${NCUT_I}
)
target_include_directories(distributed_ffn_grid_treduce_reducesum_mixed_kernel PRIVATE
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
target_link_directories(distributed_ffn_grid_treduce_reducesum_mixed_kernel PRIVATE ${ASCEND_HOME_PATH}/lib64)
target_link_libraries(distributed_ffn_grid_treduce_reducesum_mixed_kernel PRIVATE hcomm runtime nnopbase)
target_link_options(distributed_ffn_grid_treduce_reducesum_mixed_kernel PRIVATE --cce-fatobj-link)

# AllGather split variant: hidden shards are gathered through GridPipe, and
# down projection writes one output-H shard per column.
add_library(distributed_ffn_grid_tbroadcast_allgather_mixed_kernel SHARED distributed_ffn_grid_tbroadcast_allgather_compute_kernel.cpp)
target_compile_options(distributed_ffn_grid_tbroadcast_allgather_mixed_kernel PRIVATE
    ${CMAKE_CCE_COMPILE_OPTIONS}
    --cce-aicore-arch=dav-c220
    -DMEMORY_BASE -std=c++17
    -DCONFIG_FFN_GRID_ALLGATHER
    -DCONFIG_GRID_ROWS=${GRID_ROWS} -DCONFIG_GRID_COLS=${GRID_COLS}
    -DCONFIG_TOKEN_TILE=${TOKEN_TILE} -DCONFIG_MODEL_TILE=${MODEL_TILE}
    -DCONFIG_FFN_TILE=${FFN_TILE}
    -DCONFIG_NCUT_GRID_ROWS=${NCUT_GRID_ROWS} -DCONFIG_NCUT_GRID_COLS=${NCUT_GRID_COLS}
    -DCONFIG_NCUT_T=${NCUT_T} -DCONFIG_NCUT_H=${NCUT_H} -DCONFIG_NCUT_I=${NCUT_I}
    -DPTO_GRID_BCAST_TICKET_BATCH=${BCAST_TICKET_BATCH}u
)
target_include_directories(distributed_ffn_grid_tbroadcast_allgather_mixed_kernel PRIVATE
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
target_link_directories(distributed_ffn_grid_tbroadcast_allgather_mixed_kernel PRIVATE ${ASCEND_HOME_PATH}/lib64)
target_link_libraries(distributed_ffn_grid_tbroadcast_allgather_mixed_kernel PRIVATE hcomm runtime nnopbase)
target_link_options(distributed_ffn_grid_tbroadcast_allgather_mixed_kernel PRIVATE --cce-fatobj-link)

# =============================================================================
# 2. Host driver executable
# =============================================================================
add_executable(distributed_ffn_grid_treduce_reducesum main_treduce_reducesum.cpp)
target_compile_options(distributed_ffn_grid_treduce_reducesum PRIVATE ${CMAKE_CPP_COMPILE_OPTIONS})
target_include_directories(distributed_ffn_grid_treduce_reducesum PRIVATE
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
target_compile_definitions(distributed_ffn_grid_treduce_reducesum PRIVATE
    CONFIG_GRID_ROWS=${GRID_ROWS} CONFIG_GRID_COLS=${GRID_COLS}
    CONFIG_TOKEN_TILE=${TOKEN_TILE} CONFIG_MODEL_TILE=${MODEL_TILE}
    CONFIG_FFN_TILE=${FFN_TILE}
    CONFIG_NCUT_GRID_ROWS=${NCUT_GRID_ROWS} CONFIG_NCUT_GRID_COLS=${NCUT_GRID_COLS}
    CONFIG_NCUT_T=${NCUT_T} CONFIG_NCUT_H=${NCUT_H} CONFIG_NCUT_I=${NCUT_I}
)

target_link_directories(distributed_ffn_grid_treduce_reducesum PUBLIC
    ${ASCEND_HOME_PATH}/lib64
    ${ASCEND_HOME_PATH}/simulator/${SOC_VERSION}/lib
    ${ASCEND_HOME_PATH}/tools/simulator/${SOC_VERSION}/lib
)

target_link_libraries(distributed_ffn_grid_treduce_reducesum PRIVATE
    distributed_ffn_grid_treduce_reducesum_mixed_kernel
    $<BUILD_INTERFACE:$<$<STREQUAL:${RUN_MODE},sim>:runtime_camodel>>
    $<BUILD_INTERFACE:$<$<STREQUAL:${RUN_MODE},npu>:runtime>>
    stdc++ ascendcl hcomm m tiling_api platform c_sec dl nnopbase pthread
)

add_executable(distributed_ffn_grid_tbroadcast_allgather main_tbroadcast_allgather.cpp)
target_compile_options(distributed_ffn_grid_tbroadcast_allgather PRIVATE ${CMAKE_CPP_COMPILE_OPTIONS})
target_include_directories(distributed_ffn_grid_tbroadcast_allgather PRIVATE
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
target_compile_definitions(distributed_ffn_grid_tbroadcast_allgather PRIVATE
    CONFIG_FFN_GRID_ALLGATHER
    CONFIG_GRID_ROWS=${GRID_ROWS} CONFIG_GRID_COLS=${GRID_COLS}
    CONFIG_TOKEN_TILE=${TOKEN_TILE} CONFIG_MODEL_TILE=${MODEL_TILE}
    CONFIG_FFN_TILE=${FFN_TILE}
    CONFIG_NCUT_GRID_ROWS=${NCUT_GRID_ROWS} CONFIG_NCUT_GRID_COLS=${NCUT_GRID_COLS}
    CONFIG_NCUT_T=${NCUT_T} CONFIG_NCUT_H=${NCUT_H} CONFIG_NCUT_I=${NCUT_I}
)

target_link_directories(distributed_ffn_grid_tbroadcast_allgather PUBLIC
    ${ASCEND_HOME_PATH}/lib64
    ${ASCEND_HOME_PATH}/simulator/${SOC_VERSION}/lib
    ${ASCEND_HOME_PATH}/tools/simulator/${SOC_VERSION}/lib
)

target_link_libraries(distributed_ffn_grid_tbroadcast_allgather PRIVATE
    distributed_ffn_grid_tbroadcast_allgather_mixed_kernel
    $<BUILD_INTERFACE:$<$<STREQUAL:${RUN_MODE},sim>:runtime_camodel>>
    $<BUILD_INTERFACE:$<$<STREQUAL:${RUN_MODE},npu>:runtime>>
    stdc++ ascendcl hcomm m tiling_api platform c_sec dl nnopbase pthread
)

# =============================================================================
# 3. TPUSH verification variants.  Same FFN compute/topology as the TREDUCE /
#    TBROADCAST variants above, but the cross-cell collectives are spelled with the
#    explicit TPUSH/TPOP unicast primitives, proving them end-to-end on a real
#    distributed FFN.
# =============================================================================

# TPUSH ReduceSum variant: EAST+SOUTH reduction driven by explicit TPOP + TADD +
# TPUSH (the A3 lowering of TREDUCE) instead of the fused collective.
add_library(distributed_ffn_grid_tpush_reducesum_mixed_kernel SHARED distributed_ffn_grid_tpush_reducesum_compute_kernel.cpp)
target_compile_options(distributed_ffn_grid_tpush_reducesum_mixed_kernel PRIVATE
    ${CMAKE_CCE_COMPILE_OPTIONS}
    --cce-aicore-arch=dav-c220
    -DMEMORY_BASE -std=c++17
    -DCONFIG_GRID_ROWS=${GRID_ROWS} -DCONFIG_GRID_COLS=${GRID_COLS}
    -DCONFIG_TOKEN_TILE=${TOKEN_TILE} -DCONFIG_MODEL_TILE=${MODEL_TILE}
    -DCONFIG_FFN_TILE=${FFN_TILE}
    -DCONFIG_NCUT_GRID_ROWS=${NCUT_GRID_ROWS} -DCONFIG_NCUT_GRID_COLS=${NCUT_GRID_COLS}
    -DCONFIG_NCUT_T=${NCUT_T} -DCONFIG_NCUT_H=${NCUT_H} -DCONFIG_NCUT_I=${NCUT_I}
)
target_include_directories(distributed_ffn_grid_tpush_reducesum_mixed_kernel PRIVATE
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
target_link_directories(distributed_ffn_grid_tpush_reducesum_mixed_kernel PRIVATE ${ASCEND_HOME_PATH}/lib64)
target_link_libraries(distributed_ffn_grid_tpush_reducesum_mixed_kernel PRIVATE hcomm runtime nnopbase)
target_link_options(distributed_ffn_grid_tpush_reducesum_mixed_kernel PRIVATE --cce-fatobj-link)

# TPUSH AllGather variant: the two gather phases are nearest-neighbor TPUSH/TPOP
# relays (a fan-in-1 DAG) instead of the TBROADCAST MPSC collective.
add_library(distributed_ffn_grid_tpush_allgather_mixed_kernel SHARED distributed_ffn_grid_tpush_allgather_compute_kernel.cpp)
target_compile_options(distributed_ffn_grid_tpush_allgather_mixed_kernel PRIVATE
    ${CMAKE_CCE_COMPILE_OPTIONS}
    --cce-aicore-arch=dav-c220
    -DMEMORY_BASE -std=c++17
    -DCONFIG_FFN_GRID_ALLGATHER
    -DCONFIG_GRID_ROWS=${GRID_ROWS} -DCONFIG_GRID_COLS=${GRID_COLS}
    -DCONFIG_TOKEN_TILE=${TOKEN_TILE} -DCONFIG_MODEL_TILE=${MODEL_TILE}
    -DCONFIG_FFN_TILE=${FFN_TILE}
    -DCONFIG_NCUT_GRID_ROWS=${NCUT_GRID_ROWS} -DCONFIG_NCUT_GRID_COLS=${NCUT_GRID_COLS}
    -DCONFIG_NCUT_T=${NCUT_T} -DCONFIG_NCUT_H=${NCUT_H} -DCONFIG_NCUT_I=${NCUT_I}
)
target_include_directories(distributed_ffn_grid_tpush_allgather_mixed_kernel PRIVATE
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
target_link_directories(distributed_ffn_grid_tpush_allgather_mixed_kernel PRIVATE ${ASCEND_HOME_PATH}/lib64)
target_link_libraries(distributed_ffn_grid_tpush_allgather_mixed_kernel PRIVATE hcomm runtime nnopbase)
target_link_options(distributed_ffn_grid_tpush_allgather_mixed_kernel PRIVATE --cce-fatobj-link)

add_executable(distributed_ffn_grid_tpush_reducesum main_tpush_reducesum.cpp)
target_compile_options(distributed_ffn_grid_tpush_reducesum PRIVATE ${CMAKE_CPP_COMPILE_OPTIONS})
target_include_directories(distributed_ffn_grid_tpush_reducesum PRIVATE
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
target_compile_definitions(distributed_ffn_grid_tpush_reducesum PRIVATE
    CONFIG_GRID_ROWS=${GRID_ROWS} CONFIG_GRID_COLS=${GRID_COLS}
    CONFIG_TOKEN_TILE=${TOKEN_TILE} CONFIG_MODEL_TILE=${MODEL_TILE}
    CONFIG_FFN_TILE=${FFN_TILE}
    CONFIG_NCUT_GRID_ROWS=${NCUT_GRID_ROWS} CONFIG_NCUT_GRID_COLS=${NCUT_GRID_COLS}
    CONFIG_NCUT_T=${NCUT_T} CONFIG_NCUT_H=${NCUT_H} CONFIG_NCUT_I=${NCUT_I}
)

target_link_directories(distributed_ffn_grid_tpush_reducesum PUBLIC
    ${ASCEND_HOME_PATH}/lib64
    ${ASCEND_HOME_PATH}/simulator/${SOC_VERSION}/lib
    ${ASCEND_HOME_PATH}/tools/simulator/${SOC_VERSION}/lib
)

target_link_libraries(distributed_ffn_grid_tpush_reducesum PRIVATE
    distributed_ffn_grid_tpush_reducesum_mixed_kernel
    $<BUILD_INTERFACE:$<$<STREQUAL:${RUN_MODE},sim>:runtime_camodel>>
    $<BUILD_INTERFACE:$<$<STREQUAL:${RUN_MODE},npu>:runtime>>
    stdc++ ascendcl hcomm m tiling_api platform c_sec dl nnopbase pthread
)

add_executable(distributed_ffn_grid_tpush_allgather main_tpush_allgather.cpp)
target_compile_options(distributed_ffn_grid_tpush_allgather PRIVATE ${CMAKE_CPP_COMPILE_OPTIONS})
target_include_directories(distributed_ffn_grid_tpush_allgather PRIVATE
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
target_compile_definitions(distributed_ffn_grid_tpush_allgather PRIVATE
    CONFIG_FFN_GRID_ALLGATHER
    CONFIG_GRID_ROWS=${GRID_ROWS} CONFIG_GRID_COLS=${GRID_COLS}
    CONFIG_TOKEN_TILE=${TOKEN_TILE} CONFIG_MODEL_TILE=${MODEL_TILE}
    CONFIG_FFN_TILE=${FFN_TILE}
    CONFIG_NCUT_GRID_ROWS=${NCUT_GRID_ROWS} CONFIG_NCUT_GRID_COLS=${NCUT_GRID_COLS}
    CONFIG_NCUT_T=${NCUT_T} CONFIG_NCUT_H=${NCUT_H} CONFIG_NCUT_I=${NCUT_I}
)

target_link_directories(distributed_ffn_grid_tpush_allgather PUBLIC
    ${ASCEND_HOME_PATH}/lib64
    ${ASCEND_HOME_PATH}/simulator/${SOC_VERSION}/lib
    ${ASCEND_HOME_PATH}/tools/simulator/${SOC_VERSION}/lib
)

target_link_libraries(distributed_ffn_grid_tpush_allgather PRIVATE
    distributed_ffn_grid_tpush_allgather_mixed_kernel
    $<BUILD_INTERFACE:$<$<STREQUAL:${RUN_MODE},sim>:runtime_camodel>>
    $<BUILD_INTERFACE:$<$<STREQUAL:${RUN_MODE},npu>:runtime>>
    stdc++ ascendcl hcomm m tiling_api platform c_sec dl nnopbase pthread
)

# =============================================================================
# 4. GridPipe single-source broadcast smoke test (Tier 1).  Vec-only data
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
    -DCONFIG_BCAST_SUBRECT=${BCAST_SUBRECT}
    -DCONFIG_BCAST_RECT_R0=${BCAST_RECT_R0} -DCONFIG_BCAST_RECT_R1=${BCAST_RECT_R1}
    -DCONFIG_BCAST_RECT_C0=${BCAST_RECT_C0} -DCONFIG_BCAST_RECT_C1=${BCAST_RECT_C1}
    -DCONFIG_BCAST_RECT_SRC=${BCAST_RECT_SRC}
    -DCONFIG_BCAST_ROUNDS=${BCAST_ROUNDS} -DCONFIG_BCAST_ALL_SOURCES=${BCAST_ALL_SOURCES}
    -DCONFIG_BCAST_MAX_SPINS=${BCAST_MAX_SPINS}u
    -DCONFIG_BCAST_T=${BCAST_T} -DCONFIG_BCAST_W=${BCAST_W}
    -DCONFIG_BCAST_SLOT_COUNT=${BCAST_SLOT_COUNT}
    -DPTO_GRID_BCAST_TICKET_BATCH=${BCAST_TICKET_BATCH}u
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
    CONFIG_BCAST_SUBRECT=${BCAST_SUBRECT}
    CONFIG_BCAST_RECT_R0=${BCAST_RECT_R0} CONFIG_BCAST_RECT_R1=${BCAST_RECT_R1}
    CONFIG_BCAST_RECT_C0=${BCAST_RECT_C0} CONFIG_BCAST_RECT_C1=${BCAST_RECT_C1}
    CONFIG_BCAST_RECT_SRC=${BCAST_RECT_SRC}
    CONFIG_BCAST_ROUNDS=${BCAST_ROUNDS} CONFIG_BCAST_ALL_SOURCES=${BCAST_ALL_SOURCES}
    CONFIG_BCAST_MAX_SPINS=${BCAST_MAX_SPINS}u
    CONFIG_BCAST_T=${BCAST_T} CONFIG_BCAST_W=${BCAST_W}
    CONFIG_BCAST_SLOT_COUNT=${BCAST_SLOT_COUNT}
    PTO_GRID_BCAST_TICKET_BATCH=${BCAST_TICKET_BATCH}u
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

# =============================================================================
# 5. GridPipe unicast HANDOVER smoke test (Tier 1).  Vec-only: two producers take
#    turns on ONE consumer channel, and the second takes it over while the first
#    one's tiles are still in the ring -- the case relay counting exists for, and
#    the one no other test in this tree reaches.
# =============================================================================
add_library(unicast_smoke_kernel SHARED smoke/unicast_smoke_kernel.cpp)
target_compile_options(unicast_smoke_kernel PRIVATE
    ${CMAKE_CCE_COMPILE_OPTIONS}
    --cce-aicore-arch=dav-c220
    -DMEMORY_BASE -std=c++17
    -DCONFIG_UNICAST_TILES=${UNICAST_TILES}
    -DCONFIG_UNICAST_SLOT_COUNT=${UNICAST_SLOT_COUNT}
    -DCONFIG_UNICAST_MAX_SPINS=${UNICAST_MAX_SPINS}u
    -DCONFIG_UNICAST_T=${UNICAST_T} -DCONFIG_UNICAST_W=${UNICAST_W}
)
target_include_directories(unicast_smoke_kernel PRIVATE
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
target_link_directories(unicast_smoke_kernel PRIVATE ${ASCEND_HOME_PATH}/lib64)
target_link_libraries(unicast_smoke_kernel PRIVATE hcomm runtime nnopbase)
target_link_options(unicast_smoke_kernel PRIVATE --cce-fatobj-link)

add_executable(unicast_smoke smoke/main_unicast_smoke.cpp)
target_compile_options(unicast_smoke PRIVATE ${CMAKE_CPP_COMPILE_OPTIONS})
target_include_directories(unicast_smoke PRIVATE
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
target_compile_definitions(unicast_smoke PRIVATE
    CONFIG_UNICAST_TILES=${UNICAST_TILES}
    CONFIG_UNICAST_SLOT_COUNT=${UNICAST_SLOT_COUNT}
    CONFIG_UNICAST_MAX_SPINS=${UNICAST_MAX_SPINS}u
    CONFIG_UNICAST_T=${UNICAST_T} CONFIG_UNICAST_W=${UNICAST_W}
)
target_link_directories(unicast_smoke PUBLIC
    ${ASCEND_HOME_PATH}/lib64
    ${ASCEND_HOME_PATH}/simulator/${SOC_VERSION}/lib
    ${ASCEND_HOME_PATH}/tools/simulator/${SOC_VERSION}/lib
)
target_link_libraries(unicast_smoke PRIVATE
    unicast_smoke_kernel
    $<BUILD_INTERFACE:$<$<STREQUAL:${RUN_MODE},sim>:runtime_camodel>>
    $<BUILD_INTERFACE:$<$<STREQUAL:${RUN_MODE},npu>:runtime>>
    stdc++ ascendcl hcomm m tiling_api platform c_sec dl nnopbase pthread
)


# =============================================================================
# 6. GridPipe reduce <-> unicast CHANNEL RELAY smoke test (Tier 1).  Vec-only:
#    ONE channel in the shared pool carries a group reduce, then a unicast flow,
#    then a group reduce again -- first one stage per launch, then all stages in
#    one launch.  It exercises both handover rules and the mid-launch release.
# =============================================================================
add_library(relay_smoke_kernel SHARED smoke/relay_smoke_kernel.cpp)
target_compile_options(relay_smoke_kernel PRIVATE
    ${CMAKE_CCE_COMPILE_OPTIONS}
    --cce-aicore-arch=dav-c220
    -DMEMORY_BASE -std=c++17
    -DCONFIG_RELAY_ROUNDS=${RELAY_ROUNDS}
    -DCONFIG_RELAY_TILES=${RELAY_TILES}
    -DCONFIG_RELAY_SIDE_TILES=${RELAY_SIDE_TILES}
    -DCONFIG_RELAY_SLOT_COUNT=${RELAY_SLOT_COUNT}
    -DCONFIG_RELAY_MAX_SPINS=${RELAY_MAX_SPINS}u
    -DCONFIG_RELAY_T=${RELAY_T} -DCONFIG_RELAY_W=${RELAY_W}
)
target_include_directories(relay_smoke_kernel PRIVATE
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
target_link_directories(relay_smoke_kernel PRIVATE ${ASCEND_HOME_PATH}/lib64)
target_link_libraries(relay_smoke_kernel PRIVATE hcomm runtime nnopbase)
target_link_options(relay_smoke_kernel PRIVATE --cce-fatobj-link)

add_executable(relay_smoke smoke/main_relay_smoke.cpp)
target_compile_options(relay_smoke PRIVATE ${CMAKE_CPP_COMPILE_OPTIONS})
target_include_directories(relay_smoke PRIVATE
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
target_compile_definitions(relay_smoke PRIVATE
    CONFIG_RELAY_ROUNDS=${RELAY_ROUNDS}
    CONFIG_RELAY_TILES=${RELAY_TILES}
    CONFIG_RELAY_SIDE_TILES=${RELAY_SIDE_TILES}
    CONFIG_RELAY_SLOT_COUNT=${RELAY_SLOT_COUNT}
    CONFIG_RELAY_MAX_SPINS=${RELAY_MAX_SPINS}u
    CONFIG_RELAY_T=${RELAY_T} CONFIG_RELAY_W=${RELAY_W}
)
target_link_directories(relay_smoke PUBLIC
    ${ASCEND_HOME_PATH}/lib64
    ${ASCEND_HOME_PATH}/simulator/${SOC_VERSION}/lib
    ${ASCEND_HOME_PATH}/tools/simulator/${SOC_VERSION}/lib
)
target_link_libraries(relay_smoke PRIVATE
    relay_smoke_kernel
    $<BUILD_INTERFACE:$<$<STREQUAL:${RUN_MODE},sim>:runtime_camodel>>
    $<BUILD_INTERFACE:$<$<STREQUAL:${RUN_MODE},npu>:runtime>>
    stdc++ ascendcl hcomm m tiling_api platform c_sec dl nnopbase pthread
)
