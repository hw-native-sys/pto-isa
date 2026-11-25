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
if(NOT DEFINED ENV{ASCEND_HOME_PATH})
    message(FATAL_ERROR "Set ASCEND_HOME_PATH to the CANN toolkit before configuring")
endif()
set(ASCEND_HOME_PATH "$ENV{ASCEND_HOME_PATH}")
if(NOT DEFINED CMAKE_CXX_COMPILER)
    find_program(FLASH_MLA_COMPILER bisheng HINTS "${ASCEND_HOME_PATH}/tools/bisheng_compiler/bin" REQUIRED)
    set(CMAKE_CXX_COMPILER "${FLASH_MLA_COMPILER}" CACHE FILEPATH "CANN compiler")
endif()
project(flash_mla_bf16 LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(RUN_MODE sim CACHE STRING "Runtime: sim or npu")
set(SOC_VERSION Ascend950PR_9599 CACHE STRING "Simulator platform")
set(FLASH_MLA_TILE_N 128 CACHE STRING "KV token tile size: 64 or 128")
set_property(CACHE FLASH_MLA_TILE_N PROPERTY STRINGS 64 128)
if(NOT RUN_MODE MATCHES "^(sim|npu)$")
    message(FATAL_ERROR "RUN_MODE must be sim or npu")
endif()
if(NOT SOC_VERSION MATCHES "^Ascend950")
    message(FATAL_ERROR "BF16 FlashMLA requires an A5 / Ascend950 platform")
endif()
if(NOT "${FLASH_MLA_TILE_N}" MATCHES "^(64|128)$")
    message(FATAL_ERROR "FLASH_MLA_TILE_N must be 64 or 128")
endif()
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
add_compile_options(-O2 -fstack-protector-strong -Wno-macro-redefined -Wno-ignored-attributes)
add_link_options(-Wl,-z,relro -Wl,-z,now)
include_directories(
    "${PROJECT_SOURCE_DIR}/../../../../include"
    "${ASCEND_HOME_PATH}/include"
    "${ASCEND_HOME_PATH}/pkg_inc"
    "${ASCEND_HOME_PATH}/pkg_inc/profiling"
    "${ASCEND_HOME_PATH}/pkg_inc/runtime"
    "${ASCEND_HOME_PATH}/include/experiment/runtime"
    "${ASCEND_HOME_PATH}/include/experiment/msprof"
)
add_library(flash_mla_bf16_kernel SHARED flash_mla_kernel.cpp)
add_library(flash_mla_bf16_host STATIC host_runner.cpp)
add_executable(flash_mla_bf16_test main.cpp)
if(RUN_MODE STREQUAL "npu")
    add_executable(flash_mla_bf16_bench bench_main.cpp)
    target_link_libraries(flash_mla_bf16_bench PRIVATE flash_mla_bf16_host)
endif()
target_link_libraries(flash_mla_bf16_test PRIVATE flash_mla_bf16_host)
foreach(target flash_mla_bf16_kernel flash_mla_bf16_host)
    target_compile_definitions(${target} PRIVATE
        FLASH_MLA_TILE_N=${FLASH_MLA_TILE_N}
        FLASH_MLA_SIM_RUNTIME=$<STREQUAL:${RUN_MODE},sim>
    )
endforeach()
# Compile device code as C++17 for compatibility with the CANN frontend.
set_property(TARGET flash_mla_bf16_kernel PROPERTY CXX_STANDARD 17)
target_compile_options(flash_mla_bf16_kernel PRIVATE
    -xcce --cce-aicore-arch=dav-c310 -Xhost-start -Xhost-end
    "SHELL:-mllvm -cce-aicore-stack-size=0x8000"
    "SHELL:-mllvm -cce-aicore-function-stack-size=0x8000"
    "SHELL:-mllvm -cce-aicore-record-overflow=true"
    "SHELL:-mllvm -cce-aicore-addr-transform"
    "SHELL:-mllvm -cce-aicore-dcci-insert-for-scalar=false"
    -U__ADVSIMD_VEC_MATH_SUPPORTED -U__SVE_VEC_MATH_SUPPORTED
)
target_link_options(flash_mla_bf16_kernel PRIVATE --cce-fatobj-link)
target_compile_options(flash_mla_bf16_host PRIVATE -xc++ "SHELL:-include stdint.h" "SHELL:-include stddef.h")
target_link_directories(flash_mla_bf16_host PUBLIC
    "${ASCEND_HOME_PATH}/lib64"
    "${ASCEND_HOME_PATH}/simulator/${SOC_VERSION}/lib"
    "${ASCEND_HOME_PATH}/tools/simulator/${SOC_VERSION}/lib"
)
if(RUN_MODE STREQUAL "sim")
    set(FLASH_MLA_RUNTIME runtime_camodel)
else()
    set(FLASH_MLA_RUNTIME runtime)
endif()
target_link_libraries(flash_mla_bf16_host PUBLIC flash_mla_bf16_kernel ${FLASH_MLA_RUNTIME}
    stdc++ ascendcl m tiling_api platform c_sec dl nnopbase pthread unified_dlog)
file(WRITE "${CMAKE_BINARY_DIR}/flash_mla_run_mode.txt" "${RUN_MODE}\n")
file(WRITE "${CMAKE_BINARY_DIR}/flash_mla_soc_version.txt" "${SOC_VERSION}\n")
file(WRITE "${CMAKE_BINARY_DIR}/flash_mla_tile_n.txt" "${FLASH_MLA_TILE_N}\n")
