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
project(pto_test)

# PTO_RDMA_BACKEND is a configure-time input.  Capture it once and translate
# the only currently supported value into one consistent set of host/device
# compile definitions.  The generated binaries do not read this variable.
set(PTO_RDMA_BACKEND_CONFIGURED "$ENV{PTO_RDMA_BACKEND}")
if("${PTO_RDMA_BACKEND_CONFIGURED}" STREQUAL "HNS_1825")
    add_compile_definitions(PTO_RDMA_SUPPORTED PTO_RDMA_BACKEND_HNS_1825_SUPPORTED)
    message(STATUS "PTO RDMA backend compiled into this build: HNS_1825")
elseif("${PTO_RDMA_BACKEND_CONFIGURED}" STREQUAL "")
    message(STATUS "PTO RDMA disabled for this build: PTO_RDMA_BACKEND is unset or empty")
else()
    message(WARNING
        "Unsupported PTO_RDMA_BACKEND='${PTO_RDMA_BACKEND_CONFIGURED}'; "
        "RDMA is disabled for this build (only HNS_1825 is currently supported)")
endif()

# bisheng/cce-ld does not support --dependency-file; disable linker dep tracking
# that CMake 3.27+ enables by default.
set(CMAKE_LINK_DEPENDS_USE_LINKER FALSE)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

if(NOT DEFINED ENV{ASCEND_HOME_PATH})
    message(FATAL_ERROR "Cannot find ASCEND_HOME_PATH, please run set_env.sh.")
else()
    set(ASCEND_HOME_PATH $ENV{ASCEND_HOME_PATH})
endif()



set(ASCEND_DRIVER_PATH /usr/local/Ascend/driver)

set(CMAKE_COMPILER bisheng)
set(CMAKE_C_COMPILER ${CMAKE_COMPILER})
set(CMAKE_CXX_COMPILER ${CMAKE_COMPILER})

find_package(GTest REQUIRED)

add_compile_options(
    -D_FORTIFY_SOURCE=2
    -O2 -std=c++17
    -Wno-macro-redefined -Wno-ignored-attributes
    -fstack-protector-strong
)
# add_compile_definitions(_GLIBCXX_USE_CXX11_ABI=0)
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

set(CMAKE_CPP_COMPILE_OPTIONS
    -xc++
    "SHELL:-include stdint.h"
    "SHELL:-include stddef.h"
)

if(DEBUG_MODE)
    message(STATUS "Debug Mode Enabled, Add Debug Options")
    list(APPEND CMAKE_CPP_COMPILE_OPTIONS -DCOMM_DEBUG)
    list(APPEND CMAKE_CCE_COMPILE_OPTIONS -DCOMM_DEBUG)
endif()

# 公共 include 路径
include_directories(
    ${PROJECT_SOURCE_DIR}/../../../../../include
    ${PROJECT_SOURCE_DIR}/../../../../../pkg_inc
    ${PROJECT_SOURCE_DIR}/../../../../..
    ${ASCEND_HOME_PATH}/include
    ${ASCEND_DRIVER_PATH}/kernel/inc
)

# hccl include paths
include_directories(
    ${ASCEND_HOME_PATH}/include/hccl
)

option(HIGH_PRECISION_IMPL "Use high-precision algorithms (usually slower) for some operations" OFF)

add_subdirectory(testcase)
