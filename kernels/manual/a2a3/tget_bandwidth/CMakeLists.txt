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
project(pto_example)

set(CMAKE_LINK_DEPENDS_USE_LINKER FALSE)

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

set(ASCEND_DRIVER_PATH /usr/local/Ascend/driver)

set(CMAKE_COMPILER bisheng)
set(CMAKE_C_COMPILER ${CMAKE_COMPILER})
set(CMAKE_CXX_COMPILER ${CMAKE_COMPILER})

add_compile_options(
    -D_FORTIFY_SOURCE=2
    -O2 -std=c++17
    -Wno-macro-redefined -Wno-ignored-attributes
    -fstack-protector-strong
)

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
set(CMAKE_CCE_COMPILE_OPTIONS "${CMAKE_CCE_COMPILE_OPTIONS} --cce-pto-enable")

if(DEBUG_MODE)
    message(STATUS "Debug Mode Enabled, Add Debug Options")
    add_compile_definitions(_DEBUG)
    list(APPEND CMAKE_CCE_COMPILE_OPTIONS -DCOMM_DEBUG)
endif()

set(CMAKE_CPP_COMPILE_OPTIONS
    -xc++
    "SHELL:-include stdint.h"
    "SHELL:-include stddef.h"
)

set(PTO_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/../../../../include)
set(COMM_TESTCASE_DIR ${PROJECT_SOURCE_DIR}/../../../../tests/npu/a2a3/comm/st/testcase)

include_directories(
    ${PTO_INCLUDE_DIR}
    ${COMM_TESTCASE_DIR}
    ${ASCEND_HOME_PATH}/include
    ${ASCEND_HOME_PATH}/include/hccl
    ${ASCEND_DRIVER_PATH}/kernel/inc
)

# Host-arch aware include root: x86_64 first, fallback to aarch64.
set(PTO_TOOLKIT_DIR "${ASCEND_HOME_PATH}/x86_64-linux")
if(NOT EXISTS "${PTO_TOOLKIT_DIR}")
    set(PTO_TOOLKIT_DIR "${ASCEND_HOME_PATH}/aarch64-linux")
endif()

set(PTO_ASCENDC_KERNEL_INCLUDE "")
if(EXISTS "${ASCEND_HOME_PATH}/tools/tikcpp/ascendc_kernel/include")
    set(PTO_ASCENDC_KERNEL_INCLUDE "${ASCEND_HOME_PATH}/tools/tikcpp/ascendc_kernel/include")
elseif(EXISTS "${ASCEND_HOME_PATH}/compiler/tikcpp/ascendc_kernel/include")
    set(PTO_ASCENDC_KERNEL_INCLUDE "${ASCEND_HOME_PATH}/compiler/tikcpp/ascendc_kernel/include")
endif()

add_library(tget_bandwidth_kernel SHARED tget_bandwidth_kernel.cpp)
target_compile_options(tget_bandwidth_kernel PRIVATE
    ${CMAKE_CCE_COMPILE_OPTIONS} --cce-aicore-arch=dav-c220-vec -DMEMORY_BASE -std=c++17
    -Wno-non-c-typedef-for-linkage -Wno-braced-scalar-init
)
target_include_directories(tget_bandwidth_kernel PRIVATE
    ${PTO_INCLUDE_DIR}
    ${COMM_TESTCASE_DIR}
    ${ASCEND_HOME_PATH}/pkg_inc/
    ${ASCEND_HOME_PATH}/pkg_inc/profiling/
    ${ASCEND_HOME_PATH}/pkg_inc/runtime/runtime
    ${PTO_TOOLKIT_DIR}/asc
    ${PTO_TOOLKIT_DIR}/asc/include
    ${PTO_TOOLKIT_DIR}/ascendc/include
    ${PTO_TOOLKIT_DIR}/ascendc/include/basic_api
    ${PTO_TOOLKIT_DIR}/include/ascendc
    ${PTO_TOOLKIT_DIR}/include/ascendc/basic_api
    ${PTO_TOOLKIT_DIR}/asc/impl/basic_api
    ${PTO_TOOLKIT_DIR}/ascendc/include/basic_api/impl
    ${PTO_TOOLKIT_DIR}/asc/include/interface
    ${PTO_TOOLKIT_DIR}/asc/include/utils
    ${PTO_ASCENDC_KERNEL_INCLUDE}
)
target_link_directories(tget_bandwidth_kernel PRIVATE
    ${ASCEND_HOME_PATH}/lib64
)
target_link_libraries(tget_bandwidth_kernel PRIVATE
    hccl hcomm runtime nnopbase
)
target_link_options(tget_bandwidth_kernel PRIVATE --cce-fatobj-link)

add_executable(tget_bandwidth main.cpp)
target_compile_options(tget_bandwidth PRIVATE
    ${CMAKE_CPP_COMPILE_OPTIONS}
    -Wno-non-c-typedef-for-linkage -Wno-braced-scalar-init
)
target_include_directories(tget_bandwidth PRIVATE
    ${PTO_INCLUDE_DIR}
    ${COMM_TESTCASE_DIR}
    ${PROJECT_SOURCE_DIR}/../../../../tests/common
)

target_link_directories(tget_bandwidth PUBLIC
    ${ASCEND_HOME_PATH}/lib64
    ${ASCEND_HOME_PATH}/simulator/${SOC_VERSION}/lib
    ${ASCEND_HOME_PATH}/tools/simulator/${SOC_VERSION}/lib
)

target_link_libraries(tget_bandwidth PRIVATE
    tget_bandwidth_kernel
    $<BUILD_INTERFACE:$<$<STREQUAL:${RUN_MODE},sim>:runtime_camodel>>
    $<BUILD_INTERFACE:$<$<STREQUAL:${RUN_MODE},npu>:runtime>>
    stdc++ ascendcl hcomm m tiling_api platform c_sec dl nnopbase pthread
)
