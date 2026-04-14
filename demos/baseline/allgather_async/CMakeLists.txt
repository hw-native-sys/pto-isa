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

set(CMAKE_COMPILER bisheng)
set(CMAKE_C_COMPILER ${CMAKE_COMPILER})
set(CMAKE_CXX_COMPILER ${CMAKE_COMPILER})

project(allgather_async_demo)

set(CMAKE_LINK_DEPENDS_USE_LINKER FALSE)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

if(NOT DEFINED ENV{ASCEND_HOME_PATH})
    message(FATAL_ERROR "ASCEND_HOME_PATH not set. Please source set_env.sh first.")
else()
    set(ASCEND_HOME_PATH $ENV{ASCEND_HOME_PATH})
endif()

set(ASCEND_DRIVER_PATH /usr/local/Ascend/driver)

get_filename_component(PTO_LIB_PATH "${CMAKE_CURRENT_LIST_DIR}/../../.." ABSOLUTE)
message(STATUS "PTO_LIB_PATH = ${PTO_LIB_PATH}")

set(COMM_COMMON_DIR "${PTO_LIB_PATH}/tests/npu/a2a3/comm/st/testcase")
message(STATUS "COMM_COMMON_DIR = ${COMM_COMMON_DIR}")

set(SOC_VERSION "ascend910b1" CACHE STRING "SoC version (ascend910b1 or Ascend910_9599)")

add_compile_options(
    -D_FORTIFY_SOURCE=2
    -O2 -std=c++17
    -Wno-macro-redefined -Wno-ignored-attributes
    -fstack-protector-strong
)
add_link_options(-s -Wl,-z,relro -Wl,-z,now)

if(SOC_VERSION STREQUAL "Ascend910_9599")
    set(CCE_AICORE_ARCH "dav-c310-vec")
    set(CCE_MEMORY_MODEL "REGISTER_BASE")
else()
    set(CCE_AICORE_ARCH "dav-c220-vec")
    set(CCE_MEMORY_MODEL "MEMORY_BASE")
endif()

set(CCE_OPTS
    -xcce
    -Xhost-start -Xhost-end
    "SHELL:-mllvm -cce-aicore-stack-size=0x8000"
    "SHELL:-mllvm -cce-aicore-function-stack-size=0x8000"
    "SHELL:-mllvm -cce-aicore-record-overflow=true"
    "SHELL:-mllvm -cce-aicore-addr-transform"
    "SHELL:-mllvm -cce-aicore-dcci-insert-for-scalar=false"
)

set(CPP_OPTS
    -xc++
    "SHELL:-include stdint.h"
    "SHELL:-include stddef.h"
)

include_directories(
    ${PTO_LIB_PATH}/include
    ${PTO_LIB_PATH}
    ${ASCEND_HOME_PATH}/include
    ${ASCEND_DRIVER_PATH}/kernel/inc
    ${ASCEND_HOME_PATH}/include/hccl
)

set(TOOLKIT_DIR "${ASCEND_HOME_PATH}/x86_64-linux")
if(NOT EXISTS "${TOOLKIT_DIR}")
    set(TOOLKIT_DIR "${ASCEND_HOME_PATH}/aarch64-linux")
endif()

set(ASCENDC_KERNEL_INCLUDE "")
if(EXISTS "${ASCEND_HOME_PATH}/tools/tikcpp/ascendc_kernel/include")
    set(ASCENDC_KERNEL_INCLUDE "${ASCEND_HOME_PATH}/tools/tikcpp/ascendc_kernel/include")
elseif(EXISTS "${ASCEND_HOME_PATH}/compiler/tikcpp/ascendc_kernel/include")
    set(ASCENDC_KERNEL_INCLUDE "${ASCEND_HOME_PATH}/compiler/tikcpp/ascendc_kernel/include")
endif()

# === Kernel shared library ===
add_library(allgather_kernel SHARED csrc/kernel/allgather_kernel.cpp)

target_compile_options(allgather_kernel PRIVATE
    ${CCE_OPTS}
    --cce-aicore-arch=${CCE_AICORE_ARCH}
    -D${CCE_MEMORY_MODEL}
    -std=c++17
    -Wno-non-c-typedef-for-linkage
    -Wno-braced-scalar-init
)

target_include_directories(allgather_kernel PRIVATE
    ${ASCEND_HOME_PATH}/pkg_inc/
    ${ASCEND_HOME_PATH}/pkg_inc/profiling/
    ${ASCEND_HOME_PATH}/pkg_inc/runtime/runtime
    ${PTO_LIB_PATH}/include
    ${COMM_COMMON_DIR}
    ${TOOLKIT_DIR}/asc
    ${TOOLKIT_DIR}/asc/include
    ${TOOLKIT_DIR}/ascendc/include
    ${TOOLKIT_DIR}/ascendc/include/basic_api
    ${TOOLKIT_DIR}/include/ascendc
    ${TOOLKIT_DIR}/include/ascendc/basic_api
    ${TOOLKIT_DIR}/asc/impl/basic_api
    ${TOOLKIT_DIR}/ascendc/include/basic_api/impl
    ${TOOLKIT_DIR}/asc/include/interface
    ${TOOLKIT_DIR}/asc/include/utils
    ${ASCENDC_KERNEL_INCLUDE}
)

target_link_directories(allgather_kernel PRIVATE ${ASCEND_HOME_PATH}/lib64)
target_link_libraries(allgather_kernel PRIVATE hcomm runtime nnopbase)
target_link_options(allgather_kernel PRIVATE --cce-fatobj-link)

# === Host executable ===
add_executable(allgather_demo csrc/host/main.cpp)

target_compile_options(allgather_demo PRIVATE
    ${CPP_OPTS}
    -Wno-non-c-typedef-for-linkage
    -Wno-braced-scalar-init
)

target_include_directories(allgather_demo PRIVATE
    ${PTO_LIB_PATH}/include
    ${COMM_COMMON_DIR}
)

target_link_directories(allgather_demo PUBLIC ${ASCEND_HOME_PATH}/lib64)

target_link_libraries(allgather_demo PRIVATE
    allgather_kernel
    runtime
    stdc++ ascendcl hcomm m tiling_api platform c_sec dl nnopbase pthread
)
