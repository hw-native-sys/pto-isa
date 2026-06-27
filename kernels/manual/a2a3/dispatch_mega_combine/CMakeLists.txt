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

set(CMAKE_COMPILER "bisheng" CACHE STRING "Compiler used for host and CCE sources")
if(NOT DEFINED CMAKE_C_COMPILER)
    set(CMAKE_C_COMPILER "${CMAKE_COMPILER}")
endif()
if(NOT DEFINED CMAKE_CXX_COMPILER)
    set(CMAKE_CXX_COMPILER "${CMAKE_COMPILER}")
endif()

project(pto_example LANGUAGES C CXX)

# bisheng/cce-ld does not support the linker depfile option that CMake 3.27+ may enable.
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

if(DEBUG_MODE)
    message(STATUS "Debug Mode Enabled, Add Debug Options")
    add_compile_definitions(_DEBUG)
    set(CMAKE_CCE_COMPILE_OPTIONS "${CMAKE_CCE_COMPILE_OPTIONS} --cce-enable-print")
endif()

set(A2A3_SOC_VERSIONS
    Ascend910B1
    Ascend910B
    Ascend910C
    Ascend910_93
    Ascend910_9391
    Ascend910_9381
    Ascend910_9372
    Ascend910_9392
    Ascend910_9382
    Ascend910_9362
    ascend910_93
    ascend910_9391
    ascend910_9381
    ascend910_9372
    ascend910_9392
    ascend910_9382
    ascend910_9362
)

if("${SOC_VERSION}" IN_LIST A2A3_SOC_VERSIONS)
    set(CCE_AICORE_ARCH "dav-c220")
elseif("${SOC_VERSION}" STREQUAL "Ascend910_9599")
    set(CCE_AICORE_ARCH "dav-c310")
else()
    message(FATAL_ERROR "Unsupported SOC_VERSION: ${SOC_VERSION}")
endif()

set(CMAKE_CPP_COMPILE_OPTIONS
    -xc++
    "SHELL:-include stdint.h"
    "SHELL:-include stddef.h"
)

get_filename_component(PTO_REPO_INCLUDE_DIR "${PROJECT_SOURCE_DIR}/../../../../include" ABSOLUTE)

include_directories(
    ${PTO_REPO_INCLUDE_DIR}
    ${ASCEND_HOME_PATH}/include
    ${ASCEND_HOME_PATH}/pkg_inc
    ${ASCEND_HOME_PATH}/pkg_inc/runtime
    ${ASCEND_HOME_PATH}/pkg_inc/profiling
    ${ASCEND_DRIVER_PATH}/kernel/inc
    ${ASCEND_HOME_PATH}/include/experiment/runtime
    ${ASCEND_HOME_PATH}/include/experiment/msprof    
)

function(pto_example_mixed NAME)
    add_library(${NAME}_kernel SHARED kernel_launch.cpp)
    target_compile_options(${NAME}_kernel PRIVATE ${CMAKE_CCE_COMPILE_OPTIONS} --cce-aicore-arch=${CCE_AICORE_ARCH} -std=c++17)
    target_include_directories(${NAME}_kernel BEFORE PRIVATE
        ${PTO_REPO_INCLUDE_DIR}
        ${PROJECT_SOURCE_DIR}
        ${PROJECT_SOURCE_DIR}/op_kernel
        ${PROJECT_SOURCE_DIR}/op_kernel/utils
        ${PROJECT_BINARY_DIR}
        ${ASCEND_HOME_PATH}/aarch64-linux/pkg_inc/runtime/runtime
        ${ASCEND_HOME_PATH}/aarch64-linux/include/ascendc/basic_api
        ${ASCEND_HOME_PATH}/aarch64-linux/asc
        ${ASCEND_HOME_PATH}/aarch64-linux/asc/include/basic_api
        ${ASCEND_HOME_PATH}/aarch64-linux/asc/include/interface
        ${ASCEND_HOME_PATH}/aarch64-linux/ascendc/include/basic_api/impl
    )
    target_link_options(${NAME}_kernel PRIVATE --cce-fatobj-link)

    add_executable(${NAME}
        main.cpp
        runtime_context.cpp
        tiling_builder.cpp
        data_utils.cpp
    )
    target_compile_options(${NAME} PRIVATE ${CMAKE_CPP_COMPILE_OPTIONS})
    target_include_directories(${NAME} BEFORE PRIVATE
        ${PTO_REPO_INCLUDE_DIR}
        ${PROJECT_SOURCE_DIR}
        ${PROJECT_SOURCE_DIR}/op_kernel
        ${PROJECT_SOURCE_DIR}/op_kernel/utils
        ${PROJECT_BINARY_DIR}
    )

    target_link_directories(${NAME} PUBLIC
        ${ASCEND_HOME_PATH}/lib64
    )

    target_link_libraries(${NAME} PRIVATE
        ${NAME}_kernel
        stdc++ ascendcl hcomm runtime m tiling_api platform c_sec dl nnopbase pthread
    )
endfunction()

pto_example_mixed(dispatch_mega_combine)
