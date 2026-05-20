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
project(pto_example)

option(PTO_CPU_SIM_STANDALONE "Build a local CPU-sim runner for this kernel example." OFF)
option(PTO_CPU_SIM_TRACE_MODE "Enable PTO instruction tracing in local CPU-sim runner." OFF)

if (PTO_CPU_SIM_STANDALONE)
    set(CMAKE_CXX_STANDARD 20)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
    set(CMAKE_POSITION_INDEPENDENT_CODE ON)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

    add_compile_definitions(__CPU_SIM)
    add_compile_definitions(__DAV_C220_CUBE__)
    add_compile_definitions(__DAV_CUBE__)
    add_compile_definitions(__DAV_VEC__)
    add_compile_definitions(PTO_CPU_SIM_RUN_ORIGINAL_FLASH=1)
    if (PTO_CPU_SIM_TRACE_MODE)
        add_compile_definitions(PTO_CPU_SIM_TRACE_MODE=1)
    endif()

    find_package(Threads REQUIRED)
    find_package(Python3 REQUIRED COMPONENTS Interpreter)

    set(GENERATED_CASES_H ${CMAKE_BINARY_DIR}/generated_cases.h)
    set(GENERATED_CASES_JSON ${CMAKE_BINARY_DIR}/generated_cases.json)
    add_custom_command(
        OUTPUT ${GENERATED_CASES_H} ${GENERATED_CASES_JSON}
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/scripts/generate_cases.py
                --output-header ${GENERATED_CASES_H}
                --output-json ${GENERATED_CASES_JSON}
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/scripts/generate_cases.py
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        VERBATIM
    )
    add_custom_target(fa_generated_cases DEPENDS ${GENERATED_CASES_H} ${GENERATED_CASES_JSON})

    add_executable(fa_performance_cpu_sim main.cpp fa_performance_kernel.cpp ${GENERATED_CASES_H})
    add_dependencies(fa_performance_cpu_sim fa_generated_cases)
    target_compile_options(fa_performance_cpu_sim PRIVATE
        -D_FORTIFY_SOURCE=2
        -Wno-macro-redefined -Wno-ignored-attributes
        -fstack-protector-strong
    )
    target_include_directories(fa_performance_cpu_sim PRIVATE
        ${PROJECT_SOURCE_DIR}/../../../../include/
        ${PROJECT_SOURCE_DIR}/../../../../tests/common
        ${CMAKE_BINARY_DIR}
    )
    target_link_libraries(fa_performance_cpu_sim PRIVATE Threads::Threads)
    if (UNIX)
        target_link_libraries(fa_performance_cpu_sim PRIVATE m)
    endif()
    if (UNIX AND NOT APPLE)
        target_link_libraries(fa_performance_cpu_sim PRIVATE dl)
    endif()
    return()
endif()

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

if(DEBUG_MODE)
    message(STATUS "Debug Mode Enabled, Add Debug Options")
    add_compile_definitions(_DEBUG)
    set(CMAKE_CCE_COMPILE_OPTIONS "${CMAKE_CCE_COMPILE_OPTIONS} --cce-enable-print")
endif()

if(${SOC_VERSION} STREQUAL "Ascend910B1")
    set(CCE_AICORE_ARCH "dav-c220")
elseif(${SOC_VERSION} STREQUAL "Ascend910_9599")
    set(CCE_AICORE_ARCH "dav-c310")
else()
    message(FATAL_ERROR "Unsupported SOC_VERSION: ${SOC_VERSION}")
endif()

set(CMAKE_CPP_COMPILE_OPTIONS
    -xc++
    "SHELL:-include stdint.h"
    "SHELL:-include stddef.h"
)

include_directories(
    ${PROJECT_SOURCE_DIR}/../../../../include/
    ${ASCEND_HOME_PATH}/include
    ${ASCEND_HOME_PATH}/pkg_inc
    ${ASCEND_HOME_PATH}/pkg_inc/runtime
    ${ASCEND_HOME_PATH}/pkg_inc/profiling
    ${ASCEND_DRIVER_PATH}/kernel/inc
    ${ASCEND_HOME_PATH}/include/experiment/runtime
    ${ASCEND_HOME_PATH}/include/experiment/msprof
)

function(pto_example_mixed NAME)
    add_library(${NAME}_kernel SHARED ${NAME}_kernel.cpp)
    target_compile_options(${NAME}_kernel PRIVATE ${CMAKE_CCE_COMPILE_OPTIONS} --cce-aicore-arch=${CCE_AICORE_ARCH} -std=c++17)
    target_include_directories(${NAME}_kernel PRIVATE
        ${PROJECT_BINARY_DIR}
    )
    target_link_options(${NAME}_kernel PRIVATE --cce-fatobj-link)

    add_executable(${NAME} main.cpp)
    target_compile_options(${NAME} PRIVATE ${CMAKE_CPP_COMPILE_OPTIONS})
    target_include_directories(${NAME} PRIVATE
        ${PROJECT_SOURCE_DIR}/../../../../tests/common
        ${PROJECT_BINARY_DIR}
    )

    target_link_directories(${NAME} PUBLIC
        ${ASCEND_HOME_PATH}/lib64
        ${ASCEND_HOME_PATH}/simulator/${SOC_VERSION}/lib
        ${ASCEND_HOME_PATH}/tools/simulator/${SOC_VERSION}/lib
    )

    target_link_libraries(${NAME} PRIVATE
        ${NAME}_kernel
        $<BUILD_INTERFACE:$<$<STREQUAL:${RUN_MODE},sim>:runtime_camodel>>
        $<BUILD_INTERFACE:$<$<STREQUAL:${RUN_MODE},npu>:runtime>>
        stdc++ ascendcl m tiling_api platform c_sec dl nnopbase pthread
    )
endfunction()

pto_example_mixed(fa_performance)
