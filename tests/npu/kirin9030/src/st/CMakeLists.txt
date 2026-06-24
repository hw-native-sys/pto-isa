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
get_target_property(GTEST_INCLUDE GTest::gtest INTERFACE_INCLUDE_DIRECTORIES)

set(PTO_GLIBCXX_USE_CXX11_ABI "auto" CACHE STRING "Force _GLIBCXX_USE_CXX11_ABI (auto/0/1) for linking with system libs such as GTest.")
set_property(CACHE PTO_GLIBCXX_USE_CXX11_ABI PROPERTY STRINGS auto 0 1)
if (PTO_GLIBCXX_USE_CXX11_ABI STREQUAL "auto")
    if (UNIX AND NOT APPLE AND TARGET GTest::gtest AND (CMAKE_NM OR EXISTS "/usr/bin/nm"))
        get_target_property(_pto_gtest_loc GTest::gtest IMPORTED_LOCATION)
        if (NOT _pto_gtest_loc)
            get_target_property(_pto_gtest_loc GTest::gtest LOCATION)
        endif()
        if (_pto_gtest_loc AND EXISTS "${_pto_gtest_loc}")
            execute_process(
                COMMAND bash -lc "nm -C \"${_pto_gtest_loc}\" 2>/dev/null | grep -q 'GetBoolAssertionFailureMessage\\[abi:cxx11\\]'"
                RESULT_VARIABLE _pto_gtest_is_abi1
            )
            if (NOT _pto_gtest_is_abi1 EQUAL 0)
                set(PTO_GLIBCXX_USE_CXX11_ABI "0" CACHE STRING "" FORCE)
            endif()
        endif()
        unset(_pto_gtest_loc)
        unset(_pto_gtest_is_abi1)
    endif()
endif()
if (PTO_GLIBCXX_USE_CXX11_ABI STREQUAL "0")
    add_compile_definitions(_GLIBCXX_USE_CXX11_ABI=0)
endif()

add_compile_options(
    -D_FORTIFY_SOURCE=2
    -O2 -std=c++17
    -Wno-macro-redefined -Wno-ignored-attributes -Wno-unknown-attributes
    -fstack-protector-strong
)
add_link_options(
    -s
    -Wl,-z,relro
    -Wl,-z,now
)
if (SOC_VERSION STREQUAL "Kirin9030")
    set(CCE_AICORE_ARCH dav-l311)
endif()

if(RUN_MODE STREQUAL "npu")
    if(NOT DEFINED ENV{NATIVE})
        message(FATAL_ERROR "Cannot find NATIVE, please set the Environment Variable.")
    else()
        set(NATIVE $ENV{NATIVE})
    endif()

    if (NOT DEFINED ENV{ARM_GTEST})
        message(FATAL_ERROR "Cannot find ARM_GTEST, please set the Environment Variable.")
    else()
        set(ARM_GTEST $ENV{ARM_GTEST})
    endif()
    set(CMAKE_CCE_COMPILE_OPTIONS
        -xcce
        -cce-kernel-launch=aclrtLaunch
        --target=aarch64-linux-ohos
        --sysroot=${NATIVE}/sysroot
        -I${NATIVE}/sysroot/usr/include/aarch64-linux-ohos
        -I${NATIVE}/llvm/include/libcxx-ohos/include/c++/v1
        "SHELL:-mllvm -cce-aicore-function-stack-size=0x4000"
        "SHELL:-mllvm -cce-aicore-addr-transform"
    )
    set(CMAKE_CCE_COMPILE_OPTIONS "${CMAKE_CCE_COMPILE_OPTIONS} --cce-pto-enable")

    include_directories(
        ${PROJECT_SOURCE_DIR}/../../../../../include 
        ${ASCEND_HOME_PATH}/ohos
        ${ASCEND_DRIVER_PATH}/kernel/inc
    )
else()
    set(CMAKE_CCE_COMPILE_OPTIONS
    -xcce
    "SHELL:-mllvm -cce-aicore-function-stack-size=0x4000"
    "SHELL:-mllvm -cce-aicore-addr-transform"
    )

    include_directories(
        ${PROJECT_SOURCE_DIR}/../../../../../include 
        ${ASCEND_HOME_PATH}/include
        ${ASCEND_DRIVER_PATH}/kernel/inc
    )

endif()


if(DEBUG_MODE)
    message(STATUS "Debug Mode Enabled, Add Debug Options")
    add_compile_definitions(_DEBUG)
    set(CMAKE_CCE_COMPILE_OPTIONS "${CMAKE_CCE_COMPILE_OPTIONS} --cce-enable-print")
endif()

set(CMAKE_CPP_COMPILE_OPTIONS
    -xc++
    "SHELL:-include stdint.h"
    "SHELL:-include stddef.h"
)


add_subdirectory(testcase)
