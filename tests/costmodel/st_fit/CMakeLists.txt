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
project(pto_test_costmodel_fit_sim)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

add_definitions(-D__COSTMODEL -D__NPU_ARCH__=2201 -DPTO_COMM_NOT_SUPPORTED)

find_package(GTest QUIET)
if (NOT GTest_FOUND)
    include(FetchContent)
    message(STATUS "GTest not found; fetching googletest via FetchContent")
    set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
    set(BUILD_GMOCK OFF CACHE BOOL "" FORCE)
    FetchContent_Declare(
        googletest
        URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip
    )
    FetchContent_MakeAvailable(googletest)
    if (NOT TARGET GTest::gtest)
        add_library(GTest::gtest ALIAS gtest)
    endif()
    if (NOT TARGET GTest::gtest_main)
        add_library(GTest::gtest_main ALIAS gtest_main)
    endif()
endif()

add_compile_options(
    -D_FORTIFY_SOURCE=2
    -std=c++23
    -Wno-macro-redefined -Wno-ignored-attributes
    -fstack-protector-strong
)

if (CMAKE_BUILD_TYPE STREQUAL "Release")
    add_compile_options(-O2)
endif()
if (CMAKE_BUILD_TYPE STREQUAL "MinSizeRel")
    add_compile_options(-O2)
endif()

if (UNIX AND NOT APPLE)
    add_link_options(
        -Wl,-z,relro
        -Wl,-z,now
    )
endif()

set(CMAKE_CPP_COMPILE_OPTIONS
    -xc++
    "SHELL:-include stdint.h"
    "SHELL:-include stddef.h"
)

if (APPLE)
    execute_process(
        COMMAND xcrun --show-sdk-path
        OUTPUT_VARIABLE APPLE_SDK_PATH
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    if (APPLE_SDK_PATH)
        include_directories(SYSTEM "${APPLE_SDK_PATH}/usr/include/c++/v1")
    endif()

    if (GTest_FOUND)
        get_target_property(_gtest_lib GTest::gtest IMPORTED_LOCATION)
        if (NOT _gtest_lib)
            get_target_property(_gtest_lib GTest::gtest LOCATION)
        endif()
        if (_gtest_lib)
            get_filename_component(_gtest_lib_dir "${_gtest_lib}" DIRECTORY)
            list(APPEND CMAKE_BUILD_RPATH "${_gtest_lib_dir}")
            list(APPEND CMAKE_INSTALL_RPATH "${_gtest_lib_dir}")
        endif()
        unset(_gtest_lib)
        unset(_gtest_lib_dir)
    endif()
endif()

add_subdirectory(testcase)
