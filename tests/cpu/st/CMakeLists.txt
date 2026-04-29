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
project(pto_test_cpu_sim)

option(PTO_CPU_SIM_ENABLE_BF16 "Enable CPU-SIM BF16 coverage. Requires C++23 std::bfloat16_t support." OFF)
option(PTO_CPU_SIM_PREFER_FETCH_GTEST "Build googletest from source instead of using a preinstalled package." OFF)

if (PTO_CPU_SIM_ENABLE_BF16)
    set(CMAKE_CXX_STANDARD 23)
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 14.0)
    set(CMAKE_CXX_STANDARD 23)
else()
    set(CMAKE_CXX_STANDARD 20)
endif()

set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

add_definitions(-D__CPU_SIM)
if (PTO_CPU_SIM_ENABLE_BF16)
    add_compile_definitions(PTO_CPU_SIM_ENABLE_BF16=1)
    add_compile_definitions(CPU_SIM_BFLOAT_ENABLED=1)
    include(CheckCXXSourceCompiles)
    set(CMAKE_REQUIRED_FLAGS "-std=c++23")
    check_cxx_source_compiles(
        "
        #include <stdfloat>
        int main() {
            std::bfloat16_t x = std::bfloat16_t(1.0f);
            static_assert(sizeof(std::bfloat16_t) == 2);
            return static_cast<float>(x) > 0.0f ? 0 : 1;
        }
        "
        PTO_CPU_SIM_HAS_STDBFLOAT16
    )
    unset(CMAKE_REQUIRED_FLAGS)
    if (NOT PTO_CPU_SIM_HAS_STDBFLOAT16)
        message(FATAL_ERROR
            "PTO_CPU_SIM_ENABLE_BF16=ON requires a C++23 compiler and standard library with <stdfloat> "
            "and std::bfloat16_t support."
        )
    endif()
endif()

# GTest is required for CPU STs. Prefer a system installation, but fall back to
# fetching googletest to make macOS builds self-contained.
set(PTO_CPU_SIM_GTEST_TARGET "")
set(PTO_CPU_SIM_GTEST_MAIN_TARGET "")

if (NOT PTO_CPU_SIM_PREFER_FETCH_GTEST)
    find_package(GTest QUIET)
    if (GTest_FOUND AND APPLE AND CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND TARGET GTest::gtest)
        get_target_property(_pto_gtest_imported GTest::gtest IMPORTED)
        get_target_property(_pto_gtest_loc GTest::gtest IMPORTED_LOCATION)
        if (NOT _pto_gtest_imported)
            unset(_pto_gtest_loc)
        elseif (NOT _pto_gtest_loc)
            get_target_property(_pto_gtest_loc GTest::gtest LOCATION)
        endif()
        if (_pto_gtest_loc AND EXISTS "${_pto_gtest_loc}")
            execute_process(
                COMMAND bash -lc "nm -C \"${_pto_gtest_loc}\" 2>/dev/null | grep -q 'std::__1::'"
                RESULT_VARIABLE _pto_gtest_uses_libcpp
            )
            if (_pto_gtest_uses_libcpp EQUAL 0)
                message(STATUS "Installed GTest uses libc++; ignoring imported targets and fetching googletest for GNU/libstdc++ compatibility")
                set(GTest_FOUND OFF)
            endif()
        endif()
        unset(_pto_gtest_imported)
        unset(_pto_gtest_loc)
        unset(_pto_gtest_uses_libcpp)
    endif()
    if (GTest_FOUND)
        set(PTO_CPU_SIM_GTEST_TARGET GTest::gtest)
        set(PTO_CPU_SIM_GTEST_MAIN_TARGET GTest::gtest_main)
    endif()
endif()
if (NOT PTO_CPU_SIM_GTEST_TARGET OR NOT PTO_CPU_SIM_GTEST_MAIN_TARGET)
    include(FetchContent)
    message(STATUS "GTest not found; fetching googletest via FetchContent")
    set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
    set(BUILD_GMOCK OFF CACHE BOOL "" FORCE)
    FetchContent_Declare(
        googletest
        URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip
    )
    FetchContent_MakeAvailable(googletest)
    set(PTO_CPU_SIM_GTEST_TARGET gtest)
    set(PTO_CPU_SIM_GTEST_MAIN_TARGET gtest_main)
endif()

set(PTO_GLIBCXX_USE_CXX11_ABI "auto" CACHE STRING "Force _GLIBCXX_USE_CXX11_ABI (auto/0/1) for linking with system libs such as GTest.")
set_property(CACHE PTO_GLIBCXX_USE_CXX11_ABI PROPERTY STRINGS auto 0 1)
if (PTO_GLIBCXX_USE_CXX11_ABI STREQUAL "auto")
    if (UNIX AND NOT APPLE AND PTO_CPU_SIM_GTEST_TARGET AND TARGET ${PTO_CPU_SIM_GTEST_TARGET} AND (CMAKE_NM OR EXISTS "/usr/bin/nm"))
        get_target_property(_pto_gtest_imported ${PTO_CPU_SIM_GTEST_TARGET} IMPORTED)
        get_target_property(_pto_gtest_loc ${PTO_CPU_SIM_GTEST_TARGET} IMPORTED_LOCATION)
        if (NOT _pto_gtest_imported)
            unset(_pto_gtest_loc)
        elseif (NOT _pto_gtest_loc)
            get_target_property(_pto_gtest_loc ${PTO_CPU_SIM_GTEST_TARGET} LOCATION)
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
        unset(_pto_gtest_imported)
        unset(_pto_gtest_loc)
        unset(_pto_gtest_is_abi1)
    endif()
endif()
if (PTO_GLIBCXX_USE_CXX11_ABI STREQUAL "0")
    add_compile_definitions(_GLIBCXX_USE_CXX11_ABI=0)
endif()

if (NOT PTO_CPU_SIM_GTEST_TARGET OR NOT PTO_CPU_SIM_GTEST_MAIN_TARGET)
    message(FATAL_ERROR "Failed to resolve a usable GTest target for CPU ST builds.")
endif()

message("-- COMPILER: ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}; Using standard C++${CMAKE_CXX_STANDARD}; Force BF16 CPU-SIM: ${PTO_CPU_SIM_ENABLE_BF16}")

add_compile_options(
    -D_FORTIFY_SOURCE=2
    -std=c++${CMAKE_CXX_STANDARD}
    -Wno-macro-redefined -Wno-ignored-attributes
    -fstack-protector-strong
)

if (CMAKE_BUILD_TYPE STREQUAL "Release")
    add_compile_options(-O2)
endif()
if (CMAKE_BUILD_TYPE STREQUAL "MinSizeRel")
    add_compile_options(-O2)
endif()

# add_compile_definitions(_GLIBCXX_USE_CXX11_ABI=0)
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

add_subdirectory(testcase)
