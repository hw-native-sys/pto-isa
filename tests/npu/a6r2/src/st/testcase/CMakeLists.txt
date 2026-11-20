# --------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# --------------------------------------------------------------------------------

# A6R2 (dav-9202) ST build helpers.
#
# Core archs:
#   vec  -> dav-920r2-vec  (__NPU_ARCH__=9202 -> PTO_NPU_ARCH_A6R2)
#   cube -> dav-920r2-cube
#
# Source resolution order for each testcase NAME:
#   1) local a6r2 testcase/<NAME>/
#   2) a6 twin under tests/npu/a6/.../testcase/<NAME>/
#   3) a5 twin under tests/npu/a5/.../testcase/<NAME>/
# A local a6r2 directory always wins (needed when R2 differs, e.g. kernel launch
# chevron <<<1, 0, stream>>> instead of <<<1, nullptr, stream>>>).

set(A6_TESTCASE_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../../../../a6/src/st/testcase)
set(A5_TESTCASE_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../../../../a5/src/st/testcase)

macro(pto_st_sources NAME)
    set(_a6r2_kernel ${CMAKE_CURRENT_SOURCE_DIR}/${NAME}_kernel.cpp)
    set(_a6r2_nested ${CMAKE_CURRENT_SOURCE_DIR}/${NAME}/${NAME}_kernel.cpp)
    set(_a6_kernel ${A6_TESTCASE_ROOT}/${NAME}/${NAME}_kernel.cpp)
    set(_a5_kernel ${A5_TESTCASE_ROOT}/${NAME}/${NAME}_kernel.cpp)

    if(EXISTS ${_a6r2_kernel})
        set(PTO_KSRC ${_a6r2_kernel})
        set(PTO_MSRC ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp)
        message(STATUS "${NAME}: a6r2 local sources -> ${PTO_KSRC}")
    elseif(EXISTS ${_a6r2_nested})
        set(PTO_KSRC ${_a6r2_nested})
        set(PTO_MSRC ${CMAKE_CURRENT_SOURCE_DIR}/${NAME}/main.cpp)
        message(STATUS "${NAME}: a6r2 local sources -> ${PTO_KSRC}")
    elseif(EXISTS ${_a6_kernel})
        set(PTO_KSRC ${_a6_kernel})
        set(PTO_MSRC ${A6_TESTCASE_ROOT}/${NAME}/main.cpp)
        message(STATUS "${NAME}: a6 twin sources -> ${PTO_KSRC}")
    elseif(EXISTS ${_a5_kernel})
        set(PTO_KSRC ${_a5_kernel})
        set(PTO_MSRC ${A5_TESTCASE_ROOT}/${NAME}/main.cpp)
        message(STATUS "${NAME}: a5 twin sources -> ${PTO_KSRC}")
    else()
        message(FATAL_ERROR "testcase ${NAME}: no a6r2/a6/a5 sources found")
    endif()
endmacro()

function(pto_vec_st NAME)
    pto_st_sources(${NAME})
    add_library(${NAME}_kernel SHARED ${PTO_KSRC})
    target_compile_options(${NAME}_kernel PRIVATE
        ${CMAKE_CCE_COMPILE_OPTIONS}
        --cce-aicore-arch=dav-920r2-vec
        -DREGISTER_BASE -std=c++17)

    target_include_directories(${NAME}_kernel PRIVATE
        ${ASCEND_HOME_PATH}/pkg_inc/
        ${ASCEND_HOME_PATH}/pkg_inc/profiling/
        ${ASCEND_HOME_PATH}/pkg_inc/runtime/runtime)
    target_link_options(${NAME}_kernel PRIVATE --cce-fatobj-link)

    add_executable(${NAME} ${PTO_MSRC})
    target_compile_options(${NAME} PRIVATE ${CMAKE_CPP_COMPILE_OPTIONS})
    target_include_directories(${NAME} PRIVATE
        ${PROJECT_SOURCE_DIR}/../../../../common)


    target_link_directories(${NAME} PUBLIC ${ASCEND_HOME_PATH}/lib64)
    foreach(simdir
            ${ASCEND_HOME_PATH}/tools/simulator/${SOC_VERSION}/lib
            ${ASCEND_HOME_PATH}/tools/simulator/${SOC_VERSION}/camodel)
        if(EXISTS ${simdir})
            target_link_directories(${NAME} PUBLIC ${simdir})
        endif()
    endforeach()

    target_link_libraries(${NAME} PRIVATE
        ${NAME}_kernel
        $<BUILD_INTERFACE:$<$<STREQUAL:${RUN_MODE},sim>:runtime_camodel>>
        $<BUILD_INTERFACE:$<$<STREQUAL:${RUN_MODE},npu>:runtime>>
        stdc++ ascendcl m tiling_api platform c_sec dl nnopbase GTest::gtest GTest::gtest_main pthread)
endfunction()

function(pto_cube_st NAME)
    pto_st_sources(${NAME})
    add_library(${NAME}_kernel SHARED ${PTO_KSRC})
    target_compile_options(${NAME}_kernel PRIVATE
        ${CMAKE_CCE_COMPILE_OPTIONS}
        --cce-aicore-arch=dav-920r2-cube
        -DREGISTER_BASE -std=c++17 -Wno-c++20-extensions)
    target_include_directories(${NAME}_kernel PRIVATE
        ${ASCEND_HOME_PATH}/pkg_inc/
        ${ASCEND_HOME_PATH}/pkg_inc/profiling/
        ${ASCEND_HOME_PATH}/pkg_inc/runtime/runtime)
    target_link_options(${NAME}_kernel PRIVATE --cce-fatobj-link)

    add_executable(${NAME} ${PTO_MSRC})
    target_compile_options(${NAME} PRIVATE ${CMAKE_CPP_COMPILE_OPTIONS})
    target_include_directories(${NAME} PRIVATE
        ${PROJECT_SOURCE_DIR}/../../../../common)


    target_link_directories(${NAME} PUBLIC ${ASCEND_HOME_PATH}/lib64)
    foreach(simdir
            ${ASCEND_HOME_PATH}/tools/simulator/${SOC_VERSION}/lib
            ${ASCEND_HOME_PATH}/tools/simulator/${SOC_VERSION}/camodel)
        if(EXISTS ${simdir})
            target_link_directories(${NAME} PUBLIC ${simdir})
        endif()
    endforeach()

    target_link_libraries(${NAME} PRIVATE
        ${NAME}_kernel
        $<BUILD_INTERFACE:$<$<STREQUAL:${RUN_MODE},sim>:runtime_camodel>>
        $<BUILD_INTERFACE:$<$<STREQUAL:${RUN_MODE},npu>:runtime>>
        stdc++ ascendcl m tiling_api platform c_sec dl nnopbase GTest::gtest GTest::gtest_main pthread)
endfunction()

function(pto_mix_st NAME)
    pto_st_sources(${NAME})
    add_library(${NAME}_kernel_mix SHARED ${PTO_KSRC})
    target_compile_options(${NAME}_kernel_mix PRIVATE
        ${CMAKE_CCE_COMPILE_OPTIONS}
        --cce-aicore-arch=dav-920r2
        -DREGISTER_BASE -std=c++17 -Wno-c++20-extensions)
    target_include_directories(${NAME}_kernel_mix PRIVATE
        ${ASCEND_HOME_PATH}/pkg_inc/
        ${ASCEND_HOME_PATH}/pkg_inc/profiling/
        ${ASCEND_HOME_PATH}/pkg_inc/runtime/runtime)
    target_link_options(${NAME}_kernel_mix PRIVATE --cce-fatobj-link)

    add_executable(${NAME} ${PTO_MSRC})
    target_compile_options(${NAME} PRIVATE ${CMAKE_CPP_COMPILE_OPTIONS})
    target_include_directories(${NAME} PRIVATE
        ${PROJECT_SOURCE_DIR}/../../../../common)


    target_link_directories(${NAME} PUBLIC ${ASCEND_HOME_PATH}/lib64)
    foreach(simdir
            ${ASCEND_HOME_PATH}/tools/simulator/${SOC_VERSION}/lib
            ${ASCEND_HOME_PATH}/tools/simulator/${SOC_VERSION}/camodel)
        if(EXISTS ${simdir})
            target_link_directories(${NAME} PUBLIC ${simdir})
        endif()
    endforeach()

    target_link_libraries(${NAME} PRIVATE
        ${NAME}_kernel_mix
        $<BUILD_INTERFACE:$<$<STREQUAL:${RUN_MODE},sim>:runtime_camodel>>
        $<BUILD_INTERFACE:$<$<STREQUAL:${RUN_MODE},npu>:runtime>>
        stdc++ ascendcl m tiling_api platform c_sec dl nnopbase GTest::gtest GTest::gtest_main pthread)
endfunction()

# A6R2 cases — start with TADD; add more as they are adopted from A6/A5.
set(ALL_TESTCASES
    tadd
    tpushpop_c2v_maxtrack
)

if (AUTO_MODE)
  set(AUTO_MODE_WHITELIST
  )
  foreach(whitelisted_case ${AUTO_MODE_WHITELIST})
    list(REMOVE_ITEM ALL_TESTCASES ${whitelisted_case})
  endforeach()
endif()

if((TEST_CASE IN_LIST ALL_TESTCASES) OR (TEST_CASE STREQUAL "all"))
    message(STATUS "run : ${TEST_CASE}")
else()
    message(FATAL_ERROR "not found TEST_CASE: ${TEST_CASE}, support case is : ${ALL_TESTCASES}")
endif()

set(A6R2_SHARED_CUBE_CASES tmov tmatmul)

foreach(TESTCASE ${ALL_TESTCASES})
    if((DEFINED TEST_CASE AND TEST_CASE STREQUAL TESTCASE) OR (NOT DEFINED TEST_CASE) OR (TEST_CASE STREQUAL "all"))
        if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${TESTCASE}/CMakeLists.txt)
            add_subdirectory(${TESTCASE})
        elseif(${TESTCASE} IN_LIST A6R2_SHARED_CUBE_CASES)
            pto_cube_st(${TESTCASE})
        else()
            pto_vec_st(${TESTCASE})
        endif()
    endif()
endforeach()
