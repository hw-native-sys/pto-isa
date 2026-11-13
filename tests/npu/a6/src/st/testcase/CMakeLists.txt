# --------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# --------------------------------------------------------------------------------

# A6 (dav-9201) ST build helpers.
#
# These mirror the A5 pto_*_st helpers but target the A6 core archs:
#   vec  -> dav-920r1-vec  (--npu-arch=dav-9201)
#   cube -> dav-920r1-cube
# TLOAD / TSTORE do not yet lower correctly for A6, so vec-core kernels in this
# tree call the A6 low-level DMA intrinsics (copy_gm_to_ubuf_align_v2 /
# copy_ubuf_to_gm_align_v2) directly instead of the TLoad/TStore wrappers.

# Shared testcases are identical to their a5 twins and have NO directory in
# this tree: their targets compile the a5 sources directly for the a6 core
# (PTO_NPU_ARCH_A6 comes from pto-inst.hpp as usual). A local testcase dir
# always wins — put genuinely-a6 implementations (tquant, tshl family, ...)
# in their own directory as before.
# Root of the a5 testcase tree — resolved once here so pto_st_sources works
# from both this directory and per-testcase subdirectories.
set(A5_TESTCASE_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../../../../a5/src/st/testcase)

macro(pto_st_sources NAME)
    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${NAME}_kernel.cpp)
        set(PTO_KSRC ${NAME}_kernel.cpp)
        set(PTO_MSRC main.cpp)
    else()
        set(PTO_A5_TC ${A5_TESTCASE_ROOT}/${NAME})
        if(NOT EXISTS ${PTO_A5_TC}/${NAME}_kernel.cpp)
            message(FATAL_ERROR "testcase ${NAME}: no local dir and no a5 twin under ${PTO_A5_TC}")
        endif()
        set(PTO_KSRC ${PTO_A5_TC}/${NAME}_kernel.cpp)
        set(PTO_MSRC ${PTO_A5_TC}/main.cpp)
    endif()
endmacro()

function(pto_vec_st NAME)
    pto_st_sources(${NAME})
    add_library(${NAME}_kernel SHARED ${PTO_KSRC})
    target_compile_options(${NAME}_kernel PRIVATE
        ${CMAKE_CCE_COMPILE_OPTIONS}
        --cce-aicore-arch=dav-920r1-vec
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
        --cce-aicore-arch=dav-920r1-cube
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
        --cce-aicore-arch=dav-920r1
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

# testcase list — add new A6 cases here. tquant is the BF16 -> HiF4 boilerplate
# harness using the raw A6 DMA intrinsics (copy_gm_to_ubuf_align_v2 /
# copy_ubuf_to_gm_align_v2) directly, since PTO's TLOAD/TSTORE wrappers do not
# lower correctly for dav-920r1-vec yet (see arch_cce_intrinsic.hpp:114-123).
set(ALL_TESTCASES
    mgather_gm2l1
    mscatter
    textract_ndto2xnz
    tquant
    tload_hif4
    tmatmul_mx
    tshl
    tshls
    tshr
    tshrs
    tinterleave
    tdeinterleave
    tcvt_e8m0
    syncall
    tmov_nd2nz
    tmov_zz
    tquant_dn
    tload_large_stride
    tpushpop_cv_nosplit
    tpushpop_vc_nosplit
    tpushpop_subtile
    tmins
    tmaxs
    txors
    tors
    tmuls
    tsubs
    tands
    tadds
    tadd
    tand
    tassign
    taxpy
    tci
    tcmp
    tcmps
    tcolexpand
    tcolexpandadd
    tcolexpanddiv
    tcolexpandexpdif
    tcolexpandmax
    tcolexpandmin
    tcolexpandmul
    tcolexpandsub
    tcolmax
    tcolmin
    tcolprod
    tcolsum
    tconcat
    tdequant
    tdiv
    tdivs
    texpands
    tfmod
    tfmods
    tgather
    tgatherb
    thistogram
    tlrelu
    tmadd
    tmatmul
    tmax
    tmin
    tmov
    tmrgsort
    tmul
    tmula
    tor
    tpartargmax
    tpartargmin
    tpartmin
    tpartmul
    tpow
    tprelu
    trandom
    trem
    trems
    trsqrt
    tscatter
    tsel
    tsels
    tstore
    tsub
    ttrans
    ttri
    txor
)

if (AUTO_MODE)
  set(AUTO_MODE_WHITELIST
    tquant
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

# Shared cases that need the cube core (their a5 sources use cube-pipe
# intrinsics): everything else shared defaults to vec.
set(A6_SHARED_CUBE_CASES tmov tmatmul)

foreach(TESTCASE ${ALL_TESTCASES})
    if((DEFINED TEST_CASE AND TEST_CASE STREQUAL TESTCASE) OR (NOT DEFINED TEST_CASE) OR (TEST_CASE STREQUAL "all"))
        if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${TESTCASE}/CMakeLists.txt)
            add_subdirectory(${TESTCASE})
        elseif(${TESTCASE} IN_LIST A6_SHARED_CUBE_CASES)
            pto_cube_st(${TESTCASE})
        else()
            pto_vec_st(${TESTCASE})
        endif()
    endif()
endforeach()
