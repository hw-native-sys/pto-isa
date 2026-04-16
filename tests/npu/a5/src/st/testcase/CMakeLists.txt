# --------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# --------------------------------------------------------------------------------

function(pto_vec_st NAME)
    add_library(${NAME}_kernel SHARED ${NAME}_kernel.cpp)
    target_compile_options(${NAME}_kernel PRIVATE ${CMAKE_CCE_COMPILE_OPTIONS} --cce-aicore-arch=dav-c310-vec -DREGISTER_BASE -std=c++17)

    target_include_directories(${NAME}_kernel PRIVATE
        ${ASCEND_HOME_PATH}/pkg_inc/
        ${ASCEND_HOME_PATH}/pkg_inc/profiling/
        ${ASCEND_HOME_PATH}/pkg_inc/runtime/runtime
    )
    target_link_options(${NAME}_kernel PRIVATE --cce-fatobj-link)

    add_executable(${NAME} main.cpp)
    target_compile_options(${NAME} PRIVATE ${CMAKE_CPP_COMPILE_OPTIONS})
    target_include_directories(${NAME} PRIVATE
        ${PROJECT_SOURCE_DIR}/../../../../common
    )

    target_link_directories(${NAME} PUBLIC
        ${ASCEND_HOME_PATH}/lib64
        ${ASCEND_HOME_PATH}/tools/simulator/${SOC_VERSION}/lib
    )

    target_link_libraries(${NAME} PRIVATE
        ${NAME}_kernel
        $<BUILD_INTERFACE:$<$<STREQUAL:${RUN_MODE},sim>:runtime_camodel>>
        $<BUILD_INTERFACE:$<$<STREQUAL:${RUN_MODE},npu>:runtime>>
        stdc++ ascendcl m tiling_api platform c_sec dl nnopbase GTest::gtest GTest::gtest_main pthread
    )
endfunction()

function(pto_cube_st NAME)
    add_library(${NAME}_kernel SHARED ${NAME}_kernel.cpp)
    target_compile_options(${NAME}_kernel PRIVATE ${CMAKE_CCE_COMPILE_OPTIONS} --cce-aicore-arch=dav-c310-cube -DREGISTER_BASE -std=c++17)
    target_include_directories(${NAME}_kernel PRIVATE
        ${ASCEND_HOME_PATH}/pkg_inc/
        ${ASCEND_HOME_PATH}/pkg_inc/profiling/
        ${ASCEND_HOME_PATH}/pkg_inc/runtime/runtime
    )
    target_link_options(${NAME}_kernel PRIVATE --cce-fatobj-link)

    add_executable(${NAME} main.cpp)
    target_compile_options(${NAME} PRIVATE ${CMAKE_CPP_COMPILE_OPTIONS})
    target_include_directories(${NAME} PRIVATE
        ${PROJECT_SOURCE_DIR}/../../../../common
    )

    target_link_directories(${NAME} PUBLIC
        ${ASCEND_HOME_PATH}/lib64
        ${ASCEND_HOME_PATH}/tools/simulator/${SOC_VERSION}/lib
    )

    target_link_libraries(${NAME} PRIVATE
        ${NAME}_kernel
        $<BUILD_INTERFACE:$<$<STREQUAL:${RUN_MODE},sim>:runtime_camodel>>
        $<BUILD_INTERFACE:$<$<STREQUAL:${RUN_MODE},npu>:runtime>>
        stdc++ ascendcl m tiling_api platform c_sec dl nnopbase GTest::gtest GTest::gtest_main pthread
    )
endfunction()

function(pto_mix_st NAME)
    add_library(${NAME}_kernel_mix SHARED ${NAME}_kernel.cpp)
    target_compile_options(${NAME}_kernel_mix PRIVATE ${CMAKE_CCE_COMPILE_OPTIONS} --cce-aicore-arch=dav-c310 -DREGISTER_BASE -std=c++17)

    target_include_directories(${NAME}_kernel_mix PRIVATE
        ${ASCEND_HOME_PATH}/pkg_inc/
        ${ASCEND_HOME_PATH}/pkg_inc/profiling/
        ${ASCEND_HOME_PATH}/pkg_inc/runtime/runtime
    )
    target_link_options(${NAME}_kernel_mix PRIVATE --cce-fatobj-link)

    add_executable(${NAME} main.cpp)
    target_compile_options(${NAME} PRIVATE ${CMAKE_CPP_COMPILE_OPTIONS})
    target_include_directories(${NAME} PRIVATE
        ${PROJECT_SOURCE_DIR}/../../../../common
    )

    target_link_directories(${NAME} PUBLIC
        ${ASCEND_HOME_PATH}/lib64
        ${ASCEND_HOME_PATH}/tools/simulator/${SOC_VERSION}/lib
    )

    target_link_libraries(${NAME} PRIVATE
        ${NAME}_kernel_mix
        $<BUILD_INTERFACE:$<$<STREQUAL:${RUN_MODE},sim>:runtime_camodel>>
        $<BUILD_INTERFACE:$<$<STREQUAL:${RUN_MODE},npu>:runtime>>
        stdc++ ascendcl m tiling_api platform c_sec dl nnopbase GTest::gtest GTest::gtest_main pthread
    )
endfunction()

# 定义测试用例列表 testcase下如有新增用例目录需要在此加
set(ALL_TESTCASES
    tsubs
    tmaxs
    tci
    tmatmul
    tmatmul_mx
    tload_mix
    tload_shape2d
    tload
    tload_mx_ND_DN
    tload_mx_NZ
    tload_mx_gmtensor
    tmrgsort
    ttrans
    ttrans_conv
    tstore
    textract
    textract_compact
    trowsum
    trowprod
    trowmax
    trowargmax
    trowmin
    trowargmin
    tcolsum
    tcolprod
    tcolmax
    tcolargmax
    tcolmin
    tcolargmin
    tsort32
    trowexpand
    trowexpandadd
    trowexpanddiv
    trowexpandmax
    trowexpandmin
    trowexpandmul
    trowexpandsub
    trowexpandexpdif
    tcolgather
    tgather
    tcvt
    tsub
    tsubreluconv
    trem
    tfmod
    tprelu
    tand
    tor
    tands
    tors
    txor
    txors
    tshl
    tshls
    tshr
    tshrs
    tadd
    taddreluconv
    tadddeqrelu
    tpartadd
    tpartmul
    tfillpad
    tmin
    tmax
    tmins
    tmov
    tmov_mx
    tmul
    tlrelu
    tdiv
    tcmps
    tmov_acc2vec
    tsel
    tsels
    trsqrt
    tsqrt
    texp
    tabs
    tlog
    trecip
    tstore_acc2gm
    tdivs
    tmuls
    tadds
    taxpy
    tgatherb
    tpartmin
    tpartmax
    tpartargmin
    tpartargmax
    tmov_vect
    texpands
    tmov_acc2mat
    timg2col
    tassign
    tmov_ub2l1
    tcmp
    tadd_tdiv
    tmul_tadds
    tsub_texp
    tmuls_trowsum
    trowexpand_tsqrt
    trowexpand_trowsum
    trowsum_trowexpand
    trowexpand_tdiv
    trems
    tfmods
    tcolscatter
    tscatter
    tneg
    tcolexpandadd
    tcolexpandmax
    tcolexpandmin
    tcolexpanddiv
    tcolexpandmul
    tcolexpandsub
    tcolexpandexpdif
    trowexpandbrcb
    ttri
    tcolexpand
    tnot
    trelu
    tquant
    tdequant
    trandom
    mgather
    mgather_gm2l1
    mscatter
    t_dhrystone

    tconcatidx
    tconcat
    thistogram
    tinsert
    tinsert_acc2vec
    textract_nd_vec
    textract_vec
    tmov_zz
    tmov_nd2nz
    textract_acc2vec
    syncall
    tpushpop_cv
    tpushpop_cv_nosplit
    tpushpop_fixpipe
    tpushpop_vc
    tpushpop_vc_nosplit
    tpushpop_dir_both
    texpands_mat
    tpushpop_subtile
    tinterleave
    tdeinterleave
    tpairreducesum
    tfusedmuladd
    tmuladddst
    tsubrelu
    tfusedmuladdrelu
)

# Opt-in testcases: built ONLY when TEST_CASE=<name> is explicitly supplied.
# These are excluded from the default "build everything" path so they do not
# slow down CI. They are still valid TEST_CASE targets for local ST runs.
set(OPT_IN_TESTCASES
# Device-side async L2 prefetch: correctness + focused perf comparison
tprefetch_async
tquant_dn
)

if (AUTO_MODE)
  set(AUTO_MODE_WHITELIST
  # TPUSH and TPOP currently directly use TASSIGN inside their implementations,
  # which doesn't work for auto mode. They need to be disabled for auto mode
  # until rewritten to adapt to auto mode
    tpushpop_cv
    tpushpop_cv_nosplit
    tpushpop_fixpipe
    tpushpop_vc
    tpushpop_vc_nosplit
    tpushpop_dir_both
    tpushpop_subtile
  # the following case wait to be fix
    tcvt
    ttrans
    ttrans_conv
    tci
    tquant
    texpands_mat
    tinsert
    tscatter
    tcolargmin
    tcolargmax
    tmov
    tmatmul
    tmatmul_mx
    tmov_zz
    tquant_dn
    thistogram
    tsubreluconv
    tsubrelu
    taddreluconv
    tcolgather
    tshrs
    tcolscatter
    tdequant
    tci
    tcolargmax
    tcolargmin
    tcolsum
    tcolprod
    tcvt
    textract
    textract_acc2vec
    tfillpad
    tgather
    tgatherb
    tload_mix
    tload_mx_ND_DN
    tload_mx_gmtensor
    tload_shape2d
    tmov
    tpartadd
    tpartmul
    tpartmax
    tpartmin
    ttrans
    ttrans_conv
    tscatter
    tquant
    tinsert
    tinsert_acc2vec
    tmov_zz
    tmatmul
    tmatmul_mx
    tmov_acc2vec
    txor
    txors
    mgather_gm2l1
    tcmp
  )

  foreach(whitelisted_case ${AUTO_MODE_WHITELIST})
    list(REMOVE_ITEM ALL_TESTCASES ${whitelisted_case})
  endforeach()

  # TPREFETCH_ASYNC relies on manual-mode Tile::data() returning a raw __ubuf__
  # pointer. In auto mode the CCE tile_size type system prevents this cast, and
  # auto-mode scheduling makes manual prefetch unnecessary.
  list(REMOVE_ITEM OPT_IN_TESTCASES tprefetch_async)
endif()

if((TEST_CASE IN_LIST ALL_TESTCASES) OR (TEST_CASE IN_LIST OPT_IN_TESTCASES) OR (TEST_CASE STREQUAL "all"))
    message(STATUS "run : ${TEST_CASE}")
else()
    message(FATAL_ERROR "not found TEST_CASE: ${TEST_CASE}, support case is : ${ALL_TESTCASES} ${OPT_IN_TESTCASES}")
endif()


foreach(TESTCASE ${ALL_TESTCASES})
    if((DEFINED TEST_CASE AND TEST_CASE STREQUAL TESTCASE) OR (NOT DEFINED TEST_CASE) OR (TEST_CASE STREQUAL "all"))
        add_subdirectory(${TESTCASE})
    endif()
endforeach()

# Opt-in cases are built only when the user asked for them by name (or "all").
foreach(TESTCASE ${OPT_IN_TESTCASES})
if((DEFINED TEST_CASE AND TEST_CASE STREQUAL TESTCASE) OR (TEST_CASE STREQUAL "all"))
    add_subdirectory(${TESTCASE})
endif()
endforeach()
