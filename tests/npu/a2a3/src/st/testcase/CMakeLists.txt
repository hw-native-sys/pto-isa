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
    target_compile_options(${NAME}_kernel PRIVATE ${CMAKE_CCE_COMPILE_OPTIONS} --cce-aicore-arch=dav-c220-vec -DMEMORY_BASE -std=c++17)
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
        ${ASCEND_HOME_PATH}/simulator/${SOC_VERSION}/lib
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
    target_compile_options(${NAME}_kernel PRIVATE ${CMAKE_CCE_COMPILE_OPTIONS} --cce-aicore-arch=dav-c220-cube -DMEMORY_BASE -std=c++17)
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
        ${ASCEND_HOME_PATH}/simulator/${SOC_VERSION}/lib
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
    target_compile_options(${NAME}_kernel_mix PRIVATE ${CMAKE_CCE_COMPILE_OPTIONS} --cce-aicore-arch=dav-c220 -DMEMORY_BASE -std=c++17)
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
tmatmul
trowsum
trowprod
trowmax
trowmin
trowargmax
trowargmin
textract
timg2col
tmov
tload
ttrans
ttrans_conv
ttrans_3d
tmrgsort
tfillpad
tcolgather
tgather
tsort32
tcmps
tstore
tsels
tdiv
tmov_vect
tmov_acc2mat
trowexpand
trowexpandadd
trowexpanddiv
trowexpandmax
trowexpandmin
trowexpandmul
trowexpandsub
tcvt
tmul
tadd
taddplus
tprelu
tand
tor
tshl
tshr
tpartadd
tpartmul
tpartmin
tpartmax
tpartargmax
tpartargmin
tpow
tpows
tsel
tsub
tsubs
trem
trems
tfmod
tfmods
tlrelu
tmaxs
tmins
tmin
tmax
tload_gm2mat
tstore_acc2gm
trsqrt
tsqrt
texp
tabs
tlog
trecip
tdivs
tmuls
tneg
tadds
taxpy
tgatherb
tci
tcolmax
tcolargmax
tcolmin
tcolargmin
texpands
texpands_mat
tcolsum
tcolprod
tcmp
tassign
tstore_mat2gm
tcolscatter
tscatter
tcolexpand
ttri
tnot
trelu
taddreluconv
tadddeqrelu
tsubreluconv
tands
tors
tshls
tshrs
txor
txors
trowexpandexpdif
tcolexpandadd
tcolexpandmax
tcolexpandmin
tcolexpanddiv
tcolexpandexpdif
tcolexpandmul
tcolexpandsub
tquant
tdequant
t_dhrystone
tconcatdstidx
tconcatidx
tconcat
textract_vec
tinsert_vec
mgather
mgather_gm2l1
mscatter
tpushpop_cv
tpushpop_vc
tpushpop_cv_nosplit
tpushpop_vc_nosplit
tpushpop_dir_both
tpushpop_subtile
tfusedmuladd
tmuladddst
tsubrelu
tfusedmuladdrelu
syncall
tpairreducesum
)

# Opt-in testcases: built ONLY when TEST_CASE=<name> is explicitly supplied.
# These are excluded from the default "build everything" path so they do not
# slow down CI. They are still valid TEST_CASE targets for local ST runs.
set(OPT_IN_TESTCASES
# Device-side async L2 prefetch: correctness + focused perf comparison
tprefetch_async
)

if (AUTO_MODE)
  set(AUTO_MODE_WHITELIST
  # TPUSH and TPOP currently directly use TASSIGN inside their implementations,
  # which doesn't work for auto mode. They need to be disabled for auto mode
  # until rewritten to adapt to auto mode
    tpushpop_cv
    tpushpop_vc
    tpushpop_cv_nosplit
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
