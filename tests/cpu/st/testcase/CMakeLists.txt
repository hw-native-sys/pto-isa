# --------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# --------------------------------------------------------------------------------

function(pto_cpu_sim_st NAME)
    set(FILES_LIST main.cpp)
    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${NAME}_kernel.cpp)
        list(APPEND FILES_LIST ${NAME}_kernel.cpp)
    endif()
    add_executable(${NAME} ${FILES_LIST})
    target_compile_options(${NAME} PRIVATE ${CMAKE_CPP_COMPILE_OPTIONS} -std=c++${CMAKE_CXX_STANDARD})
    target_include_directories(${NAME} PRIVATE
        ${PROJECT_SOURCE_DIR}/../../../include/
        ${PROJECT_SOURCE_DIR}/../../../include/common
        ${PROJECT_SOURCE_DIR}/../stubs
        ${PROJECT_SOURCE_DIR}/../../common
    )

    target_link_libraries(${NAME} PRIVATE
        GTest::gtest_main
        Threads::Threads
    )
    if (UNIX)
        target_link_libraries(${NAME} PRIVATE m)
    endif()
    if (UNIX AND NOT APPLE)
        target_link_libraries(${NAME} PRIVATE dl)
    endif()
endfunction()

find_package(Threads REQUIRED)

set(ALL_TESTCASES
hashfind
mgather
mscatter
setgetval
tabs
tadd
taddc
tadds
taddsc
tand
tands
tbroadcast
tci
tcmp
tcmps
tcolexpand
tcolexpandop
tcolmax
tcolmin
tcolsum
tcvt
tdiv
tdivs
texp
texpands
textract
tfillpad
tflashattn
tgather
tgatherb
tget
tload
tloadconv
tlog
tlrelu
tmatmul
tmax
tmaxs
tmin
tmins
tmov
tmrgsort
tmul
tmuls
tneg
tnot
tnotify
tor
tors
tpartadd
tpartmax
tpartmin
tprefetch
tprelu
tpushpop
tput
trecip
treduce
trelu
trem
trems
trowexpand
trowexpandop
trowmax
trowmin
trowsum
trsqrt
tscatter
tsel
tsels
tshl
tshr
tsort32
tsqrt
tstore
tsub
tsubc
tsubs
tsubsc
ttest
ttrans
ttri
twait
treshape
tpushpop_cv
tmatmul_layout
trandom
thistogram
timg2col
tquant
tisa_coverage
txor
txors
)

foreach(TESTCASE ${ALL_TESTCASES})
    if(NOT DEFINED TEST_CASE OR TEST_CASE STREQUAL "${TESTCASE}")
        add_subdirectory(${TESTCASE})
    endif()
endforeach()
