# --------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# --------------------------------------------------------------------------------
function(pto_st NAME)	 
     add_library(${NAME}_kernel SHARED ${NAME}_kernel.cpp)	 
     target_compile_options(${NAME}_kernel PRIVATE ${CMAKE_CCE_COMPILE_OPTIONS}	 
         --cce-aicore-arch=${CCE_AICORE_ARCH} -DREGISTER_BASE -std=c++17)	 
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
         $<BUILD_INTERFACE:$<$<STREQUAL:${RUN_MODE},npu>:ai_runtime>>	 
         $<BUILD_INTERFACE:$<$<STREQUAL:${RUN_MODE},npu>:runtime>>	 
         stdc++ ascendcl m tiling_api platform c_sec dl nnopbase GTest::gtest GTest::gtest_main pthread 
     ) 
 endfunction()


function(pto_st_npu NAME)
    add_library(${NAME}_kernel SHARED ${NAME}_kernel.cpp)
    target_compile_options(${NAME}_kernel PRIVATE ${CMAKE_CCE_COMPILE_OPTIONS}
        --cce-aicore-arch=${CCE_AICORE_ARCH} -DREGISTER_BASE -std=c++17)
    target_include_directories(${NAME}_kernel PRIVATE 
        ${ASCEND_HOME_PATH}/pkg_inc/
        ${ASCEND_HOME_PATH}/pkg_inc/profiling/
        ${ASCEND_HOME_PATH}/pkg_inc/runtime/runtime
    )
    target_link_options(${NAME}_kernel PRIVATE 
        --cce-fatobj-link
        --target=aarch64-linux-ohos
        --sysroot=${NATIVE}/sysroot
        -nostdlib)
    
    set(OUTPUT_OBJ "${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_FILES_DIRECTORY}/${NAME}.dir/main.cpp.o")
    set(DEP_FILE "${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_FILES_DIRECTORY}/${NAME}.dir/main.cpp.o.d")
    
    add_custom_command(
        OUTPUT ${OUTPUT_OBJ}
        DEPFILE ${DEP_FILE}
        COMMAND ${NATIVE}/llvm/bin/clang-15
                -I${PROJECT_SOURCE_DIR}/../../../../../include 
                -I${ASCEND_HOME_PATH}/ohos
                -I${ASCEND_DRIVER_PATH}/kernel/inc
                -I${PROJECT_SOURCE_DIR}/../../../../common
                -fPIE -D_FORTIFY_SOURCE=2 -O2 -std=c++17
                -Wno-macro-redefined -Wno-ignored-attributes -Wno-unknown-attributes
                -fstack-protector-strong -xc++ -include stdint.h -include stddef.h
                --sysroot=${NATIVE}/sysroot
                -I${NATIVE}/sysroot/usr/include/aarch64-linux-ohos
                -I${NATIVE}/llvm/include/libcxx-ohos/include/c++/v1
                -I${GTEST_INCLUDE} --target=aarch64-linux-ohos
                -MD -MT ${OUTPUT_OBJ}
                -MF ${DEP_FILE} -o ${OUTPUT_OBJ}
                -c ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp
        VERBATIM
    )

    file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
    file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
    set(EXECUTABLE_OUTPUT "${CMAKE_BINARY_DIR}/bin/${NAME}")

    set(KERNEL_LIB_PATH "$<TARGET_FILE:${NAME}_kernel>")

    add_custom_command(
        OUTPUT ${EXECUTABLE_OUTPUT}
        COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/bin
        COMMAND ${NATIVE}/llvm/bin/clang-15
            -s
            -Wl,-z,relro
            -Wl,-z,now
            --target=aarch64-linux-ohos
            --sysroot=${NATIVE}/sysroot
            ${OUTPUT_OBJ}
            -o ${EXECUTABLE_OUTPUT}
            -Wl,-rpath,${CMAKE_BINARY_DIR}/lib
            ${KERNEL_LIB_PATH}
            -L${NATIVE}/lib
            -L${ARM_GTEST}/lib
            -lstdc++ -lascendcl -lm -ldl -lgtest_main -lgtest -lpthread
            -lai_runtime
        DEPENDS ${OUTPUT_OBJ} ${NAME}_kernel ${NATIVE}/lib/libai_runtime.so
        VERBATIM
    )

    add_custom_target(
            ${NAME} ALL
            DEPENDS ${EXECUTABLE_OUTPUT}
        )
    
endfunction()

function(pto_test NAME)	
    if(RUN_MODE STREQUAL "npu")
        pto_st_npu(${NAME})
        target_compile_definitions(${NAME} PRIVATE PTO_RUN_MODE_NPU)
    else()
        pto_st(${NAME})
    endif()
endfunction()

function(pto_vec_st NAME)
    pto_test(${NAME})
endfunction()

function(pto_cube_st NAME)
    pto_test(${NAME})
endfunction()

function(pto_mix_st NAME)
    pto_test(${NAME})
endfunction()

# 定义测试用例列表 testcase下如有新增用例目录需要在此加
set(ALL_TESTCASES
    tsubs
    tmaxs
    tci
    tmatmul
    tload_mix
    # tload_shape2d
    tload
    tmrgsort
    ttrans
    # ttrans_conv
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
    tgather
    tcvt
    tsub
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
    tpartadd
    tpartmul
    tfillpad
    tmin
    tmax
    tmins
    tmov
    tmul
    tlrelu
    tdiv
    tcmps
    # tmov_acc2vec
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
    # t_dhrystone
    tpow
    tpows
    tconcatdstidx
    tconcatidx
    tconcat
    tinsert
    tinsert_acc2vec
    textract_nd_vec
    textract_vec
    # tmov_zz
    tmov_nd2nz
    textract_acc2vec
    texpands_mat
    timg2col
)

if((TEST_CASE IN_LIST ALL_TESTCASES) OR (TEST_CASE STREQUAL "all"))
    message(STATUS "run : ${TEST_CASE}")
else()
    message(FATAL_ERROR "not found TEST_CASE: ${TEST_CASE}, support case is : ${ALL_TESTCASES}")
endif()


foreach(TESTCASE ${ALL_TESTCASES})
    if((DEFINED TEST_CASE AND TEST_CASE STREQUAL TESTCASE) OR (NOT DEFINED TEST_CASE) OR (TEST_CASE STREQUAL "all"))
        add_subdirectory(${TESTCASE})
    endif()
endforeach()