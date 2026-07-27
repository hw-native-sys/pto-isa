# --------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# --------------------------------------------------------------------------------

add_library(syncall_soft_kernel SHARED syncall_soft_kernel.cpp)
target_compile_options(syncall_soft_kernel PRIVATE ${CMAKE_CCE_COMPILE_OPTIONS} --cce-aicore-arch=dav-c310-vec
                                                     -DREGISTER_BASE -std=c++17)
target_include_directories(syncall_soft_kernel PRIVATE
    ${ASCEND_HOME_PATH}/pkg_inc/
    ${ASCEND_HOME_PATH}/pkg_inc/profiling/
    ${ASCEND_HOME_PATH}/pkg_inc/runtime/runtime
)
target_link_options(syncall_soft_kernel PRIVATE --cce-fatobj-link)

add_library(syncall_hard_kernel SHARED syncall_hard_kernel.cpp)
target_compile_options(syncall_hard_kernel PRIVATE ${CMAKE_CCE_COMPILE_OPTIONS} --cce-aicore-arch=dav-c310-vec
                                                     -DREGISTER_BASE -std=c++17)
target_include_directories(syncall_hard_kernel PRIVATE
    ${ASCEND_HOME_PATH}/pkg_inc/
    ${ASCEND_HOME_PATH}/pkg_inc/profiling/
    ${ASCEND_HOME_PATH}/pkg_inc/runtime/runtime
)
target_link_options(syncall_hard_kernel PRIVATE --cce-fatobj-link)

# dav-c310 auto-split chevron kernel (single .so per source). arch35 bare dav-c310
# compiles one source for both cube and vector and packs a MIX-launchable binary,
# the same model dispatch_mega_combine uses. Only ratios that ccec can auto-split
# (1:N with N>=1) may use this; true 1:1 still needs a separate path.
function(pto_syncall_chevron_kernel TARGET SOURCE)
    add_library(${TARGET} SHARED ${SOURCE})
    target_compile_options(${TARGET} PRIVATE ${CMAKE_CCE_COMPILE_OPTIONS} --cce-aicore-arch=dav-c310
                                             -DREGISTER_BASE -std=c++17)
    target_include_directories(${TARGET} PRIVATE
        ${ASCEND_HOME_PATH}/pkg_inc/
        ${ASCEND_HOME_PATH}/pkg_inc/profiling/
        ${ASCEND_HOME_PATH}/pkg_inc/runtime/runtime
    )
    target_link_options(${TARGET} PRIVATE --cce-fatobj-link)
    target_link_libraries(syncall PRIVATE ${TARGET})
endfunction()

# Pure cube-only chevron kernel (single .so per source), dav-c310-cube. Used for
# AIC-only kernels: one __global__ symbol launched with <<<blocks>>> on cube cores,
# mirroring the AIV-only syncall_soft_kernel but on the cube arch.
function(pto_syncall_cube_chevron_kernel TARGET SOURCE)
    add_library(${TARGET} SHARED ${SOURCE})
    target_compile_options(${TARGET} PRIVATE ${CMAKE_CCE_COMPILE_OPTIONS} --cce-aicore-arch=dav-c310-cube
                                             -DREGISTER_BASE -std=c++17)
    target_include_directories(${TARGET} PRIVATE
        ${ASCEND_HOME_PATH}/pkg_inc/
        ${ASCEND_HOME_PATH}/pkg_inc/profiling/
        ${ASCEND_HOME_PATH}/pkg_inc/runtime/runtime
    )
    target_link_options(${TARGET} PRIVATE --cce-fatobj-link)
    target_link_libraries(syncall PRIVATE ${TARGET})
endfunction()

# True MIX 1:1 (dual-arch, dual-stream chevron). dav-c310 auto-split is physically
# 1:2, and the register-ELF path cannot run on A5 because it needs an FFTS base that
# rtGetC2cCtrlAddr does not provide here. So cube and vector are compiled as two
# independent kernels into one .so and each gets its own chevron on its own stream;
# the soft GM barrier is what synchronizes them.
function(pto_syncall_mix11_soft_kernel TARGET SOURCE)
    set(AIC_OBJ ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/${TARGET}_aic.dir/${SOURCE}.o)
    set(AIV_OBJ ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/${TARGET}_aiv.dir/${SOURCE}.o)
    set(SYNCALL_MIX_CCE_OPTIONS
        -xcce
        -fPIC
        -Xhost-start -Xhost-end
        -mllvm -cce-aicore-stack-size=0x8000
        -mllvm -cce-aicore-function-stack-size=0x8000
        -mllvm -cce-aicore-record-overflow=true
        -mllvm -cce-aicore-addr-transform
        -mllvm -cce-aicore-dcci-insert-for-scalar=false
    )
    # Library headers the kernels compile against; listing them keeps the custom
    # commands from reusing stale .o when only the pto headers change.
    file(GLOB SYNCALL_PTO_HEADERS
        ${PROJECT_SOURCE_DIR}/../../../../../include/pto/npu/a5/*.hpp
        ${PROJECT_SOURCE_DIR}/../../../../../include/pto/common/*.hpp)
    add_custom_command(
        OUTPUT ${AIC_OBJ}
        COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/${TARGET}_aic.dir
        COMMAND ${CMAKE_CXX_COMPILER} ${SYNCALL_MIX_CCE_OPTIONS} --cce-aicore-arch=dav-c310-cube
                -DREGISTER_BASE -std=c++17 -D__MIX_CORE_AIC_BLOCKS__=18 -D__MIX_CORE_AIV_RATIO__=1
                -DSYNCALL_MIX_BUILD_AIC
                -I${PROJECT_SOURCE_DIR}/../../../../../include
                -I${ASCEND_HOME_PATH}/include
                -I${ASCEND_DRIVER_PATH}/kernel/inc
                -I${ASCEND_HOME_PATH}/pkg_inc/
                -I${ASCEND_HOME_PATH}/pkg_inc/profiling/
                -I${ASCEND_HOME_PATH}/pkg_inc/runtime
                -I${ASCEND_HOME_PATH}/pkg_inc/runtime/runtime
                -xcce -o ${AIC_OBJ} -c ${CMAKE_CURRENT_SOURCE_DIR}/${SOURCE}
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/${SOURCE}
                ${CMAKE_CURRENT_SOURCE_DIR}/syncall_mix_common.hpp
                ${SYNCALL_PTO_HEADERS}
        VERBATIM
    )
    add_custom_command(
        OUTPUT ${AIV_OBJ}
        COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/${TARGET}_aiv.dir
        COMMAND ${CMAKE_CXX_COMPILER} ${SYNCALL_MIX_CCE_OPTIONS} --cce-aicore-arch=dav-c310-vec
                -DREGISTER_BASE -std=c++17 -D__MIX_CORE_AIC_BLOCKS__=18 -D__MIX_CORE_AIV_RATIO__=1
                -DSYNCALL_MIX_BUILD_AIV
                -I${PROJECT_SOURCE_DIR}/../../../../../include
                -I${ASCEND_HOME_PATH}/include
                -I${ASCEND_DRIVER_PATH}/kernel/inc
                -I${ASCEND_HOME_PATH}/pkg_inc/
                -I${ASCEND_HOME_PATH}/pkg_inc/profiling/
                -I${ASCEND_HOME_PATH}/pkg_inc/runtime
                -I${ASCEND_HOME_PATH}/pkg_inc/runtime/runtime
                -xcce -o ${AIV_OBJ} -c ${CMAKE_CURRENT_SOURCE_DIR}/${SOURCE}
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/${SOURCE}
                ${CMAKE_CURRENT_SOURCE_DIR}/syncall_mix_common.hpp
                ${SYNCALL_PTO_HEADERS}
        VERBATIM
    )
    add_library(${TARGET} SHARED ${AIC_OBJ} ${AIV_OBJ})
    set_target_properties(${TARGET} PROPERTIES LINKER_LANGUAGE CXX)
    set_source_files_properties(${AIC_OBJ} ${AIV_OBJ} PROPERTIES EXTERNAL_OBJECT TRUE GENERATED TRUE)
    target_include_directories(${TARGET} PRIVATE
        ${ASCEND_HOME_PATH}/pkg_inc/
        ${ASCEND_HOME_PATH}/pkg_inc/profiling/
        ${ASCEND_HOME_PATH}/pkg_inc/runtime/runtime
    )
    target_link_options(${TARGET} PRIVATE --cce-fatobj-link -Wl,--strip-debug)
    target_link_libraries(syncall PRIVATE ${TARGET})
endfunction()

add_executable(syncall main.cpp)
target_compile_options(syncall PRIVATE ${CMAKE_CPP_COMPILE_OPTIONS})
target_include_directories(syncall PRIVATE
    ${PROJECT_SOURCE_DIR}/../../../../common
)

target_link_directories(syncall PUBLIC
    ${ASCEND_HOME_PATH}/lib64
    ${ASCEND_HOME_PATH}/tools/simulator/${SOC_VERSION}/lib
)

target_link_libraries(syncall PRIVATE
    syncall_soft_kernel
    syncall_hard_kernel
    $<BUILD_INTERFACE:$<$<STREQUAL:${RUN_MODE},sim>:runtime_camodel>>
    $<BUILD_INTERFACE:$<$<STREQUAL:${RUN_MODE},npu>:runtime>>
    stdc++ ascendcl m tiling_api platform c_sec dl nnopbase GTest::gtest GTest::gtest_main pthread
)

pto_syncall_mix11_soft_kernel(syncall_mix_1_1_kernel syncall_mix_1_1_kernel.cpp)
pto_syncall_chevron_kernel(syncall_mix_1_2_kernel syncall_mix_1_2_kernel.cpp)
pto_syncall_chevron_kernel(syncall_hard_mix_1_2_kernel syncall_hard_mix_1_2_kernel.cpp)
pto_syncall_cube_chevron_kernel(syncall_aic_hard_kernel syncall_aic_hard_kernel.cpp)
pto_syncall_cube_chevron_kernel(syncall_aic_soft_kernel syncall_aic_soft_kernel.cpp)
