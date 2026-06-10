# --------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# --------------------------------------------------------------------------------

pto_vec_st(syncall)
find_package(Python3 REQUIRED COMPONENTS Interpreter)

add_library(syncall_soft_kernel SHARED syncall_soft_kernel.cpp)
target_compile_options(syncall_soft_kernel PRIVATE ${CMAKE_CCE_COMPILE_OPTIONS} --cce-aicore-arch=dav-c220-vec
                                                     -DMEMORY_BASE -std=c++17)
target_include_directories(syncall_soft_kernel PRIVATE
    ${ASCEND_HOME_PATH}/pkg_inc/
    ${ASCEND_HOME_PATH}/pkg_inc/profiling/
    ${ASCEND_HOME_PATH}/pkg_inc/runtime/runtime
)
target_link_options(syncall_soft_kernel PRIVATE --cce-fatobj-link)
target_link_libraries(syncall PRIVATE syncall_soft_kernel)

# dav-c220 auto-split chevron kernel (single .so per source).
function(pto_syncall_chevron_kernel TARGET SOURCE)
    add_library(${TARGET} SHARED ${SOURCE})
    target_compile_options(${TARGET} PRIVATE ${CMAKE_CCE_COMPILE_OPTIONS} --cce-aicore-arch=dav-c220 -DMEMORY_BASE -std=c++17)
    target_include_directories(${TARGET} PRIVATE
        ${ASCEND_HOME_PATH}/pkg_inc/
        ${ASCEND_HOME_PATH}/pkg_inc/profiling/
        ${ASCEND_HOME_PATH}/pkg_inc/runtime/runtime
    )
    target_link_options(${TARGET} PRIVATE --cce-fatobj-link)
    target_link_libraries(syncall PRIVATE ${TARGET})
endfunction()

pto_syncall_chevron_kernel(syncall_mix_1_2_kernel syncall_mix_1_2_kernel.cpp)
pto_syncall_chevron_kernel(syncall_aic_hard_kernel syncall_aic_hard_kernel.cpp)

# register-ELF path for hard MIX 1:1 only (true 1:1 cannot use dav-c220 auto-split on ccec).
function(pto_syncall_mix_kernel TARGET SOURCE RATIO_DEF)
    set(AIC_OBJ ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/${TARGET}_aic.dir/${SOURCE}.o)
    set(AIV_OBJ ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/${TARGET}_aiv.dir/${SOURCE}.o)
    set(REGISTER_AIC_OBJ ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/${TARGET}_register_aic.dir/${SOURCE}.o)
    set(REGISTER_AIV_OBJ ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/${TARGET}_register_aiv.dir/${SOURCE}.o)
    set(REGISTER_ELF ${CMAKE_CURRENT_BINARY_DIR}/${TARGET}_register.elf)
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
    add_custom_command(
        OUTPUT ${AIC_OBJ}
        COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/${TARGET}_aic.dir
        COMMAND ${CMAKE_CXX_COMPILER} ${SYNCALL_MIX_CCE_OPTIONS} --cce-aicore-arch=dav-c220-cube
                -DMEMORY_BASE -std=c++17 ${RATIO_DEF} -DSYNCALL_MIX_BUILD_AIC
                -DSYNCALL_MIX_REGISTER_OBJECT_PATH=\"${REGISTER_ELF}\"
                -I${PROJECT_SOURCE_DIR}/../../../../../include
                -I${ASCEND_HOME_PATH}/include
                -I${ASCEND_DRIVER_PATH}/kernel/inc
                -I${ASCEND_HOME_PATH}/pkg_inc/
                -I${ASCEND_HOME_PATH}/pkg_inc/profiling/
                -I${ASCEND_HOME_PATH}/pkg_inc/runtime
                -I${ASCEND_HOME_PATH}/pkg_inc/runtime/runtime
                -xcce -o ${AIC_OBJ} -c ${CMAKE_CURRENT_SOURCE_DIR}/${SOURCE}
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/${SOURCE}
        VERBATIM
    )
    add_custom_command(
        OUTPUT ${AIV_OBJ}
        COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/${TARGET}_aiv.dir
        COMMAND ${CMAKE_CXX_COMPILER} ${SYNCALL_MIX_CCE_OPTIONS} --cce-aicore-arch=dav-c220-vec
                -DMEMORY_BASE -std=c++17 ${RATIO_DEF} -DSYNCALL_MIX_BUILD_AIV
                -DSYNCALL_MIX_REGISTER_OBJECT_PATH=\"${REGISTER_ELF}\"
                -I${PROJECT_SOURCE_DIR}/../../../../../include
                -I${ASCEND_HOME_PATH}/include
                -I${ASCEND_DRIVER_PATH}/kernel/inc
                -I${ASCEND_HOME_PATH}/pkg_inc/
                -I${ASCEND_HOME_PATH}/pkg_inc/profiling/
                -I${ASCEND_HOME_PATH}/pkg_inc/runtime
                -I${ASCEND_HOME_PATH}/pkg_inc/runtime/runtime
                -xcce -o ${AIV_OBJ} -c ${CMAKE_CURRENT_SOURCE_DIR}/${SOURCE}
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/${SOURCE}
        VERBATIM
    )
    add_custom_command(
        OUTPUT ${REGISTER_ELF}
        COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/${TARGET}_register_aic.dir
        COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/${TARGET}_register_aiv.dir
        COMMAND ${CMAKE_CXX_COMPILER} ${SYNCALL_MIX_CCE_OPTIONS} --cce-aicore-arch=dav-c220-cube
                -DMEMORY_BASE -std=c++17 ${RATIO_DEF} -DSYNCALL_MIX_BUILD_AIC -DSYNCALL_MIX_REGISTER_BUILD
                -I${PROJECT_SOURCE_DIR}/../../../../../include
                -I${ASCEND_HOME_PATH}/include
                -I${ASCEND_DRIVER_PATH}/kernel/inc
                -I${ASCEND_HOME_PATH}/pkg_inc/
                -I${ASCEND_HOME_PATH}/pkg_inc/profiling/
                -I${ASCEND_HOME_PATH}/pkg_inc/runtime
                -I${ASCEND_HOME_PATH}/pkg_inc/runtime/runtime
                -xcce -o ${REGISTER_AIC_OBJ} -c ${CMAKE_CURRENT_SOURCE_DIR}/${SOURCE}
        COMMAND ${CMAKE_CXX_COMPILER} ${SYNCALL_MIX_CCE_OPTIONS} --cce-aicore-arch=dav-c220-vec
                -DMEMORY_BASE -std=c++17 ${RATIO_DEF} -DSYNCALL_MIX_BUILD_AIV -DSYNCALL_MIX_REGISTER_BUILD
                -I${PROJECT_SOURCE_DIR}/../../../../../include
                -I${ASCEND_HOME_PATH}/include
                -I${ASCEND_DRIVER_PATH}/kernel/inc
                -I${ASCEND_HOME_PATH}/pkg_inc/
                -I${ASCEND_HOME_PATH}/pkg_inc/profiling/
                -I${ASCEND_HOME_PATH}/pkg_inc/runtime
                -I${ASCEND_HOME_PATH}/pkg_inc/runtime/runtime
                -xcce -o ${REGISTER_AIV_OBJ} -c ${CMAKE_CURRENT_SOURCE_DIR}/${SOURCE}
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/make_mix_register_elf.py
                ${REGISTER_AIC_OBJ} ${REGISTER_AIV_OBJ} ${REGISTER_ELF}
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/${SOURCE} ${CMAKE_CURRENT_SOURCE_DIR}/make_mix_register_elf.py
        VERBATIM
    )
    add_custom_target(${TARGET}_register_obj DEPENDS ${REGISTER_ELF})
    add_library(${TARGET} SHARED ${AIC_OBJ} ${AIV_OBJ})
    add_dependencies(${TARGET} ${TARGET}_register_obj)
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

pto_syncall_mix_kernel(syncall_mix_1_1_kernel syncall_mix_1_1_kernel.cpp -D__MIX_CORE_AIV_RATIO__=1)

# Soft MIX 1:1: dual-arch (cube + vec) dual-stream chevron; no register ELF.
function(pto_syncall_mix11_soft_kernel TARGET SOURCE)
    set(AIC_OBJ ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/${TARGET}_aic.dir/${SOURCE}.o)
    set(AIV_OBJ ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/${TARGET}_aiv.dir/${SOURCE}.o)
    set(MIX11_CCE_OPTIONS
        -xcce
        -fPIC
        -Xhost-start -Xhost-end
        -mllvm -cce-aicore-stack-size=0x8000
        -mllvm -cce-aicore-function-stack-size=0x8000
        -mllvm -cce-aicore-record-overflow=true
        -mllvm -cce-aicore-addr-transform
        -mllvm -cce-aicore-dcci-insert-for-scalar=false
    )
    set(MIX11_INCLUDES
        -I${PROJECT_SOURCE_DIR}/../../../../../include
        -I${ASCEND_HOME_PATH}/include
        -I${ASCEND_DRIVER_PATH}/kernel/inc
        -I${ASCEND_HOME_PATH}/pkg_inc/
        -I${ASCEND_HOME_PATH}/pkg_inc/profiling/
        -I${ASCEND_HOME_PATH}/pkg_inc/runtime
        -I${ASCEND_HOME_PATH}/pkg_inc/runtime/runtime
        -I${CMAKE_CURRENT_SOURCE_DIR}
    )
    add_custom_command(
        OUTPUT ${AIC_OBJ}
        COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/${TARGET}_aic.dir
        COMMAND ${CMAKE_CXX_COMPILER} ${MIX11_CCE_OPTIONS} --cce-aicore-arch=dav-c220-cube
                -DMEMORY_BASE -std=c++17 -D__MIX_CORE_AIV_RATIO__=1 -DSYNCALL_MIX_BUILD_AIC
                ${MIX11_INCLUDES}
                -xcce -o ${AIC_OBJ} -c ${CMAKE_CURRENT_SOURCE_DIR}/${SOURCE}
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/${SOURCE}
        VERBATIM
    )
    add_custom_command(
        OUTPUT ${AIV_OBJ}
        COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/${TARGET}_aiv.dir
        COMMAND ${CMAKE_CXX_COMPILER} ${MIX11_CCE_OPTIONS} --cce-aicore-arch=dav-c220-vec
                -DMEMORY_BASE -std=c++17 -D__MIX_CORE_AIV_RATIO__=1 -DSYNCALL_MIX_BUILD_AIV
                ${MIX11_INCLUDES}
                -xcce -o ${AIV_OBJ} -c ${CMAKE_CURRENT_SOURCE_DIR}/${SOURCE}
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/${SOURCE}
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

pto_syncall_mix11_soft_kernel(syncall_mix_1_1_soft_kernel syncall_mix_1_1_soft_kernel.cpp)

add_library(syncall_aic_kernel SHARED syncall_aic_kernel.cpp)
target_compile_options(syncall_aic_kernel PRIVATE ${CMAKE_CCE_COMPILE_OPTIONS} --cce-aicore-arch=dav-c220-cube
                                                   -DMEMORY_BASE -std=c++17)
target_include_directories(syncall_aic_kernel PRIVATE
    ${ASCEND_HOME_PATH}/pkg_inc/
    ${ASCEND_HOME_PATH}/pkg_inc/profiling/
    ${ASCEND_HOME_PATH}/pkg_inc/runtime/runtime
)
target_link_options(syncall_aic_kernel PRIVATE --cce-fatobj-link)
target_link_libraries(syncall PRIVATE syncall_aic_kernel)
