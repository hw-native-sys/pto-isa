# --------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# --------------------------------------------------------------------------------

# syncall needs several kernels behind one test binary, so it does not use the
# pto_vec_st / pto_cube_st helpers (those build one executable per kernel).

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
    $<BUILD_INTERFACE:$<$<STREQUAL:${RUN_MODE},sim>:runtime_camodel>>
    $<BUILD_INTERFACE:$<$<STREQUAL:${RUN_MODE},npu>:runtime>>
    stdc++ ascendcl m tiling_api platform c_sec dl nnopbase GTest::gtest GTest::gtest_main pthread
)

# ARCH selects the core type: dav-920r1-vec / dav-920r1-cube build a single core
# type, while bare dav-920r1 compiles one source for both and packs a MIX-launchable
# binary. A6 auto-split pairs one AIC with one AIV, which is exactly the 1:1 MIX
# configuration under test.
function(pto_syncall_a6_kernel TARGET SOURCE ARCH)
    add_library(${TARGET} SHARED ${SOURCE})
    target_compile_options(${TARGET} PRIVATE ${CMAKE_CCE_COMPILE_OPTIONS}
                                             --cce-aicore-arch=${ARCH}
                                             -DREGISTER_BASE -std=c++17 -Wno-c++20-extensions)
    target_include_directories(${TARGET} PRIVATE
        ${ASCEND_HOME_PATH}/pkg_inc/
        ${ASCEND_HOME_PATH}/pkg_inc/profiling/
        ${ASCEND_HOME_PATH}/pkg_inc/runtime/runtime
    )
    target_link_options(${TARGET} PRIVATE --cce-fatobj-link)
    target_link_libraries(syncall PRIVATE ${TARGET})
endfunction()

pto_syncall_a6_kernel(syncall_hard_kernel syncall_hard_kernel.cpp dav-920r1-vec)
pto_syncall_a6_kernel(syncall_aic_hard_kernel syncall_aic_hard_kernel.cpp dav-920r1-cube)
pto_syncall_a6_kernel(syncall_hard_mix_kernel syncall_hard_mix_kernel.cpp dav-920r1)
pto_syncall_a6_kernel(syncall_soft_kernel syncall_soft_kernel.cpp dav-920r1-vec)
pto_syncall_a6_kernel(syncall_aic_soft_kernel syncall_aic_soft_kernel.cpp dav-920r1-cube)
pto_syncall_a6_kernel(syncall_soft_mix_kernel syncall_soft_mix_kernel.cpp dav-920r1)
