# --------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# --------------------------------------------------------------------------------

add_library(tprefetch_async_kernel SHARED
    tprefetch_async_kernel.cpp
)
target_compile_options(tprefetch_async_kernel PRIVATE ${CMAKE_CCE_COMPILE_OPTIONS} --cce-aicore-arch=dav-c310-vec -DREGISTER_BASE -std=c++17)
target_include_directories(tprefetch_async_kernel PRIVATE
    ${ASCEND_HOME_PATH}/pkg_inc/
    ${ASCEND_HOME_PATH}/pkg_inc/profiling/
    ${ASCEND_HOME_PATH}/pkg_inc/runtime/runtime
)
target_link_directories(tprefetch_async_kernel PRIVATE
    ${ASCEND_HOME_PATH}/lib64
)
target_link_libraries(tprefetch_async_kernel PRIVATE
    runtime nnopbase
)
target_link_options(tprefetch_async_kernel PRIVATE --cce-fatobj-link)

add_executable(tprefetch_async main.cpp)
target_compile_options(tprefetch_async PRIVATE ${CMAKE_CPP_COMPILE_OPTIONS})
target_include_directories(tprefetch_async PRIVATE
    ${PROJECT_SOURCE_DIR}/../../../../common
)

target_link_directories(tprefetch_async PUBLIC
    ${ASCEND_HOME_PATH}/lib64
    ${ASCEND_HOME_PATH}/tools/simulator/${SOC_VERSION}/lib
)

target_link_libraries(tprefetch_async PRIVATE
    tprefetch_async_kernel
    $<BUILD_INTERFACE:$<$<STREQUAL:${RUN_MODE},sim>:runtime_camodel>>
    $<BUILD_INTERFACE:$<$<STREQUAL:${RUN_MODE},npu>:runtime>>
    stdc++ ascendcl m tiling_api platform c_sec dl nnopbase GTest::gtest GTest::gtest_main pthread
)
