# --------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# --------------------------------------------------------------------------------

pto_cube_st(textract)

add_library(textract_acc2mat_kernel SHARED textract_acc2mat_kernel.cpp)
target_compile_options(textract_acc2mat_kernel PRIVATE
    ${CMAKE_CCE_COMPILE_OPTIONS} --cce-aicore-arch=dav-c310 -DREGISTER_BASE -std=c++17)
target_include_directories(textract_acc2mat_kernel PRIVATE
    ${ASCEND_HOME_PATH}/pkg_inc/
    ${ASCEND_HOME_PATH}/pkg_inc/profiling/
    ${ASCEND_HOME_PATH}/pkg_inc/runtime/runtime)
target_link_options(textract_acc2mat_kernel PRIVATE --cce-fatobj-link)
target_link_libraries(textract PRIVATE textract_acc2mat_kernel)
