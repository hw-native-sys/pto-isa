# --------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# --------------------------------------------------------------------------------

if(SOC_VERSION STREQUAL "Kirin9050" OR SOC_VERSION STREQUAL "dav_9201")
    pto_vec_st(tload_large_stride)
    target_compile_definitions(tload_large_stride_kernel PRIVATE PTO_TLOAD_VEC_ONLY)
    if(SOC_VERSION STREQUAL "dav_9201")
        target_compile_definitions(tload_large_stride PRIVATE PTO_TLOAD_VEC_ONLY)
    endif()
else()
    pto_mix_st(tload_large_stride)
endif()

if(SOC_VERSION MATCHES "^Ascend950" OR SOC_VERSION STREQUAL "dav_9201")
    target_compile_definitions(tload_large_stride PRIVATE PTO_TLOAD_WIDE_STRIDE)
    if(TARGET tload_large_stride_kernel_mix)
        target_compile_definitions(tload_large_stride_kernel_mix PRIVATE PTO_TLOAD_WIDE_STRIDE)
    else()
        target_compile_definitions(tload_large_stride_kernel PRIVATE PTO_TLOAD_WIDE_STRIDE)
    endif()
endif()

if(SOC_VERSION MATCHES "^Ascend950")
    target_compile_definitions(tload_large_stride PRIVATE PTO_TLOAD_CONV_WIDE)
    target_compile_definitions(tload_large_stride_kernel_mix PRIVATE PTO_TLOAD_CONV_WIDE)
endif()

if(SOC_VERSION MATCHES "^Ascend910" OR SOC_VERSION STREQUAL "KirinX90")
    target_compile_definitions(tload_large_stride PRIVATE PTO_TLOAD_VECTOR_LENGTH)
    if(TARGET tload_large_stride_kernel_mix)
        target_compile_definitions(tload_large_stride_kernel_mix PRIVATE PTO_TLOAD_VECTOR_LENGTH)
    else()
        target_compile_definitions(tload_large_stride_kernel PRIVATE PTO_TLOAD_VECTOR_LENGTH)
    endif()
endif()
