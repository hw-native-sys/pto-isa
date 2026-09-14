# --------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# --------------------------------------------------------------------------------

pto_cpu_sim_st(ffts)
foreach(role IN ITEMS aic aiv aiv_second)
    add_library(ffts_${role} SHARED kernel.cpp)
    target_include_directories(ffts_${role} PRIVATE ${PROJECT_SOURCE_DIR}/../../../include)
    target_compile_options(ffts_${role} PRIVATE ${CMAKE_CPP_COMPILE_OPTIONS})
    set_target_properties(ffts_${role} PROPERTIES CXX_VISIBILITY_PRESET hidden VISIBILITY_INLINES_HIDDEN YES)
    if(role STREQUAL "aic")
        target_compile_definitions(ffts_${role} PRIVATE __DAV_CUBE__)
    else()
        target_compile_definitions(ffts_${role} PRIVATE __DAV_VEC__)
    endif()
    target_link_libraries(ffts_${role} PRIVATE Threads::Threads ${CMAKE_DL_LIBS})
endforeach()
add_dependencies(ffts ffts_aic ffts_aiv ffts_aiv_second)
set_target_properties(ffts PROPERTIES ENABLE_EXPORTS YES)
target_compile_definitions(ffts PRIVATE
    FFTS_AIC_PATH="$<TARGET_FILE:ffts_aic>"
    FFTS_AIV_PATH="$<TARGET_FILE:ffts_aiv>"
    FFTS_AIV_SECOND_PATH="$<TARGET_FILE:ffts_aiv_second>"
)
