# --------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# --------------------------------------------------------------------------------

cmake_minimum_required(VERSION 3.10)
include(cmake/fetch_cann_cmake.cmake)
project(pto-isa)

init_cann_project()
add_cann_target_options()

# Package type: run / rpm / deb. Default "run" keeps backward-compatible makeself run package.
# Forwarded to set_cann_cpack_config(PACKAGE_TYPE ...) in cmake/package.cmake.
if(NOT DEFINED PACKAGE_TYPE)
    set(PACKAGE_TYPE "run" CACHE STRING "Package type: run/rpm/deb")
endif()

include(cmake/package.cmake)
include(cmake/func.cmake)
include(version.cmake)
check_cann_pkg_build_deps("pto-isa")
add_cann_version_info_targets()
pack_built_in()