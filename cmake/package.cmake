# ----------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
#### CPACK to package run #####

function(pack_built_in)
  #### built-in package ####
  message(STATUS "System processor: ${CMAKE_SYSTEM_PROCESSOR}")
  if (CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64")
      message(STATUS "Detected architecture: x86_64")
      set(ARCH x86_64)
  elseif (CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64|arm")
      message(STATUS "Detected architecture: ARM64")
      set(ARCH aarch64)
  else ()
      message(WARNING "Unknown architecture: ${CMAKE_SYSTEM_PROCESSOR}")
  endif ()

  set(script_prefix ${CMAKE_SOURCE_DIR}/scripts/package/pto_isa/scripts)
  install(DIRECTORY ${script_prefix}/
      DESTINATION share/info/pto_isa/script
      FILE_PERMISSIONS
      OWNER_READ OWNER_WRITE OWNER_EXECUTE  # 文件权限
      GROUP_READ GROUP_EXECUTE
      WORLD_READ WORLD_EXECUTE
      DIRECTORY_PERMISSIONS
      OWNER_READ OWNER_WRITE OWNER_EXECUTE  # 目录权限
      GROUP_READ GROUP_EXECUTE
      WORLD_READ WORLD_EXECUTE
  )

  set(SCRIPTS_FILES
      ${CANN_CMAKE_DIR}/scripts/install/check_version_required.awk
      ${CANN_CMAKE_DIR}/scripts/install/common_func.inc
      ${CANN_CMAKE_DIR}/scripts/install/common_interface.sh
      ${CANN_CMAKE_DIR}/scripts/install/common_interface.csh
      ${CANN_CMAKE_DIR}/scripts/install/common_interface.fish
      ${CANN_CMAKE_DIR}/scripts/install/version_compatiable.inc
      ${CANN_CMAKE_DIR}/scripts/package/merge_binary_info_config.py
  )

  install(FILES ${SCRIPTS_FILES}
      DESTINATION share/info/pto_isa/script
  )
  set(COMMON_FILES
      ${CANN_CMAKE_DIR}/scripts/install/install_common_parser.sh
      ${CANN_CMAKE_DIR}/scripts/install/common_func_v2.inc
      ${CANN_CMAKE_DIR}/scripts/install/common_installer.inc
      ${CANN_CMAKE_DIR}/scripts/install/script_operator.inc
      ${CANN_CMAKE_DIR}/scripts/install/version_cfg.inc
  )

  set(PACKAGE_FILES
      ${COMMON_FILES}
      ${CANN_CMAKE_DIR}/scripts/install/multi_version.inc
  )
  set(CONF_FILES
      ${CANN_CMAKE_DIR}/scripts/package/cfg/path.cfg
  )
  install(FILES ${CMAKE_BINARY_DIR}/version.pto-isa.info
      DESTINATION share/info/pto_isa
      RENAME version.info
  )
  install(FILES ${CONF_FILES}
      DESTINATION ${CMAKE_SYSTEM_PROCESSOR}-linux/conf
  )
  install(FILES ${PACKAGE_FILES}
      DESTINATION share/info/pto_isa/script
  )

  set(pto_source ${CMAKE_SOURCE_DIR}/include)
  install(DIRECTORY ${pto_source}/
      DESTINATION ${CMAKE_SYSTEM_PROCESSOR}-linux/include
      FILE_PERMISSIONS
      OWNER_READ OWNER_WRITE
      GROUP_READ GROUP_EXECUTE
      REGEX "include/README\\.md$" EXCLUDE
      REGEX "include/README_zh\\.md$" EXCLUDE
      REGEX "include/pto/README\\.md$" EXCLUDE
      REGEX "include/pto/README_zh\\.md$" EXCLUDE
  )

  string(FIND "${ASCEND_COMPUTE_UNIT}" ";" SEMICOLON_INDEX)
  if (SEMICOLON_INDEX GREATER -1)
      # 截取分号前的字串
      math(EXPR SUBSTRING_LENGTH "${SEMICOLON_INDEX}")
      string(SUBSTRING "${ASCEND_COMPUTE_UNIT}" 0 "${SUBSTRING_LENGTH}" compute_unit)
  else()
      # 没有分号取全部内容
      set(compute_unit "${ASCEND_COMPUTE_UNIT}")
  endif()

  message(STATUS "current compute_unit is: ${compute_unit}")

  set_cann_cpack_config(pto-isa
      NO_COMPONENT_INSTALL
      COMPUTE_UNIT "${compute_unit}"
      SHARE_INFO_NAME pto_isa
      OUTPUT "${CMAKE_SOURCE_DIR}/build_out"
  )
endfunction()
