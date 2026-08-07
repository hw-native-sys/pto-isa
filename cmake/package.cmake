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

  set(script_prefix ${CMAKE_CURRENT_SOURCE_DIR}/scripts/package/pto_isa/scripts)
  install(DIRECTORY ${script_prefix}/
      DESTINATION share/info/pto_isa/script
      COMPONENT pto-isa
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
      COMPONENT pto-isa
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
  install(FILES ${CMAKE_BINARY_DIR}/version.pto-isa.info
      DESTINATION share/info/pto_isa
      RENAME version.info
      COMPONENT pto-isa
  )
  install(FILES ${PACKAGE_FILES}
      DESTINATION share/info/pto_isa/script
      COMPONENT pto-isa
  )

  set(pto_source ${CMAKE_CURRENT_SOURCE_DIR}/include)
  install(DIRECTORY ${pto_source}/
      DESTINATION ${CMAKE_SYSTEM_PROCESSOR}-linux/include
      COMPONENT pto-isa
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

  # For rpm/deb packages, prefix each RUN_DEPENDENCIES_LIST entry's version with
  # ">=" so convert_dependencies_to_package_formats produces well-formed Requires/
  # Depends ("pkg >= 9.1") instead of "pkg 9.1" (which RPM splits into a phantom
  # "9.1" standalone dependency). This must NOT touch version.info: that file is
  # generated from CANN_VERSION_*_RUN_DEPS (a separate list), not from
  # RUN_DEPENDENCIES_LIST, so the run package's version check still sees the plain
  # version "9.1" expected by the install.sh version_vaild logic.
  if(PACKAGE_TYPE STREQUAL "rpm" OR PACKAGE_TYPE STREQUAL "deb"
     OR PACKAGE_TYPE STREQUAL "deb,rpm" OR PACKAGE_TYPE STREQUAL "all")
      set(_patched_deps "")
      foreach(_dep_entry ${RUN_DEPENDENCIES_LIST})
          # _dep_entry is "pkg version" (space-separated); prefix version with >=.
          string(REGEX REPLACE "^([^ ]+) +(.*)" "\\1 >= \\2" _patched "${_dep_entry}")
          list(APPEND _patched_deps "${_patched}")
      endforeach()
      set(RUN_DEPENDENCIES_LIST "${_patched_deps}")
  endif()

  # cann-cmake sets CPACK_PACKAGE_NAME to "cann-${component}" = "cann-pto-isa".
  # With CPACK_RPM_COMPONENT_INSTALL/CPACK_DEB_COMPONENT_INSTALL ON, CPack appends
  # the component name ("pto-isa") to the package name AND to the file name,
  # producing the duplicated "cann-pto-isa-pto-isa". Set the per-component package
  # name and file name explicitly before set_cann_cpack_config so the cann-cmake
  # include(CPack) picks them up. The <COMPONENT> part must be UPPER-CASE per CMake
  # docs (CPackDeb only honors the upper-case form; CPackRPM accepts both).
  # We use the historical "cann-pto-isa" (hyphen) for the Name field to stay
  # consistent with the makeself run package filename.
  set(CPACK_RPM_PTO_ISA_PACKAGE_NAME "cann-pto-isa")
  set(CPACK_DEBIAN_PTO_ISA_PACKAGE_NAME "cann-pto-isa")
  string(TOLOWER "${CMAKE_SYSTEM_NAME}" _sys_name_lower)
  set(_file_name_base "cann-pto-isa_${CANN_VERSION_pto-isa_VERSION}_${_sys_name_lower}-${CMAKE_SYSTEM_PROCESSOR}")
  set(CPACK_RPM_PTO_ISA_FILE_NAME "${_file_name_base}.rpm")
  set(CPACK_DEBIAN_PTO_ISA_FILE_NAME "${_file_name_base}.deb")
  # DEB control "Package" field also honors the non-component variable when the
  # generator builds a single component; set it to the same value to be sure.
  set(CPACK_DEBIAN_PACKAGE_NAME "cann-pto-isa")

  set_cann_cpack_config(pto-isa
      COMPUTE_UNIT "${compute_unit}"
      SHARE_INFO_NAME pto_isa
      OUTPUT "${CMAKE_CURRENT_SOURCE_DIR}/build_out"
      PACKAGE_TYPE "${PACKAGE_TYPE}"
  )
endfunction()
