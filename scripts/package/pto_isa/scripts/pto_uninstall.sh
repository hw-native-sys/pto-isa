#!/bin/bash
# --------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# --------------------------------------------------------------------------------

OPERATE_FAILED="0x0001"
PARAM_INVALID="0x0002"
PARAM_INVALID_DES="Invalid input parameter."
FILE_NOT_EXIST="0x0080"
FILE_NOT_EXIST_DES="File not found."
FILE_READ_FAILED="0x0082"
FILE_READ_FAILED_DES="File read failed."
CURR_PATH=$(dirname $(readlink -f $0))
COMMON_INC_FILE="${CURR_PATH}/common_func.inc"
PTO_COMMON_FILE="${CURR_PATH}/pto_common.sh"

. "${COMMON_INC_FILE}"
. "${PTO_COMMON_FILE}"

ARCH_INFO=$(uname -m)
PTO_PLATFORM_DIR=pto_isa
PTO_PLATFORM_UPPER=$(echo "${PTO_PLATFORM_DIR}" | tr '[:lower:]' '[:upper:]')
FILELIST_FILE="${CURR_PATH}/filelist.csv"
COMMON_PARSER_FILE="${CURR_PATH}/install_common_parser.sh"
TARGET_INSTALL_PATH=""
TARGET_VERSION_DIR="${CURR_PATH}/../.."
TARGET_VERSION_DIR=$(readlink -f ${TARGET_VERSION_DIR})     # TARGET_INSTALL_PATH + PKG_VERSION_DIR
TARGET_MOULDE_DIR=${TARGET_VERSION_DIR}/${PTO_PLATFORM_DIR} # TARGET_INSTALL_PATH + PKG_VERSION_DIR + PTO_PLATFORM_DIR
ASCEND_INSTALL_INFO="ascend_install.info"
# init log file path
INSTALL_INFO_FILE="${TARGET_MOULDE_DIR}/${ASCEND_INSTALL_INFO}"

VERSION_INFO_FILE="${TARGET_MOULDE_DIR}/version.info"

# keys of infos in ascend_install.info
KEY_INSTALLED_UNAME="USERNAME"
KEY_INSTALLED_UGROUP="USERGROUP"
KEY_INSTALLED_TYPE="${PTO_PLATFORM_UPPER}_INSTALL_TYPE"
KEY_INSTALLED_FEATURE="${PTO_PLATFORM_UPPER}_INSTALL_FEATURE"
KEY_INSTALLED_PATH="${PTO_PLATFORM_UPPER}_INSTALL_PATH_VAL"
KEY_INSTALLED_VERSION="${PTO_PLATFORM_UPPER}_VERSION"

get_opts() {
  INSTALLED_PATH="$1"
  UNINSTALL_MODE="$2"
  IS_QUIET="$3"
  IN_FEATURE="$4"
  IS_DOCKER_INSTALL="$5"
  DOCKER_ROOT="$6"
  PKG_VERSION_DIR="$7"
  local parameter_num="$#"

  if [ "${parameter_num}" != 0 ]; then
    if [ "${INSTALLED_PATH}" = "" ] ||
      [ "${UNINSTALL_MODE}" = "" ] ||
      [ "${IS_QUIET}" = "" ]; then
      logandprint "[ERROR]: ERR_NO:${PARAM_INVALID};ERR_DES:Empty parameters is invalid\
for call uninstall functions."
      exit 1
    fi
  fi
}

get_docker_install_path() {
  local docker_root_tmp="$(echo "${DOCKER_ROOT}" | sed "s#/\+\$##g")"
  local docker_root_regex="$(echo "${docker_root_tmp}" | sed "s#\/#\\\/#g")"
  relative_path_val=$(echo "${TARGET_VERSION_DIR}" | sed "s/^${docker_root_regex}//g" | sed "s/\/\+\$//g")
  return
}

log_with_errorlevel() {
  local ret_status="$1"
  local level="$2"
  local msg="$3"
  if [ "${ret_status}" != 0 ]; then
    if [ "${level}" = "error" ]; then
      logandprint "${msg}"
      exit 1
    else
      logandprint "${msg}"
    fi
  fi
}

check_directory_exist() {
  local path="${1}"
  if [ ! -d "${path}" ]; then
    logandprint "[ERROR]: ERR_NO:${FILE_NOT_EXIST};ERR_DES:Installation directory [${path}] does not exist, uninstall failed."
    exit 1
  fi
}

check_file_exist() {
  local path_param="${1}"
  if [ ! -f "${path_param}" ]; then
    logandprint "[ERROR]: ERR_NO:${FILE_NOT_EXIST};ERR_DES:The file (${path_param}) does not existed."
    exit 1
  fi
}

check_installed_files() {
  # check install folder existed
  check_file_exist "${INSTALL_INFO_FILE}"

  check_file_exist "${FILELIST_FILE}"

  check_file_exist "${COMMON_PARSER_FILE}"

  check_directory_exist "${TARGET_MOULDE_DIR}"
}

check_installed_type() {
  local type="$1"
  if [ "${type}" != "run" ] &&
    [ "${type}" != "full" ] &&
    [ "${type}" != "devel" ]; then
    logandprint "[ERROR]: ERR_NO:${UNAME_NOT_EXIST};ERR_DES:Install type of pto module is not right!"
    exit 1
  fi
}

unsetenv() {
  logandprint "[INFO]: Unset the environment path [ export ASCEND_PTO_ISA_PATH=${relative_path_val}/${PTO_PLATFORM_DIR}]."
  if [ "${IS_DOCKER_INSTALL}" = y ]; then
    UNINSTALL_OPTION="--docker-root=${DOCKER_ROOT}"
  else
    UNINSTALL_OPTION=""
  fi
}

get_installed_info() {
  local key="$1"
  local res=""
  if [ -f "${INSTALL_INFO_FILE}" ]; then
    chmod 644 "${INSTALL_INFO_FILE}" >/dev/null 2>&1
    res=$(cat ${INSTALL_INFO_FILE} | grep "${key}" | awk -F = '{print $2}')
  fi
  echo "${res}"
}

get_installed_param() {
  INSTALLED_TYPE=$(get_installed_info "${KEY_INSTALLED_TYPE}")
  TARGET_USERNAME=$(get_installed_info "${KEY_INSTALLED_UNAME}")
  TARGET_USERGROUP=$(get_installed_info "${KEY_INSTALLED_UGROUP}")
  get_package_version "RUN_PKG_VERSION" "$VERSION_INFO_FILE"
  if [ "${PKG_VERSION_DIR}" = "" ]; then
    TARGET_INSTALL_PATH=${TARGET_VERSION_DIR}
  else
    TARGET_INSTALL_PATH=$(readlink -f "${TARGET_VERSION_DIR}/../../../")
  fi
}

remove_module() {
  chmod u+w ${TARGET_MOULDE_DIR}/scene.info

  logandprint "[INFO]: Delete the installed pto source files in (${TARGET_VERSION_DIR})."

  bash "${COMMON_PARSER_FILE}" --package="${PTO_PLATFORM_DIR}" --uninstall --remove-install-info \
    --username="${TARGET_USERNAME}" --usergroup="${TARGET_USERGROUP}" --version=$RUN_PKG_VERSION \
    --use-share-info --version-dir=$PKG_VERSION_DIR  ${UNINSTALL_OPTION} "${INSTALLED_TYPE}" "${TARGET_INSTALL_PATH}" \
    "${FILELIST_FILE}" "${IN_FEATURE}" --recreate-softlink
  log_with_errorlevel "$?" "error" "[ERROR]: ERR_NO:${OPERATE_FAILED};ERR_DES:Uninstall pto module failed."
}

remove_pto() {
  local ori_mod=$(stat -c %a ${TARGET_MOULDE_DIR})
  if [ "$(id -u)" != 0 ] && [ ! -w "${TARGET_MOULDE_DIR}" ]; then
    chmod u+w "${TARGET_MOULDE_DIR}" 2>/dev/null
  fi

  remove_module

  if [ "${UNINSTALL_MODE}" != "upgrade" ]; then
    logandprint "[INFO]: Delete the install info file (${INSTALL_INFO_FILE})."
    rm -f "${INSTALL_INFO_FILE}"
    log_with_errorlevel "$?" "warn" "[WARNING] Delete ops install info file failed, please delete it by yourself."
  fi
}

remote_all_soft_link() {
  local lib_dir=${TARGET_INSTALL_PATH}/cann/${ARCH_INFO}-linux/lib64/
  local ori_mod=$(stat -c %a ${lib_dir})
  if [ "$(id -u)" != 0 ] && [ ! -w "${lib_dir}" ]; then
    chmod u+w "${lib_dir}" 2>/dev/null
  fi

  local pto_isa_lib_files="opapi_math opgraph_math ophost_math common_math"
  for lib_name in ${pto_isa_lib_files}; do
    local so_name=${lib_dir}/lib${lib_name}.so
    remove_softlink "${so_name}"
  done

  chmod ${ori_mod} ${lib_dir}
  # remove aclnn_kernels
  local arch_include_dir=${TARGET_INSTALL_PATH}/cann/${ARCH_INFO}-linux/include
  ori_mod=$(stat -c %a ${arch_include_dir})
  if [ "$(id -u)" != 0 ] && [ ! -w "${arch_include_dir}" ]; then
    chmod u+w -R "${arch_include_dir}" 2>/dev/null
  fi
  [ -d ${arch_include_dir}/aclnn_kernels ] && rm -rf "${arch_include_dir}/aclnn_kernels"
  # remove all softlink
  find ${TARGET_INSTALL_PATH}/cann/ -type l -lname "*/${PTO_PLATFORM_DIR}/*" -delete
  rm -rf ${TARGET_INSTALL_PATH}/cann/${PTO_PLATFORM_DIR}
}

logandprint "[INFO]: Begin uninstall the pto module."

main() {
  get_opts "$@"

  get_docker_install_path

  check_installed_files

  get_installed_param

  check_installed_type "${INSTALLED_TYPE}"

  unsetenv

  remove_pto

 # remote_all_soft_link

  if [ "${UNINSTALL_MODE}" != "upgrade" ]; then
    remove_dir_if_empty ${TARGET_VERSION_DIR}/${PTO_PLATFORM_DIR}
  fi

  remove_dir_if_empty ${INSTALLED_PATH}

  logandprint "[INFO]: Pto package uninstalled successfully! Uninstallation takes effect immediately."
}

main "$@"
exit 0
