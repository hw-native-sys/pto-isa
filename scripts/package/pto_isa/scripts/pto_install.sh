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

PARAM_INVALID="0x0002"
INSTALL_FAILED="0x0000"
INSTALL_FAILED_DES="Update successfully."
FILE_NOT_EXIST="0x0080"
FILE_NOT_EXIST_DES="File not found."
FILE_READ_FAILED="0x0082"
FILE_READ_FAILED_DES="File read failed."
FILE_WRITE_FAILED="0x0081"
FILE_WRITE_FAILED_DES="File write failed."
PERM_DENIED="0x0093"
PERM_DENIED_DES="Permission denied."

# run package's files info
CURR_PATH=$(dirname $(readlink -f $0))
VERSION_INFO_FILE="${CURR_PATH}/../version.info"
FILELIST_FILE="${CURR_PATH}/filelist.csv"
COMMON_PARSER_FILE="${CURR_PATH}/install_common_parser.sh"
SCENE_FILE="${CURR_PATH}/../scene.info"
ASCEND_INSTALL_INFO="ascend_install.info"

ARCH_INFO=$(uname -m)
PTO_PLATFORM_DIR=pto_isa
PTO_PLATFORM_UPPER=$(echo "${PTO_PLATFORM_DIR}" | tr '[:lower:]' '[:upper:]')

TARGET_INSTALL_PATH=""
TARGET_MOULDE_DIR=""  # TARGET_INSTALL_PATH + PKG_VERSION_DIR + PTO_PLATFORM_DIR
TARGET_VERSION_DIR="" # TARGET_INSTALL_PATH + PKG_VERSION_DIR

COMMON_INC_FILE="${CURR_PATH}/common_func.inc"
COMMON_FUNC_V2_PATH="${CURR_PATH}/common_func_v2.inc"
VERSION_CFG="${CURR_PATH}/version_cfg.inc"
PTO_COMMON_FILE="${CURR_PATH}/pto_common.sh"

. "${COMMON_INC_FILE}"
. "${COMMON_FUNC_V2_PATH}"
. "${VERSION_CFG}"
. "${PTO_COMMON_FILE}"

# keys of infos in ascend_install.info
KEY_INSTALLED_UNAME="USERNAME"
KEY_INSTALLED_UGROUP="USERGROUP"
KEY_INSTALLED_TYPE="${PTO_PLATFORM_UPPER}_INSTALL_TYPE"
KEY_INSTALLED_FEATURE="${PTO_PLATFORM_UPPER}_INSTALL_FEATURE"
KEY_INSTALLED_CHIP="${PTO_PLATFORM_UPPER}_INSTALL_CHIP"
KEY_INSTALLED_PATH="${PTO_PLATFORM_UPPER}_INSTALL_PATH_VAL"
KEY_INSTALLED_VERSION="${PTO_PLATFORM_UPPER}_VERSION"

get_opts() {
  TARGET_INSTALL_PATH="$1"
  TARGET_USERNAME="$2"
  TARGET_USERGROUP="$3"
  IN_FEATURE="$4"
  INSTALL_TYPE="$5"
  IS_FOR_ALL="$6"
  IS_SETENV="$7"
  IS_DOCKER_INSTALL="$8"
  DOCKER_ROOT="$9"
  PKG_VERSION_DIR="${10}"

  if [ "${TARGET_INSTALL_PATH}" = "" ] || [ "${TARGET_USERNAME}" = "" ] ||
    [ "${TARGET_USERGROUP}" = "" ] || [ "${INSTALL_TYPE}" = "" ]; then
    logandprint "[ERROR]: ERR_NO:${PARAM_INVALID};ERR_DES:Empty paramters is invalid for install."
    exit 1
  fi

  INSTALL_FOR_ALL=""
  if [ "${IS_FOR_ALL}" = "y" ]; then
    INSTALL_FOR_ALL="--install_for_all"
  fi
}

init_install_env() {
  get_package_version "RUN_PKG_VERSION" "$VERSION_INFO_FILE"
  if [ "${PKG_VERSION_DIR}" = "" ]; then
    TARGET_VERSION_DIR=${TARGET_INSTALL_PATH}
  else
    TARGET_VERSION_DIR=${TARGET_INSTALL_PATH}/${PKG_VERSION_DIR}
  fi
  TARGET_MOULDE_DIR=${TARGET_VERSION_DIR}/share/info/${PTO_PLATFORM_DIR}
  INSTALL_INFO_FILE=${TARGET_MOULDE_DIR}/${ASCEND_INSTALL_INFO}

  if [ "$(id -u)" != "0" ]; then
    LOG_PATH_PERM="740"
    LOG_FILE_PERM="640"
    INSTALL_INFO_PERM="600"
  else
    LOG_PATH_PERM="750"
    LOG_FILE_PERM="640"
    INSTALL_INFO_PERM="644"
  fi

  if [ "${IS_FOR_ALL}" = "y" ]; then
    BUILTIN_PERM="555"
    CUSTOM_PERM="755"
    CREATE_DIR_PERM="755"
    ONLYREAD_PERM="444"
  else
    BUILTIN_PERM="550"
    CUSTOM_PERM="750"
    CREATE_DIR_PERM="750"
    ONLYREAD_PERM="440"
  fi
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

get_installed_info() {
  local key="$1"
  local res=""
  if [ -f "${INSTALL_INFO_FILE}" ]; then
    res=$(cat ${INSTALL_INFO_FILE} | grep "${key}" | awk -F = '{print $2}')
  fi
  echo "${res}"
}

update_install_info() {
  local key_val="$1"
  local val="$2"
  local old_val=$(get_installed_info "${key_val}")
  if [ -f "${INSTALL_INFO_FILE}" ]; then
    if [ "x${old_val}" = "x" ]; then
      echo "${key_val}=${val}" >>"${INSTALL_INFO_FILE}"
    else
      sed -i "/${key_val}/c ${key_val}=${val}" "${INSTALL_INFO_FILE}"
    fi
  else
    echo "${key_val}=${val}" >"${INSTALL_INFO_FILE}"
  fi
}

update_install_infos() {
  local uname="$1"
  local ugroup="$2"
  local type="$3"
  local path="$4"
  local version
  get_package_version "version" "$VERSION_INFO_FILE"
  comm_create_file "${INSTALL_INFO_FILE}" "${INSTALL_INFO_PERM}" "${TARGET_USERNAME}:${TARGET_USERGROUP}" "${IS_FOR_ALL}"

  update_install_info "${KEY_INSTALLED_UNAME}" "${uname}"
  update_install_info "${KEY_INSTALLED_UGROUP}" "${ugroup}"
  update_install_info "${KEY_INSTALLED_TYPE}" "${type}"
  update_install_info "${KEY_INSTALLED_PATH}" "${path}"
  update_install_info "${KEY_INSTALLED_VERSION}" "${version}"
}

check_file_exist() {
  local path_param="${1}"
  if [ ! -f "${path_param}" ]; then
    logandprint "[ERROR]: ERR_NO:${FILE_NOT_EXIST};ERR_DES:The file (${path_param}) does not existed."
    exit 1
  fi
}

check_env() {
  check_file_exist "${FILELIST_FILE}"
  check_file_exist "${COMMON_PARSER_FILE}"
}

createsoftlink() {
  local src_path="$1"
  local dst_path="$2"
  if [ -e "$dst_path" ]; then
    if [ -L "$dst_path" ]; then
      `rm -f $dst_path`
    else
      return 0
    fi
  fi
  ln -s "${src_path}" "${dst_path}" 2>/dev/null
  log_with_errorlevel "$?" "error" "[ERROR]: ERR_NO:${PERM_DENIED};ERR_DES:${src_path} Create softlink to  ${dst_path} failed."
}

get_install_path() {
  docker_root_tmp="$(echo "${DOCKER_ROOT}" | sed "s#/\+\$##g")"
  docker_root_regex="$(echo "${docker_root_tmp}" | sed "s#\/#\\\/#g")"
  relative_path_val=$(echo "${TARGET_VERSION_DIR}" | sed "s/^${docker_root_regex}//g" | sed "s/\/\+\$//g")
  return
}

setenv() {
  logandprint "[INFO]: Set the environment path [ export ASCEND_PTO_ISA_PATH=${relative_path_val}/${PTO_PLATFORM_DIR} ]."
  if [ "${IS_DOCKER_INSTALL}" = y ]; then
    INSTALL_OPTION="--docker-root=${DOCKER_ROOT}"
  else
    INSTALL_OPTION=""
  fi
  if [ "${IS_SETENV}" = "y" ]; then
    INSTALL_OPTION="${INSTALL_OPTION} --setenv"
  fi
}

# 创建单个文件的软链接，链接文件级别
create_file_softlink() {
  local src_file=$1
  local dst_file=$2
  local base_dir=$(dirname ${dst_file})

  comm_create_dir "${base_dir}" "${CREATE_DIR_PERM}" "${TARGET_USERNAME}:${TARGET_USERGROUP}" "${IS_FOR_ALL}"

  local relative_file_path=$(realpath -s --relative-to="${base_dir}" "${src_file}")
  # 创建软连接
 # createsoftlink "${relative_file_path}" "${dst_file}"
}

# 创建单个目录的软连接，链接目录级别
create_dir_softlink() {
  local src_dir=$1
  local dst_dir=$2
  local base_dir=$(dirname ${dst_dir})

  comm_create_dir "${base_dir}" "${CREATE_DIR_PERM}" "${TARGET_USERNAME}:${TARGET_USERGROUP}" "${IS_FOR_ALL}"

  if [ ! -d "${src_dir}" ]; then
    logandprint "[ERROR]: src dir ["${src_dir}"] not exists to create soft link."
  fi

  # 获取相对路径
  relative_dir_path=$(realpath -s --relative-to="${base_dir}" "${src_dir}")
  # 创建软连接
 # createsoftlink "${relative_dir_path}" "${dst_dir}"
}

# 创建目录下子目录的软连接，链接子目录级别
create_softlink_for_dirs() {
  local src_dir=$1
  local dst_dir=$2

  if [ ! -d "${src_dir}" ]; then
    logandprint "[ERROR]: src dir ["${src_dir}"] not exists to create soft link."
    exit 1
  fi

  comm_create_dir "${dst_dir}" "${CREATE_DIR_PERM}" "${TARGET_USERNAME}:${TARGET_USERGROUP}" "${IS_FOR_ALL}"

  find "${src_dir}" -mindepth 1 -maxdepth 1 -type d -print0 | while IFS= read -r -d '' src_dir_path; do
    local sub_dir_name=$(basename ${src_dir_path})
    local dst_dir_path="${dst_dir}/${sub_dir_name}"
    # 计算目录相对路径
    local relative_dir_path=$(realpath -s --relative-to="${dst_dir}" "${src_dir_path}")
    # 创建软连接
 #   createsoftlink "${relative_dir_path}" "${dst_dir_path}"
  done
}

# 创建某目录下所有文件的软链接，链接文件级别，要求目录中不能有子目录
create_softlink_for_files() {
  local src_dir=$1
  local dst_dir=$2
  local exclude_list="$3"

  if [ ! -d "${src_dir}" ]; then
    logandprint "[ERROR]: src dir ["${src_dir}"] not exists, cannot create soft link."
    exit 1
  fi

  comm_create_dir "${dst_dir}" "${CREATE_DIR_PERM}" "${TARGET_USERNAME}:${TARGET_USERGROUP}" "${IS_FOR_ALL}"

  find "${src_dir}" -mindepth 1 -maxdepth 1 -type f -print0 | while IFS= read -r -d '' src_file_path; do
    # 文件名
    local file_name=$(basename ${src_file_path})
    if $(echo ${exclude_list} | grep -wq ${file_name}); then
      continue
    fi
    local dst_file_path="${dst_dir}/${file_name}"
    # 获取相对路径
    local relative_file_path=$(realpath --relative-to="${dst_dir}" "${src_file_path}")
    # 创建软连接
  #  createsoftlink "${relative_file_path}" "${dst_file_path}"
  done
}

# 递归创建目录下所有文件的软链接，链接文件级别，目录中可以有子目录，对于子目录会创建对应目录不是链接
create_softlink_for_files_and_dirs() {
  local src_dir=$1
  local dst_dir=$2

  if [ ! -d "${src_dir}" ]; then
    logandprint "[ERROR]: src dir ["${src_dir}"] not exists, cannot create soft link."
    exit 1
  fi

  comm_create_dir "${dst_dir}" "${CREATE_DIR_PERM}" "${TARGET_USERNAME}:${TARGET_USERGROUP}" "${IS_FOR_ALL}"
  find "${src_dir}" -mindepth 1 -maxdepth 1 -type d -print0 | while IFS= read -r -d '' src_dir_path; do
    local base_dir=$(basename ${src_dir_path})
    local dst_dir_path="${dst_dir}/${base_dir}"
  done
}

#create softlink for TARGET_MOULDE_DIR include
create_module_include_softlink() {
  local dir_mode=""
  local dst_path=${TARGET_MOULDE_DIR}/include
  if [ -d "${dst_path}" ]; then
    dir_mode=$(stat -c %a ${dst_path})
    if [ "$(id -u)" != 0 ] && [ ! -w "${dir_mode}" ]; then
      chmod u+w "${dst_path}" 2>/dev/null
    fi
  fi
  comm_create_dir "${dst_path}" "${CREATE_DIR_PERM}" "${TARGET_USERNAME}:${TARGET_USERGROUP}" "${IS_FOR_ALL}"

  if [ -n "$dir_mode" ]; then
    chmod ${dir_mode} ${dst_path} 2>/dev/null
  fi
}

#create cann [x86-64|aarch64]/lib64
create_arch_lib_softlink() {
  local dir_mode=""
  local dst_lib_path="${TARGET_INSTALL_PATH}/cann/${ARCH_INFO}-linux/lib64"
  if [ -d "${dst_lib_path}" ]; then
    dir_mode=$(stat -c %a ${dst_lib_path})
    if [ "$(id -u)" != 0 ] && [ ! -w "${dir_mode}" ]; then
      chmod u+w "${dst_lib_path}" 2>/dev/null
    fi
  fi

  if [ -n "$dir_mode" ]; then
    chmod ${dir_mode} ${dst_lib_path} 2>/dev/null
  fi

  dir_mode=""
  local dst_inc_path="${TARGET_INSTALL_PATH}/cann/${ARCH_INFO}-linux/include"
  if [ -d "${dst_inc_path}" ]; then
    dir_mode=$(stat -c %a ${dst_inc_path})
    if [ "$(id -u)" != 0 ] && [ ! -w "${dir_mode}" ]; then
      chmod u+w "${dst_inc_path}" 2>/dev/null
    fi
  fi

  if [ -n "$dir_mode" ]; then
    chmod ${dir_mode} ${dst_inc_path} 2>/dev/null
  fi
}

create_opgraph_softlink() {
  local dir_mode=""
  local dst_path=${TARGET_INSTALL_PATH}/cann/pto/built-in/op_graph
  if [ -d "${dst_path}" ]; then
    dir_mode=$(stat -c %a ${dst_path})
    if [ "$(id -u)" != 0 ] && [ ! -w "${dir_mode}" ]; then
      chmod u+w "${dst_path}" 2>/dev/null
    fi
  fi
  comm_create_dir "${dst_path}" "${CREATE_DIR_PERM}" "${TARGET_USERNAME}:${TARGET_USERGROUP}" "${IS_FOR_ALL}"

  if [ -n "$dir_mode" ]; then
    chmod ${dir_mode} ${dst_path} 2>/dev/null
  fi
}

create_tbe_kernel_softlink() {
  local kernel_src_path=${TARGET_INSTALL_PATH}/cann/${PTO_PLATFORM_DIR}/built-in/op_impl/ai_core/tbe/kernel
  if [ ! -d ${kernel_src_path} ]; then
    logandprint "[INFO]: Package with no pto kernel, no need to install kernel."
    return 0
  fi
  local dir_mode=""
  local dst_path=${TARGET_INSTALL_PATH}/cann/pto_isa/built-in/op_impl/ai_core/tbe/kernel
  if [ -d "${dst_path}" ]; then
    dir_mode=$(stat -c %a ${dst_path})
    if [ "$(id -u)" != 0 ] && [ ! -w "${dir_mode}" ]; then
      chmod u+w "${dst_path}" 2>/dev/null
    fi
  fi
  comm_create_dir "${dst_path}" "${CREATE_DIR_PERM}" "${TARGET_USERNAME}:${TARGET_USERGROUP}" "${IS_FOR_ALL}"

  local kernel_dirs=$(ls ${TARGET_INSTALL_PATH}/cann/${PTO_PLATFORM_DIR}/built-in/op_impl/ai_core/tbe/kernel)
  for dir in ${kernel_dirs}; do
    local kernel_src_dir=${kernel_src_path}/${dir}
    local kernel_dst_dir_math=${dst_path}/${dir}/${PTO_PLATFORM_DIR}
    local kernel_dst_dir=${dst_path}/${dir}
    if [[ ${dir} =~ ascend* ]]; then
      echo ""
      #create_softlink_for_dirs "${kernel_src_dir}" "${kernel_dst_dir}"
    else
      local kernel_socs=$(ls ${kernel_src_dir})
      for soc in ${kernel_socs}; do
        soc_src_path=${TARGET_INSTALL_PATH}/cann/${PTO_PLATFORM_DIR}/built-in/op_impl/ai_core/tbe/kernel/${dir}/${soc}

        local json_files_src_path=${kernel_src_path}/${dir}/${soc}
        local json_files_dst_path=${dst_path}/${dir}/${soc}/
        local json_files_dst_path_math=${dst_path}/${dir}/${soc}/${PTO_PLATFORM_DIR}

        local binary_json_src_path=${json_files_src_path}/binary_info_config.json
        local binary_json_dst_path=${json_files_dst_path}/binary_info_config.json
        if [ -L ${binary_json_dst_path} ]; then
          rm -f ${binary_json_dst_path}
          cp ${binary_json_src_path} ${binary_json_dst_path}
        elif [ ! -e ${binary_json_dst_path} ]; then
          cp ${binary_json_src_path} ${binary_json_dst_path}
        else
          local binary_json_mode=$(stat -c %a ${binary_json_dst_path})
          if [ "$(id -u)" != 0 ] && [ ! -w "${binary_json_dst_path}" ]; then
            chmod u+w "${binary_json_dst_path}" 2>/dev/null
          fi
          python3 ${CURR_PATH}/merge_binary_info_config.py --base-file ${binary_json_dst_path} \
            --update-file ${binary_json_src_path} --output-file ${binary_json_dst_path}
          if [ $? -ne 0 ]; then
            log_with_errorlevel "$?" "error" "[ERROR]: ERR_NO:${INSTALL_FAILED};ERR_DES:Update binary_info_config info failed."
            exit 1
          fi
          chmod ${binary_json_mode} ${binary_json_dst_path} 2>/dev/null
        fi
      done
    fi
  done

  if [ -n "$dir_mode" ]; then
    chmod ${dir_mode} ${dst_path} 2>/dev/null
  fi
}

create_tbe_impl_softlink() {
  local dir_mode=""
  local dst_path=${TARGET_INSTALL_PATH}/cann/pto_isa/built-in/op_impl/ai_core/tbe/impl
  if [ -d "${dst_path}" ]; then
    dir_mode=$(stat -c %a ${dst_path})
    if [ "$(id -u)" != 0 ] && [ ! -w "${dir_mode}" ]; then
      chmod u+w "${dst_path}" 2>/dev/null
    fi
  fi

  local impl_src=${TARGET_INSTALL_PATH}/cann/${PTO_PLATFORM_DIR}/built-in/op_impl/ai_core/tbe/impl

  ascendc_src_dir=${impl_src}/ascendc
  ascendc_dst_dir=${dst_path}/ascendc

  dynamic_src_dir=${impl_src}/dynamic
  dynamic_dst_dir=${dst_path}/dynamic

  if [ -n "$dir_mode" ]; then
    chmod ${dir_mode} ${dst_path} 2>/dev/null
  fi
}

create_tbe_softlink() {
  local dir_mode=""
  local dst_path=${TARGET_INSTALL_PATH}/cann/pto_isa/built-in/op_impl/ai_core/tbe
  if [ -d "${dst_path}" ]; then
    dir_mode=$(stat -c %a ${dst_path})
    if [ "$(id -u)" != 0 ] && [ ! -w "${dir_mode}" ]; then
      chmod u+w "${dst_path}" 2>/dev/null
    fi
  fi
  comm_create_dir "${dst_path}" "${CREATE_DIR_PERM}" "${TARGET_USERNAME}:${TARGET_USERGROUP}" "${IS_FOR_ALL}"

  create_tbe_impl_softlink

  create_tbe_kernel_softlink

  if [ -n "$dir_mode" ]; then
    chmod ${dir_mode} ${dst_path} 2>/dev/null
  fi
}

#create softlink cann/pto_isa/built-in
create_latest_builtin_softlink() {
  local dir_mode=""
  local dst_path=${TARGET_INSTALL_PATH}/cann/pto_isa/built-in
  if [ -d "${dst_path}" ]; then
    dir_mode=$(stat -c %a ${dst_path})
    if [ "$(id -u)" != 0 ] && [ ! -w "${dir_mode}" ]; then
      chmod u+w "${dst_path}" 2>/dev/null
    fi
  fi
  comm_create_dir "${dst_path}" "${CREATE_DIR_PERM}" "${TARGET_USERNAME}:${TARGET_USERGROUP}" "${IS_FOR_ALL}"

  create_opgraph_softlink

  create_tbe_softlink

  if [ -n "$dir_mode" ]; then
    chmod ${dir_mode} ${dst_path} 2>/dev/null
  fi
}

#create softlink cann/pto_isa/lib64
create_latest_lib_softlink() {
  local dir_mode=""
  local dst_path=${TARGET_INSTALL_PATH}/cann/pto_isa/lib64
  if [ -d "${dst_path}" ]; then
    dir_mode=$(stat -c %a ${dst_path})
    if [ "$(id -u)" != 0 ] && [ ! -w "${dir_mode}" ]; then
      chmod u+w "${dst_path}" 2>/dev/null
    fi
  fi
  comm_create_dir "${dst_path}" "${CREATE_DIR_PERM}" "${TARGET_USERNAME}:${TARGET_USERGROUP}" "${IS_FOR_ALL}"

  if [ -n "$dir_mode" ]; then
    chmod ${dir_mode} ${dst_path} 2>/dev/null
  fi
}

#create softlink cann/pto_isa/include
create_latest_include_softlink() {
  local dir_mode=""
  local dst_path=${TARGET_INSTALL_PATH}/cann/pto_isa/include
  if [ -d "${dst_path}" ]; then
    dir_mode=$(stat -c %a ${dst_path})
    if [ "$(id -u)" != 0 ] && [ ! -w "${dir_mode}" ]; then
      chmod u+w "${dst_path}" 2>/dev/null
    fi
  fi
  comm_create_dir "${dst_path}" "${CREATE_DIR_PERM}" "${TARGET_USERNAME}:${TARGET_USERGROUP}" "${IS_FOR_ALL}"

  local pto_aclnnop_src_dir=${TARGET_INSTALL_PATH}/cann/${PTO_PLATFORM_DIR}/built-in/op_impl/ai_core/tbe/op_api/include/aclnnop
  local pto_aclnnop_dst_dir=${dst_path}/aclnnop
  local pto_aclnnop_dst_dir_math=${dst_path}/aclnnop/${PTO_PLATFORM_DIR}

  if [ -n "$dir_mode" ]; then
    chmod ${dir_mode} ${dst_path} 2>/dev/null
  fi
}

#create softlink cann/pto_isa
create_latest_pto_softlink() {
  local dir_mode=""
  local dst_path=${TARGET_INSTALL_PATH}/cann/pto_isa/
  if [ -d "${dst_path}" ]; then
    dir_mode=$(stat -c %a ${dst_path})
    if [ "$(id -u)" != 0 ] && [ ! -w "${dir_mode}" ]; then
      chmod u+w "${dst_path}" 2>/dev/null
    fi
  fi
  comm_create_dir "${dst_path}" "${CREATE_DIR_PERM}" "${TARGET_USERNAME}:${TARGET_USERGROUP}" "${IS_FOR_ALL}"

  create_latest_lib_softlink

  create_latest_include_softlink

  create_latest_builtin_softlink

  if [ -n "$dir_mode" ]; then
    chmod ${dir_mode} ${dst_path} 2>/dev/null
  fi
}

#create cann [x86-64|aarch64]/include
create_arch_include_softlink() {
  local dir_mode=""
  local dst_path=${TARGET_INSTALL_PATH}/cann/${ARCH_INFO}-linux/include
  if [ -d "${dst_path}" ]; then
    dir_mode=$(stat -c %a ${dst_path})
    if [ "$(id -u)" != 0 ] && [ ! -w "${dir_mode}" ]; then
      chmod u+w "${dst_path}" 2>/dev/null
    fi
  fi
  comm_create_dir "${dst_path}" "${CREATE_DIR_PERM}" "${TARGET_USERNAME}:${TARGET_USERGROUP}" "${IS_FOR_ALL}"

  if [ -n "$dir_mode" ]; then
    chmod ${dir_mode} ${dst_path} 2>/dev/null
  fi
}

create_latest_softlink() {
  local dir_mode=""
  local dst_path=${TARGET_INSTALL_PATH}/cann
  if [ -d "${dst_path}" ]; then
    dir_mode=$(stat -c %a ${dst_path})
    if [ "$(id -u)" != 0 ] && [ ! -w "${dir_mode}" ]; then
      chmod u+w "${dst_path}" 2>/dev/null
    fi
  fi
  comm_create_dir "${dst_path}" "${CREATE_DIR_PERM}" "${TARGET_USERNAME}:${TARGET_USERGROUP}" "${IS_FOR_ALL}"

  if [ -n "$dir_mode" ]; then
    chmod ${dir_mode} ${dst_path} 2>/dev/null
  fi
}

install_pto() {
  logandprint "[INFO]: Begin install pto module."
  local version_mod=""
  local module_mod=""
  if [ -d ${TARGET_VERSION_DIR} ]; then
    version_mod=$(stat -c %a ${TARGET_VERSION_DIR})
    if [ "$(id -u)" != 0 ] && [ ! -w "${TARGET_VERSION_DIR}" ]; then
      chmod u+w "${TARGET_VERSION_DIR}" 2>/dev/null
    fi
  fi
  if [ -d ${TARGET_MOULDE_DIR} ]; then
    module_mod=$(stat -c %a ${TARGET_MOULDE_DIR})
    if [ "$(id -u)" != 0 ] && [ ! -w "${TARGET_MOULDE_DIR}" ]; then
      chmod u+w "${TARGET_MOULDE_DIR}" 2>/dev/null
    fi
  fi
  comm_create_dir "${TARGET_VERSION_DIR}" "${CREATE_DIR_PERM}" "${TARGET_USERNAME}:${TARGET_USERGROUP}" "${IS_FOR_ALL}"
  comm_create_dir "${TARGET_MOULDE_DIR}" "${CREATE_DIR_PERM}" "${TARGET_USERNAME}:${TARGET_USERGROUP}" "${IS_FOR_ALL}"

  setenv

  logandprint "[INFO]: Update the pto install info."

  update_install_infos "${TARGET_USERNAME}" "${TARGET_USERGROUP}" "${INSTALL_TYPE}" "${relative_path_val}"
  log_with_errorlevel "$?" "error" "[ERROR]: ERR_NO:${INSTALL_FAILED};ERR_DES:Update pto install info failed."

  bash "${COMMON_PARSER_FILE}" --copy_all --package="${PTO_PLATFORM_DIR}" --install --username="${TARGET_USERNAME}" \
    --usergroup="${TARGET_USERGROUP}" --set-cann-uninstall --version=$RUN_PKG_VERSION \
    --version-dir=$PKG_VERSION_DIR --use-share-info $INSTALL_OPTION ${INSTALL_FOR_ALL} "--feature=all" "--chip=all" \
    "${INSTALL_TYPE}" "${TARGET_INSTALL_PATH}" "${FILELIST_FILE}"
  log_with_errorlevel "$?" "error" "[ERROR]: ERR_NO:${INSTALL_FAILED};ERR_DES:Install pto module files failed."

  logandprint "[INFO]: upgradePercentage:30%"

  if [ -n "${module_mod}" ]; then
    chmod ${module_mod} "${TARGET_MOULDE_DIR}" 2>/dev/null
  fi
  if [ -n "${version_mod}" ]; then
    chmod ${version_mod} "${TARGET_VERSION_DIR}" 2>/dev/null
  fi

  logandprint "[INFO]: upgradePercentage:50%"
}

main() {
  logandprint "[INFO]: Command pto_install"

  get_opts "$@"

  init_install_env

  get_package_upgrade_version_dir "upgrade_version_dir" "$TARGET_INSTALL_PATH" "${PTO_PLATFORM_DIR}"
  get_package_last_installed_version "last_installed" "$TARGET_INSTALL_PATH" "${PTO_PLATFORM_DIR}"
  last_installed_version=$(echo ${last_installed} | cut --only-delimited -d":" -f2-)

  get_install_path

  check_env

  install_pto

  #chmod to support copy
  if [ -d "${TARGET_MOULDE_DIR}/vendors" ] && [ "$(id -u)" != "0" ]; then
    chmod -R "${CUSTOM_PERM}" ${TARGET_MOULDE_DIR}/vendors
  fi

  # change log dir and file owner and rights
  chmod "${LOG_PATH_PERM}" "${COMM_LOG_DIR}" 2>/dev/null
  chmod "${LOG_FILE_PERM}" "${COMM_LOGFILE}" 2>/dev/null
  chmod "${LOG_FILE_PERM}" "${COMM_OPERATION_LOGFILE}" 2>/dev/null

  # change installed folder's permission except aicpu
  chmod -R "${BUILTIN_PERM}" "${TARGET_MOULDE_DIR}/built-in/op_impl/ai_core/tbe/op_tiling/lib" 2>/dev/null
  chmod -R "${BUILTIN_PERM}" "${TARGET_MOULDE_DIR}/built-in/op_proto/lib" 2>/dev/null

  if [ "$(id -u)" = "0" ]; then
    chmod "755" "${TARGET_MOULDE_DIR}" 2>/dev/null
  else
    chmod "${BUILTIN_PERM}" "${TARGET_MOULDE_DIR}" 2>/dev/null
  fi

  chmod "${ONLYREAD_PERM}" "${TARGET_MOULDE_DIR}/scene.info" 2>/dev/null
  chmod "${ONLYREAD_PERM}" "${TARGET_MOULDE_DIR}/version.info" 2>/dev/null
  chmod "${ONLYREAD_PERM}" "${TARGET_MOULDE_DIR}/ascend_install.info" 2>/dev/null

  # change installed folder's owner and group except aicpu
  chown "${TARGET_USERNAME}":"${TARGET_USERGROUP}" "${TARGET_MOULDE_DIR}" 2>/dev/null
  log_with_errorlevel "$?" "error" "[ERROR]: ERR_NO:${INSTALL_FAILED};ERR_DES:Change pto onwership failed.."

  logandprint "[INFO]: upgradePercentage:100%"

  logandprint "[INFO]: Installation information listed below:"
  logandprint "[INFO]: Install path: (${TARGET_MOULDE_DIR})"
  logandprint "[INFO]: Install log file path: (${COMM_LOGFILE})"
  logandprint "[INFO]: Operation log file path: (${COMM_OPERATION_LOGFILE})"

  if [ "${IS_SETENV}" != "y" ]; then
    logandprint "[INFO]: Using requirements: when pto module install finished or \
 before you run the pto module, execute the command \
 [ export ASCEND_PTO_ISA_PATH=${TARGET_INSTALL_PATH}/cann/${PTO_PLATFORM_DIR} ] to set the environment path."
  fi

  logandprint "[INFO]: Pto package installed successfully! The new version takes effect immediately."
}

main "$@"
exit 0
