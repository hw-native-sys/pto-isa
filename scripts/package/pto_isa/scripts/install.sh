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
# error number and description
OPERATE_FAILED="0x0001"
PARAM_INVALID="0x0002"
FILE_NOT_EXIST="0x0080"
FILE_NOT_EXIST_DES="File not found."
PTO_COMPATIBILITY_CHECK_ERR="0x0092"
PTO_COMPATIBILITY_CHECK_ERR_DES="PtoMath compatibility check error."
PERM_DENIED="0x0093"
PERM_DENIED_DES="Permission denied."

PTO_PLATFORM_DIR=pto_isa
PTO_PLATFORM_UPPER=$(echo "${PTO_PLATFORM_DIR}" | tr '[:lower:]' '[:upper:]')
CURR_OPERATE_USER="$(id -nu 2>/dev/null)"
CURR_OPERATE_GROUP="$(id -ng 2>/dev/null)"
# defaults for general user
if [ "$(id -u)" != "0" ]; then
  DEFAULT_INSTALL_PATH="${HOME}/Ascend"
else
  IS_FOR_ALL="y"
  DEFAULT_INSTALL_PATH="/usr/local/Ascend"
fi

# run package's files info, CURR_PATH means current temp path
CURR_PATH=$(dirname $(readlink -f $0))
INSTALL_SHELL_FILE="${CURR_PATH}/pto_install.sh"
RUN_PKG_INFO_FILE="${CURR_PATH}/../scene.info"
VERSION_INFO_FILE="${CURR_PATH}/../version.info"
COMMON_INC_FILE="${CURR_PATH}/common_func.inc"
VERCHECK_FILE="${CURR_PATH}/ver_check.sh"
VERSION_COMPAT_FUNC_PATH="${CURR_PATH}/version_compatiable.inc"
COMMON_FUNC_V2_PATH="${CURR_PATH}/common_func_v2.inc"
VERSION_CFG_PATH="${CURR_PATH}/version_cfg.inc"
PTO_COMMON_FILE="${CURR_PATH}/pto_common.sh"

. "${VERSION_COMPAT_FUNC_PATH}"
. "${COMMON_INC_FILE}"
. "${COMMON_FUNC_V2_PATH}"
. "${VERSION_CFG_PATH}"
. "${PTO_COMMON_FILE}"

ARCH_INFO=$(grep -e "arch" "$RUN_PKG_INFO_FILE" | cut --only-delimited -d"=" -f2-)

# defaluts info determined by user's inputs
ASCEND_INSTALL_INFO="ascend_install.info"
TARGET_INSTALL_PATH="${DEFAULT_INSTALL_PATH}" #--input-path
TARGET_USERNAME="${CURR_OPERATE_USER}"
TARGET_USERGROUP="${CURR_OPERATE_GROUP}"
TARGET_MOULDE_DIR=""  # TARGET_INSTALL_PATH + PKG_VERSION_DIR + PTO_PLATFORM_DIR
TARGET_VERSION_DIR="" # TARGET_INSTALL_PATH + PKG_VERSION_DIR

# keys of infos in ascend_install.info
KEY_INSTALLED_UNAME="USERNAME"
KEY_INSTALLED_UGROUP="USERGROUP"
KEY_INSTALLED_TYPE="${PTO_PLATFORM_UPPER}_INSTALL_TYPE"
KEY_INSTALLED_PATH="${PTO_PLATFORM_UPPER}_INSTALL_PATH_VAL"
KEY_INSTALLED_VERSION="${PTO_PLATFORM_UPPER}_VERSION"
KEY_INSTALLED_FEATURE="${PTO_PLATFORM_UPPER}_INSTALL_FEATURE"
KEY_INSTALLED_CHIP="${PTO_PLATFORM_UPPER}_INSTALL_CHIP"

# keys of infos in run package
KEY_RUNPKG_VERSION="Version"

# init install cmd status, set default as n
CMD_LIST="$*"
IS_UNINSTALL=n
IS_INSTALL=n
IS_UPGRADE=n
IS_QUIET=n
IS_INPUT_PATH=n
IS_CHECK=n
IS_PRE_CHECK=n
IN_INSTALL_TYPE=""
IN_INSTALL_PATH=""
IS_DOCKER_INSTALL=n
IS_SETENV=n
IS_JIT=n
DOCKER_ROOT=""
CONFLICT_CMD_NUMS=0
IN_FEATURE="All"

# log functions
# start info before shell executing
startlog() {
  echo "[PTO] [$(getdate)] [INFO]: Start Time: $(getdate)"
}

exitlog() {
  echo "[PTO] [$(getdate)] [INFO]: End Time: $(getdate)"
}

#check ascend_install.info for the change in code warning
get_installed_info() {
  local key="$1"
  local res=""
  if [ -f "${INSTALL_INFO_FILE}" ]; then
    chmod 644 "${INSTALL_INFO_FILE}" >/dev/null 2>&1
    res=$(cat ${INSTALL_INFO_FILE} | grep "${key}" | awk -F = '{print $2}')
  fi
  echo "${res}"
}

clean_before_reinstall() {
  local installed_path=$(get_installed_info "${KEY_INSTALLED_PATH}")
  local existed_files=$(find ${TARGET_MOULDE_DIR} -type f -print 2>/dev/null)
  if [ -z "${existed_files}" ]; then
    logandprint "[INFO]: Directory is empty, directly install pto module."
    return 0
  fi

  if [ "${IS_QUIET}" = "y" ]; then
    logandprint "[WARNING]: Directory has file existed or installed pto\
 module, are you sure to keep installing pto module in it? y"
  else
    if [ ! -f "${TARGET_MOULDE_DIR}/ascend_install.info" ]; then
      logandprint "[INFO]: Directory has file existed, do you want to continue? [y/n]"
    else
      logandprint "[INFO]: Pto package has been installed on the path $(get_installed_info "${KEY_INSTALLED_PATH}"),\
 the version is $(get_installed_info "${KEY_INSTALLED_VERSION}"),\
 and the version of this package is ${RUN_PKG_VERSION}, do you want to continue? [y/n]"
    fi
    while true; do
      read yn
      if [ "$yn" = "n" ]; then
        logandprint "[INFO]: Exit to install pto module."
        exitlog
        exit 0
      elif [ "$yn" = "y" ]; then
        break
      else
        echo "[WARNING]: Input error, please input y or n to choose!"
      fi
    done
  fi

  if [ "${installed_path}" = "${TARGET_VERSION_DIR}" ]; then
    logandprint "[INFO]: Clean the installed pto module before install."
    if [ ! -f "${UNINSTALL_SHELL_FILE}" ]; then
      logandprint "[ERROR]: ERR_NO:${FILE_NOT_EXIST};ERR_DES:${FILE_NOT_EXIST_DES}.The file\
 (${UNINSTALL_SHELL_FILE}) not exists. Please set the correct install \
 path or clean the previous version pto install info (${INSTALL_INFO_FILE}) and then reinstall it."
      return 1
    fi
    bash "${UNINSTALL_SHELL_FILE}" "${TARGET_VERSION_DIR}" "upgrade" "${IS_QUIET}" ${IN_FEATURE} "${IS_DOCKER_INSTALL}" "${DOCKER_ROOT}" "$pkg_version_dir"
    if [ "$?" != 0 ]; then
      logandprint "[ERROR]: ERR_NO:${INSTALL_FAILED};ERR_DES:Clean the installed directory failed."
      return 1
    fi
  fi
  return 0
}

select_last_dir_component() {
  path="$1"
  last_component=$(basename ${path})
  if [ "${last_component}" = "atc" ]; then
    last_component="atc"
    return
  elif [ "${last_component}" = "fwkacllib" ]; then
    last_component="fwkacllib"
    return
  elif [ "${last_component}" = "compiler" ]; then
    last_component="compiler"
    return
  fi
}

check_version_file() {
  pkg_path="$1"
  component_ret="$2"
  run_pkg_path_temp=$(dirname "${pkg_path}")
  run_pkg_path_temp2=${run_pkg_path_temp%/*}
  run_pkg_path="${run_pkg_path_temp}""/${component_ret}"
  run_pkg_path_temp2=${run_pkg_path%/*}
  version_file="${run_pkg_path}""/version.info"
  version_file_tmp="${run_pkg_path_temp2}""/version.info"
  if [ -f "${version_file_tmp}" ]; then
    version_file=${version_file_tmp}
  fi
  if [ -f "${version_file}" ]; then
    echo "${version_file}" 2 >>/dev/null
  else
    logandprint "[ERROR]: ERR_NO:${FILE_NOT_EXIST}; The [${component_ret}] version.info in path [${pkg_path}] not exists."
    exitlog
    exit 1
  fi
  return
}

check_pto_version_file() {
  if [ -f "${CURR_PATH}/../version.info" ]; then
    pto_ver_info="${CURR_PATH}/../version.info"
  elif [ -f "${DEFAULT_INSTALL_PATH}/${PTO_PLATFORM_DIR}/version.info" ]; then
    pto_ver_info="${DEFAULT_INSTALL_PATH}/${PTO_PLATFORM_DIR}/version.info"
  else
    logandprint "[ERROR]: ERR_NO:${FILE_NOT_EXIST}; The [${PTO_PLATFORM_DIR}] version.info not exists."
    exitlog
    exit 1
  fi
  return
}

check_relation() {
  pto_ver_info_val="$1"
  req_pkg_name="$2"
  req_pkg_version="$3"
  if [ -f "${COMMON_INC_FILE}" ]; then
    . "${COMMON_INC_FILE}"
    check_pkg_ver_deps "${pto_ver_info_val}" "${req_pkg_name}" "${req_pkg_version}"
    ret_situation=$ver_check_status
  else
    logandprint "[ERROR]: ERR_NO:${FILE_NOT_EXIST}; The ${COMMON_INC_FILE} not exists."
    exitlog
    exit 1
  fi
  return
}

show_relation() {
  relation_situation="$1"
  req_pkg_name_val="$2"
  req_pkg_path="$3"
  if [ "$relation_situation" = "SUCC" ]; then
    logandprint "[INFO]: Relationship of pto with ${req_pkg_name_val} in path ${req_pkg_path} checked successfully"
  else
    logandprint "[WARNING]: Relationship of pto with ${req_pkg_name_val} in path ${req_pkg_path} checked failed."
  fi
  return
}

find_version_check() {
  if [ "$(id -u)" != "0" ]; then
    atc_res=$(find ${HOME} -name "ccec_compiler" | grep Ascend | grep atc)
    fwk_res=$(find ${HOME} -name "ccec_compiler" | grep Ascend | grep fwk)
    comp_res=$(find ${HOME} -name "ccec_compiler" | grep Ascend | grep Ascend/compiler)
    ccec_compiler_path="$atc_res $fwk_res $comp_res"
  else
    atc_res=$(find /usr/local -name "ccec_compiler" | grep Ascend | grep atc)
    fwk_res=$(find /usr/local -name "ccec_compiler" | grep Ascend | grep fwk)
    comp_res=$(find /usr/local -name "ccec_compiler" | grep Ascend | grep Ascend/compiler)
    ccec_compiler_path="$atc_res $fwk_res $comp_res"
  fi
  check_pto_version_file
  ret_check_pto_version_file=$pto_ver_info
  for var in ${ccec_compiler_path}; do
    run_pkg_path_val=$(dirname "${var}")
    # find run pkg name
    select_last_dir_component "${run_pkg_path_val}"
    ret_pkg_name=$last_component
    #get check version
    check_version_file "${run_pkg_path_val}" "${ret_pkg_name}"
    ret_check_version_file=$version_file
    #check relation
    check_relation "${ret_check_pto_version_file}" "${ret_pkg_name}" "${ret_check_version_file}"
    ret_check_relation_val=$ret_situation
    #show relation
    show_relation "${ret_check_relation_val}" "${ret_pkg_name}" "${run_pkg_path_val}"
  done
  return
}

path_version_check() {
  path_env_list="$1"
  check_pto_version_file
  ret_check_pto_version_file_name=$pto_ver_info
  path_list=$(echo "${path_env_list}" | cut -d"=" -f2)
  array=$(echo ${path_list} | awk '{split($0,arr,":");for(i in arr) print arr[i]}')
  for var in ${array}; do
    path_ccec_compile=$(echo ${var} | grep -w "ccec_compiler")
    if [ "${path_ccec_compile}" != "" ]; then
      pkg_path_val=$(dirname $(dirname "${path_ccec_compile}"))
      # find run pkg name
      select_last_dir_component "${pkg_path_val}"
      ret_pkg_name_val=$last_component
      #get check version
      check_version_file "${pkg_path_val}" "${ret_pkg_name_val}"
      ret_check_version_file_val=$version_file
      #check relation
      check_relation "${ret_check_pto_version_file_name}" "${ret_pkg_name}" "${ret_check_version_file_val}"
      ret_check_relation=$ret_situation
      #show relation
      show_relation "${ret_check_relation}" "${ret_pkg_name}" "${pkg_path_val}"
    else
      echo "the var_case does not contains ccec_compiler" 2 >>/dev/null
    fi
  done
  return
}

check_docker_path() {
  docker_path="$1"
  if [ "${docker_path}" != "/"* ]; then
    echo "[PTO] [ERROR]: ERR_NO:${PARAM_INVALID};ERR_DES:Parameter --docker-root\
 must with absolute path that which is start with root directory /. Such as --docker-root=/${docker_path}"
    exitlog
    exit 1
  fi
  if [ ! -d "${docker_path}" ]; then
    echo "[PTO] [ERROR]: ERR_NO:${FILE_NOT_EXIST}; The directory:${docker_path} not exist, please create this directory."
    exitlog
    exit 1
  fi
}

judgment_path() {
  . "${COMMON_INC_FILE}"
  check_install_path_valid "${1}"
  if [ $? -ne 0 ]; then
    echo "[PTO][ERROR]: The pto install path ${1} is invalid, only characters in [a-z,A-Z,0-9,-,_] are supported!"
    exitlog
    exit 1
  fi
}

check_install_path() {
  TARGET_INSTALL_PATH="$1"
  # empty patch check
  if [ "x${TARGET_INSTALL_PATH}" = "x" ]; then
    echo "[PTO] [ERROR]: ERR_NO:${PARAM_INVALID};ERR_DES:Parameter --install-path\
 not support that the install path is empty."
    exitlog
    exit 1
  fi
  # space check
  if echo "x${TARGET_INSTALL_PATH}" | grep -q " "; then
    echo "[PTO] [ERROR]: ERR_NO:${PARAM_INVALID};ERR_DES:Parameter --install-path\
 not support that the install path contains space character."
    exitlog
    exit 1
  fi
  # delete last "/"
  local temp_path="${TARGET_INSTALL_PATH}"
  temp_path=$(echo "${temp_path%/}")
  if [ x"${temp_path}" = "x" ]; then
    temp_path="/"
  fi
  # convert relative path to absolute path
  local prefix=$(echo "${temp_path}" | cut -d"/" -f1 | cut -d"~" -f1)
  if [ "x${prefix}" = "x" ]; then
    TARGET_INSTALL_PATH="${temp_path}"
  else
    prefix=$(echo "${RUN_PATH}" | cut -d"/" -f1 | cut -d"~" -f1)
    if [ x"${prefix}" = "x" ]; then
      TARGET_INSTALL_PATH="${RUN_PATH}/${temp_path}"
    else
      echo "[PTO] [ERROR]: ERR_NO:${PARAM_INVALID};ERR_DES: Run package path is invalid: $RUN_PATH"
      exitlog
      exit 1
    fi
  fi
  # convert '~' to home path
  local home=$(echo "${TARGET_INSTALL_PATH}" | cut -d"~" -f1)
  if [ "x${home}" = "x" ]; then
    local temp_path_value=$(echo "${TARGET_INSTALL_PATH}" | cut -d"~" -f2)
    if [ "$(id -u)" -eq 0 ]; then
      TARGET_INSTALL_PATH="/root$temp_path_value"
    else
      local home_path=$(eval echo "${USER}")
      home_path=$(echo "${home_path}%/")
      TARGET_INSTALL_PATH="$home_path$temp_path_value"
    fi
  fi
}

# execute prereq_check file
exec_pre_check() {
  bash "${PRE_CHECK_FILE}"
}

# execute prereq_check file and interact with user
interact_pre_check() {
  exec_pre_check
  if [ "$?" != 0 ]; then
    if [ "${IS_QUIET}" = y ]; then
      logandprint "[WARNING]: Precheck of pto module execute failed! do you want to continue install? y"
    else
      logandprint "[WARNING]: Precheck of pto module execute failed! do you want to continue install?  [y/n] "
      while true; do
        read yn
        if [ "$yn" = "n" ]; then
          echo "stop install pto module!"
          exit 1
        elif [ "$yn" = y ]; then
          break
        else
          echo "[WARNING]: Input error, please input y or n to choose!"
        fi
      done
    fi
  fi
}

#get the dir of xxx.run
#pto_install_path_curr=`echo "$2" | cut -d"/" -f2- `
# cut first two params from *.run
get_run_path() {
  RUN_PATH=$(echo "$2" | cut -d"-" -f3-)
  if [ x"${RUN_PATH}" = x"" ]; then
    RUN_PATH=$(pwd)
  else
    # delete last "/"
    RUN_PATH=$(echo "${RUN_PATH%/}")
    if [ "x${RUN_PATH}" = "x" ]; then
      # root path
      RUN_PATH=$(pwd)
    fi
  fi
}

check_arch() {
  local architecture=$(uname -m)
  # check platform
  if [ "${architecture}" != "${ARCH_INFO}" ]; then
    logandprint "[ERROR]: ERR_NO:${OPERATE_FAILED};ERR_DES:the architecture of the run package arch:${ARCH_INFO}\
  is inconsistent with that of the current environment ${architecture}. "
    exitlog
    exit 1
  fi
}

get_opts() {
  i=0
  while true
  do
    if [ "x$1" = "x" ]; then
      break
    fi
    if [ "$(expr substr "$1" 1 2)" = "--" ]; then
      i=$(expr $i + 1)
    fi
    if [ $i -gt 2 ]; then
      break
    fi
    shift 1
  done

  if [ "$*" = "" ]; then
    echo "[ERROR]: ERR_NO:${PARAM_INVALID}; ERR_DES:Unrecognized parameters.Try './xxx.run --help for more information.'"
    exitlog
    exit 1
  fi

  while true; do
    # skip 2 parameters avoid run pkg and directory as input parameter
    case "$1" in
      --version)
        if [ -e "${VERSION_INFO_FILE}" ]; then
          . "${VERSION_INFO_FILE}"
          echo ${Version}
          exitlog
          exit 0
        else
          echo "[ERROR]: ERR_NO:${FILE_NOT_EXIST};ERR_DES:${FILE_NOT_EXIST_DES}.\
 The version file (${VERSION_INFO_FILE}) not exists or without execute permission."
          exitlog
          exit 1
        fi
        ;;
      --run | --full | --devel)
        IN_INSTALL_TYPE=$(echo ${1} | awk -F"--" '{print $2}')
        IS_INSTALL="y"
        CONFLICT_CMD_NUMS=$(expr $CONFLICT_CMD_NUMS + 1)
        shift
        ;;
      --upgrade)
        IS_UPGRADE="y"
        CONFLICT_CMD_NUMS=$(expr $CONFLICT_CMD_NUMS + 1)
        shift
        ;;
      --uninstall)
        IS_UNINSTALL="y"
        CONFLICT_CMD_NUMS=$(expr $CONFLICT_CMD_NUMS + 1)
        shift
        ;;
      --install-path=*)
        IS_INPUT_PATH="y"
        IN_INSTALL_PATH=$(echo ${1} | cut -d"=" -f2-)
        # check path
        judgment_path "${IN_INSTALL_PATH}"
        check_install_path "${IN_INSTALL_PATH}"
        shift
        ;;
      --quiet)
        IS_QUIET="y"
        shift
        ;;
      --install-for-all)
        IS_FOR_ALL="y"
        shift
        ;;
      --check)
        IS_CHECK="y"
        CONFLICT_CMD_NUMS=$(expr $CONFLICT_CMD_NUMS + 1)
        shift
        ;;
      --check-path=*)
        check_path=$1
        CONFLICT_CMD_NUMS=$(expr $CONFLICT_CMD_NUMS + 1)
        shift
        ;;
      --pre-check)
        IS_PRE_CHECK="y"
        shift
        ;;
      --jit)
        IS_JIT="y"
        shift
        ;;
      --setenv)
        IS_SETENV="y"
        shift
        ;;
      --docker-root=*)
        IS_DOCKER_INSTALL="y"
        DOCKER_ROOT=$(echo $1 | cut -d"=" -f2)
        check_docker_path ${DOCKER_ROOT}
        shift
        ;;
      -*)
        echo "[PTO] [ERROR]: ERR_NO:${PARAM_INVALID};ERR_DES:Unsupported parameters [$1],\
 operation execute failed. Please use [--help] to see the usage."
        exitlog
        exit 1
        ;;
      *)
        break
        ;;
    esac
  done
}

# pre-check
check_opts() {
  if [ "${CONFLICT_CMD_NUMS}" -eq 0 ] && [ "x${IS_PRE_CHECK}" = "xy" ]; then
    interact_pre_check
    exitlog
    exit 0
  fi

  if [ "${CONFLICT_CMD_NUMS}" != 1 ]; then
    echo "[PTO] [ERROR]: ERR_NO:${PARAM_INVALID};ERR_DES:\
 only support one type: full/run/devel/upgrade/uninstall/check, operation execute failed!\
 Please use [--help] to see the usage."
    exitlog
    exit 1
  fi
}

# init target_dir and log for install
init_env() {
  if is_version_dirpath "$TARGET_INSTALL_PATH"; then
    pkg_version_dir="$(basename "$TARGET_INSTALL_PATH")"
    TARGET_INSTALL_PATH="$(dirname "$TARGET_INSTALL_PATH")"
  else
    pkg_version_dir="cann"
  fi
  TARGET_VERSION_DIR="$TARGET_INSTALL_PATH/$pkg_version_dir"
  # Splicing docker-root and install-path
  if [ "${IS_DOCKER_INSTALL}" = "y" ]; then
    # delete last "/"
    local temp_path_param="${DOCKER_ROOT}"
    local temp_path_val=$(echo "${temp_path_param%/}")
    if [ "x${temp_path_val}" = "x" ]; then
      temp_path_val="/"
    fi
    TARGET_VERSION_DIR=${temp_path_val}${TARGET_VERSION_DIR}
  fi

  UNINSTALL_SHELL_FILE="${TARGET_VERSION_DIR}/share/info/pto_isa/script/pto_uninstall.sh"
  INSTALL_INFO_FILE="${TARGET_VERSION_DIR}/share/info/pto_isa/${ASCEND_INSTALL_INFO}"
  is_multi_version_pkg "pkg_is_multi_version" "$VERSION_INFO_FILE"
  get_package_version "RUN_PKG_VERSION" "$VERSION_INFO_FILE"

  # create log folder and log file
  comm_init_log

  logandprint "[INFO]: Execute the pto run package."
  logandprint "[INFO]: OperationLogFile path: ${COMM_LOGFILE}."
  logandprint "[INFO]: Input params: $CMD_LIST"

  local installed_version=$(get_installed_info "${KEY_INSTALLED_VERSION}")
  if [ "${installed_version}" = "" ]; then
    logandprint "[INFO]: Version of installing pto module is ${RUN_PKG_VERSION}."
  else
    if [ "${RUN_PKG_VERSION}" != "" ]; then
      logandprint "[INFO]: Existed pto module version is ${installed_version},\
 the new pto module version is ${RUN_PKG_VERSION}."
    fi
  fi
}

check_pre_install() {
  if [ "${IS_CHECK}" = "y" ] && [ "${check_path}" = "" ]; then
    path_env_list_val=$(env | grep -w PATH)
    path_ccec_compile_val=$(echo ${path_env_list} | grep -w "ccec_compiler")
    if [ "${path_ccec_compile_val}" != "" ]; then
      path_version_check "${path_env_list_val}"
    else
      find_version_check
    fi
    exitlog
    exit 0
  fi

  if [ "${IS_CHECK}"="y" ] && [ "${check_path}" != "" ]; then
    VERCHECK_FILE="${CURR_PATH}""/ver_check.sh"
    if [ ! -f "${VERCHECK_FILE}" ]; then
      logandprint "[ERROR]: ERR_NO:${FILE_NOT_EXIST};ERR_DES:${FILE_NOT_EXIST_DES}.\
      The file (${VERCHECK_FILE}) not exists.\
 Please make sure that the pto module installed in (${VERCHECK_FILE}) and then set the correct install path."
    fi
    bash "${VERCHECK_FILE}" "${check_path}"
    exitlog
    exit 0
  fi

  local installed_user=$(get_installed_info "${KEY_INSTALLED_UNAME}")
  local installed_group=$(get_installed_info "${KEY_INSTALLED_UGROUP}")
  if [ "${installed_user}" != "" ] || [ "${installed_group}" != "" ]; then
    if [ "${installed_user}" != "${TARGET_USERNAME}" ] || [ "${installed_group}" != "${TARGET_USERGROUP}" ]; then
      logandprint "[ERROR]: The user and group are not same with last installation,\
 do not support overwriting installation!"
      exitlog
      exit 1
    fi
  fi
}

#Support the installation script when the specified path (relative path and absolute path) does not exist
mkdir_install_path() {
  local base_dir=$(dirname ${TARGET_INSTALL_PATH})
  if [ ! -d ${base_dir} ]; then
    logandprint "[ERROR]: ERR_NO:${FILE_NOT_EXIST}; The directory:${base_dir} not exist, please create this directory."
    exitlog
    exit 1
  fi

  if [ -d "${TARGET_INSTALL_PATH}" ]; then
    test -w ${TARGET_INSTALL_PATH} >>/dev/null 2>&1
    if [ "$?" -ne 0 ]; then
      #All paths exist with write permission
      logandprint "[ERROR]: ERR_NO:${PERM_DENIED};ERR_DES:${PERM_DENIED_DES}. The ${TARGET_USERNAME} do\
 access ${TARGET_INSTALL_PATH} failed, please reset the directory to a right permission."
      exit 1
    fi
  else
    test -w ${base_dir} >>/dev/null 2>&1
    if [ "$?" -ne 0 ]; then
      #All paths exist with write permission
      logandprint "[ERROR]: ERR_NO:${PERM_DENIED};ERR_DES:${PERM_DENIED_DES}. The ${TARGET_USERNAME} do\
 access ${base_dir} failed, please reset the directory to a right permission."
      exit 1
    else
      comm_create_dir "${TARGET_INSTALL_PATH}" "750" "${TARGET_USERNAME}:${TARGET_USERGROUP}" "${IS_FOR_ALL}"
    fi
  fi
}

install_package() {
  if [ "${IS_INSTALL}" = "n" ] && [ "${IS_UPGRADE}" = "n" ]; then
    return
  fi
  # precheck before install pto module
  if [ "${IS_PRE_CHECK}" = "y" ]; then
    interact_pre_check
  fi

  # use uninstall to clean the install folder
  clean_before_reinstall
  if [ "$?" != 0 ]; then
    comm_log_operation "Install" "${IN_INSTALL_TYPE}" "PTO" "$?" "${CMD_LIST}"
  fi

  bash "${INSTALL_SHELL_FILE}" "${TARGET_INSTALL_PATH}" "${TARGET_USERNAME}" "${TARGET_USERGROUP}" "${IN_FEATURE}" \
    "${IN_INSTALL_TYPE}" "${IS_FOR_ALL}" "${IS_SETENV}" "${IS_DOCKER_INSTALL}" "${DOCKER_ROOT}" "$pkg_version_dir"
  if [ "$?" != 0 ]; then
    comm_log_operation "Install" "${IN_INSTALL_TYPE}" "PTO" "$?" "${CMD_LIST}"
  fi
  if [ $(id -u) -eq 0 ]; then
    chown -R "root":"root" "${TARGET_MOULDE_DIR}/script" 2>/dev/null
    chown "root":"root" "${TARGET_MOULDE_DIR}" 2>/dev/null
  else
    chmod -R 550 "${TARGET_MOULDE_DIR}/script" 2>/dev/null
    chmod 440 "${TARGET_MOULDE_DIR}/script/filelist.csv" 2>/dev/null
  fi
  comm_log_operation "Install" "${IN_INSTALL_TYPE}" "PTO" "$?" "${CMD_LIST}"
}

uninstall_package() {
  if [ "${IS_UNINSTALL}" = "n" ]; then
    return
  fi

  if [ ! -f "${UNINSTALL_SHELL_FILE}" ]; then
    logandprint "[ERROR]: ERR_NO:${FILE_NOT_EXIST};ERR_DES:The file\
 (${UNINSTALL_SHELL_FILE}) not exists. Please make sure that the pto module\
 installed in (${TARGET_VERSION_DIR}) and then set the correct install path."
    uninstall_path=$(ls "${TARGET_INSTALL_PATH}" 2>/dev/null)
    if [ "${uninstall_path}" = "" ]; then
      rm -rf "${TARGET_INSTALL_PATH}"
    fi
    comm_log_operation "Uninstall" "${IN_INSTALL_TYPE}" "PTO" "$?" "${CMD_LIST}"
    exit 0
  fi
  bash "${UNINSTALL_SHELL_FILE}" "${TARGET_INSTALL_PATH}" "uninstall" "${IS_QUIET}" ${IN_FEATURE} "${IS_DOCKER_INSTALL}" "${DOCKER_ROOT}" "$pkg_version_dir"
  # remove precheck info in ${TARGET_VERSION_DIR}/bin/prereq_check.bash
  logandprint "[INFO]: Remove precheck info."

  comm_log_operation "Uninstall" "${IN_INSTALL_TYPE}" "PTO" "$?" "${CMD_LIST}"
}

pre_check_only() {
  if [ "${IS_PRE_CHECK}" = "y" ]; then
    exec_pre_check
    exit $?
  fi
}

main() {
  check_arch

  get_run_path "$@"

  startlog

  get_opts "$@"

  check_opts

  init_env

  check_pre_install

  mkdir_install_path

  install_package

  uninstall_package

  pre_check_only
}

main "$@"
exit 0
