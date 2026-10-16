#!/bin/bash
# --------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# --------------------------------------------------------------------------------

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

TOOLKIT_INSTALL_PATH=""
TOOLKIT_PACKAGE_PATH=""
DOWNLOAD_DIR="./downloads"
PTO_ISA_PKG=""
GTEST_PKG=""
LOG_FILE="./setup_pto_isa_env.log"

CURRENT_DATE=$(date '+%Y%m%d')
CURRENT_YEAR=$(date '+%Y')
CURRENT_MONTH=$(date '+%m')
CURRENT_DAY=$(date + '%d')

MAX_BACKTRACK_DAYS=3

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(dirname "$SCRIPT_DIR")

log() {
  local level="$1"
  local message="$2"
  echo -e "${level}[$(date '+%Y-%m-%d %H:%M:%S')] $message${NC}" | tee -a "$LOG_FILE"
}

info() { log "${GREEN}[INFO]${NC}" "$1"; }
warn() { log "${YELLOW}[WARN]${NC}" "$1"; }
error() { log "${RED}[ERROR]${NC}" "$1"; }

detect_arch() {
  local arch=$(uname -m)
  case $arch in
    aarch64|arm64)
      echo "aarch64"
      ;;
    x86_64|amd64)
      echo "x86_64"
      ;;
    *)
      error "not support arch: $arch"
      exit 1
      ;;
  esac
}

check_toolkit_installed() {
  local path="$1"
  local setenv_path="${path}/cann/bin/setenv.bash"

  if [[ -f "$setenv_path" ]]; then
    info "toolkit installed: $path"
    return 0
  else
    warn "toolkit has not installed"
    return 1
  fi
}

install_toolkit() {
  local package_path="$1"
  local install_path="$2"

  if [[ ! -f "$package_path" ]]; then
    error "toolkit package not exist: $package_path"
    exit 1
  fi
  info "install toolkit package to $install_path"

  if [[ "$package_path" == *.run ]]; then
    chmod +x "$package_path"
    "$package_path" --full --quiet --install-path="$install_path"
  else
    error "not support format: $package_path"
    exit 1
  fi

  if [[ $? -ne 0 ]]; then
    error "install toolkit failed!"
    exit 1
  fi

  info "install toolkit success!"
}

download_pto_isa_run() {
  local arch="$1"
  local base_url="https://container-obsfs-filesystem.obs.cn-north-4.myhuaweicloud.com/package/cann/pto-isa/version_compile/master/${CURRENT_YEAR}${CURRENT_MONTH}"

  local version_cmake="${REPO_ROOT}/version.cmake"
  if [[ ! -f "$version_cmake" ]]; then
    error "version.cmake not found (expected: ${version_cmake}); package version cannot be determined. Please run this script from the pto-isa repository."
    exit 1
  fi
  local pto_isa_version
  pto_isa_version=$(sed -n 's/.*set_cann_package(pto-isa VERSION "\([0-9][0-9]*\.[0-9][0-9]*\.[0-9][0-9]*\)".*/\1/p' "$version_cmake" | head -n 1)
  if [[ -z "$pto_isa_version" ]]; then
    error "cannot parse package version from ${version_cmake}: set_cann_package(pto-isa VERSION \"x.y.z\") not found."
    exit 1
  fi

  local try_dates=()
  local current=$(date -d "$CURRENT_DATE" +%s)

  for i in $(seq 0 $((MAX_BACKTRACK_DAYS - 1))); do
    local target_date=$(date -d "@$((current - i * 86400))" '+%Y%m%d')
    try_dates+=("$target_date")
  done

  local filename="cann-pto-isa_${pto_isa_version}_linux-${arch}.run"
  PTO_ISA_PKG="${DOWNLOAD_DIR}/${filename}"

  mkdir -p  "$DOWNLOAD_DIR"

  local forbidden=0
  for date in "${try_dates[@]}"; do
    local url="${base_url}/${date}/ubuntu_${arch}/${filename}"

    if [[ -f "$PTO_ISA_PKG" ]]; then
      info "pto isa package exist: $PTO_ISA_PKG"
      return 0
    fi

    info "downloading $url"

    local http_code
    http_code=$(curl -sS -o "$PTO_ISA_PKG" -w '%{http_code}' --connect-timeout 15 "${url}" 2>/dev/null)
    if [[ "$http_code" == "200" ]]; then
      info "download successful!"
      return 0
    elif [[ "$http_code" == "403" ]]; then
      rm -f "$PTO_ISA_PKG"
      warn "download forbidden (403): $url"
      forbidden=1
    else
      rm -f "$PTO_ISA_PKG"
      warn "download failed (http $http_code): $url"
    fi
  done

  if [[ "$forbidden" == "1" ]]; then
    error "PTO ISA package download was denied (HTTP 403): the OBS bucket rejects anonymous public access or the package is not published yet. Please build the package locally with './build.sh --pkg' (output in build_out/) or contact the release owner."
    exit 1
  fi

  error "please check network!"
  exit 1
}

install_pto_isa_run() {
  local toolkit_path="$1"
  local install_path="${toolkit_path}"

  chmod +x "$PTO_ISA_PKG"
  "$PTO_ISA_PKG" --full --quiet --install-path="$install_path"

  if [[ $? -ne 0 ]]; then
    error "install PTO ISA package failed!"
    exit 1
  fi
  info "install PTO ISA package success!"
}

install_gtest() {
  git config --global http.sslverify false
  git clone https://github.com/google/googletest.git -b v1.14.x
  cd googletest
  mkdir build
  cd build
  cmake .. -DCMAKE_CXX_FLAGS="-fPIC"

  if make -j$(nproc); then
    info "build gtest success!"
  else
    error "build gtest failed!"
    exit 1
  fi

  if sudo make install; then
    info "install gtest success!"
  else
    error "install gtest failed!"
    exit 1
  fi
}

main() {
  if [[ $# -lt 1 ]]; then
    error "usage: $0 <toolkit_install_path> [toolkit_package_path]"
    exit 1
  fi

  TOOLKIT_INSTALL_PATH="$1"
  TOOLKIT_PACKAGE_PATH="$2"

  local ARCH=$(detect_arch)
  info "arch:$ARCH"

  if check_toolkit_installed "$TOOLKIT_INSTALL_PATH"; then
    download_pto_isa_run "$ARCH"
    install_pto_isa_run "$DOWNLOAD_DIR" "$TOOLKIT_INSTALL_PATH"
  else
    if [[ -z "$TOOLKIT_PACKAGE_PATH" ]]; then
      error "no toolkit package info"
      exit 1
    fi

    install_toolkit "$TOOLKIT_PACKAGE_PATH" "$TOOLKIT_INSTALL_PATH"
    download_pto_isa_run "$ARCH"
    install_pto_isa_run "$TOOLKIT_INSTALL_PATH"
  fi

  install_gtest

  info "set environment successfully!"
}

main "$@"
