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

set -e

dotted_line="----------------------------------------------------------------"
COLOR_RESET="\033[0m"
COLOR_GREEN="\033[32m"
COLOR_RED="\033[31m"

export BASE_PATH=$(
  cd "$(dirname $0)"
  pwd
)

export INCLUDE_PATH="${ASCEND_HOME_PATH}/include"
export ASCEND_ENV_PATH="${ASCEND_HOME_PATH}/bin"
export BUILD_PATH="${BASE_PATH}/build"
export BUILD_OUT_PATH="${BASE_PATH}/build_out"
CANN_3RD_LIB_PATH="${BASE_PATH}/third_party"
CMAKE_ARGS=""

#print usage message
usage() {
  echo "Usage:"
  echo ""
  echo "    -h, --help  Print usage"
  echo "    --pkg Build run package"
  echo "    --run_all run all st on sim"
  echo "    --run_simple run some st on board"
  echo ""
}

print_success() {
  echo
  echo $dotted_line
  local msg="$1"
  echo -e "${COLOR_GREEN}[SUCCESS] ${msg}${COLOR_RESET}"
  echo $dotted_line
  echo
}

print_error() {
  echo
  echo $dotted_line
  local msg="$1"
  echo -e "${COLOR_RED}[ERROR] ${msg}${COLOR_RESET}"
  echo $dotted_line
  echo
}

checkopts() {
  ENABLE_SIMPLE_ST=FALSE
  ENABLE_BUILD_ALL=FALSE
  ENABLE_BUILD_ONLY=FALSE
  ENABLE_RUN_EXAMPLE=FALSE
  ENABLE_PACKAGE=FALSE
  ENABLE_A3=FALSE
  ENABLE_A5=FALSE
  ENABLE_CPU=FALSE
  ENABLE_COMM=FALSE
  RUN_TYPE="npu"
  EXAMPLE_NAME=""
  EXAMPLE_MODE=""
  PLATFORM_MODE=""
  INST_NAME=""
  AUTO_MODE=FALSE

  parsed_args=$(getopt -a -o j:hvuO: -l help,verbose,cov,make_clean,noexec,pkg,run_all,a3,a5,sim,npu,comm,cpu,auto_mode,run_simple,build,cann_3rd_lib_path: -- "$@") || {
  usage
  exit 1
  }

  eval set -- "$parsed_args"

  while true; do
    case "$1" in
      -h | --help)
        usage
        exit 0
        ;;
      --run_all)
        ENABLE_BUILD_ALL=TRUE
        shift
        ;;
      --run_simple)
        ENABLE_SIMPLE_ST=TRUE
        shift
        ;;
      --pkg)
        ENABLE_PACKAGE=TRUE
        shift
        ;;
      --a3)
        ENABLE_A3=TRUE
        shift
        ;;
      --a5)
        ENABLE_A5=TRUE
        shift
        ;;
      --comm)
        ENABLE_COMM=TRUE
        shift
        ;;
      --sim)
        RUN_TYPE=sim
        shift
        ;;
      --npu)
        RUN_TYPE=npu
        shift
        ;;
      --cpu)
        ENABLE_CPU=TRUE
        shift
        ;;
      --cann_3rd_lib_path)
        shift
        CANN_3RD_LIB_PATH="$1"
        shift
        ;;
      --build)
        shift
        ENABLE_BUILD_ONLY=TRUE
        ;;
      --auto_mode)
        shift
        AUTO_MODE=TRUE
        ;;
      --)
        shift
        break
        ;;
      *)
        usage
        exit 1
        ;;
    esac
  done
  CMAKE_ARGS="$CMAKE_ARGS -DCANN_3RD_LIB_PATH=${CANN_3RD_LIB_PATH}"
}

build_only() {
  echo $dotted_line
  echo "build only"
  if [ "$ENABLE_A3" = "TRUE" ] && [ "$ENABLE_A5" = "FALSE" ]; then
    python3 tests/script/build_st.py -r npu -v a3 -t all
  elif [ "$ENABLE_A3" = "FALSE" ] && [ "$ENABLE_A5" = "TRUE" ]; then
    python3 tests/script/build_st.py -r npu -v a5 -t all
  elif [ "$ENABLE_A3" = "TRUE" ] && [ "$ENABLE_A5" = "TRUE" ]; then
    python3 tests/script/build_st.py -r npu -v a3 -t all
    python3 tests/script/build_st.py -r npu -v a5 -t all
  else
    python3 tests/script/build_st.py -r npu -v a5 -t all
  fi
  echo "build end"
}

run_simple_st() {
  echo $dotted_line
  echo "Start to run simple st"
  chmod +x ./tests/run_st.sh
  ARGS=" "
  if [ "$ENABLE_A3" = "TRUE" ] && [ "$ENABLE_A5" = "FALSE" ]; then
    ARGS+="--a3 "
  elif [ "$ENABLE_A3" = "FALSE" ] && [ "$ENABLE_A5" = "TRUE" ]; then
    ARGS+="--a5 "
  elif [ "$ENABLE_A3" = "TRUE" ] && [ "$ENABLE_A5" = "TRUE" ]; then
    ARGS+="--a3_a5 "
  else
    ARGS+="--a3 "
  fi
  ARGS+="--$RUN_TYPE --simple "
  if [ "$AUTO_MODE" == "TRUE" ]; then
    ARGS+="--auto_mode "
  fi
  ./tests/run_st.sh ${ARGS}
  echo "execute samples success"
}

run_comm_st() {
  echo $dotted_line
  echo "Start to run comm st"
  chmod +x ./tests/run_st.sh
  ARGS="--comm "
  if [ "$ENABLE_A3" = "TRUE" ] && [ "$ENABLE_A5" = "FALSE" ]; then
    ARGS+="--a3 "
  elif [ "$ENABLE_A3" = "FALSE" ] && [ "$ENABLE_A5" = "TRUE" ]; then
    ARGS+="--a5 "
  elif [ "$ENABLE_A3" = "TRUE" ] && [ "$ENABLE_A5" = "TRUE" ]; then
    ARGS+="--a3_a5 "
  else
    ARGS+="--a3 "
  fi
  ARGS+="--$RUN_TYPE "
  ./tests/run_st.sh ${ARGS}
  echo "execute comm samples success"
}

run_cpu_st() {
  echo $dotted_line
  echo "Start to run cpu st"
  python3 tests/run_cpu.py --cxx=clang++ --clean --verbose
  python3 tests/run_cpu.py --cxx=clang++ --demo gemm --verbose
  python3 tests/run_cpu.py --cxx=clang++ --demo flash_attn --verbose
  python3 tests/run_cpu.py --cxx=clang++ --demo mla --verbose
  bash tests/run_costmodel_tests.sh
}

run_all_st() {
  echo $dotted_line
  echo "Start to run all st"
  chmod +x ./tests/run_st.sh
  ARGS=" "
  if [ "$ENABLE_A3" = "TRUE" ] && [ "$ENABLE_A5" = "FALSE" ]; then
    ARGS+="--a3 "
  elif [ "$ENABLE_A3" = "FALSE" ] && [ "$ENABLE_A5" = "TRUE" ]; then
    ARGS+="--a5 "
  elif [ "$ENABLE_A3" = "TRUE" ] && [ "$ENABLE_A5" = "TRUE" ]; then
    ARGS+="--a3_a5 "
  else
    ARGS+="--a3 "
  fi
  ARGS+="--$RUN_TYPE --all "
  if [ "$AUTO_MODE" == "TRUE" ]; then
      ARGS+="--auto_mode "
  fi
  ./tests/run_st.sh ${ARGS}
  echo "execute samples success"
}

clean_build() {
  if [ -d "${BUILD_PATH}" ]; then
    rm -rf ${BUILD_PATH}
  fi
}

clean_build_out() {
  if [ -d "${BUILD_OUT_PATH}" ]; then
    rm -rf ${BUILD_OUT_PATH}
  fi
}


build_package() {
  echo "---------------package start-----------------"
  clean_build_out
  clean_build
  mkdir $BUILD_PATH
  mkdir $BUILD_OUT_PATH
  cd $BUILD_PATH
  cmake ${CMAKE_ARGS} ..
  make package
  echo "---------------package end------------------"
}

run_example() {
  echo $dotted_line
  echo "Start to run example"
  python3 tests/script/run_st.py -r $PLATFORM_MODE -v $EXAMPLE_MODE -t $INST_NAME -g $$EXAMPLE_NAME
  echo "execute samples success"
}

main() {
  checkopts "$@"
  if [ "$RUN_TYPE" == "sim" ]; then
      ulimit -n 65535
  fi
  if [ "$ENABLE_SIMPLE_ST" == "TRUE" ]; then
      run_simple_st
  fi
  if [ "$ENABLE_BUILD_ALL" == "TRUE" ]; then
      run_all_st
  fi
  if [ "$ENABLE_RUN_EXAMPLE" == "TRUE" ]; then
      run_example
  fi
  if [ "$ENABLE_PACKAGE" == "TRUE" ]; then
    build_package
  fi
  if [ "$ENABLE_BUILD_ONLY" == "TRUE" ]; then
      build_only
  fi
  if [ "$ENABLE_CPU" == "TRUE" ]; then
    run_cpu_st
  fi
  if [ "$ENABLE_COMM" == "TRUE" ]; then
    run_comm_st
  fi
}

set -o pipefail
main "$@" | gawk '{print strftime("[%Y-%m-%d %H:%M:%S]"), $0}'
