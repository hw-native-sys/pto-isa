#!/bin/bash
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

set -e
set -o pipefail

echo "package_name=${package_name:-}"
echo "ge_st_rt2=${ge_st_rt2:-}"
echo "task_name=${task_name:-}"

REPOSITORY_NAME="pto-isa"


# Print and execute a command, capturing its exit code in the global variable ${ret}
function LOG_DO() {
   local cmd="$*"
   date_time=$(date +%Y%m%d-%H%M%S)
   echo -e "[Command] ${date_time} ${cmd}$"
   ${cmd}
}

function DP_ASSERT_EQUAL() {
    local actual_value=${1}
    local expect_value=${2}
    local assert_msg=${3}
    echo "actual_value:${actual_value}"
    echo "expect_value:${expect_value}"
    if [ "${actual_value}" != "${expect_value}" ]; then
        echo "${assert_msg} is failed."
        exit 1
    else
        echo "${assert_msg} is success."
    fi
}

#########
# Install
#########
cd "${WORKSPACE}" || exit
source /home/jenkins/Ascend/cann/bin/setenv.bash

#########
# Build
#########
echo "Y" | apt install libgtest-dev libgmock-dev
gcc --version
rm -rf /opt/rh/devtoolset-7
bisheng -v

set +e
if [[ "${ge_st_rt2}X" == "A5X" ]]; then
    LOG_DO bash build.sh --a5 --build
    DP_ASSERT_EQUAL "$?" "0" "Build  A5 ${REPOSITORY_NAME}"
else
    LOG_DO bash build.sh --pkg
    DP_ASSERT_EQUAL "$?" "0" "Build  ${REPOSITORY_NAME}"
fi

# Locate the generated .run package
echo "package_name=${package_name}" >> "${ATOMGIT_OUTPUT}"
