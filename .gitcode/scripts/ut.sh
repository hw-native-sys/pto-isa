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

echo "ut_type=${ut_type:-}"
echo "TARGET_BRANCH=${TARGET_BRANCH:-}"
echo "ge_st_rt2=${ge_st_rt2:-}"
echo "task_name=${task_name:-}"

grep -E "^VERSION_ID=" /etc/os-release | cut -d'"' -f2
export PATH=/opt/buildtools/python-3.10.2/bin:$PATH
sudo update-alternatives --set gcc /usr/bin/gcc-14
gcc --version

# Print and execute a command, capturing its exit code in the global variable ${ret}
function LOG_DO() {
    local date_time
    date_time=$(date +%Y%m%d-%H%M%S)
    echo -e "[Command] ${date_time} $*"
    "$@" && ret=0 || ret=$?
    return "${ret}"
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

main() {
    cd "${WORKSPACE}" || exit
    source /home/jenkins/Ascend/cann/bin/setenv.bash
    echo "Start run c++ testcase"
    echo "Y" | apt install libgtest-dev libgmock-dev
    gcc --version
    rm -rf /opt/rh/devtoolset-7
    bisheng -v
    sudo apt-get update && sudo apt-get install clang -y
    clang --version
    clang++ --version

    # Only run UT tests on the master branch
    if [[ "${TARGET_BRANCH}" != "master" ]]; then
        echo "Skip UT test on non-master branch"
        exit 0
    fi
    set +e
    if [[ "${ge_st_rt2}X" == "A3X" ]]; then
        LOG_DO python3 tests/script/build_st.py -a -r npu -v a3 -t all
        DP_ASSERT_EQUAL "$?" "0" "Run A3 UT TESTCASE"
    elif [[ "${ge_st_rt2}X" == "A5X" ]]; then
        LOG_DO python3 tests/script/build_st.py -r npu -v a5 -t all
        DP_ASSERT_EQUAL "$?" "0" "Run A5 UT TESTCASE"
    else
        LOG_DO bash build.sh --cpu
        DP_ASSERT_EQUAL "$?" "0" "Run A5 UT TESTCASE"
    fi
    echo "Run UT TESTCASE success"
}

main "$@"
