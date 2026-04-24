#!/bin/bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# 批量执行 costmodel 测试的脚本
# ===================== 配置区（可根据需要修改）=====================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

TARGET_DIR="${SCRIPT_DIR}/.."

# ST/ST_FIT 测试用例目录
ST_TESTCASE_DIR="${SCRIPT_DIR}/costmodel/st/testcase"
ST_FIT_TESTCASE_DIR="${SCRIPT_DIR}/costmodel/st_fit/testcase"

# 需要执行的测试用例列表（留空则自动发现 ST_TESTCASE_DIR 与 ST_FIT_TESTCASE_DIR 下所有子目录）
# 用法：TESTCASES=("tsub" "time_predict")
TESTCASES=()

# 测试命令的固定参数
TEST_ARGS="--clean --verbose"
# ==================================================================

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

if command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN=python3
else
    PYTHON_BIN=python
fi

# 函数：打印错误信息并退出
error_exit() {
    echo -e "${RED}[ERROR] $1${NC}"
    exit 1
}

# 记录结果的数组
PASSED_CASES=()
FAILED_CASES=()
RUN_ITEMS=()

# 1. 检查目标目录是否存在
if [ ! -d "${TARGET_DIR}" ]; then
    error_exit "Dir not exists：${TARGET_DIR}"
fi

is_case_in_suite() {
    local suite="$1"
    local testcase="$2"
    local base_dir=""

    if [ "${suite}" = "st" ]; then
        base_dir="${ST_TESTCASE_DIR}"
    elif [ "${suite}" = "st_fit" ]; then
        base_dir="${ST_FIT_TESTCASE_DIR}"
    else
        return 1
    fi

    [ -d "${base_dir}/${testcase}" ]
}

append_run_item() {
    local suite="$1"
    local testcase="$2"
    RUN_ITEMS+=("${suite}:${testcase}")
}

resolve_suite_for_case() {
    local testcase="$1"
    local found_st=0
    local found_st_fit=0

    if is_case_in_suite "st" "${testcase}"; then
        found_st=1
    fi
    if is_case_in_suite "st_fit" "${testcase}"; then
        found_st_fit=1
    fi

    if [ ${found_st} -eq 1 ] && [ ${found_st_fit} -eq 0 ]; then
        echo "st"
        return 0
    fi

    if [ ${found_st} -eq 0 ] && [ ${found_st_fit} -eq 1 ]; then
        echo "st_fit"
        return 0
    fi

    if [ ${found_st} -eq 1 ] && [ ${found_st_fit} -eq 1 ]; then
        error_exit "Testcase '${testcase}' exists in both st and st_fit. Please use 'st:${testcase}' or 'st_fit:${testcase}'"
    fi

    error_exit "Unknown testcase '${testcase}', not found in st/st_fit"
}

auto_discover_testcases() {
    local dir=""
    local suite=""

    for suite in "st" "st_fit"; do
        if [ "${suite}" = "st" ]; then
            dir="${ST_TESTCASE_DIR}"
        else
            dir="${ST_FIT_TESTCASE_DIR}"
        fi

        if [ ! -d "${dir}" ]; then
            error_exit "Testcase dir not exists：${dir}"
        fi

        while IFS= read -r -d '' subdir; do
            append_run_item "${suite}" "$(basename "${subdir}")"
        done < <(find "${dir}" -mindepth 1 -maxdepth 1 -type d -print0 | sort -z)
    done
}

normalize_manual_testcases() {
    local item=""
    local suite=""

    for item in "${TESTCASES[@]}"; do
        if [[ "${item}" == *":"* ]]; then
            error_exit "Invalid testcase '${item}'. run_costmodel_tests.sh does not support explicit suite."
        fi
        suite="$(resolve_suite_for_case "${item}")"
        append_run_item "${suite}" "${item}"
    done
}

# 若 TESTCASES 为空，自动发现 st/st_fit 两个目录下所有子目录
if [ ${#TESTCASES[@]} -eq 0 ]; then
    auto_discover_testcases
    echo -e "${YELLOW}[INFO] Auto-discovered run items: ${RUN_ITEMS[*]}${NC}"
else
    normalize_manual_testcases
fi

if [ ${#RUN_ITEMS[@]} -eq 0 ]; then
    error_exit "No runnable testcase found"
fi

# 2. 进入目标目录
echo -e "${YELLOW}[INFO] Enter Dir：${TARGET_DIR}${NC}"
cd "${TARGET_DIR}" || error_exit "Enter Dir Failed：${TARGET_DIR}"

# 3. 遍历测试用例并执行
for item in "${RUN_ITEMS[@]}"; do
    suite="${item%%:*}"
    testcase="${item#*:}"
    label="${suite}:${testcase}"

    echo -e "\n========================================"
    echo -e "${YELLOW}[INFO] Start Test Case:${label}${NC}"
    echo -e "========================================"

    # 构建测试命令
    test_cmd="${PYTHON_BIN} tests/run_costmodel.py --suite ${suite} --testcase ${testcase} ${TEST_ARGS}"
    echo -e "${YELLOW}[INFO] Execute cmd:${test_cmd}${NC}"

    # 执行命令并捕获退出码
    ${test_cmd}
    exit_code=$?

    # 根据退出码判断执行结果
    if [ ${exit_code} -eq 0 ]; then
        echo -e "${GREEN}[SUCCESS] Test Case ${label} Finished${NC}"
        PASSED_CASES+=("${label}")
    else
        echo -e "${RED}[FAIL] Test Case ${label} Failed (Exit Code: ${exit_code})${NC}"
        FAILED_CASES+=("${label}")
    fi
done

# 4. 输出总览
total=${#RUN_ITEMS[@]}
passed=${#PASSED_CASES[@]}
failed=${#FAILED_CASES[@]}

echo -e "\n========================================"
echo -e "           ST Results Summary"
echo -e "========================================"
echo -e "Total:  ${total}"
echo -e "${GREEN}Passed: ${passed}${NC}"
echo -e "${RED}Failed: ${failed}${NC}"

if [ ${passed} -gt 0 ]; then
    echo -e "\n${GREEN}[PASSED]${NC}"
    for tc in "${PASSED_CASES[@]}"; do
        echo -e "  ${GREEN}✔ ${tc}${NC}"
    done
fi

if [ ${failed} -gt 0 ]; then
    echo -e "\n${RED}[FAILED]${NC}"
    for tc in "${FAILED_CASES[@]}"; do
        echo -e "  ${RED}✘ ${tc}${NC}"
    done
fi

echo -e "========================================"

if [ ${failed} -gt 0 ]; then
    exit 1
fi
exit 0
