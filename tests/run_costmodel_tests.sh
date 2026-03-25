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

# ST 测试用例目录
TESTCASE_DIR="${SCRIPT_DIR}/costmodel/st/testcase"

# 需要执行的测试用例列表（留空则自动发现 TESTCASE_DIR 下所有子目录）
TESTCASES=()

# 测试命令的固定参数
TEST_ARGS="--clean --verbose"
# ==================================================================

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# 函数：打印错误信息并退出
error_exit() {
    echo -e "${RED}[ERROR] $1${NC}"
    exit 1
}

# 记录结果的数组
PASSED_CASES=()
FAILED_CASES=()

# 1. 检查目标目录是否存在
if [ ! -d "${TARGET_DIR}" ]; then
    error_exit "Dir not exists：${TARGET_DIR}"
fi

# 若 TESTCASES 为空，自动发现 TESTCASE_DIR 下所有子目录
if [ ${#TESTCASES[@]} -eq 0 ]; then
    if [ ! -d "${TESTCASE_DIR}" ]; then
        error_exit "Testcase dir not exists：${TESTCASE_DIR}"
    fi
    while IFS= read -r -d '' dir; do
        TESTCASES+=("$(basename "${dir}")")
    done < <(find "${TESTCASE_DIR}" -mindepth 1 -maxdepth 1 -type d -print0 | sort -z)
    echo -e "${YELLOW}[INFO] Auto-discovered testcases: ${TESTCASES[*]}${NC}"
fi

# 2. 进入目标目录
echo -e "${YELLOW}[INFO] Enter Dir：${TARGET_DIR}${NC}"
cd "${TARGET_DIR}" || error_exit "Enter Dir Failed：${TARGET_DIR}"

# 3. 遍历测试用例并执行
for testcase in "${TESTCASES[@]}"; do
    echo -e "\n========================================"
    echo -e "${YELLOW}[INFO] Start Test Case:${testcase}${NC}"
    echo -e "========================================"

    # 构建测试命令
    test_cmd="python tests/run_costmodel.py --testcase ${testcase} ${TEST_ARGS}"
    echo -e "${YELLOW}[INFO] Execute cmd:${test_cmd}${NC}"

    # 执行命令并捕获退出码
    ${test_cmd}
    exit_code=$?

    # 根据退出码判断执行结果
    if [ ${exit_code} -eq 0 ]; then
        echo -e "${GREEN}[SUCCESS] Test Case ${testcase} Finished${NC}"
        PASSED_CASES+=("${testcase}")
    else
        echo -e "${RED}[FAIL] Test Case ${testcase} Failed (Exit Code: ${exit_code})${NC}"
        FAILED_CASES+=("${testcase}")
    fi
done

# 4. 输出总览
total=${#TESTCASES[@]}
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