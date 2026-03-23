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

# 需要执行的测试用例列表（可按需添加/删除）
TESTCASES=("tadd" "tmul" "tsub" "tadds" "tdivs" "tmins" "tmuls" "tabs" "texp" "tsqrt" "tabs" "tcolmax" "tcolsum" "trowexpand" "trowmax" "trowsum")

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

# 1. 检查目标目录是否存在
if [ ! -d "${TARGET_DIR}" ]; then
    error_exit "Dir not exists：${TARGET_DIR}"
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
    else
        echo -e "${RED}[FAIL] Test Case ${testcase} Failed (Exit Code: ${exit_code})${NC}"
        # 可选：如果某个用例失败是否继续执行后续用例
        # error_exit "测试用例 ${testcase} 执行失败，终止脚本"
    fi
done

# 4. 脚本执行完成
echo -e "\n${GREEN}[INFO] All Test Case Finished${NC}"
exit 0