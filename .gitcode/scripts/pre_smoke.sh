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

mkdir -p /home/taskspace && cd /home/taskspace
echo "start run test case, please wait ..."

export ASCEND_GLOBAL_LOG_LEVEL=2
export ASCEND_SLOG_PRINT_TO_STDOUT=0

sudo apt update && sudo apt install -y mpich libmpich-dev
sudo apt install -y openmpi-bin openmpi-common libopenmpi-dev
mpicc --version
mpirun --version
rm -rf /opt/rh/devtoolset-7
source /usr/local/Ascend/cann/set_env.sh
echo "bash build.sh --run_simple"
set +e
bash build.sh --run_simple --a3 2>&1 | tee -a ./run_test.log
source /usr/local/Ascend/cann/set_env.sh
echo "bash build.sh --comm --a3 --npu"

bash build.sh --comm --a3 --npu 2>&1 | tee -a ./run_test.log

# Package slog
mkdir -p /root/ascend
slog_name="slog.tar.gz"
tar -zcf slog.tar.gz -C /root/ascend log
OBS_KEY="${obs_smoke_path}/plog/${slog_name}"
# Upload slog
if python3 /home/upload.py --bucket-name "ascend-ci" --action upload --local-file "slog.tar.gz" --obs-object-key "${OBS_KEY}"; then
    echo "::set-output var=plog_url:https://ascend-ci.obs.cn-north-4.myhuaweicloud.com/${OBS_KEY}"
fi

npu-smi info
echo "4. checking test results ..."
date_time=$(date +%Y%m%d.%H%M%S)
if grep -w -e "execute comm samples success" "./run_test.log" && grep -w -e "execute samples success" "./run_test.log"; then
    echo "$date_time : run test case success"
else
    echo "$date_time : run test case failed"
    exit 1
fi
