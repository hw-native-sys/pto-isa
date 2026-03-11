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

set -e

ENABLE_A3=false
ENABLE_A5=false
ENABLE_SIM=false
ENABLE_NPU=false
RUN_TYPE="npu"
CARD_NAME="Ascend910_9599"

if [ "$1" = "a3" ]; then
  ENABLE_A3=true
elif [ "$1" = "a5" ]; then
  ENABLE_A5=true
elif [ "$1" = "a3_a5" ]; then
  ENABLE_A3=true
  ENABLE_A5=true
fi

if [ "$2" = "sim" ]; then
  RUN_TYPE=sim
elif [ "$2" = "npu" ]; then
  RUN_TYPE=npu
fi

CARD_NAME="$3"

if [ "$ENABLE_A3" = "true" ]; then
  cd kernels/manual/a2a3/flash_atten
  python3 scripts/gen_data.py
  bash run.sh -r $RUN_TYPE -v $CARD_NAME
  cd ../../../../

  cd kernels/manual/a2a3/gemm_performance
  python3 scripts/gen_data.py
  bash run.sh -r $RUN_TYPE -v $CARD_NAME
  cd ../../../../

  cd kernels/manual/a2a3/topk
  python3 scripts/gen_data.py
  bash run.sh -r $RUN_TYPE -v $CARD_NAME
  cd ../../../../
  echo "run kernels success"
fi

if [ "$ENABLE_A5" = "true" ]; then
  cd kernels/manual/a5/flash_atten
  python3 scripts/gen_data.py
  bash run.sh -r $RUN_TYPE -v $CARD_NAME --cases "128,16384,16384,128,128" --qk-preload 2
  cd ../../../../

  cd kernels/manual/a5/matmul_mxfp4_performance
  python3 scripts/gen_data.py
  bash run.sh -r $RUN_TYPE -v $CARD_NAME -n 0 --cases "128,16384,16384,128,128" --qk-preload 2 --mode 1
  cd ../../../../

  cd kernels/manual/a5/matmul_mxfp8_performance
  python3 scripts/gen_data.py
  bash run.sh -r $RUN_TYPE -v $CARD_NAME -n 0 --cases "128,16384,16384,128,128" --qk-preload 2 --mode 1
  cd ../../../../

  cd kernels/manual/a5/engram_simt
  bash run.sh -r $RUN_TYPE -v $CARD_NAME -c "ENGRAMSIMTTest.baseline_E128_B1_T64K"
  cd ../../../../
  
  echo "run kernels success"
fi

