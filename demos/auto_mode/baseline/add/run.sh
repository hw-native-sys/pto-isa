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

#example script

set -e
# export ASCEND_HOME_PATH=/usr/local/Ascend/
# source /usr/local/Ascend/ascend-toolkit/set_env.sh
# specify your path to pto-isa
#export PTO_LIB_PATH=[YOUR_PATH]/pto-isa

rm -fr build op_extension.egg-info
python3 setup.py bdist_wheel
cd dist &&
python3 -m pip uninstall op_extension-0.0.0-cp39-cp39-linux_aarch64.whl &&
python3 -m pip install op_extension-0.0.0-cp39-cp39-linux_aarch64.whl &&
cd ../test && python3 test.py && cd ..
