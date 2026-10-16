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

# cleanup.sh is expected to run with CWD set to the install directory (package
# manager <cleanup> hook). Resolve the physical path first and refuse to remove
# anything when it is empty, "/" or the user's home, so an accidental run from
# an arbitrary directory cannot recursively wipe user data.
target_dir="$(pwd -P)"
if [ -z "${target_dir}" ] || [ "${target_dir}" = "/" ] || [ "${target_dir}" = "${HOME}" ]; then
    echo "[ERROR] Refuse to clean unsafe directory: ${target_dir}" >&2
    exit 1
fi

echo "[INFO] Cleaning up install directory: ${target_dir}"
rm -rf -- "${target_dir}"
