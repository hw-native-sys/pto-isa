#!/usr/bin/env bash
# --------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# --------------------------------------------------------------------------------

# Source from compile.sh/run.sh before enabling nounset for the vendor environment.
if [[ -z "${ASCEND_HOME_PATH:-}" ]]; then
    # Use the installation's selected toolkit version.
    for sfmla_toolkit_root in \
        "${HOME}/Ascend/cann" \
        "${HOME}/Ascend/ascend-toolkit/latest" \
        /usr/local/Ascend/cann \
        /usr/local/Ascend/ascend-toolkit/latest; do
        if [[ -f "${sfmla_toolkit_root}/set_env.sh" ]]; then
            export ASCEND_HOME_PATH="$sfmla_toolkit_root"
            break
        fi
    done
    unset sfmla_toolkit_root
fi
if [[ -z "${ASCEND_HOME_PATH:-}" || ! -f "${ASCEND_HOME_PATH}/set_env.sh" ]]; then
    echo "Cannot find CANN set_env.sh at ASCEND_HOME_PATH=${ASCEND_HOME_PATH:-<unset>}." >&2
    echo "Set ASCEND_HOME_PATH to your CANN toolkit root containing set_env.sh." >&2
    exit 2
fi
source "${ASCEND_HOME_PATH}/set_env.sh"
