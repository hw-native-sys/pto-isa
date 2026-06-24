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

#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
build="${SCRIPT_DIR}/../build"

set -euo pipefail
python3 "${SCRIPT_DIR}/../scripts/pipeline_log_analysis.py" \
	--device-addrs "${build}/device_addrs.toml" \
	--cube-start "${build}/core0.cubecore0.instr_popped_log.dump" \
	--cube-end "${build}/core0.cubecore0.instr_log.dump" \
	--vec-start "${build}/core0.veccore0.instr_popped_log.dump" \
	--vec-end "${build}/core0.veccore0.instr_log.dump" \
	--out-csv timeline.csv \
	--out-json timeline.json \
	--out-agg timeline_agg.csv \
	--out-svg timeline.svg
