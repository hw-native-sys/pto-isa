/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef SYNCALL_CORE_CONFIG_HPP
#define SYNCALL_CORE_CONFIG_HPP

#include <cstdint>
#include <fstream>
#include "runtime/rt.h"

// Host-side AI Core count detection for SYNCALL ST.
//
// run_st.py -v a3 always builds with SOC_VERSION=Ascend910B1 and never tells the
// binary which physical chip it runs on, so the launch block count must be decided
// at runtime. We must not call ascendc interfaces here, only cce/runtime (rt.h).
//
// Source of truth (in order):
//   1) syncall_core_cfg.txt written by gen_data.py (runs before the binary in run_st.py);
//   2) rtGetAiCoreCount() runtime fallback (requires the runtime to be ready);
//   3) Ascend910B1 full-die defaults.
namespace syncall_cfg {

struct CoreConfig {
    int32_t aic; // Cube (AI Core) count: 910B1=24, 910B4=20
    int32_t aiv; // Vector core count: 910B1=48, 910B4=40 (= 2 * aic on 910B)
};

inline bool ReadCfgFile(CoreConfig& cfg)
{
    const char* candidates[] = {"syncall_core_cfg.txt", "../syncall_core_cfg.txt"};
    for (const char* path : candidates) {
        std::ifstream file(path);
        if (!file.is_open()) {
            continue;
        }
        int32_t aic = 0;
        int32_t aiv = 0;
        if ((file >> aic >> aiv) && aic > 0 && aiv > 0) {
            cfg.aic = aic;
            cfg.aiv = aiv;
            return true;
        }
    }
    return false;
}

inline CoreConfig GetCoreConfig()
{
    CoreConfig cfg{24, 48};
    if (ReadCfgFile(cfg)) {
        return cfg;
    }
    uint32_t aiCoreCnt = 0;
    if (rtGetAiCoreCount(&aiCoreCnt) == RT_ERROR_NONE && aiCoreCnt > 0) {
        cfg.aic = static_cast<int32_t>(aiCoreCnt);
        cfg.aiv = static_cast<int32_t>(aiCoreCnt) * 2;
    }
    return cfg;
}

} // namespace syncall_cfg

#endif // SYNCALL_CORE_CONFIG_HPP
