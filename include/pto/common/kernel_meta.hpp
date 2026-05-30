/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_COMMON_KERNEL_META_HPP
#define PTO_COMMON_KERNEL_META_HPP

#include <stdint.h>

namespace pto {
namespace detail {
constexpr uint16_t PTO_META_F_TYPE_KTYPE = 1;
constexpr uint16_t PTO_META_F_TYPE_MIX_TASK_RATION = 3;
constexpr uint32_t PTO_META_K_TYPE_AIC_ONLY = 1;
constexpr uint32_t PTO_META_K_TYPE_MIX_AIC_MAIN = 4;
constexpr uint32_t PTO_META_K_TYPE_MIX_AIV_MAIN = 5;

struct PtoMetaBaseTlv {
    uint16_t type;
    uint16_t len;
};

struct PtoMetaKType {
    PtoMetaBaseTlv head;
    uint32_t ktype;
};

struct PtoMetaMixCoreType {
    PtoMetaBaseTlv head;
    uint16_t taskRation0;
    uint16_t taskRation1;
};

struct PtoMetaFunLevelMixCoreType {
    PtoMetaKType ktypeMeta;
    PtoMetaMixCoreType mixCoreType;
};
} // namespace detail
} // namespace pto

#define PTO_DETAIL_CONCAT_INNER(a, b) a##b
#define PTO_DETAIL_CONCAT(a, b) PTO_DETAIL_CONCAT_INNER(a, b)
#define PTO_SYNCALL_AIV_KERNEL_META(kernelName)                                                                      \
    static const ::pto::detail::PtoMetaFunLevelMixCoreType PTO_DETAIL_CONCAT(g_pto_syncall_aiv_meta_, __COUNTER__)   \
        __attribute__((used, section(".ascend.meta." #kernelName))) = {                                              \
            {{::pto::detail::PTO_META_F_TYPE_KTYPE, sizeof(uint32_t)}, ::pto::detail::PTO_META_K_TYPE_MIX_AIV_MAIN}, \
            {{::pto::detail::PTO_META_F_TYPE_MIX_TASK_RATION, sizeof(uint32_t)}, 0, 1}}

#define PTO_SYNCALL_MIX_AIC_KERNEL_META(kernelName, aicRatio, aivRatio)                                                \
    static const ::pto::detail::PtoMetaFunLevelMixCoreType PTO_DETAIL_CONCAT(g_pto_syncall_mix_aic_meta_, __COUNTER__) \
        __attribute__((used, section(".ascend.meta." #kernelName))) = {                                                \
            {{::pto::detail::PTO_META_F_TYPE_KTYPE, sizeof(uint32_t)}, ::pto::detail::PTO_META_K_TYPE_MIX_AIC_MAIN},   \
            {{::pto::detail::PTO_META_F_TYPE_MIX_TASK_RATION, sizeof(uint32_t)}, aicRatio, aivRatio}}

#define PTO_SYNCALL_AIC_KERNEL_META(kernelName)                                                                    \
    static const ::pto::detail::PtoMetaFunLevelMixCoreType PTO_DETAIL_CONCAT(g_pto_syncall_aic_meta_, __COUNTER__) \
        __attribute__((used, section(".ascend.meta." #kernelName))) = {                                            \
            {{::pto::detail::PTO_META_F_TYPE_KTYPE, sizeof(uint32_t)}, ::pto::detail::PTO_META_K_TYPE_AIC_ONLY},   \
            {{::pto::detail::PTO_META_F_TYPE_MIX_TASK_RATION, sizeof(uint32_t)}, 1, 0}}

#endif
