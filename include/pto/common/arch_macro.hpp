/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/
// Implementation of interface adaptation layer for device-side and cloud-side compatibility
#ifndef ARCH_MACRO_HPP
#define ARCH_MACRO_HPP

#if __NPU_ARCH__ == 2201
#define PTO_NPU_ARCH_A2A3
#elif (__NPU_ARCH__ == 3101) || (__NPU_ARCH__ == 3510)
#define PTO_NPU_ARCH_A5
#elif __NPU_ARCH__ == 3113
#define PTO_COMM_NOT_SUPPORTED
#define PTO_NPU_ARCH_KIRIN9030
#elif __NPU_ARCH__ == 3003
#define PTO_COMM_NOT_SUPPORTED
#define PTO_NPU_ARCH_KIRINX90
#endif

#if defined(PTO_NPU_ARCH_KIRIN9030) || defined(PTO_NPU_ARCH_KIRINX90)
#define __tf__
#define __in__
#define __out__
#define __cce_get_tile_ptr
#endif
#endif // ARCH_MACRO_HPP